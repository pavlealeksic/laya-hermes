"""Backend detection and lazy model loading for the Laya Hermes plugin.

Three backends, selected by the ``LAYA_BACKEND`` env var (``auto`` default):

- ``mlx``    — ``pip install laya-mlx``    (Apple Silicon, macOS 14+; best Mac default)
- ``coreml`` — ``pip install laya-coreml`` (Apple Silicon, macOS 15+; lowest latency/energy
               on short decisions, ANE bundles cap input at ~96 tokens)
- ``torch``  — ``pip install laya``        (upstream PyTorch implementation, CUDA/CPU)

Model aliases (``LAYA_MODEL`` env var or per-call ``model`` arg):
``english`` (421M, 512 tok), ``multilingual`` (322M, 100+ languages, 1024 tok — default),
``typed-decisions`` (421M fine-tuned for the typed-decision workflows, 1024 tok).
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import platform
import sys
import threading
import time
from typing import Any, Dict, Optional, Tuple

try:
    from . import config as _config
except ImportError:  # standalone import (tests, smoke scripts)
    import config as _config  # type: ignore

logger = logging.getLogger(__name__)

BACKENDS = ("mlx", "coreml", "torch")

_MODULES = {"mlx": "laya_mlx", "coreml": "laya_coreml", "torch": "laya"}

_INSTALL_HINTS = {
    "mlx": "pip install laya-mlx   # Apple Silicon, macOS 14+, Python 3.11+",
    "coreml": "pip install laya-coreml   # Apple Silicon, macOS 15+, Python 3.11-3.13",
    "torch": "pip install laya   # PyTorch backend, CUDA or CPU",
}

# alias -> per-backend Hugging Face checkpoint (torch uses repo + optional subfolder)
_CHECKPOINTS: Dict[str, Dict[str, Any]] = {
    "english": {
        "mlx": "aac6fef/laya-mlx",
        "coreml": "aac6fef/laya-coreml",
        "torch": ("convaiinnovations/laya", None),
    },
    "multilingual": {
        "mlx": "aac6fef/laya-multilingual-mlx",
        "coreml": "aac6fef/laya-multilingual-coreml",
        "torch": ("convaiinnovations/laya", "multilingual"),
    },
    "typed-decisions": {
        "mlx": "aac6fef/laya-typed-decisions-mlx",
        "coreml": "aac6fef/laya-typed-decisions-coreml",
        "torch": ("convaiinnovations/laya", "typed-decisions"),
    },
}

DEFAULT_MODEL = "multilingual"

PRESETS = ("router", "guard", "moderation", "triage")


class BackendUnavailableError(RuntimeError):
    """Raised when the selected backend package is not installed."""


_agents: Dict[Tuple[str, str], Any] = {}
_lock = threading.Lock()


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _module_available(backend: str) -> bool:
    try:
        return importlib.util.find_spec(_MODULES[backend]) is not None
    except (ImportError, ValueError):
        return False


def available_backends() -> Dict[str, bool]:
    return {b: _module_available(b) for b in BACKENDS}


def detect_backend() -> str:
    """Resolve the backend to use. Honors LAYA_BACKEND; 'auto' prefers MLX on
    Apple Silicon (full context + batching), then Core ML, then PyTorch."""
    forced = str(_config.get_value("backend")).strip().lower()
    if forced in BACKENDS:
        return forced
    if _is_apple_silicon():
        for candidate in ("mlx", "coreml", "torch"):
            if _module_available(candidate):
                return candidate
        return "mlx"  # nothing installed; report the recommended install hint
    return "torch"


def default_model() -> str:
    return str(_config.get_value("model")).strip().lower()


def _checkpoint(alias: str, backend: str) -> Any:
    if alias not in _CHECKPOINTS:
        raise ValueError(
            f"Unknown Laya model alias {alias!r}; valid: {', '.join(sorted(_CHECKPOINTS))}"
        )
    ref = _CHECKPOINTS[alias][backend]
    if backend == "coreml" and _config.get_value("coreml_ane"):
        # Neural Engine bundles: fastest and most energy-efficient, but capped at
        # ~96 total tokens (question + options + state).
        ref = f"{ref}-ane"
    return ref


_BACKEND_PACKAGES = {"mlx": "laya-mlx", "coreml": "laya-coreml", "torch": "laya"}

# Guard against concurrent pip installs when several tool calls race on first use.
_install_lock = threading.Lock()


def _auto_install_enabled() -> bool:
    return bool(_config.get_value("auto_install"))


def auto_install(backend: str) -> Tuple[bool, str]:
    """pip-install the backend package into the current Python environment.

    Returns (ok, message). No-op when the module is already importable.
    """
    if _module_available(backend):
        return True, f"{_MODULES[backend]} already installed"
    package = _BACKEND_PACKAGES[backend]
    if not _auto_install_enabled():
        return False, (
            f"auto-install disabled (LAYA_AUTO_INSTALL=0); "
            f"install manually: {_INSTALL_HINTS[backend]}"
        )
    import subprocess

    with _install_lock:
        if _module_available(backend):  # another thread beat us to it
            return True, f"{_MODULES[backend]} already installed"
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pip", "install", package],
                capture_output=True, text=True, timeout=600,
            )
        except Exception as exc:
            return False, f"pip install {package} failed to run: {exc}"
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()
            detail = tail[-1] if tail else f"exit code {proc.returncode}"
            return False, f"pip install {package} failed: {detail}"
        importlib.invalidate_caches()
        if _module_available(backend):
            logger.info("laya: auto-installed %s into %s", package, sys.executable)
            return True, f"installed {package}"
        return False, (
            f"pip installed {package} but {_MODULES[backend]} is still not importable; "
            f"restart Hermes or install manually: {_INSTALL_HINTS[backend]}"
        )


def _import_backend_module(backend: str):
    module_name = _MODULES[backend]
    if not _module_available(backend):
        ok, message = auto_install(backend)
        if not ok:
            raise BackendUnavailableError(
                f"Laya backend {backend!r} is not available. {message}"
            )
    try:
        if backend == "torch":
            # Upstream laya deadlocks on import when TensorFlow is also installed.
            os.environ.setdefault("USE_TF", "0")
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise BackendUnavailableError(
            f"Laya backend {backend!r} failed to import ({exc}). "
            f"Try reinstalling: {_INSTALL_HINTS[backend]}"
        ) from exc


def get_agent(model: Optional[str] = None, backend: Optional[str] = None) -> Tuple[Any, str, str]:
    """Load (and cache) a Laya agent. Returns (agent, backend, model_alias)."""
    backend = (backend or detect_backend()).strip().lower()
    if backend not in BACKENDS:
        raise ValueError(f"Unknown Laya backend {backend!r}; valid: {', '.join(BACKENDS)}")
    alias = (model or default_model()).strip().lower()
    key = (backend, alias)
    with _lock:
        if key in _agents:
            return _agents[key], backend, alias
        mod = _import_backend_module(backend)
        ref = _checkpoint(alias, backend)
        if backend == "mlx":
            agent = mod.load(ref, dtype=str(_config.get_value("dtype")))
        elif backend == "coreml":
            agent = mod.load(ref)
        else:  # torch
            repo, subfolder = ref
            agent = mod.load(repo, subfolder=subfolder) if subfolder else mod.load(repo)
        _agents[key] = agent
        return agent, backend, alias


def loaded_agents() -> Dict[str, str]:
    """Map of 'backend/model' -> repr for status reporting."""
    with _lock:
        return {f"{b}/{m}": type(a).__name__ for (b, m), a in _agents.items()}


def get_preset(name: str, backend: Optional[str] = None) -> Dict[str, Any]:
    """Return one of Laya's built-in question presets (router/guard/moderation/triage)."""
    name = name.strip().lower()
    if name not in PRESETS:
        raise ValueError(f"Unknown preset {name!r}; valid: {', '.join(PRESETS)}")
    mod = _import_backend_module((backend or detect_backend()).strip().lower())
    fn = getattr(mod, f"{name}_questions", None)
    if fn is None:
        raise BackendUnavailableError(
            f"Backend module {mod.__name__!r} does not expose {name}_questions(); "
            f"pass explicit questions instead."
        )
    return fn()


def predict(
    state: Any,
    questions: Dict[str, Any],
    model: Optional[str] = None,
    backend: Optional[str] = None,
) -> Tuple[Dict[str, Any], float, str, str]:
    """Run one typed-decision call. Returns (raw_result, latency_ms, backend, model)."""
    agent, backend, alias = get_agent(model=model, backend=backend)
    start = time.perf_counter()
    result = agent.predict(state, questions)
    latency_ms = round((time.perf_counter() - start) * 1000.0, 2)
    if not isinstance(result, dict):
        result = {"answers": result}
    try:
        from . import metrics
    except ImportError:  # standalone import (tests, smoke scripts)
        import metrics  # type: ignore
    metrics.record_decision(latency_ms)
    return result, latency_ms, backend, alias
