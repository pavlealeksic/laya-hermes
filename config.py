"""Plugin settings, readable and writable from inside Hermes.

Storage: ``plugins.entries.laya.settings.<key>`` in Hermes' config.yaml via
``ctx.get_config`` / ``ctx.set_config`` (declared in plugin.yaml's ``config_schema``,
so Hermes' UI can render them). Precedence: env var > Hermes setting > default.
Reads are live — changes via ``/laya set`` apply immediately, no restart.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

# key -> {env, type, default, description}
SETTINGS: Dict[str, Dict[str, Any]] = {
    "backend": {
        "env": "LAYA_BACKEND", "type": "str", "default": "auto",
        "description": "Decision backend: auto, mlx, coreml, or torch.",
    },
    "model": {
        "env": "LAYA_MODEL", "type": "str", "default": "multilingual",
        "description": "Checkpoint alias: english, multilingual, or typed-decisions.",
    },
    "dtype": {
        "env": "LAYA_DTYPE", "type": "str", "default": "float16",
        "description": "MLX dtype: float16 or float32.",
    },
    "coreml_ane": {
        "env": "LAYA_COREML_ANE", "type": "bool", "default": False,
        "description": "Use Neural Engine Core ML bundles (short inputs only).",
    },
    "routing_hint": {
        "env": "LAYA_ROUTING_HINT", "type": "bool", "default": False,
        "description": "pre_llm_call hook: inject a hint when Laya is confident a request is simple.",
    },
    "auto_install": {
        "env": "LAYA_AUTO_INSTALL", "type": "bool", "default": True,
        "description": "Self-install the backend package (pip) on first use.",
    },
    "filter_output": {
        "env": "LAYA_FILTER_OUTPUT", "type": "bool", "default": False,
        "description": "Truncate large *successful* tool/terminal outputs Laya judges disposable.",
    },
    "filter_min_chars": {
        "env": "LAYA_FILTER_MIN_CHARS", "type": "int", "default": 6000,
        "description": "Minimum output size before filtering is considered.",
    },
}

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}

_ctx: Any = None


def init(ctx: Any) -> None:
    """Capture the plugin context at register() time so settings work in-session."""
    global _ctx
    _ctx = ctx


def _coerce(key: str, raw: Any) -> Any:
    spec = SETTINGS[key]
    stype = spec["type"]
    if stype == "bool":
        if isinstance(raw, bool):
            return raw
        text = str(raw).strip().lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
        raise ValueError(f"{key} must be a boolean (true/false), got {raw!r}")
    if stype == "int":
        if isinstance(raw, bool):
            raise ValueError(f"{key} must be an integer, got {raw!r}")
        try:
            return int(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be an integer, got {raw!r}") from None
    return str(raw)


def get(key: str) -> Tuple[Any, str]:
    """Return (value, source) where source is 'env' | 'settings' | 'default'."""
    spec = SETTINGS[key]
    raw = os.environ.get(spec["env"])
    if raw is not None and raw.strip() != "":
        return _coerce(key, raw), "env"
    if _ctx is not None:
        try:
            value = _ctx.get_config(key, None)
            if value is not None:
                return _coerce(key, value), "settings"
        except Exception:
            pass
    return spec["default"], "default"


def get_value(key: str) -> Any:
    return get(key)[0]


def set_value(key: str, raw: Any) -> Tuple[Any, str]:
    """Persist a setting via the Hermes plugin context. Returns (value, message)."""
    if key not in SETTINGS:
        raise KeyError(f"Unknown setting {key!r}; valid: {', '.join(sorted(SETTINGS))}")
    value = _coerce(key, raw)
    if _ctx is None:
        raise RuntimeError("settings can only be changed inside Hermes (no plugin context)")
    _ctx.set_config(key, value)
    return value, f"{key} = {value!r} (saved to Hermes config; applies immediately)"


def describe() -> List[Dict[str, Any]]:
    """Effective value + source + description for every setting."""
    out = []
    for key, spec in SETTINGS.items():
        value, source = get(key)
        out.append({
            "key": key, "value": value, "source": source,
            "default": spec["default"], "env": spec["env"],
            "description": spec["description"],
        })
    return out
