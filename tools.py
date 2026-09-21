"""Tool handlers for the Laya Hermes plugin.

Hermes handler contract: ``def handler(args: dict, **kwargs) -> str`` — always
return a JSON string, never raise.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

try:
    from . import backend, metrics
except ImportError:  # standalone import (tests, smoke scripts)
    import backend  # type: ignore
    import metrics  # type: ignore

_QUESTION_TYPES = ("choice", "score", "noul")


def _dump(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, default=str, ensure_ascii=False)


def _err(message: str, **extra: Any) -> str:
    return _dump({"success": False, "error": message, **extra})


def _validate_questions(questions: Any) -> Optional[str]:
    if not isinstance(questions, dict) or not questions:
        return "'questions' must be a non-empty object mapping names to question specs."
    for name, spec in questions.items():
        if not isinstance(spec, dict):
            return f"Question {name!r} must be an object with 'type' and 'instructions'."
        qtype = spec.get("type")
        if qtype not in _QUESTION_TYPES:
            return (
                f"Question {name!r} has invalid type {qtype!r}; "
                f"valid: {', '.join(_QUESTION_TYPES)}."
            )
        if qtype in ("choice", "score") and not spec.get("criteria"):
            return f"Question {name!r} of type {qtype!r} needs 'criteria' (the options/levels)."
    return None


def _run_decide(state: Any, questions: Dict[str, Any], model: Optional[str],
                backend_name: Optional[str]) -> str:
    result, latency_ms, used_backend, used_model = backend.predict(
        state, questions, model=model, backend=backend_name
    )
    return _dump({
        "success": True,
        "answers": result.get("answers", result),
        "action": result.get("action"),
        "meta": {
            "backend": used_backend,
            "model": used_model,
            "latency_ms": latency_ms,
            "routing": result.get("routing"),
        },
    })


def handle_decide(args: Dict[str, Any], **kwargs: Any) -> str:
    """laya_decide tool handler."""
    metrics.record_decision(0.0, feature="tool_calls")
    try:
        state = args.get("state")
        if state is None or (isinstance(state, str) and not state.strip()):
            return _err("'state' is required (text, JSON object, or conversation list).")

        questions = args.get("questions")
        preset = args.get("preset")
        model = args.get("model")
        backend_name = args.get("backend")
        if backend_name in (None, "", "auto"):
            backend_name = None

        if preset and questions:
            return _err("Pass either 'questions' or 'preset', not both.")
        if not preset and questions is None:
            return _err("Pass 'questions' (custom typed questions) or 'preset' (router/guard/moderation/triage).")

        if preset:
            questions = backend.get_preset(str(preset), backend=backend_name)
        else:
            problem = _validate_questions(questions)
            if problem:
                return _err(problem)

        return _run_decide(state, questions, model, backend_name)
    except backend.BackendUnavailableError as exc:
        return _err(str(exc), hint="Run laya_status for backend diagnostics.")
    except Exception as exc:  # never raise across the tool boundary
        return _err(f"{type(exc).__name__}: {exc}")


def handle_status(args: Dict[str, Any], **kwargs: Any) -> str:
    """laya_status tool handler."""
    try:
        detected = backend.detect_backend()
        availability = backend.available_backends()
        installed = [b for b, ok in availability.items() if ok]
        return _dump({
            "success": True,
            "ready": bool(installed),
            "detected_backend": detected,
            "backends_installed": availability,
            "default_model": backend.default_model(),
            "loaded_models": backend.loaded_agents(),
            "config": {
                "LAYA_BACKEND": os.environ.get("LAYA_BACKEND", "auto"),
                "LAYA_MODEL": os.environ.get("LAYA_MODEL", backend.DEFAULT_MODEL),
                "LAYA_DTYPE": os.environ.get("LAYA_DTYPE", "float16"),
                "LAYA_COREML_ANE": os.environ.get("LAYA_COREML_ANE", "0"),
                "LAYA_ROUTING_HINT": os.environ.get("LAYA_ROUTING_HINT", "0"),
                "LAYA_AUTO_INSTALL": os.environ.get("LAYA_AUTO_INSTALL", "1"),
                "LAYA_FILTER_OUTPUT": os.environ.get("LAYA_FILTER_OUTPUT", "0"),
            },
            "metrics": metrics.snapshot(),
            "install_hints": (
                {} if installed
                else {b: backend._INSTALL_HINTS[b] for b in backend.BACKENDS}
            ),
        })
    except Exception as exc:
        return _err(f"{type(exc).__name__}: {exc}")


def handle_setup() -> str:
    """Ensure the detected backend package is installed (auto-installs if needed)."""
    detected = backend.detect_backend()
    ok, message = backend.auto_install(detected)
    return _dump({
        "success": ok,
        "backend": detected,
        "message": message,
        "note": "Model checkpoints download from Hugging Face on first use.",
    })


def handle_slash(raw_args: str) -> str:
    """``/laya`` slash command.

    Forms:
      /laya status | stats | setup
      /laya preset <router|guard|moderation|triage>; <state>
      /laya noul; <yes/no question>; <state>
      /laya choice|score; <instructions>; <opt1, opt2, ...>; <state>
    """
    raw = (raw_args or "").strip()
    if not raw or raw == "help":
        return _dump({
            "success": True,
            "usage": [
                "/laya status",
                "/laya stats",
                "/laya setup",
                "/laya preset triage; <text>",
                "/laya noul; Does the user ask for a refund?; <text>",
                "/laya choice; Which department?; billing, technical, other; <text>",
                "/laya score; How urgent?; low, medium, high, critical; <text>",
            ],
        })
    if raw == "status":
        return handle_status({})
    if raw == "stats":
        return metrics.render()
    if raw == "setup":
        return handle_setup()

    parts = [p.strip() for p in raw.split(";")]
    head = parts[0]

    if head == "preset" and len(parts) >= 2:
        preset_and_rest = parts[0]
        # "preset triage" arrives in the first segment
        tokens = preset_and_rest.split(None, 1)
        if len(tokens) != 2:
            return _err("Usage: /laya preset <router|guard|moderation|triage>; <state>")
        preset_name = tokens[1].strip()
        state = "; ".join(parts[1:])
        return handle_decide({"preset": preset_name, "state": state})

    qtype = head.lower()
    if qtype not in _QUESTION_TYPES:
        return _err(f"Unknown form. First segment must be 'status', 'preset', or one of "
                    f"{', '.join(_QUESTION_TYPES)}. Try /laya help.")

    if qtype == "noul":
        if len(parts) < 3:
            return _err("Usage: /laya noul; <yes/no question>; <state>")
        questions = {"answer": {"type": "noul", "instructions": parts[1]}}
        state = "; ".join(parts[2:])
    else:
        if len(parts) < 4:
            return _err(f"Usage: /laya {qtype}; <instructions>; <opt1, opt2, ...>; <state>")
        options = [o.strip() for o in parts[2].split(",") if o.strip()]
        if len(options) < 2:
            return _err("Provide at least two comma-separated options.")
        questions = {"answer": {"type": qtype, "instructions": parts[1], "criteria": options}}
        state = "; ".join(parts[3:])

    return handle_decide({"state": state, "questions": questions})
