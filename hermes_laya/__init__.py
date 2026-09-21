"""Laya plugin for Hermes Agent.

Exposes the local Laya "System 1" typed-decision model as agent tools
(``laya_decide``, ``laya_status``), a ``/laya`` slash command (``help``,
``status``, ``stats``, ``setup``, ``config``, ``set``, decision forms), a bundled
``laya:laya-decisions`` skill, session metrics, and two hooks: a ``pre_llm_call``
routing hint and conservative output truncation. All hooks are gated by live
plugin settings (``/laya set <key> <value>`` — no restart needed); backend
packages self-install on first use unless ``auto_install`` is off.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from . import config, filters, schemas, tools

logger = logging.getLogger(__name__)

_ROUTING_HINT_QUESTIONS = {
    "complexity": {
        "type": "choice",
        "instructions": "How much reasoning effort does this user request need from an AI assistant?",
        "criteria": {
            "simple": "A short, routine request: lookup, small edit, simple question, formatting, quick command.",
            "complex": "Needs multi-step reasoning, design trade-offs, debugging, or long-form generation.",
        },
    }
}

_ROUTING_HINT_THRESHOLD = 0.8


def _routing_hint_hook(
    session_id: str = "",
    user_message: str = "",
    is_first_turn: bool = False,
    model: str = "",
    **kwargs: Any,
) -> Optional[dict]:
    """pre_llm_call hook, gated live on the ``routing_hint`` setting.

    Runs Laya's complexity check on the user message; when Laya is confident the
    request is simple, injects a short hint into the turn's user message. Never
    raises, never blocks — hint only, no model override.
    """
    try:
        if not config.get_value("routing_hint"):
            return None
        if not isinstance(user_message, str) or not user_message.strip():
            return None
        from . import backend

        result, _, _, _ = backend.predict(user_message, _ROUTING_HINT_QUESTIONS)
        answer = (result.get("answers") or {}).get("complexity") or {}
        confidence = answer.get("confidence") or 0.0
        if answer.get("choice") == "simple" and confidence >= _ROUTING_HINT_THRESHOLD:
            try:
                from . import metrics

                metrics.record_decision(0.0, feature="routing_hints")
            except Exception:
                pass
            return {"context": (
                f"[laya] Local decision model rates this request as simple "
                f"(confidence {confidence:.2f}). Prefer the most direct, minimal path."
            )}
    except Exception as exc:  # a hook must never break the turn
        logger.debug("laya routing hint skipped: %s", exc)
    return None


def register(ctx) -> None:
    config.init(ctx)
    ctx.register_tool(
        name="laya_decide",
        toolset="laya",
        schema=schemas.DECIDE_SCHEMA,
        handler=tools.handle_decide,
        description="Local System-1 typed decisions (choice/score/boolean) via Laya.",
        emoji="⚡",
    )
    ctx.register_tool(
        name="laya_status",
        toolset="laya",
        schema=schemas.STATUS_SCHEMA,
        handler=tools.handle_status,
        description="Laya backend/model status and diagnostics.",
        emoji="⚡",
    )
    ctx.register_command(
        "laya",
        handler=tools.handle_slash,
        description="Quick local decisions and settings via Laya (try /laya help).",
        args_hint="<type>; <question>; <state>",
    )
    skill_path = Path(__file__).parent / "SKILL.md"
    if skill_path.exists():
        ctx.register_skill(
            "laya-decisions",
            skill_path,
            description="When and how to use Laya for fast local typed decisions.",
        )
    # Hooks register unconditionally and gate on live settings at call time, so
    # `/laya set routing_hint true` / `set filter_output true` take effect without
    # a restart. When disabled the callbacks return None immediately.
    ctx.register_hook("pre_llm_call", _routing_hint_hook)
    ctx.register_hook("transform_terminal_output", filters.transform_terminal_output)
    ctx.register_hook("transform_tool_result", filters.transform_tool_result)
    # Context engine: opt-in via `hermes config set context.engine laya` (+ /reset).
    try:
        from . import engine as laya_engine

        ctx.register_context_engine(laya_engine.LayaContextCompressor())
    except Exception as exc:
        logger.warning("laya: context engine registration failed: %s", exc)
