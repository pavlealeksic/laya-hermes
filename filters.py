"""Opt-in output filtering hooks for the Laya Hermes plugin (default OFF).

Enabled with ``LAYA_FILTER_OUTPUT=1``. When a *successful* command/tool produces
very large output (``LAYA_FILTER_MIN_CHARS``, default 6000), Laya decides whether
the output looks disposable (progress bars, install spam, boilerplate). Only when
Laya is confident it is NOT needed (P(needed) < threshold) is the output truncated
to head + tail with a marker. Failures (nonzero returncode, error status) and
ambiguous cases always pass through untouched.

Laya is non-generative — this truncates, it never summarizes. Both hooks are
fail-open: any exception returns None so Hermes keeps the original output.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

try:
    from . import backend, metrics
except ImportError:  # standalone import (tests, smoke scripts)
    import backend  # type: ignore
    import metrics  # type: ignore

logger = logging.getLogger(__name__)

_HEAD_CHARS = 1000
_TAIL_CHARS = 500
# Truncate only when P(output is needed) is below this.
_NEEDED_THRESHOLD = 0.3

_QUESTIONS = {
    "needed": {
        "type": "noul",
        "instructions": (
            "Does this output contain errors, failures, stack traces, test results, "
            "diffs, or other information an AI coding assistant would need for its "
            "next step? Answer true if in doubt."
        ),
    }
}


def filter_enabled() -> bool:
    return os.environ.get("LAYA_FILTER_OUTPUT", "0").strip() == "1"


def _min_chars() -> int:
    try:
        return int(os.environ.get("LAYA_FILTER_MIN_CHARS", "6000"))
    except ValueError:
        return 6000


def _maybe_truncate(output: Any, context: str) -> Optional[str]:
    """Return a truncated replacement, or None to leave the output untouched."""
    if not isinstance(output, str) or len(output) <= _min_chars():
        return None
    probe = output[:_HEAD_CHARS] + "\n...\n" + output[-_TAIL_CHARS:]
    result, _, _, _ = backend.predict(
        f"Command/tool output excerpt ({context}):\n{probe}", _QUESTIONS
    )
    answer = (result.get("answers") or {}).get("needed") or {}
    p_needed = answer.get("noul")
    if not isinstance(p_needed, (int, float)) or p_needed >= _NEEDED_THRESHOLD:
        return None
    removed = len(output) - _HEAD_CHARS - _TAIL_CHARS
    metrics.record_truncation(removed)
    return (
        output[:_HEAD_CHARS]
        + f"\n\n[laya: truncated {removed} chars of low-value output; "
        f"P(needed)={p_needed:.2f} — rerun the command if you need the rest]\n\n"
        + output[-_TAIL_CHARS:]
    )


def transform_terminal_output(
    command: str = "",
    output: Any = None,
    returncode: Any = None,
    task_id: str = "",
    **kwargs: Any,
) -> Optional[str]:
    """Hook: transform_terminal_output. Only touches successful, huge outputs."""
    try:
        if not filter_enabled() or returncode != 0:
            return None
        return _maybe_truncate(output, f"command: {str(command)[:200]}")
    except Exception as exc:
        metrics.record_error()
        logger.debug("laya terminal filter skipped: %s", exc)
        return None


def transform_tool_result(
    tool_name: str = "",
    args: Any = None,
    result: Any = None,
    status: str = "",
    **kwargs: Any,
) -> Optional[str]:
    """Hook: transform_tool_result. Only touches successful, huge results.

    Terminal output is already handled by transform_terminal_output upstream,
    so the terminal tool is skipped here.
    """
    try:
        if not filter_enabled() or tool_name == "terminal" or status != "success":
            return None
        return _maybe_truncate(result, f"tool: {tool_name}")
    except Exception as exc:
        metrics.record_error()
        logger.debug("laya tool-result filter skipped: %s", exc)
        return None
