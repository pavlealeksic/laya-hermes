"""Laya-powered context engine for Hermes Agent.

Design ported from ``hermes-jev-compact`` (TheEpTic, MIT) — same seam (override
``_prune_old_tool_results`` on the built-in ContextCompressor), same decision ladder
(keep / drop_result / drop_call), same commit gates (validity + minimum reduction +
deterministic fallback). The difference: judgments come from a LOCAL Laya model
instead of the hosted Jev API, so no transcript leaves the machine and each verdict
is free. Because Laya's context is ~1024 tokens (vs Jev's 25k), the state is built
per candidate unit (goal + recent tail + the call under judgment) rather than from
the whole transcript.

Safety: the proactive ``prune_tool_results_only`` path is untouched (deterministic),
and any exception or failed gate falls back to the built-in prune — worst case is
exactly stock Hermes behavior.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, List, Optional, Tuple

try:
    from . import backend, config, metrics
except ImportError:  # standalone import (tests, smoke scripts)
    import backend  # type: ignore
    import config  # type: ignore
    import metrics  # type: ignore

logger = logging.getLogger(__name__)

try:  # real base class inside Hermes; minimal stand-in for standalone tests
    from agent.context_compressor import ContextCompressor as _Base
except Exception:  # pragma: no cover - exercised only outside Hermes

    class _Base:  # type: ignore[no-redef]
        def __init__(self, model: str, **kwargs: Any) -> None:
            self.model = model
            self.quiet_mode = kwargs.get("quiet_mode", False)

        def _prune_boundary(self, messages, protect_tail_count, protect_tail_tokens=None):
            return max(1, len(messages) - max(1, protect_tail_count))

        def _prune_old_tool_results(self, messages, protect_tail_count,
                                    protect_tail_tokens=None, min_prune_chars=200):
            return list(messages), 0

        def prune_tool_results_only(self, messages, current_tokens=None):
            return messages, 0


_STATE_CONTEXT = (
    "A coding assistant conversation is being compacted to free context. Judge whether "
    "one earlier tool call, or its full output, still needs to stay in the history "
    "verbatim. Whatever is dropped is gone; re-running a tool costs time and may have "
    "side effects, so prefer keeping outputs the assistant will likely need again."
)

_ERROR_MARKERS = ("error", "failed", "failure", "traceback", "exception")

_QUESTION_KEEP_CALL = (
    "Knowing this tool call was made, with its input, still matters for what the "
    "assistant does next."
)
_QUESTION_KEEP_RESULT = (
    "The assistant is likely to need the exact contents of this tool output again, "
    "and dropping them would lose information."
)

_TRUNCATION_MARKER = (
    "[laya-compaction truncated {removed} chars of this tool result{error_note}; "
    "re-run the tool if needed]"
)


def _tc_get(tc: Any, key: str) -> Any:
    """Read a key from a dict- or object-shaped tool_call entry."""
    if isinstance(tc, dict):
        if key in ("name", "arguments"):
            fn = tc.get("function")
            return fn.get(key) if isinstance(fn, dict) else None
        return tc.get(key)
    fn = getattr(tc, "function", None)
    if key in ("name", "arguments"):
        return getattr(fn, key, None) if fn is not None else None
    return getattr(tc, key, None)


def _text_of(msg: Dict[str, Any]) -> str:
    content = msg.get("content")
    return content if isinstance(content, str) else ""


class _Candidate:
    __slots__ = ("call_id", "tool", "arguments", "call_idx", "result_idx",
                 "result_text", "is_error")

    def __init__(self, call_id, tool, arguments, call_idx, result_idx, result_text):
        self.call_id = call_id
        self.tool = tool
        self.arguments = arguments
        self.call_idx = call_idx
        self.result_idx = result_idx
        self.result_text = result_text
        head = result_text[:2000].lower()
        self.is_error = any(m in head for m in _ERROR_MARKERS)


def _collect_candidates(messages: List[Dict[str, Any]], boundary: int,
                        min_result_chars: int) -> List[_Candidate]:
    """Pair assistant tool_calls with their tool result rows before the boundary."""
    calls: Dict[str, Tuple[int, str, str]] = {}  # id -> (index, tool, arguments)
    results: Dict[str, Tuple[int, str]] = {}     # id -> (index, text)
    duplicates: set = set()
    for i, msg in enumerate(messages[:max(0, boundary)]):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "assistant":
            for tc in msg.get("tool_calls") or []:
                call_id, tool = _tc_get(tc, "id"), _tc_get(tc, "name")
                args = _tc_get(tc, "arguments")
                if not isinstance(call_id, str) or not isinstance(tool, str):
                    continue
                if not isinstance(args, str):
                    args = str(args) if args is not None else ""
                if call_id in calls:
                    duplicates.add(call_id)  # ambiguous address — fail closed
                else:
                    calls[call_id] = (i, tool, args)
        elif role == "tool":
            call_id = msg.get("tool_call_id")
            text = _text_of(msg)
            if not isinstance(call_id, str) or not text:
                continue
            if call_id in results:
                duplicates.add(call_id)
            else:
                results[call_id] = (i, text)
    out = []
    for call_id, (call_idx, tool, args) in calls.items():
        if call_id in duplicates or call_id not in results:
            continue
        result_idx, text = results[call_id]
        if result_idx <= call_idx or len(text) < min_result_chars:
            continue
        out.append(_Candidate(call_id, tool, args, call_idx, result_idx, text))
    return out


def _redact(text: str) -> str:
    """Best-effort secret redaction via the host; failure -> empty (never raw)."""
    try:
        from agent.redact import redact  # type: ignore

        return redact(text, force=True)
    except Exception:
        try:
            from agent import redact as redact_mod  # type: ignore

            fn = getattr(redact_mod, "redact", None)
            return fn(text, force=True) if callable(fn) else ""
        except Exception:
            return ""


def _valid_openai_sequence(messages: List[Dict[str, Any]]) -> bool:
    """Every tool row references an earlier call; no dup ids; no orphan calls."""
    seen_calls: Dict[str, int] = {}
    answered: set = set()
    for msg in messages:
        if not isinstance(msg, dict):
            return False
        role = msg.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            return False
        if role == "assistant":
            for tc in msg.get("tool_calls") or []:
                call_id = _tc_get(tc, "id")
                if not isinstance(call_id, str) or call_id in seen_calls:
                    return False
                seen_calls[call_id] = 1
        elif role == "tool":
            call_id = msg.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in seen_calls:
                return False
            if call_id in answered:
                return False
            answered.add(call_id)
    return answered == set(seen_calls)


def _total_chars(messages: List[Dict[str, Any]]) -> int:
    return sum(len(_text_of(m)) for m in messages if isinstance(m, dict))


class LayaContextCompressor(_Base):
    """ContextCompressor whose phase-1 prune is targeted by local Laya decisions."""

    @property
    def name(self) -> str:
        return "laya"

    def __init__(self, model: str = "laya", **kwargs: Any) -> None:
        base_params = set(inspect.signature(_Base.__init__).parameters) - {"self"}
        super().__init__(model, **{k: v for k, v in kwargs.items() if k in base_params})
        self.laya_calls = 0            # Laya inferences made for compaction
        self.laya_pruned_units = 0     # units truncated or dropped
        self.laya_kept_units = 0
        self.laya_fallbacks = 0        # times the built-in prune ran instead

    def __deepcopy__(self, memo: Dict[int, Any]) -> "LayaContextCompressor":
        import copy

        cls = type(self)
        new = cls.__new__(cls)
        memo[id(self)] = new
        for key, value in self.__dict__.items():
            try:
                setattr(new, key, copy.deepcopy(value, memo))
            except Exception:
                setattr(new, key, value)  # locks/db handles: share, never fail the copy
        return new

    # -- proactive path stays deterministic (documented no-LLM) --------------------

    def prune_tool_results_only(self, messages, current_tokens=None):
        out, n = super().prune_tool_results_only(messages, current_tokens=current_tokens)
        return list(out), int(n)

    # -- the seam -----------------------------------------------------------------

    def _prune_old_tool_results(self, messages, protect_tail_count,
                                protect_tail_tokens=None, min_prune_chars=200):
        if protect_tail_tokens is None:
            # Proactive path calls us with no token budget — stay deterministic there.
            return super()._prune_old_tool_results(
                messages, protect_tail_count, protect_tail_tokens, min_prune_chars)
        applied = None
        try:
            applied = self._laya_prune(messages, protect_tail_count,
                                       protect_tail_tokens, min_prune_chars)
        except Exception as exc:
            logger.info("laya compaction aborted (%s); built-in prune", exc)
        if applied is None:
            self.laya_fallbacks += 1
            metrics.record_compaction(0, 0, fallback=True)
            return super()._prune_old_tool_results(
                messages, protect_tail_count, protect_tail_tokens, min_prune_chars)
        return applied

    # -- internals -----------------------------------------------------------------

    def _cancelled(self) -> bool:
        check = getattr(self, "_compression_cancelled_check", None)
        try:
            return bool(check and check())
        except Exception:
            return False

    def _build_state(self, messages: List[Dict[str, Any]], c: _Candidate) -> Dict[str, Any]:
        goal = ""
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                text = _text_of(msg).strip()
                if text:
                    goal = text[:300]
                    break
        tail = []
        for msg in messages[-4:]:
            if isinstance(msg, dict):
                tail.append({"role": msg.get("role", "?"), "text": _text_of(msg)[:120]})
        excerpt = _redact(c.result_text[: int(config.get_value("result_excerpt_chars"))])
        return {
            "context": _STATE_CONTEXT,
            "goal": goal,
            "tail": tail,
            "call": {
                "id": c.call_id, "tool": c.tool, "input": c.arguments[:200],
                "result": f"{'error' if c.is_error else 'ok'}, {len(c.result_text)} chars"
                          + (f", head: {excerpt}" if excerpt else ""),
            },
        }

    def _judge(self, messages: List[Dict[str, Any]], c: _Candidate) -> Tuple[float, float]:
        """One Laya call per candidate, both questions in a single forward pass."""
        questions = {
            "keep_call": {"type": "noul", "instructions": _QUESTION_KEEP_CALL},
            "keep_result": {"type": "noul", "instructions": _QUESTION_KEEP_RESULT},
        }
        result, _, _, _ = backend.predict(self._build_state(messages, c), questions)
        self.laya_calls += 1
        answers = result.get("answers") or {}
        keep_call = (answers.get("keep_call") or {}).get("noul")
        keep_result = (answers.get("keep_result") or {}).get("noul")
        for value in (keep_call, keep_result):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"laya returned no usable probability: {value!r}")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"laya probability out of range: {value!r}")
        return float(keep_call), float(keep_result)

    def _laya_prune(self, messages, protect_tail_count, protect_tail_tokens,
                    min_prune_chars) -> Optional[Tuple[List[Dict[str, Any]], int]]:
        boundary = self._prune_boundary(list(messages), protect_tail_count, protect_tail_tokens)
        candidates = _collect_candidates(messages, boundary,
                                         int(config.get_value("min_result_chars")))
        if not candidates:
            return None
        keep_bar = float(config.get_value("keep_threshold"))
        error_bar = float(config.get_value("error_keep_threshold"))
        head_chars = int(config.get_value("truncate_head_chars"))

        decisions: Dict[str, str] = {}  # call_id -> keep | drop_result | drop_call
        for c in candidates:
            if self._cancelled():
                return None
            keep_call, keep_result = self._judge(messages, c)
            bar = error_bar if c.is_error else keep_bar
            if keep_result >= bar:
                decisions[c.call_id] = "keep"
            elif keep_call >= bar:
                decisions[c.call_id] = "drop_result"
            else:
                decisions[c.call_id] = "drop_call"

        applied, units = self._apply(messages, decisions, head_chars)
        # Commit gates: progress, validity, minimum reduction.
        if units == 0 or applied == list(messages):
            return None
        if not _valid_openai_sequence(applied):
            logger.info("laya compaction invalid transcript; built-in prune")
            return None
        before, after = _total_chars(messages), _total_chars(applied)
        if before <= 0:
            return None
        ratio = (before - after) / before
        if ratio < float(config.get_value("min_reduction_ratio")):
            logger.info("laya compaction reduction %.2f under gate; built-in prune", ratio)
            return None
        self.laya_pruned_units += units
        self.laya_kept_units += sum(1 for d in decisions.values() if d == "keep")
        metrics.record_compaction(units, before - after)
        if not getattr(self, "quiet_mode", False):
            logger.info("laya compaction: %d units pruned, %d kept, %.0f%% smaller",
                        units, self.laya_kept_units, ratio * 100)
        return applied, units

    @staticmethod
    def _apply(messages, decisions: Dict[str, str], head_chars: int):
        drop_ids = {cid for cid, d in decisions.items() if d == "drop_call"}
        truncate_ids = {cid for cid, d in decisions.items() if d == "drop_result"}
        out: List[Dict[str, Any]] = []
        units = 0
        for msg in messages:
            if not isinstance(msg, dict):
                out.append(msg)
                continue
            role = msg.get("role")
            if role == "tool" and msg.get("tool_call_id") in drop_ids:
                units += 1
                continue  # row removed; matching call stripped below
            if role == "tool" and msg.get("tool_call_id") in truncate_ids:
                text = _text_of(msg)
                removed = max(0, len(text) - head_chars)
                if removed > 120:
                    marker = _TRUNCATION_MARKER.format(
                        removed=removed,
                        error_note=" (error)" if _is_error_text(text) else "")
                    out.append({**msg, "content": text[:head_chars] + marker})
                    units += 1
                    continue
                out.append(msg)
                continue
            if role == "assistant" and msg.get("tool_calls"):
                kept = [tc for tc in msg["tool_calls"]
                        if _tc_get(tc, "id") not in drop_ids]
                if len(kept) != len(msg["tool_calls"]):
                    if not kept and not _text_of(msg).strip():
                        continue  # emptied assistant row goes with its calls
                    out.append({**msg, "tool_calls": kept})
                    continue
            out.append(msg)
        return out, units


def _is_error_text(text: str) -> bool:
    head = text[:2000].lower()
    return any(m in head for m in _ERROR_MARKERS)
