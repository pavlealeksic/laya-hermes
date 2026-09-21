"""In-memory metrics for the Laya Hermes plugin.

Counters reset when Hermes restarts (v1 by design — no persistence).
All recording functions are no-throw; metrics must never break a turn.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict

_lock = threading.Lock()
_started_at = time.time()

_counters: Dict[str, int] = {
    "decisions": 0,           # total Laya inferences
    "tool_calls": 0,          # laya_decide tool invocations
    "routing_hints": 0,       # pre_llm_call hints injected
    "outputs_truncated": 0,   # filter hook truncations
    "chars_saved": 0,         # chars removed by filtering
    "errors": 0,              # swallowed failures across hooks/tools
    "compaction_runs": 0,     # committed Laya compaction passes
    "compaction_fallbacks": 0,  # compaction passes that fell back to built-in prune
    "compaction_units": 0,    # tool units truncated/dropped by compaction
    "compaction_chars_saved": 0,
}
_latency_ms_total = 0.0


def record_decision(latency_ms: float = 0.0, feature: str = "") -> None:
    global _latency_ms_total
    try:
        with _lock:
            _counters["decisions"] += 1
            _latency_ms_total += max(0.0, float(latency_ms))
            if feature in _counters:
                _counters[feature] += 1
    except Exception:
        pass


def record_truncation(chars_removed: int) -> None:
    try:
        with _lock:
            _counters["outputs_truncated"] += 1
            _counters["chars_saved"] += max(0, int(chars_removed))
    except Exception:
        pass


def record_error() -> None:
    try:
        with _lock:
            _counters["errors"] += 1
    except Exception:
        pass


def record_compaction(units: int, chars_saved: int, fallback: bool = False) -> None:
    try:
        with _lock:
            if fallback:
                _counters["compaction_fallbacks"] += 1
            else:
                _counters["compaction_runs"] += 1
                _counters["compaction_units"] += max(0, int(units))
                _counters["compaction_chars_saved"] += max(0, int(chars_saved))
    except Exception:
        pass


def snapshot() -> Dict[str, Any]:
    with _lock:
        counters = dict(_counters)
        latency_total = _latency_ms_total
    decisions = counters["decisions"]
    return {
        "decisions": decisions,
        "avg_latency_ms": round(latency_total / decisions, 2) if decisions else 0.0,
        "tool_calls": counters["tool_calls"],
        "routing_hints": counters["routing_hints"],
        "outputs_truncated": counters["outputs_truncated"],
        "chars_saved": counters["chars_saved"],
        "est_tokens_saved": (counters["chars_saved"] + counters["compaction_chars_saved"]) // 4,
        "errors": counters["errors"],
        "compaction_runs": counters["compaction_runs"],
        "compaction_fallbacks": counters["compaction_fallbacks"],
        "compaction_units": counters["compaction_units"],
        "compaction_chars_saved": counters["compaction_chars_saved"],
        "uptime_s": int(time.time() - _started_at),
    }


def render() -> str:
    s = snapshot()
    lines = [
        "Laya — since Hermes start",
        f"  decisions:            {s['decisions']}",
        f"  avg latency:          {s['avg_latency_ms']} ms",
        f"  tool calls:           {s['tool_calls']}",
        f"  routing hints:        {s['routing_hints']}",
        f"  outputs truncated:    {s['outputs_truncated']}",
        f"  compaction runs:      {s['compaction_runs']} ({s['compaction_fallbacks']} fallbacks)",
        f"  compaction units:     {s['compaction_units']}",
        f"  est. tokens saved:    {s['est_tokens_saved']}",
        f"  swallowed errors:     {s['errors']}",
    ]
    return "\n".join(lines)
