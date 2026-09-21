"""End-to-end smoke test against the real Laya model (downloads the checkpoint
on first run). Usage: .venv/bin/python tests/smoke.py [backend] [model]"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tools  # noqa: E402

backend_arg = sys.argv[1] if len(sys.argv) > 1 else None
model_arg = sys.argv[2] if len(sys.argv) > 2 else None

extra = {}
if backend_arg:
    extra["backend"] = backend_arg
if model_arg:
    extra["model"] = model_arg

print("== laya_status ==")
print(tools.handle_status({}))
print()

print("== laya_decide: choice + score + noul in one call ==")
out = tools.handle_decide({
    "state": "I was billed twice this month and support never replied. I want my money back.",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this support request?",
            "criteria": {
                "billing": "invoices, charges, refunds, payments",
                "technical": "bugs, crashes, outages",
                "sales": "upgrades, quotes, new plans",
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this request?",
            "criteria": ["low", "medium", "high", "critical"],
        },
        "refund_requested": {
            "type": "noul",
            "instructions": "Does the customer explicitly ask for money back?",
        },
    },
    **extra,
})
parsed = json.loads(out)
print(json.dumps(parsed, indent=2)[:2000])
assert parsed["success"], "decide call failed"

print()
print("== laya_decide: triage preset ==")
out2 = tools.handle_decide({"state": "App crashes on launch after update.", "preset": "triage", **extra})
parsed2 = json.loads(out2)
print(json.dumps(parsed2, indent=2)[:1500])

print()
print("== /laya slash command ==")
print(tools.handle_slash("noul; Is the user angry?; This is ridiculous, fix it NOW"))
print()
print("SMOKE OK" if parsed["success"] else "SMOKE FAILED")
