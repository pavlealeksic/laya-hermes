---
name: laya-decisions
description: "Fast local typed decisions (choice/score/boolean) with the Laya System-1 model — use instead of in-LLM reasoning for routing, triage, gating, and moderation."
version: 1.0.0
author: pavlealeksic
license: Apache-2.0
platforms: [macos, linux, windows]
metadata:
  hermes:
    tags: [laya, decisions, routing, classification, triage, moderation, local]
---

# Laya Typed Decisions

Laya is a local, non-generative "System 1" decision model. Given a **state** plus
**typed questions**, it returns calibrated probabilities in one forward pass
(~5–15 ms on Apple Silicon, fully offline). It does **not** generate text.

## When to use `laya_decide`

Prefer it over reasoning in-LLM whenever the task is a *decision*, not a *composition*:

- **Routing** — which tool, model, queue, or workflow should handle this?
- **Triage** — department, priority, SLA for a ticket or message.
- **Gating** — yes/no checks before acting ("is this destructive?", "is this about billing?").
- **Scoring** — ordinal levels (urgency low→critical, sentiment, confidence rubrics).
- **Moderation / guardrails** — jailbreak or abuse screening (preset: `guard`, `moderation`).
- **Escalation** — the result's `action.act_probability` signals "act now" vs "escalate".

If a task needs explanation, code, or long text — do that yourself; Laya only decides.

## Question types

- `choice` — pick among labeled options; needs `criteria` (list or `{label: description}`).
- `score` — ordinal rubric; `criteria` is the ordered list of levels (low → high).
- `noul` — boolean; returns `P(true)`. `criteria` not needed.

Multiple questions in one call are batched in a single forward pass — ask everything
at once.

## Example

```json
{
  "state": "I was billed twice this month and nobody replied to my emails!",
  "questions": {
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"billing": "invoices, charges, refunds", "technical": "bugs, outages", "sales": "upgrades, quotes"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["low", "medium", "high", "critical"]},
    "refund_requested": {"type": "noul", "instructions": "Does the customer ask for money back?"}
  }
}
```

Read `answers.<name>.choice` / `.score` / `.noul` plus per-option `probabilities`
and `confidence`. Low confidence → fall back to your own reasoning.

## Built-in presets

Pass `preset` instead of `questions`: `router` (small vs frontier model),
`guard` (jailbreak/prompt-injection), `moderation`, `triage` (support tickets).

## Writing good questions

- 2–8 clear, mutually exclusive options; descriptive instructions beat bare labels.
- Keep the state focused — the token budget (~512 English / ~1024 multilingual)
  is shared by state + question + options.
- Known limits: weak above ~20 options; `score` is the weakest primitive (prefer
  `choice` when levels are few); no vision, no arithmetic, no long-document reasoning.

## Operations

- Unsure which backend/model is active or whether Laya is installed? Call `laya_status`.
- Quick ad-hoc checks from chat: `/laya help` shows the slash-command forms;
  `/laya setup` verifies/installs the backend; `/laya stats` shows usage metrics.
- The plugin may also run automatically in the background when the user enabled it:
  `LAYA_ROUTING_HINT=1` injects complexity hints before LLM calls, and
  `LAYA_FILTER_OUTPUT=1` truncates large *successful* tool outputs Laya judges disposable
  (a `[laya: truncated …]` marker appears in the output — rerun the command if you need
  the full text).
