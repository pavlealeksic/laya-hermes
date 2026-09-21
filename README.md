# laya-hermes

A [Hermes Agent](https://hermes-agent.nousresearch.com/) plugin that gives the agent access to
**[Laya](https://github.com/NandhaKishorM/laya)** — a local, non-generative "System 1" typed-decision
model. Laya answers `choice` / `score` / boolean (`noul`) questions with calibrated probabilities in
a single forward pass (~5–15 ms on Apple Silicon), fully offline. Use it for routing, triage, gating,
and moderation decisions instead of spending LLM tokens.

## What the plugin provides

- **`laya_decide` tool** — run typed questions against a state (text, JSON, or conversation list).
  Custom questions or built-in presets (`router`, `guard`, `moderation`, `triage`). Multiple
  questions are batched in one forward pass.
- **`laya_status` tool** — backend detection, installed packages, config, loaded models, metrics.
- **`/laya` slash command** — ad-hoc decisions plus `status`, `stats`, `setup` (try `/laya help`).
- **`laya:laya-decisions` skill** — teaches the agent when to prefer Laya over in-LLM reasoning.
- **Opt-in `pre_llm_call` routing hint** — with `LAYA_ROUTING_HINT=1`, Laya rates each user
  message's complexity and injects a short hint when it's confident the request is simple.
  Hint only; it never blocks or overrides your model.
- **Opt-in output filtering** — with `LAYA_FILTER_OUTPUT=1`, Laya screens *successful* oversized
  tool/terminal outputs and truncates ones it is confident are disposable (install spam, progress
  bars) to head+tail with a marker. Failures and ambiguous output always pass through untouched.
  Laya doesn't generate text, so this truncates — it never summarizes.
- **Session metrics** — `/laya stats` shows decisions, avg latency, truncations, and estimated
  tokens saved. In-memory; resets when Hermes restarts.

## Backends

| Backend | Package | Platform | Notes |
|---|---|---|---|
| `mlx` (default on Mac) | `laya-mlx` | Apple Silicon, macOS 14+ | Full 512/1024-token context, batching, FP16. |
| `coreml` | `laya-coreml` | Apple Silicon, macOS 15+ | Lowest latency/energy; no ML-framework deps. `LAYA_COREML_ANE=1` uses Neural Engine bundles (capped at ~96 total tokens). |
| `torch` | `laya` | CUDA / CPU, any OS | Upstream PyTorch implementation. |

`auto` (default) prefers `mlx` → `coreml` → `torch` on Apple Silicon, `torch` elsewhere.

## Install

```bash
hermes plugins install <owner>/laya-hermes --enable
```

That's it. The right backend package for your platform (`laya-mlx` on Apple Silicon)
**self-installs into Hermes' Python environment on first use** — no manual pip step.
The model checkpoint (~650 MB) then downloads from Hugging Face on the first decision.
Set `LAYA_AUTO_INSTALL=0` to opt out and install manually, or run `/laya setup` to
trigger the install on demand.

For local development, clone this repo into `~/.hermes/plugins/laya/` and
`hermes plugins enable laya`.

## Configuration (env vars, all optional)

| Var | Default | Meaning |
|---|---|---|
| `LAYA_BACKEND` | `auto` | `auto` / `mlx` / `coreml` / `torch` |
| `LAYA_MODEL` | `multilingual` | `english` (421M, 512 tok), `multilingual` (322M, 100+ langs, 1024 tok), `typed-decisions` (fine-tuned) |
| `LAYA_DTYPE` | `float16` | MLX dtype (`float16` / `float32`) |
| `LAYA_COREML_ANE` | `0` | `1` = use Neural Engine bundles (short inputs only) |
| `LAYA_ROUTING_HINT` | `0` | `1` = enable the `pre_llm_call` complexity hint |
| `LAYA_AUTO_INSTALL` | `1` | `0` = don't self-install the backend package; require manual pip install |
| `LAYA_FILTER_OUTPUT` | `0` | `1` = enable conservative truncation of large successful tool/terminal outputs |
| `LAYA_FILTER_MIN_CHARS` | `6000` | minimum output size before filtering is considered |

## Deliberately not included

- **Per-turn main-model routing** — Hermes v0.21 has no plugin seam for switching the main
  loop's model (`llm.model_override` covers only a plugin's own `ctx.llm` calls). The routing
  hint is the honest approximation until Hermes adds one.
- **Skill routing** — Hermes already progressive-discloses skills (compact index, load on
  demand); there is no skill-context bloat to fix.
- **Output summarization** — Laya is non-generative; filtering truncates, it can't rewrite.

## Example

Ask the agent something like *"use laya to triage this ticket: …"*, or call the tool shape directly:

```json
{
  "state": "I was billed twice this month and support never replied.",
  "questions": {
    "department": {"type": "choice", "instructions": "Which team handles this?",
                   "criteria": {"billing": "charges and refunds", "technical": "bugs and outages"}},
    "urgency": {"type": "score", "instructions": "How urgent?",
                "criteria": ["low", "medium", "high", "critical"]},
    "refund_requested": {"type": "noul", "instructions": "Does the customer ask for money back?"}
  }
}
```

→ `answers.department.choice = "billing"`, `answers.refund_requested.noul ≈ 0.9`, plus per-option
probabilities, confidence, and `action.act_probability`.

## Development

```bash
python3 -m unittest discover -s tests -v     # unit tests (stubbed backend, no download)
python3 -m venv .venv && .venv/bin/pip install laya-mlx
.venv/bin/python tests/smoke.py              # end-to-end against the real model
```

## Credits & license

Plugin code: Apache-2.0. Laya model and runtimes by
[Convai Innovations](https://huggingface.co/convaiinnovations/laya) (Apache-2.0);
MLX/Core ML ports by [mizorewww](https://github.com/mizorewww/laya-mlx).
Not affiliated with Nous Research, Convai Innovations, or Apple.
