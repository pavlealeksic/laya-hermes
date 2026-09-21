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
- **`laya_status` tool** — backend detection, installed packages, live settings, loaded models, metrics.
- **`/laya` slash command** — ad-hoc decisions plus `status`, `stats`, `setup`, `config`, `set`
  (try `/laya help`).
- **`laya:laya-decisions` skill** — teaches the agent when to prefer Laya over in-LLM reasoning.
- **Opt-in `pre_llm_call` routing hint** — with `routing_hint` on, Laya rates each user
  message's complexity and injects a short hint when it's confident the request is simple.
  Hint only; it never blocks or overrides your model.
- **Opt-in output filtering** — with `filter_output` on, Laya screens *successful* oversized
  tool/terminal outputs and truncates ones it is confident are disposable (install spam, progress
  bars) to head+tail with a marker. Failures and ambiguous output always pass through untouched.
  Laya doesn't generate text, so this truncates — it never summarizes.
- **Context engine (smart compaction)** — opt in with `hermes config set context.engine laya`
  then `/reset`. During context compression, Laya judges each stale tool call/result pair
  (keep / truncate / drop) instead of Hermes pruning by age alone — so a test failure from three
  turns ago survives while install spam is dropped. Fully local: unlike
  [hermes-jev-compact](https://pypi.org/project/hermes-jev-compact/) (which pioneered this
  design with the hosted Jev API), no transcript leaves the machine and each verdict is free.
  Safety: the proactive hot path stays deterministic, and any error, invalid transcript, or
  under-`min_reduction_ratio` pass falls back to the built-in prune — worst case is stock
  Hermes behavior. Caveat: Laya's ~1024-token window means per-unit judgments (goal + recent
  tail + the call), not whole-transcript reasoning.
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
Opt out with `/laya set auto_install false` and install manually, or run `/laya setup`
to trigger the install on demand.

For local development, clone this repo into `~/.hermes/plugins/laya/` and
`hermes plugins enable laya`.

## Configuration

All settings are adjustable **live from inside Hermes** — no restart needed:

```
/laya config                          # show every setting, its value, and its source
/laya set filter_output true          # enable output filtering immediately
/laya set routing_hint true           # enable pre-LLM-call complexity hints
/laya set model typed-decisions       # switch checkpoint (applies to next decision)
```

Settings persist in Hermes' `config.yaml` under `plugins.entries.laya.settings` and are
declared in the plugin's `config_schema`, so Hermes' settings UI can render them too.
Environment variables still work and **override** settings: precedence is
env var (`LAYA_*`) → Hermes setting → default.

| Key | Env var | Default | Meaning |
|---|---|---|---|
| `backend` | `LAYA_BACKEND` | `auto` | `auto` / `mlx` / `coreml` / `torch` |
| `model` | `LAYA_MODEL` | `multilingual` | `english` (421M, 512 tok), `multilingual` (322M, 100+ langs, 1024 tok), `typed-decisions` (fine-tuned) |
| `dtype` | `LAYA_DTYPE` | `float16` | MLX dtype (`float16` / `float32`) |
| `coreml_ane` | `LAYA_COREML_ANE` | `false` | use Neural Engine bundles (short inputs only) |
| `routing_hint` | `LAYA_ROUTING_HINT` | `false` | `pre_llm_call` complexity hint |
| `auto_install` | `LAYA_AUTO_INSTALL` | `true` | self-install the backend package on first use |
| `filter_output` | `LAYA_FILTER_OUTPUT` | `false` | truncate large successful tool/terminal outputs Laya judges disposable |
| `filter_min_chars` | `LAYA_FILTER_MIN_CHARS` | `6000` | minimum output size before filtering is considered |
| `keep_threshold` | `LAYA_KEEP_THRESHOLD` | `0.5` | compaction: keep-probability at/above this keeps the unit |
| `error_keep_threshold` | `LAYA_ERROR_KEEP_THRESHOLD` | `0.25` | compaction: lower keep bar for error results |
| `min_result_chars` | `LAYA_MIN_RESULT_CHARS` | `2000` | compaction: smaller tool results are never candidates |
| `result_excerpt_chars` | `LAYA_RESULT_EXCERPT_CHARS` | `300` | compaction: result head chars shown to Laya per unit |
| `truncate_head_chars` | `LAYA_TRUNCATE_HEAD_CHARS` | `300` | compaction: head kept when a result is truncated |
| `min_reduction_ratio` | `LAYA_MIN_REDUCTION_RATIO` | `0.10` | compaction: pass must shrink the transcript by this, else built-in prune runs |

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
