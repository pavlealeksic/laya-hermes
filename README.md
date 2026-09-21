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
- **`laya_status` tool** — backend detection, installed packages, config, loaded models.
- **`/laya` slash command** — ad-hoc decisions from CLI or chat gateways (try `/laya help`).
- **`laya:laya-decisions` skill** — teaches the agent when to prefer Laya over in-LLM reasoning.
- **Opt-in `pre_llm_call` routing hint** — with `LAYA_ROUTING_HINT=1`, Laya rates each user
  message's complexity and injects a short hint when it's confident the request is simple.
  Hint only; it never blocks or overrides your model.

## Backends

| Backend | Package | Platform | Notes |
|---|---|---|---|
| `mlx` (default on Mac) | `laya-mlx` | Apple Silicon, macOS 14+ | Full 512/1024-token context, batching, FP16. |
| `coreml` | `laya-coreml` | Apple Silicon, macOS 15+ | Lowest latency/energy; no ML-framework deps. `LAYA_COREML_ANE=1` uses Neural Engine bundles (capped at ~96 total tokens). |
| `torch` | `laya` | CUDA / CPU, any OS | Upstream PyTorch implementation. |

`auto` (default) prefers `mlx` → `coreml` → `torch` on Apple Silicon, `torch` elsewhere.

## Install

1. Install a backend into the Hermes Python environment (pick one):

   ```bash
   ~/.hermes/hermes-agent/venv/bin/pip install laya-mlx      # recommended on Apple Silicon
   # or: .../pip install laya-coreml                        # Core ML / Neural Engine
   # or: .../pip install laya                               # PyTorch, CUDA/CPU
   ```

2. Install the plugin and enable it:

   ```bash
   hermes plugins install <owner>/laya-hermes
   hermes plugins enable laya
   ```

   For local development, clone this repo into `~/.hermes/plugins/laya/` instead.

The model checkpoint (~650 MB) downloads from Hugging Face on first use.

## Configuration (env vars, all optional)

| Var | Default | Meaning |
|---|---|---|
| `LAYA_BACKEND` | `auto` | `auto` / `mlx` / `coreml` / `torch` |
| `LAYA_MODEL` | `multilingual` | `english` (421M, 512 tok), `multilingual` (322M, 100+ langs, 1024 tok), `typed-decisions` (fine-tuned) |
| `LAYA_DTYPE` | `float16` | MLX dtype (`float16` / `float32`) |
| `LAYA_COREML_ANE` | `0` | `1` = use Neural Engine bundles (short inputs only) |
| `LAYA_ROUTING_HINT` | `0` | `1` = enable the `pre_llm_call` complexity hint |

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
