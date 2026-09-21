"""Tool schemas for the Laya Hermes plugin (what the LLM sees)."""

DECIDE_SCHEMA = {
    "name": "laya_decide",
    "description": (
        "Fast local typed decision using the Laya 'System 1' model — returns calibrated "
        "probabilities in a single forward pass (~5-15 ms on Apple Silicon), fully offline, "
        "no LLM tokens. Use for classification, routing, triage, urgency scoring, yes/no "
        "gating, moderation and escalation checks instead of reasoning it out yourself. "
        "Three question types: 'choice' (pick among labeled options), 'score' (ordinal rubric "
        "level), 'noul' (boolean, returns P(true)). Provide either 'questions' or a built-in "
        "'preset'. Cannot generate text; keep the state under ~500 tokens (English) / ~1000 "
        "(multilingual) and choices under ~20 options."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "state": {
                "description": "The situation to judge: plain text, a JSON object, or a list of conversation messages.",
            },
            "questions": {
                "type": "object",
                "description": (
                    "Map of question name -> {type: choice|score|noul, instructions: string, "
                    "criteria: [option labels] or {label: description}}. 'criteria' is required "
                    "for choice/score (the options or rubric levels), optional for noul. "
                    "Multiple questions are batched in one forward pass."
                ),
            },
            "preset": {
                "type": "string",
                "enum": ["router", "guard", "moderation", "triage"],
                "description": "Use a built-in question set instead of 'questions': router (small vs frontier model), guard (jailbreak/prompt-injection), moderation, triage (support tickets).",
            },
            "model": {
                "type": "string",
                "enum": ["english", "multilingual", "typed-decisions"],
                "description": "Checkpoint alias. Default: env LAYA_MODEL or 'multilingual'.",
            },
            "backend": {
                "type": "string",
                "enum": ["auto", "mlx", "coreml", "torch"],
                "description": "Runtime backend. Default: env LAYA_BACKEND or auto (mlx on Apple Silicon).",
            },
        },
        "required": ["state"],
    },
}

STATUS_SCHEMA = {
    "name": "laya_status",
    "description": (
        "Report Laya plugin status: which backend was detected, which backend packages are "
        "installed, current config (env vars), and which models are already loaded in memory."
    ),
    "parameters": {"type": "object", "properties": {}},
}
