"""Virtual AI Assistant — scoped to ONCOLENS / oncology."""

from __future__ import annotations

import re
from typing import Any

from llm.prompts import OUT_OF_SCOPE_REPLY

# Fast pre-check for clearly off-topic queries (agent also enforces scope via system prompt).
_OFF_TOPIC_PATTERN = re.compile(
    r"\b("
    r"weather|forecast|temperature|rain|snow|"
    r"cricket|football|soccer|basketball|nba|nfl|world cup|"
    r"python code|javascript|programming|write code|debug|"
    r"recipe|cooking|movie|netflix|song|music|celebrity|"
    r"stock market|crypto|bitcoin|politics|election|"
    r"homework|math problem|calculate \d|"
    r"capital of|president of|who won"
    r")\b",
    re.IGNORECASE,
)

_IN_SCOPE_PATTERN = re.compile(
    r"\b("
    r"oncolens|oncology|oncolog|cancer|cancerous|tumor|tumour|carcinoma|luad|lung|"
    r"adenocarcinoma|biomarker|mutation|mutant|gene|expression|histopath|"
    r"predict|prediction|model|modality|multimodal|survival|prognos|"
    r"tp53|egfr|kras|alk|stk11|tmb|stage|staging|"
    r"\bos\b|\bpfs\b|progression|deceased|alive|"
    r"roc|auc|metric|confidence|risk|"
    r"treatment|therapy|chemo|immuno|targeted|drug|nccn|"
    r"patient|tcga|clinical|patholog|"
    r"explain this prediction|explain prediction|"
    r"this project|this app|this tool"
    r")\b",
    re.IGNORECASE,
)


def _obviously_out_of_scope(message: str) -> bool:
    """Return True when the message clearly has no oncology/ONCOLENS relevance."""
    text = message.strip()
    if not text:
        return True
    if _OFF_TOPIC_PATTERN.search(text):
        return True
    if _IN_SCOPE_PATTERN.search(text):
        return False
    # Very short greetings are allowed through so the agent can respond in scope.
    if text.lower() in {"hi", "hello", "hey", "thanks", "thank you", "bye"}:
        return False
    return True


def answer_question(
    message: str,
    context: dict[str, Any] | None = None,
    *,
    chat_history: list[dict[str, str]] | None = None,
) -> str:
    """Answer a user message using the ONCOLENS assistant."""
    from app.services.llm_agents import generate_llm_assistant_reply, llm_available

    context = context or {}

    if not llm_available():
        return "The AI assistant is not available. Please try again later."

    if _obviously_out_of_scope(message):
        return OUT_OF_SCOPE_REPLY

    try:
        return generate_llm_assistant_reply(
            message,
            chat_history=chat_history,
            patient=context.get("patient"),
            last_prediction=context.get("last_prediction"),
            default_target=context.get("default_target", "OS_STATUS"),
            prediction_history=context.get("prediction_history"),
        )
    except Exception as exc:
        return f"The assistant encountered an error: {exc}. Please try again later."
