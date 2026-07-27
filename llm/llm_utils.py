"""Local LLM inference via Ollama — shared config for ONCOLENS agents."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from langchain_ollama import ChatOllama

DEFAULT_MODEL = os.getenv("ONCOLENS_LLM_MODEL", "gemma4:26b")
DEFAULT_BASE_URL = os.getenv("ONCOLENS_OLLAMA_URL", "http://localhost:11434")

DEFAULT_LLM_KWARGS: dict[str, Any] = {
    "temperature": 0,
    "top_p": 0.9,
    "top_k": 40,
    "num_predict": 2048,
    "repeat_penalty": 1.1,
}


@lru_cache(maxsize=4)
def get_llm(
    model: str | None = None,
    *,
    base_url: str | None = None,
    num_predict: int | None = None,
    **overrides: Any,
) -> ChatOllama:
    """Return a cached ChatOllama instance for agent inference."""
    kwargs = {**DEFAULT_LLM_KWARGS, **overrides}
    if num_predict is not None:
        kwargs["num_predict"] = num_predict

    return ChatOllama(
        model=model or DEFAULT_MODEL,
        base_url=base_url or DEFAULT_BASE_URL,
        **kwargs,
    )


# Primary model used by clinical-report and treatment ReAct agents.
llm_gemma = get_llm()
