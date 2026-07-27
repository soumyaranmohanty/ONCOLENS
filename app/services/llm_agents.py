"""Streamlit-facing wrappers for local LLM ReAct agents."""

from __future__ import annotations

from typing import Any


def llm_available() -> bool:
    """True when langchain/ollama/langgraph optional deps are installed."""
    try:
        import langchain_ollama  # noqa: F401
        import langgraph  # noqa: F401

        return True
    except ImportError:
        return False


def generate_llm_clinical_report(
    patient: dict[str, Any],
    predictions: list[dict[str, Any]],
    *,
    target: str,
) -> str:
    from llm.agents.clinical_report_agent import generate_clinical_report

    return generate_clinical_report(patient, predictions, target=target)


def generate_llm_treatment_text(
    patient: dict[str, Any],
    predictions: list[dict[str, Any]],
    *,
    target: str,
) -> str:
    from llm.agents.treatment_agent import generate_treatment_with_agent

    return generate_treatment_with_agent(patient, predictions, target=target)


def generate_llm_assistant_reply(
    user_message: str,
    *,
    chat_history: list[dict[str, str]] | None = None,
    patient: dict[str, Any] | None = None,
    last_prediction: dict[str, Any] | None = None,
    default_target: str = "OS_STATUS",
    prediction_history: list[dict[str, Any]] | None = None,
) -> str:
    from llm.agents.assistant_agent import generate_assistant_reply

    return generate_assistant_reply(
        user_message,
        chat_history=chat_history,
        patient=patient,
        last_prediction=last_prediction,
        default_target=default_target,
        prediction_history=prediction_history,
    )
