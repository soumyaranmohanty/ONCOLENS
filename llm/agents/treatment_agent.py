"""ReAct agent for LLM-generated treatment recommendations."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from llm.agents.base import build_react_agent, extract_agent_text
from llm.agents.tools import build_patient_tools
from llm.llm_utils import llm_gemma
from llm.prompts import TREATMENT_SYSTEM_PROMPT


def create_treatment_agent(
    patient: dict[str, Any],
    predictions: list[dict[str, Any]],
    *,
    target: str,
    llm: BaseChatModel | None = None,
):
    """Build a patient-scoped ReAct agent for treatment recommendations."""
    tools = build_patient_tools(patient, predictions, target=target)
    return build_react_agent(
        llm or llm_gemma,
        tools,
        system_prompt=TREATMENT_SYSTEM_PROMPT,
    )


def generate_treatment_with_agent(
    patient: dict[str, Any],
    predictions: list[dict[str, Any]],
    *,
    target: str,
    llm: BaseChatModel | None = None,
) -> str:
    """Run the treatment ReAct agent and return narrative recommendations."""
    agent = create_treatment_agent(
        patient,
        predictions,
        target=target,
        llm=llm,
    )
    pid = patient.get("patient_id", "Unknown")
    result = agent.invoke(
        {
            "messages": [
                (
                    "user",
                    f"Generate educational treatment recommendations for patient {pid}. "
                    f"Call your tools first, then produce all required sections.",
                )
            ]
        }
    )
    return extract_agent_text(result)
