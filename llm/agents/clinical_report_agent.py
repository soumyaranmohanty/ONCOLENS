"""ReAct agent for LLM-generated clinical reports."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from llm.agents.base import build_react_agent, extract_agent_text
from llm.agents.tools import build_patient_tools
from llm.llm_utils import llm_gemma
from llm.prompts import CLINICAL_REPORT_SYSTEM_PROMPT


def create_clinical_report_agent(
    patient: dict[str, Any],
    predictions: list[dict[str, Any]],
    *,
    target: str,
    llm: BaseChatModel | None = None,
):
    """Build a patient-scoped ReAct agent for clinical report generation."""
    tools = build_patient_tools(patient, predictions, target=target)
    return build_react_agent(
        llm or llm_gemma,
        tools,
        system_prompt=CLINICAL_REPORT_SYSTEM_PROMPT,
    )


def generate_clinical_report(
    patient: dict[str, Any],
    predictions: list[dict[str, Any]],
    *,
    target: str,
    llm: BaseChatModel | None = None,
) -> str:
    """Run the clinical report ReAct agent and return narrative report text."""
    agent = create_clinical_report_agent(
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
                    f"Generate a complete clinical report for patient {pid}. "
                    f"Use your tools to gather all available data first, then write the report.",
                )
            ]
        }
    )
    return extract_agent_text(result)
