"""ReAct agent for the ONCOLENS Virtual AI Assistant."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from llm.agents.assistant_tools import build_assistant_tools
from llm.agents.base import build_react_agent, extract_agent_text
from llm.llm_utils import llm_gemma
from llm.prompts import ASSISTANT_SYSTEM_PROMPT

MAX_CHAT_HISTORY = 10


def _to_langchain_messages(chat_history: list[dict[str, str]]) -> list[BaseMessage]:
    """Convert Streamlit chat dicts to LangChain messages (bounded window)."""
    messages: list[BaseMessage] = []
    for msg in chat_history[-MAX_CHAT_HISTORY:]:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not content:
            continue
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def create_assistant_agent(
    *,
    patient: dict[str, Any] | None = None,
    last_prediction: dict[str, Any] | None = None,
    default_target: str = "OS_STATUS",
    prediction_history: list[dict[str, Any]] | None = None,
    llm: BaseChatModel | None = None,
):
    """Build a session-scoped ReAct assistant agent."""
    tools = build_assistant_tools(
        patient,
        last_prediction,
        default_target=default_target,
        prediction_history=prediction_history,
    )
    return build_react_agent(
        llm or llm_gemma,
        tools,
        system_prompt=ASSISTANT_SYSTEM_PROMPT,
    )


def generate_assistant_reply(
    user_message: str,
    *,
    chat_history: list[dict[str, str]] | None = None,
    patient: dict[str, Any] | None = None,
    last_prediction: dict[str, Any] | None = None,
    default_target: str = "OS_STATUS",
    prediction_history: list[dict[str, Any]] | None = None,
    llm: BaseChatModel | None = None,
) -> str:
    """Run the assistant ReAct agent for one chat turn."""
    agent = create_assistant_agent(
        patient=patient,
        last_prediction=last_prediction,
        default_target=default_target,
        prediction_history=prediction_history,
        llm=llm,
    )
    messages = _to_langchain_messages(chat_history or [])
    messages.append(HumanMessage(content=user_message))
    result = agent.invoke({"messages": messages})
    return extract_agent_text(result)
