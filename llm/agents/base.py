"""Shared helpers for ONCOLENS ReAct agents."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.prebuilt import create_react_agent


def build_react_agent(
    llm: BaseChatModel,
    tools: list,
    *,
    system_prompt: str,
):
    """Create a LangGraph ReAct agent (tool-calling loop)."""
    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
    )


def extract_agent_text(result: dict[str, Any]) -> str:
    """Return the final assistant message content from an agent invoke result."""
    messages: list[BaseMessage] = result.get("messages", [])
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content:
            content = message.content
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts = [
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                joined = "\n".join(p for p in parts if p).strip()
                if joined:
                    return joined
    return "Agent did not produce a text response."
