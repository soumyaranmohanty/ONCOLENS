"""Virtual AI Assistant page."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.components.metrics import render_disclaimer
from app.services.assistant import answer_question
from app.services.history import init_history


def render() -> None:
    st.title("Virtual AI Assistant")
    st.markdown("Ask questions about predictions, biomarkers, and model metrics.")

    init_history()

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "Hello! Ask me about TP53, PFS, ROC curves, or say 'explain this prediction'."}
        ]

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    prompt = st.chat_input("Ask a question...")
    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        context = {"last_prediction": st.session_state.get("last_prediction")}
        response = answer_question(prompt, context)
        st.session_state.chat_messages.append({"role": "assistant", "content": response})
        st.rerun()

    st.markdown("**Suggested questions:**")
    suggestions = [
        "Explain TP53 mutation",
        "What is PFS?",
        "Explain this prediction",
        "Explain ROC-AUC",
        "What are the modalities?",
    ]
    for s in suggestions:
        if st.button(s, key=f"suggest_{s}"):
            context = {"last_prediction": st.session_state.get("last_prediction")}
            response = answer_question(s, context)
            st.session_state.chat_messages.append({"role": "user", "content": s})
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
            st.rerun()

    render_disclaimer()


render()
