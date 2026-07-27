"""Virtual AI Assistant page."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.components.metrics import render_disclaimer
from app.components.patient_context import (
    get_active_patient,
    get_active_patient_id,
    render_patient_context_banner,
)
from app.services.assistant import answer_question
from app.services.history import init_history
from app.services.llm_agents import llm_available


def _assistant_context() -> dict:
    init_history()
    return {
        "patient": get_active_patient(),
        "last_prediction": st.session_state.get("last_prediction"),
        "default_target": st.session_state.get("pred_target", "OS_STATUS"),
        "prediction_history": st.session_state.get("prediction_history", []),
    }


def render() -> None:
    st.title("Virtual AI Assistant")
    st.markdown(
        "Ask questions about **ONCOLENS**, LUAD biomarkers, model predictions, and patient session data."
    )

    if not llm_available():
        st.warning("The AI assistant is not available in this environment.")
    else:
        st.caption("Only ONCOLENS and cancer/oncology-related questions are answered.")

    init_history()
    render_patient_context_banner()

    pid = get_active_patient_id()
    if pid:
        st.caption(
            f"Active patient **`{pid}`** — ask about this patient's biomarkers, predictions, or models."
        )

    if "chat_messages" not in st.session_state:
        greeting = (
            "Hello! I'm the ONCOLENS assistant. Ask me about TP53, PFS, ROC-AUC, modalities, "
            "or say **Explain this prediction** after running a model."
        )
        if pid:
            greeting = (
                f"Hello! Active patient is **`{pid}`**. Ask me about biomarkers, predictions, "
                "or ONCOLENS models."
            )
        st.session_state.chat_messages = [{"role": "assistant", "content": greeting}]

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    prompt = st.chat_input("Ask an ONCOLENS or oncology question...")
    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.spinner("Thinking…"):
            response = answer_question(
                prompt,
                _assistant_context(),
                chat_history=st.session_state.chat_messages[:-1],
            )
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
            st.session_state.chat_messages.append({"role": "user", "content": s})
            with st.spinner("Thinking…"):
                response = answer_question(
                    s,
                    _assistant_context(),
                    chat_history=st.session_state.chat_messages[:-1],
                )
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
            st.rerun()

    render_disclaimer()


render()
