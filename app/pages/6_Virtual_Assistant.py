"""Virtual AI Assistant page."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.components.cards import compact_patient_chip
from app.components.layout import ai_status_banner, empty_state, render_disclaimer, section
from app.services.assistant import answer_question
from app.services.history import init_history
from app.services.llm_agents import llm_available


def _assistant_context() -> dict:
    init_history()
    return {
        "patient": st.session_state.get("last_patient"),
        "last_prediction": st.session_state.get("last_prediction"),
        "default_target": st.session_state.get("pred_target", "OS_STATUS"),
        "prediction_history": st.session_state.get("prediction_history", []),
    }


def _last_pred_line() -> str | None:
    pred = st.session_state.get("last_prediction")
    if pred and pred.get("available", True):
        return f"{pred.get('model')} → {pred.get('predicted_label_name')}"
    return None


def render() -> None:
    section(
        "Virtual AI Assistant",
        "Ask about ONCOLENS, LUAD biomarkers, model predictions, and patient session data.",
    )

    ai_status_banner(llm_available())

    init_history()
    patient = st.session_state.get("last_patient")
    pid = patient.get("patient_id") if patient else None
    compact_patient_chip(pid, _last_pred_line())

    if "chat_messages" not in st.session_state:
        greeting = (
            "Hello! I'm the ONCOLENS assistant. Ask about TP53, PFS, ROC-AUC, modalities, "
            "or say **Explain this prediction** after running a model."
        )
        if pid:
            greeting = (
                f"Hello! Active patient is **`{pid}`**. Ask about biomarkers, predictions, "
                "or ONCOLENS models."
            )
        st.session_state.chat_messages = [{"role": "assistant", "content": greeting}]

    if len(st.session_state.chat_messages) <= 1:
        empty_state(
            "Start a conversation",
            "Ask about biomarkers, predictions, or model metrics.",
        )

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

    st.markdown("**Suggested questions**")
    suggestions = [
        "Explain TP53 mutation",
        "What is PFS?",
        "Explain this prediction",
        "Explain ROC-AUC",
        "What are the modalities?",
    ]
    s_cols = st.columns(3)
    for i, s in enumerate(suggestions):
        with s_cols[i % 3]:
            if st.button(s, key=f"suggest_{s}", use_container_width=True):
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
