"""Session prediction history."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
import streamlit as st


def init_history() -> None:
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []
    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None
    if "last_patient" not in st.session_state:
        st.session_state.last_patient = None
    if "active_patient_id" not in st.session_state:
        st.session_state.active_patient_id = None


def append_prediction(result: dict[str, Any]) -> None:
    init_history()
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "patient_id": result.get("patient_id"),
        "target": result.get("target"),
        "model": result.get("model"),
        "prediction": result.get("predicted_label_name"),
        "risk_band": result.get("risk_band"),
        "confidence": result.get("confidence"),
        "probability": result.get("probability"),
        "available": result.get("available", True),
    }
    st.session_state.prediction_history.insert(0, entry)
    if result.get("available", True):
        st.session_state.last_prediction = result


def set_last_patient(patient: dict[str, Any]) -> None:
    init_history()
    st.session_state.last_patient = patient
    st.session_state.active_patient_id = patient.get("patient_id")


def clear_patient() -> None:
    init_history()
    st.session_state.last_patient = None
    st.session_state.active_patient_id = None
    st.session_state.last_prediction = None


def history_dataframe() -> pd.DataFrame:
    init_history()
    if not st.session_state.prediction_history:
        return pd.DataFrame()
    return pd.DataFrame(st.session_state.prediction_history)


def clear_history() -> None:
    st.session_state.prediction_history = []
    st.session_state.last_prediction = None
    clear_patient()


def render_history(expanded: bool = False) -> None:
    df = history_dataframe()
    if df.empty:
        st.info("No predictions yet. Run a prediction from the Patient Prediction page.")
        return
    with st.expander("Prediction History", expanded=expanded):
        display = df.copy()
        if "patient_id" in display.columns:
            cols = ["patient_id", "timestamp", "target", "model", "prediction", "risk_band", "confidence", "probability"]
            display = display[[c for c in cols if c in display.columns]]
        st.dataframe(display, use_container_width=True, hide_index=True)
        if st.button("Clear history", key="clear_history_btn"):
            clear_history()
            st.rerun()
