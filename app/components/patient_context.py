"""Active patient context banner and sidebar — shown on patient-centric pages."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import streamlit as st

from app.services.history import init_history

# Pages that show model/cohort metrics only — no session patient UI.
MODEL_ONLY_URL_PATHS = frozenset({"/", "/dashboard", "/model-analytics", "/research"})


def is_model_only_view() -> bool:
    """True on cohort/model pages with no session patient UI."""
    path = urlparse(st.context.url or "").path.rstrip("/") or "/"
    return path in MODEL_ONLY_URL_PATHS


def _modalities(patient: dict[str, Any]) -> list[str]:
    skip = {"patient_id", "from_feature_store"}
    return sorted(k for k in patient if k not in skip)


def get_active_patient() -> dict[str, Any] | None:
    init_history()
    return st.session_state.get("last_patient")


def get_active_patient_id() -> str | None:
    patient = get_active_patient()
    if not patient:
        return None
    return str(patient.get("patient_id", "")) or None


def render_patient_context_sidebar() -> None:
    """Compact patient summary for the sidebar (call from entry script)."""
    if is_model_only_view():
        return

    init_history()
    patient = get_active_patient()
    pred = st.session_state.get("last_prediction")

    st.markdown("---")
    st.markdown("**Session patient**")
    if patient is None:
        st.caption("No patient selected")
        st.caption("Load one on Patient Prediction")
        return

    pid = patient.get("patient_id", "Unknown")
    st.markdown(f"`{pid}`")
    source = "TCGA Demo" if patient.get("from_feature_store") else "Upload"
    st.caption(f"Source: {source}")
    st.caption(f"Modalities: {', '.join(_modalities(patient)) or 'None'}")
    if pred and pred.get("available", True):
        st.caption(
            f"Last: {pred.get('model')} → {pred.get('predicted_label_name')} "
            f"({pred.get('target')})"
        )


def render_patient_context_banner(
    *,
    cohort_level: bool = False,
    require_patient: bool = False,
) -> dict[str, Any] | None:
    """
    Prominent patient context block at the top of each page.

    cohort_level=True adds a note that page content is model/cohort metrics, not patient-specific.
    require_patient=True shows a warning and returns None when no patient is loaded.
    """
    init_history()
    patient = get_active_patient()
    pred = st.session_state.get("last_prediction")

    if patient is None:
        st.warning(
            "**No patient selected.** Open **Patient Prediction**, choose a TCGA demo patient "
            "or upload CSVs, then click **Load** / **Run Prediction**."
        )
        if require_patient:
            return None
        return None

    pid = patient.get("patient_id", "Unknown")
    source = "TCGA Demo" if patient.get("from_feature_store") else "Uploaded CSV"
    mods = _modalities(patient)

    st.markdown("### Active patient")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patient ID", pid)
    c2.metric("Data source", source)
    c3.metric("Modalities loaded", len(mods))
    if pred and pred.get("available", True):
        c4.metric(
            "Last prediction",
            f"{pred.get('predicted_label_name', '—')}",
            delta=f"{pred.get('model')} · {pred.get('target')}",
            delta_color="off",
        )
    else:
        c4.metric("Last prediction", "Not run yet")

    with st.expander("Patient & prediction details", expanded=False):
        st.markdown(f"**Patient ID:** `{pid}`")
        st.markdown(f"**Data source:** {source}")
        st.markdown(f"**Modalities:** {', '.join(mods) if mods else 'None'}")
        if pred:
            if pred.get("available", True):
                st.markdown(
                    f"**Latest result:** {pred.get('model')} · **{pred.get('target')}** → "
                    f"**{pred.get('predicted_label_name')}** "
                    f"(confidence {pred.get('confidence', 0):.1%}, risk **{pred.get('risk_band')}**)"
                )
            else:
                st.markdown(f"**Latest attempt:** {pred.get('model')} — unavailable ({pred.get('reason', '')})")
        else:
            st.markdown("**Latest result:** None — run a prediction on the Patient Prediction page.")

    if cohort_level:
        st.info(
            f"Charts and tables on this page are **cohort-level model metrics**. "
            f"The active patient above (`{pid}`) is your current session context for reports and recommendations."
        )

    st.divider()
    return patient
