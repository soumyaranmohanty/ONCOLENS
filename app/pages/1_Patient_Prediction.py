"""Patient Prediction page — predictions, clinical report, and treatment tabs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from app.components.cards import patient_summary_card
from app.components.layout import render_disclaimer, section, workflow_steps
from app.components.patient_workflow import (
    render_clinical_report_tab,
    render_predictions_tab,
    render_treatment_tab,
)
from app.config import TARGETS
from app.inference.patient import make_demo_patient, make_uploaded_patient
from app.inference.tables import demo_patient_ids
from app.services.histopathology_embed import (
    embedding_from_upload,
    histopathology_inference_available,
)
from app.services.history import init_history, set_last_patient


def render() -> None:
    init_history()

    section(
        "Patient Prediction",
        "Load a patient, run model predictions, and generate clinical reports and treatment guidance.",
    )

    patient = st.session_state.get("last_patient")
    has_results = bool(st.session_state.get("last_results"))
    active_step = 2 if has_results else (1 if patient else 0)
    workflow_steps(
        [("1", "Load patient"), ("2", "Choose target"), ("3", "Run analysis")],
        active_step,
    )

    col_target, _ = st.columns([1, 2])
    with col_target:
        target = st.selectbox("Target", TARGETS, key="pred_target")

    col_demo, col_upload = st.columns(2)

    with col_demo:
        with st.container(border=True):
            st.markdown("**TCGA Demo (test set)**")
            total_demo = demo_patient_ids(target)
            histo_demo = demo_patient_ids(target, require_modality="Histopathology")
            st.caption(
                f"{len(total_demo)} held-out test patients · "
                f"{len(histo_demo)} with histopathology embeddings"
            )
            histo_only = st.checkbox(
                "Only patients with histopathology",
                key="demo_histo_only",
                help="Required for Histopathology and 4-Modality predictions.",
            )
            ids = demo_patient_ids(
                target,
                require_modality="Histopathology" if histo_only else None,
            )
            if not ids:
                st.warning("No demo patients for this cohort.")
            else:
                patient_id = st.selectbox("Patient ID", ids, key="demo_patient")
                if st.session_state.get("loaded_demo_id") != patient_id:
                    set_last_patient(make_demo_patient(patient_id, target))
                    st.session_state.loaded_demo_id = patient_id
                if st.button("Reload demo patient", key="load_demo", use_container_width=True):
                    set_last_patient(make_demo_patient(patient_id, target))
                    st.session_state.loaded_demo_id = patient_id
                    st.rerun()

    with col_upload:
        with st.container(border=True):
            st.markdown("**Upload data**")
            pid = st.text_input("Patient ID", value="UPLOAD-001", key="upload_pid")
            expr_f = st.file_uploader("Expression (CSV)", type=["csv"], key="up_expr")
            mut_f = st.file_uploader("Mutation (CSV)", type=["csv"], key="up_mut")
            clin_f = st.file_uploader("Clinical (CSV)", type=["csv"], key="up_clin")
            histo_f = st.file_uploader(
                "Histopathology slide (optional)",
                type=["svs", "csv"],
                key="up_histo",
                help="Upload a diagnostic whole-slide image (.svs). "
                "Precomputed 2048-D embedding CSVs are also accepted.",
            )
            if not histopathology_inference_available():
                st.caption(
                    "`.svs` processing requires `uv sync --extra histopathology` "
                    "(torch, OpenSlide). CSV embeddings work without it."
                )
            if st.button("Build from uploads", key="build_upload", use_container_width=True):
                histo_df = None
                if histo_f is not None:
                    with st.spinner("Extracting histopathology features from slide…"):
                        try:
                            histo_df = embedding_from_upload(histo_f, pid)
                        except Exception as exc:
                            st.error(f"Histopathology processing failed: {exc}")
                            st.stop()
                set_last_patient(
                    make_uploaded_patient(
                        pid,
                        expression=pd.read_csv(expr_f) if expr_f else None,
                        mutation=pd.read_csv(mut_f) if mut_f else None,
                        clinical=pd.read_csv(clin_f) if clin_f else None,
                        histopathology=histo_df,
                    )
                )
                st.session_state.loaded_demo_id = None
                st.rerun()

    patient = st.session_state.get("last_patient")
    if patient is None:
        st.info("Select a demo patient or upload data above to continue.")
        render_disclaimer()
        return

    pid = patient.get("patient_id")
    mods = [k for k in patient if k not in ("patient_id", "from_feature_store")]
    source = "TCGA Demo" if patient.get("from_feature_store") else "Uploaded CSV"
    patient_summary_card(pid, source, mods)

    tab_predictions, tab_report, tab_treatment = st.tabs(
        ["Predictions", "AI Clinical Report", "Treatment Recommendation"]
    )

    with tab_predictions:
        render_predictions_tab(patient, target)

    with tab_report:
        render_clinical_report_tab(patient, target)

    with tab_treatment:
        render_treatment_tab(patient, target)

    render_disclaimer()


render()
