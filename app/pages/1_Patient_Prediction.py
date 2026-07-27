"""Patient Prediction page — predictions, clinical report, and treatment tabs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from app.components.metrics import render_disclaimer
from app.components.model_help import render_models_overview
from app.components.patient_workflow import (
    render_clinical_report_tab,
    render_predictions_tab,
    render_treatment_tab,
)
from app.config import TARGETS
from app.inference.patient import make_demo_patient, make_uploaded_patient
from app.inference.tables import demo_patient_ids
from app.services.history import init_history, set_last_patient


def render() -> None:
    init_history()

    st.title("Patient Prediction")
    st.markdown(
        "Load a patient, run model predictions, and generate clinical reports and "
        "treatment recommendations in one workflow."
    )
    render_models_overview()

    st.markdown("### Patient & target")

    target = st.selectbox("Target", TARGETS, key="pred_target")

    tab_demo, tab_upload = st.tabs(["TCGA Demo", "Upload CSVs"])

    with tab_demo:
        ids = demo_patient_ids(target, four_mod=False)
        if not ids:
            st.warning("No demo patients available for this cohort.")
        else:
            patient_id = st.selectbox("Patient ID", ids, key="demo_patient")
            if st.session_state.get("loaded_demo_id") != patient_id:
                set_last_patient(make_demo_patient(patient_id, target))
                st.session_state.loaded_demo_id = patient_id
            if st.button("Reload demo patient", key="load_demo"):
                set_last_patient(make_demo_patient(patient_id, target))
                st.session_state.loaded_demo_id = patient_id
                st.success(f"Loaded {patient_id}")

    with tab_upload:
        st.markdown("Upload CSV files with one patient row. Include `PATIENT_ID` or `patient_id` column.")
        pid = st.text_input("Patient ID", value="UPLOAD-001", key="upload_pid")
        expr_f = st.file_uploader("Gene Expression CSV", type=["csv"], key="up_expr")
        mut_f = st.file_uploader("Mutation CSV", type=["csv"], key="up_mut")
        clin_f = st.file_uploader("Clinical CSV", type=["csv"], key="up_clin")
        histo_f = st.file_uploader("Histopathology CSV (optional)", type=["csv"], key="up_histo")

        if st.button("Build patient from uploads", key="build_upload"):
            set_last_patient(
                make_uploaded_patient(
                    pid,
                    expression=pd.read_csv(expr_f) if expr_f else None,
                    mutation=pd.read_csv(mut_f) if mut_f else None,
                    clinical=pd.read_csv(clin_f) if clin_f else None,
                    histopathology=pd.read_csv(histo_f) if histo_f else None,
                )
            )
            st.session_state.loaded_demo_id = None
            st.success(f"Built patient {pid}")

    patient = st.session_state.get("last_patient")
    if patient is None:
        st.info("Select a demo patient or upload CSVs above to continue.")
        render_disclaimer()
        return

    pid = patient.get("patient_id")
    mods = [k for k in patient if k not in ("patient_id", "from_feature_store")]
    source = "TCGA Demo" if patient.get("from_feature_store") else "Uploaded CSV"
    st.success(
        f"Active patient **`{pid}`** ({source}) · data loaded: {', '.join(mods) or 'None'}"
    )

    st.divider()

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
