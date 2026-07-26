"""Patient Prediction page."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from app.components.charts import render_comparison_bars
from app.components.metrics import render_disclaimer, render_prediction_card
from app.config import MODELS, TARGETS
from app.inference.loaders import four_mod_available
from app.inference.patient import make_demo_patient, make_uploaded_patient
from app.inference.predict import compare_all, predict_multimodal, predict_single
from app.inference.tables import demo_patient_ids
from app.services.history import append_prediction, set_last_patient


def render() -> None:
    st.title("Patient Prediction")
    st.markdown("Predict OS_STATUS, PFS_STATUS, or Stage using single or multimodal models.")

    target = st.selectbox("Target", TARGETS, key="pred_target")
    four_mod_ready = four_mod_available(target)

    model_options = MODELS + ["Compare All"]
    model_choice = st.selectbox("Model", model_options, key="pred_model")

    if model_choice == "4-Modality" and not four_mod_ready:
        st.info("4-Modality model is not yet trained. Train `multimodalv3` to enable. Histopathology standalone is available.")

    tab_demo, tab_upload = st.tabs(["TCGA Demo", "Upload CSVs"])

    patient = None

    with tab_demo:
        ids = demo_patient_ids(target, four_mod=(model_choice == "4-Modality"))
        if not ids:
            st.warning("No demo patients available for this cohort.")
        else:
            patient_id = st.selectbox("Patient ID", ids, key="demo_patient")
            if st.button("Load demo patient", key="load_demo"):
                patient = make_demo_patient(patient_id, target)
                set_last_patient(patient)
                st.success(f"Loaded {patient_id}")

    with tab_upload:
        st.markdown("Upload CSV files with one patient row. Include `PATIENT_ID` or `patient_id` column.")
        pid = st.text_input("Patient ID", value="UPLOAD-001", key="upload_pid")
        expr_f = st.file_uploader("Gene Expression CSV", type=["csv"], key="up_expr")
        mut_f = st.file_uploader("Mutation CSV", type=["csv"], key="up_mut")
        clin_f = st.file_uploader("Clinical CSV", type=["csv"], key="up_clin")
        histo_f = st.file_uploader("Histopathology CSV (optional)", type=["csv"], key="up_histo")

        if st.button("Build patient from uploads", key="build_upload"):
            patient = make_uploaded_patient(
                pid,
                expression=pd.read_csv(expr_f) if expr_f else None,
                mutation=pd.read_csv(mut_f) if mut_f else None,
                clinical=pd.read_csv(clin_f) if clin_f else None,
                histopathology=pd.read_csv(histo_f) if histo_f else None,
            )
            set_last_patient(patient)
            st.success(f"Built patient {pid}")

    if patient is None and "last_patient" in st.session_state and st.session_state.last_patient:
        patient = st.session_state.last_patient

    if patient is None:
        st.info("Load a demo patient or upload CSVs to run predictions.")
        render_disclaimer()
        return

    st.markdown(f"**Active patient:** `{patient.get('patient_id')}`")
    st.write("Modalities:", [k for k in patient if k not in ("patient_id", "from_feature_store")])

    if st.button("Run Prediction", type="primary", key="run_pred"):
        if model_choice == "Compare All":
            results = compare_all(target, patient)
            st.session_state.last_results = results
            for r in results:
                append_prediction(r)
        elif model_choice in ("3-Modality", "4-Modality"):
            result = predict_multimodal(target, patient, four_mod=(model_choice == "4-Modality"))
            st.session_state.last_results = [result]
            append_prediction(result)
        else:
            result = predict_single(model_choice, target, patient)
            st.session_state.last_results = [result]
            append_prediction(result)

    if "last_results" in st.session_state and st.session_state.last_results:
        results = [r for r in st.session_state.last_results if r.get("target") == target]
        if results:
            st.markdown("### Results")
            for r in results:
                render_prediction_card(r)
            if len(results) > 1:
                render_comparison_bars(results, target)
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "model": r["model"],
                                "prediction": r.get("predicted_label_name", "N/A"),
                                "probability": r.get("probability"),
                                "risk": r.get("risk_band"),
                                "available": r.get("available", True),
                            }
                            for r in results
                        ]
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

    render_disclaimer()


render()
