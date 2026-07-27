"""Patient Prediction page."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from app.config import MODELS, POSITIVE_CLASS_LABEL, TARGETS, TARGET_TITLES
from app.components.charts import render_comparison_bars
from app.components.metrics import render_disclaimer, render_prediction_card
from app.components.model_help import render_model_caption, render_models_overview
from app.inference.loaders import four_mod_available
from app.inference.patient import make_demo_patient, make_uploaded_patient
from app.inference.predict import compare_all, predict_multimodal, predict_single
from app.inference.tables import demo_patient_ids
from app.services.history import append_prediction, set_last_patient


def render() -> None:
    st.title("Patient Prediction")
    st.markdown(
        "Predict OS_STATUS, PFS_STATUS, or Stage using standalone models or multimodal fusion stacks."
    )
    render_models_overview()

    # ── Settings & filters ──────────────────────────────────────────────────
    st.markdown("### Prediction settings")

    col_target, col_model = st.columns(2)
    with col_target:
        target = st.selectbox("Target", TARGETS, key="pred_target")
    with col_model:
        model_choice = st.selectbox("Model", MODELS + ["Compare All"], key="pred_model")
        render_model_caption(model_choice)

    if model_choice == "4-Modality" and not four_mod_available(target):
        st.info(
            "**4-Modality is not available yet.** The `multimodal_model_v4` bundle has not been "
            "trained. You can still use the standalone Histopathology model or 3-Modality "
            "(Expression + Mutation + Clinical) for this target."
        )

    tab_demo, tab_upload = st.tabs(["TCGA Demo", "Upload CSVs"])

    with tab_demo:
        ids = demo_patient_ids(target, four_mod=(model_choice == "4-Modality"))
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
        st.info("Select a demo patient or upload CSVs above, then run a prediction.")
        render_disclaimer()
        return

    pid = patient.get("patient_id")
    mods = [k for k in patient if k not in ("patient_id", "from_feature_store")]
    st.success(f"Ready to predict for patient **`{pid}`** · modalities: {', '.join(mods) or 'None'}")

    if st.button("Run Prediction", type="primary", key="run_pred"):
        if model_choice == "Compare All":
            results = compare_all(target, patient)
        elif model_choice in ("3-Modality", "4-Modality"):
            results = [predict_multimodal(target, patient, four_mod=(model_choice == "4-Modality"))]
        else:
            results = [predict_single(model_choice, target, patient)]

        st.session_state.last_results = results
        st.session_state.last_results_patient_id = pid
        st.session_state.last_results_target = target
        for r in results:
            append_prediction(r)

    # ── Results (below settings) ──────────────────────────────────────────────
    st.divider()
    st.markdown("### Results")

    stored_pid = st.session_state.get("last_results_patient_id")
    stored_target = st.session_state.get("last_results_target")
    results = st.session_state.get("last_results", [])

    if not results or stored_pid != pid or stored_target != target:
        st.info("No results yet for this patient and target. Click **Run Prediction** above.")
        render_disclaimer()
        return

    st.caption(
        f"Patient **`{pid}`** · {TARGET_TITLES.get(target, target)} · model **{model_choice}**"
    )

    for r in results:
        render_prediction_card(r)

    if len(results) > 1:
        render_comparison_bars(results, target)
        prob_col = "confidence (predicted stage)" if target == "Stage" else (
            f"P({POSITIVE_CLASS_LABEL.get(target, 'event')})"
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "patient_id": r.get("patient_id", pid),
                        "model": r["model"],
                        "outcome": TARGET_TITLES.get(target, target),
                        "prediction": r.get("predicted_label_name", "N/A"),
                        prob_col: r.get("probability"),
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
