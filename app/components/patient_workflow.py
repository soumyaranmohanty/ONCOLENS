"""Patient-centric workflow tabs: predictions, clinical report, treatment."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from app.components.charts import render_comparison_bars
from app.components.metrics import render_prediction_card
from app.components.model_help import render_model_caption
from app.config import MODELS, POSITIVE_CLASS_LABEL, TARGET_TITLES
from app.inference.loaders import four_mod_available
from app.inference.predict import compare_all, predict_multimodal, predict_single
from app.services.history import append_prediction
from app.services.report import generate_report_text, report_to_pdf
from app.services.treatment import generate_treatment_recommendations


def render_predictions_tab(patient: dict[str, Any], target: str) -> None:
    pid = patient.get("patient_id")
    model_choice = st.selectbox("Model", MODELS + ["Compare All"], key="pred_model")
    render_model_caption(model_choice)

    if model_choice == "4-Modality" and not four_mod_available(target):
        st.info(
            "**4-Modality is not available yet.** The `multimodal_model_v4` bundle has not been "
            "trained. You can still use the standalone Histopathology model or 3-Modality "
            "(Expression + Mutation + Clinical) for this target."
        )

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

    stored_pid = st.session_state.get("last_results_patient_id")
    stored_target = st.session_state.get("last_results_target")
    results = st.session_state.get("last_results", [])

    if not results or stored_pid != pid or stored_target != target:
        st.info("No results yet for this patient and target. Click **Run Prediction** above.")
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


def render_clinical_report_tab(patient: dict[str, Any], target: str) -> None:
    st.markdown(
        "Generate a **template-based** patient report from biomarker flags and model predictions. "
        "This is not an LLM-generated narrative."
    )
    st.caption(
        "Combines mutation flags, top log1p(RSEM) genes from the selected expression panel, "
        "histopathology embedding summary, and predictions from all five backends."
    )

    with st.expander("LLM clinical report (coming soon)", expanded=False):
        st.info(
            "A future version will support an optional LLM agent for narrative clinical summaries. "
            "That integration is not enabled in this release — use the template report below."
        )

    pid = patient.get("patient_id")

    if st.button("Generate Report", type="primary", key="gen_report"):
        predictions = compare_all(target, patient)
        st.session_state.report_predictions = predictions
        st.session_state.report_text = generate_report_text(patient, predictions)
        st.session_state.report_patient_id = pid
        st.session_state.report_target = target

    report_pid = st.session_state.get("report_patient_id")
    report_target = st.session_state.get("report_target")
    if "report_text" not in st.session_state or report_pid != pid or report_target != target:
        st.info("Click **Generate Report** to build a report for the active patient and target.")
        return

    st.markdown(f"### Report for patient `{report_pid}` · {TARGET_TITLES.get(target, target)}")
    st.text_area("Report Preview", st.session_state.report_text, height=400, key="report_preview")
    pdf_bytes = report_to_pdf(st.session_state.report_text)
    st.download_button(
        "Download PDF",
        data=pdf_bytes,
        file_name=f"oncolens_report_{report_pid}.pdf",
        mime="application/pdf",
        key="report_pdf",
    )


def render_treatment_tab(patient: dict[str, Any], target: str) -> None:
    st.markdown(
        "Educational decision-support from **rule-based** biomarker and risk-band logic today. "
        "An LLM treatment agent will replace this with narrative, patient-specific recommendations."
    )
    st.caption(
        "Current output is deterministic: EGFR/KRAS/TMB flags, OS/PFS risk bands, and stage "
        "predictions map to fixed suggestion lists — not model-generated clinical advice."
    )

    with st.expander("LLM treatment agent (coming soon)", expanded=False):
        st.info(
            "After you add your LLM integration to the codebase, this tab will call a "
            "Treatment Recommendation agent that uses patient data, all five model predictions, "
            "and structured biomarker context to produce personalized options, monitoring, "
            "and questions for the care team."
        )

    pid = patient.get("patient_id")

    if st.button("Generate Recommendations", type="primary", key="gen_treatment"):
        predictions = compare_all(target, patient)
        st.session_state.treatment_rec = generate_treatment_recommendations(patient, predictions)
        st.session_state.treatment_patient_id = pid
        st.session_state.treatment_target = target

    rec_pid = st.session_state.get("treatment_patient_id")
    rec_target = st.session_state.get("treatment_target")
    rec = st.session_state.get("treatment_rec")
    if not rec or rec_pid != pid or rec_target != target:
        st.info("Click **Generate Recommendations** for the active patient and target.")
        return

    st.markdown(
        f"### Recommendations for patient `{rec_pid}` · {TARGET_TITLES.get(target, target)}"
    )
    st.markdown("#### Risk Assessment")
    st.write(rec["risk_assessment"])
    st.markdown("#### Likely Diagnosis")
    st.write(rec["likely_diagnosis"])
    st.markdown("#### Treatment Options")
    for item in rec["treatment_options"]:
        st.write(f"- {item}")
    st.markdown("#### Drug Classes")
    for item in rec["drug_classes"]:
        st.write(f"- {item}")
    st.markdown("#### Lifestyle Advice")
    for item in rec["lifestyle_advice"]:
        st.write(f"- {item}")
    st.markdown("#### Monitoring Suggestions")
    for item in rec["monitoring"]:
        st.write(f"- {item}")
    st.markdown("#### Questions for Your Oncologist")
    for item in rec["questions_for_oncologist"]:
        st.write(f"- {item}")
