"""Patient-centric workflow tabs: predictions, clinical report, treatment."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from app.components.cards import report_section_card
from app.components.charts import render_comparison_bars
from app.components.layout import empty_state, report_preview
from app.components.metrics import render_prediction_card
from app.components.model_help import render_model_caption
from app.config import MODELS, POSITIVE_CLASS_LABEL, TARGET_TITLES
from app.inference.loaders import four_mod_available
from app.inference.predict import compare_all, predict_multimodal, predict_single
from app.services.history import append_prediction
from app.services.llm_agents import generate_llm_clinical_report, generate_llm_treatment_text, llm_available
from app.services.report import generate_report_text, normalize_clinical_report, report_to_pdf
from app.services.treatment import generate_treatment_recommendations


def render_predictions_tab(patient: dict[str, Any], target: str) -> None:
    pid = patient.get("patient_id")
    c1, c2 = st.columns(2)
    with c1:
        model_choice = st.selectbox("Model", MODELS + ["Compare All"], key="pred_model")
    with c2:
        st.caption(TARGET_TITLES.get(target, target))
    render_model_caption(model_choice)

    if model_choice == "4-Modality" and not four_mod_available(target):
        st.info(
            "**4-Modality is not available yet.** Use standalone Histopathology or 3-Modality instead."
        )

    if model_choice in ("Histopathology", "4-Modality") and "Histopathology" not in patient:
        st.warning(
            "This patient has no histopathology embedding. Enable **Only patients with "
            "histopathology** in the demo panel above, or upload a histopathology CSV."
        )

    if st.button("Run Prediction", type="primary", key="run_pred", use_container_width=True):
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
        empty_state("No results yet", "Click **Run Prediction** above.")
        return

    st.caption(f"{TARGET_TITLES.get(target, target)} · **{model_choice}**")

    for r in results:
        render_prediction_card(r)

    if len(results) > 1:
        render_comparison_bars(results, target)
        prob_col = "confidence (predicted stage)" if target == "Stage" else (
            f"P({POSITIVE_CLASS_LABEL.get(target, 'event')})"
        )
        with st.container(border=True):
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
    with st.container(border=True):
        use_llm = False
        if llm_available():
            use_llm = st.toggle(
                "Use AI-generated clinical report",
                value=False,
                key="report_use_llm",
            )
        else:
            st.caption("AI reports unavailable — template mode only.")

        mode = "AI narrative" if use_llm else "Template-based"
        st.caption(f"Mode: **{mode}** · includes biomarkers, expression, and all five backend predictions.")

    pid = patient.get("patient_id")

    if st.button("Generate Report", type="primary", key="gen_report", use_container_width=True):
        predictions = compare_all(target, patient)
        st.session_state.report_predictions = predictions
        if use_llm:
            with st.spinner("Generating clinical report…"):
                try:
                    st.session_state.report_text = generate_llm_clinical_report(
                        patient, predictions, target=target
                    )
                except Exception as exc:
                    st.error(f"Report generation failed: {exc}")
                    st.session_state.report_text = generate_report_text(patient, predictions)
                    st.info("Fell back to template-based report.")
        else:
            st.session_state.report_text = generate_report_text(patient, predictions)
        st.session_state.report_patient_id = pid
        st.session_state.report_target = target
        st.session_state.report_mode = "ai" if use_llm else "template"

    report_pid = st.session_state.get("report_patient_id")
    report_target = st.session_state.get("report_target")
    if "report_text" not in st.session_state or report_pid != pid or report_target != target:
        empty_state("No report generated", "Click **Generate Report** above.")
        return

    st.markdown(f"**Report** · `{report_pid}` · {TARGET_TITLES.get(target, target)}")
    report_mode = st.session_state.get("report_mode")
    report_text = normalize_clinical_report(
        st.session_state.report_text,
        report_pid,
        mode=report_mode,
    )
    report_preview(report_text)
    pdf_bytes = report_to_pdf(report_text, patient_id=report_pid)
    st.download_button(
        "Download PDF",
        data=pdf_bytes,
        file_name=f"oncolens_report_{report_pid}.pdf",
        mime="application/pdf",
        key="report_pdf",
        use_container_width=True,
    )


def render_treatment_tab(patient: dict[str, Any], target: str) -> None:
    with st.container(border=True):
        use_llm = False
        if llm_available():
            use_llm = st.toggle(
                "Use AI-generated treatment recommendations",
                value=False,
                key="treatment_use_llm",
            )
        else:
            st.caption("AI recommendations unavailable — rule-based mode only.")
        st.caption("Educational decision-support only — not a substitute for clinical judgment.")

    pid = patient.get("patient_id")

    if st.button("Generate Recommendations", type="primary", key="gen_treatment", use_container_width=True):
        predictions = compare_all(target, patient)
        if use_llm:
            with st.spinner("Generating recommendations…"):
                try:
                    st.session_state.treatment_llm_text = generate_llm_treatment_text(
                        patient, predictions, target=target
                    )
                    st.session_state.treatment_use_llm_result = True
                except Exception as exc:
                    st.error(f"Recommendation generation failed: {exc}")
                    st.session_state.treatment_rec = generate_treatment_recommendations(
                        patient, predictions
                    )
                    st.session_state.treatment_use_llm_result = False
                    st.info("Fell back to rule-based recommendations.")
        else:
            st.session_state.treatment_rec = generate_treatment_recommendations(patient, predictions)
            st.session_state.treatment_use_llm_result = False
        st.session_state.treatment_patient_id = pid
        st.session_state.treatment_target = target

    rec_pid = st.session_state.get("treatment_patient_id")
    rec_target = st.session_state.get("treatment_target")
    if rec_pid != pid or rec_target != target:
        empty_state("No recommendations yet", "Click **Generate Recommendations** above.")
        return

    st.markdown(f"**Recommendations** · `{rec_pid}` · {TARGET_TITLES.get(target, target)}")

    if st.session_state.get("treatment_use_llm_result") and st.session_state.get("treatment_llm_text"):
        with st.container(border=True):
            st.markdown(st.session_state.treatment_llm_text)
        return

    rec = st.session_state.get("treatment_rec")
    if not rec:
        empty_state("No recommendations yet", "Click **Generate Recommendations** above.")
        return

    report_section_card("Risk Assessment", rec["risk_assessment"])
    report_section_card("Likely Diagnosis", rec["likely_diagnosis"])
    report_section_card("Treatment Options", rec["treatment_options"])
    report_section_card("Drug Classes", rec["drug_classes"])
    report_section_card("Lifestyle Advice", rec["lifestyle_advice"])
    report_section_card("Monitoring Suggestions", rec["monitoring"])
    report_section_card("Questions for Your Oncologist", rec["questions_for_oncologist"])
