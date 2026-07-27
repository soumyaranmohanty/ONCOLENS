"""AI Clinical Report page."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.components.metrics import render_disclaimer
from app.components.patient_context import render_patient_context_banner
from app.inference.predict import compare_all
from app.services.history import init_history
from app.services.report import generate_report_text, report_to_pdf


def render() -> None:
    st.title("AI Clinical Report")
    st.markdown(
        "Generate a **template-based** patient report from rule-based biomarker flags and "
        "model predictions. This is not an LLM-generated narrative."
    )
    st.caption(
        "Reports combine mutation flags, top log1p(RSEM) genes from the selected expression panel, "
        "histopathology embedding summary, and predictions from all five backends."
    )

    with st.expander("LLM clinical report (coming soon)", expanded=False):
        st.info(
            "A future version will support an optional LLM agent for narrative clinical summaries. "
            "That integration is not enabled in this release — use the template report below."
        )

    init_history()
    patient = render_patient_context_banner(require_patient=True)
    if patient is None:
        if st.button("Load sample patient TCGA-73-4659"):
            from app.inference.patient import make_demo_patient
            from app.services.history import set_last_patient

            patient = make_demo_patient("TCGA-73-4659", "OS_STATUS")
            set_last_patient(patient)
            st.rerun()
        render_disclaimer()
        return

    pid = patient.get("patient_id")
    target = st.selectbox("Primary target for report", ["OS_STATUS", "PFS_STATUS", "Stage"], key="report_target")

    if st.button("Generate Report", type="primary"):
        predictions = compare_all(target, patient)
        st.session_state.report_predictions = predictions
        text = generate_report_text(patient, predictions)
        st.session_state.report_text = text
        st.session_state.report_patient_id = pid

    if "report_text" in st.session_state:
        report_pid = st.session_state.get("report_patient_id", pid)
        st.markdown(f"### Report for patient `{report_pid}`")
        st.text_area("Report Preview", st.session_state.report_text, height=400, key="report_preview")
        pdf_bytes = report_to_pdf(st.session_state.report_text)
        st.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name=f"oncolens_report_{report_pid}.pdf",
            mime="application/pdf",
        )

    render_disclaimer()


render()
