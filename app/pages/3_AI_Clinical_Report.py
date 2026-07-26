"""AI Clinical Report page."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.components.metrics import render_disclaimer
from app.inference.predict import compare_all
from app.services.history import init_history
from app.services.report import generate_report_text, report_to_pdf


def render() -> None:
    st.title("AI Clinical Report")
    st.markdown("Generate a patient-specific clinical report from prediction results.")

    init_history()
    patient = st.session_state.get("last_patient")
    if patient is None:
        st.info("Run a prediction first on the Patient Prediction page, or load a demo patient below.")
        if st.button("Generate sample report for TCGA-73-4659"):
            from app.inference.patient import make_demo_patient

            patient = make_demo_patient("TCGA-73-4659", "OS_STATUS")
            st.session_state.last_patient = patient
            st.rerun()
        render_disclaimer()
        return

    target = st.selectbox("Primary target for report", ["OS_STATUS", "PFS_STATUS", "Stage"], key="report_target")

    if st.button("Generate Report", type="primary"):
        predictions = compare_all(target, patient)
        st.session_state.report_predictions = predictions
        text = generate_report_text(patient, predictions)
        st.session_state.report_text = text

    if "report_text" in st.session_state:
        st.text_area("Report Preview", st.session_state.report_text, height=400)
        pdf_bytes = report_to_pdf(st.session_state.report_text)
        st.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name=f"oncolens_report_{patient.get('patient_id')}.pdf",
            mime="application/pdf",
        )

    render_disclaimer()


render()
