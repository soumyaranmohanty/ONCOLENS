"""Treatment Recommendation page."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.components.metrics import render_disclaimer
from app.inference.predict import compare_all
from app.services.history import init_history
from app.services.treatment import generate_treatment_recommendations


def render() -> None:
    st.title("AI Treatment Recommendation")
    st.markdown("Educational decision-support based on predictions and biomarkers.")

    init_history()
    patient = st.session_state.get("last_patient")
    if patient is None:
        st.info("Run a prediction first to get treatment suggestions.")
        render_disclaimer()
        return

    target = st.selectbox("Target context", ["OS_STATUS", "PFS_STATUS", "Stage"], key="treat_target")

    if st.button("Generate Recommendations", type="primary"):
        predictions = compare_all(target, patient)
        rec = generate_treatment_recommendations(patient, predictions)
        st.session_state.treatment_rec = rec

    rec = st.session_state.get("treatment_rec")
    if rec:
        st.markdown("### Risk Assessment")
        st.write(rec["risk_assessment"])
        st.markdown("### Likely Diagnosis")
        st.write(rec["likely_diagnosis"])
        st.markdown("### Treatment Options")
        for item in rec["treatment_options"]:
            st.write(f"- {item}")
        st.markdown("### Drug Classes")
        for item in rec["drug_classes"]:
            st.write(f"- {item}")
        st.markdown("### Lifestyle Advice")
        for item in rec["lifestyle_advice"]:
            st.write(f"- {item}")
        st.markdown("### Monitoring Suggestions")
        for item in rec["monitoring"]:
            st.write(f"- {item}")
        st.markdown("### Questions for Your Oncologist")
        for item in rec["questions_for_oncologist"]:
            st.write(f"- {item}")

    render_disclaimer()


render()
