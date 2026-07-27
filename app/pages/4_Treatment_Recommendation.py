"""Redirect: treatment recommendations live under Patient Prediction."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

st.title("Treatment Recommendation")
st.info(
    "Treatment recommendations are now part of **Patient Prediction**. "
    "Open that page and use the **Treatment Recommendation** tab."
)
st.page_link("pages/1_Patient_Prediction.py", label="Go to Patient Prediction", icon="🔬")
