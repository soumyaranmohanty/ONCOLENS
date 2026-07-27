"""Redirect: clinical report lives under Patient Prediction."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

st.title("AI Clinical Report")
st.info(
    "Clinical reports are now part of **Patient Prediction**. "
    "Open that page and use the **AI Clinical Report** tab."
)
st.page_link("pages/1_Patient_Prediction.py", label="Go to Patient Prediction", icon="🔬")
