"""ONCOLENS Streamlit application entry point."""

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

# Streamlit puts `app/` on sys.path, not the repo root — load path setup without `app.*`.
_path_file = Path(__file__).resolve().parent / "_path.py"
_spec = spec_from_file_location("oncolens_path", _path_file)
assert _spec and _spec.loader
_path_mod = module_from_spec(_spec)
sys.modules["oncolens_path"] = _path_mod
_spec.loader.exec_module(_path_mod)

import streamlit as st

from app.components.patient_context import render_patient_context_sidebar
from app.components.theme import inject_global_css

st.set_page_config(
    page_title="ONCOLENS",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_css()

with st.sidebar:
    render_patient_context_sidebar()

pages = [
    st.Page("pages/0_Dashboard.py", title="Dashboard", icon="🏠", default=True, url_path="dashboard"),
    st.Page("pages/1_Patient_Prediction.py", title="Patient Prediction", icon="🔬", url_path="patient-prediction"),
    st.Page("pages/2_Model_Analytics.py", title="Model Analytics", icon="📊", url_path="model-analytics"),
    st.Page("pages/5_Research_Explainability.py", title="Research & Explainability", icon="🔍", url_path="research"),
    st.Page("pages/6_Virtual_Assistant.py", title="Virtual AI Assistant", icon="💬", url_path="assistant"),
    st.Page("pages/7_About.py", title="About", icon="ℹ️", url_path="about"),
]

pg = st.navigation(pages)
pg.run()
