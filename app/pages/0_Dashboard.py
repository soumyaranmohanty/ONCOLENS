"""Dashboard home page."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from app.components.charts import render_auc_comparison
from app.components.model_help import render_models_overview
from app.config import APP_VERSION, DISCLAIMER
from app.inference.loaders import all_registries_summary


def render() -> None:
    st.title("ONCOLENS")
    st.subheader("Integrated AI Precision Oncology Platform — TCGA LUAD")

    st.markdown(
        """
        ONCOLENS predicts **OS_STATUS**, **PFS_STATUS**, and **Stage** for lung adenocarcinoma
        patients using five prediction backends: three standalone models (Expression, Mutation,
        Histopathology) and two late-fusion multimodal stacks (3-Modality, 4-Modality).
        Clinical data feeds the multimodal stacks only — there is no standalone Clinical model.
        """
    )

    render_models_overview()

    col1, col2, col3 = st.columns(3)
    col1.metric("Cancer Type", "TCGA LUAD")
    col2.metric("3-Mod Cohort", "450 patients")
    col3.metric("4-Mod Cohort (provisional)", "44 patients")

    st.markdown("### Model Performance Summary")
    st.caption(
        "Test-set AUC from feature-store registries. Standalone rows are single-modality models; "
        "3-Modality and 4-Modality are fusion stacks (Clinical is included only in fusion models)."
    )
    summary = pd.DataFrame(all_registries_summary())
    if not summary.empty:
        display = summary[["model", "target", "test_auc", "total_patients", "model_type"]].copy()
        st.dataframe(display, use_container_width=True, hide_index=True)
        render_auc_comparison(summary)
    else:
        st.warning("No registry data found.")

    st.markdown("### Architecture")
    st.code(
        """
Standalone:  Expression | Mutation | Histopathology
Fusion:      Expression + Mutation + Clinical  →  3-Modality meta-learner
             Expression + Mutation + Clinical + Histopathology  →  4-Modality meta-learner

Level-0: RF + LR per modality branch  →  OOF probabilities
Level-1: Logistic Regression meta-learner
Output:  OS_STATUS | PFS_STATUS | Stage
        """,
        language="text",
    )

    st.markdown("### Workflow")
    st.markdown(
        "1. **Data** — TCGA LUAD cohort with aligned patient IDs  \n"
        "2. **Feature Store** — per-modality trained models and features  \n"
        "3. **Stacking** — late-fusion multimodal ensemble  \n"
        "4. **Inference** — Streamlit app for predictions and reports"
    )

    st.markdown("### Quick Links")
    st.page_link("pages/1_Patient_Prediction.py", label="Go to Patient Prediction", icon="🔬")
    st.page_link("pages/2_Model_Analytics.py", label="Go to Model Analytics", icon="📊")

    st.caption(f"ONCOLENS v{APP_VERSION} | {DISCLAIMER}")


render()
