"""Dashboard home page."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from app.components.charts import render_auc_comparison
from app.components.layout import (
    kpi_row,
    page_hero,
    pipeline_steps,
    render_disclaimer,
    section,
)
from app.components.model_help import render_models_overview
from app.config import APP_VERSION
from app.inference.loaders import all_registries_summary


def _highlight_best_auc(df: pd.DataFrame) -> pd.DataFrame:
    styled = df.copy()
    if "test_auc" not in styled.columns:
        return styled

    def _fmt(row):
        auc_val = row.get("test_auc")
        if pd.isna(auc_val):
            return [""] * len(row)
        target_rows = df[df["target"] == row["target"]]["test_auc"]
        if target_rows.notna().any() and auc_val == target_rows.max():
            return ["background-color: #ccfbf1; font-weight: 600"] * len(row)
        return [""] * len(row)

    return styled.style.apply(_fmt, axis=1).format({"test_auc": "{:.3f}"})


def render() -> None:
    page_hero(
        "ONCOLENS",
        "Integrated AI precision oncology platform for TCGA lung adenocarcinoma — "
        "multimodal survival and staging prediction.",
        badges=["TCGA LUAD", f"v{APP_VERSION}", "5 prediction backends"],
    )

    kpi_row([
        ("Cancer type", "TCGA LUAD", None),
        ("3-Mod cohort", "450 patients", None),
        ("4-Mod cohort", "44 patients", "provisional"),
    ])

    section("Quick actions", "Jump to the main workflows")
    q1, q2, q3 = st.columns(3)
    with q1:
        with st.container(border=True):
            st.markdown("**Patient Prediction**")
            st.caption("Load a patient, run models, reports, and treatment guidance.")
            st.page_link("pages/1_Patient_Prediction.py", label="Open workflow", icon="🔬")
    with q2:
        with st.container(border=True):
            st.markdown("**Model Analytics**")
            st.caption("Metrics, ROC curves, and confusion matrices per backend.")
            st.page_link("pages/2_Model_Analytics.py", label="View metrics", icon="📊")
    with q3:
        with st.container(border=True):
            st.markdown("**Research & Explainability**")
            st.caption("Gene importances, overlap analysis, and AUC comparison.")
            st.page_link("pages/5_Research_Explainability.py", label="Explore research", icon="🔍")

    section(
        "Model performance",
        "Test-set AUC from feature-store registries. Highlighted rows = best AUC per target.",
    )
    summary = pd.DataFrame(all_registries_summary())
    if not summary.empty:
        display = summary[["model", "target", "test_auc", "total_patients", "model_type"]].copy()
        st.dataframe(_highlight_best_auc(display), use_container_width=True, hide_index=True)
        render_auc_comparison(summary)
    else:
        st.warning("No registry data found.")

    section("Pipeline")
    pipeline_steps(["TCGA data", "Feature store", "Late-fusion stacking", "Clinical inference"])

    section("Architecture", "Standalone models and multimodal fusion stacks")
    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown(
            """
            <div class="oncolens-arch-col">
              <h4>Standalone</h4>
              <ul>
                <li>Expression (log1p RSEM)</li>
                <li>Mutation (+ TMB)</li>
                <li>Histopathology (ResNet50)</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with a2:
        st.markdown(
            """
            <div class="oncolens-arch-col">
              <h4>3-Modality fusion</h4>
              <ul>
                <li>Expression + Mutation + Clinical</li>
                <li>Level-0 RF/LR per branch</li>
                <li>Level-1 logistic meta-learner</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with a3:
        st.markdown(
            """
            <div class="oncolens-arch-col">
              <h4>4-Modality fusion</h4>
              <ul>
                <li>+ Histopathology branch</li>
                <li>Same stacking architecture</li>
                <li>Pending full cohort train</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_models_overview()

    render_disclaimer()


render()
