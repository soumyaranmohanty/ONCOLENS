"""About page."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.components.layout import render_disclaimer, section
from app.config import APP_VERSION, MODELS, MODEL_DESCRIPTIONS, MODELS_OVERVIEW


def render() -> None:
    section("About ONCOLENS", f"Version {APP_VERSION} · TCGA LUAD multimodal prediction platform")

    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            st.markdown(MODELS_OVERVIEW)
            st.markdown("**Prediction backends**")
            for name in MODELS:
                st.markdown(f"- **{name}:** {MODEL_DESCRIPTIONS[name]}")

    with right:
        with st.container(border=True):
            st.markdown("**Architecture**")
            st.markdown(
                """
                - **Standalone level-0:** RF + LR per modality (Expression, Mutation, Histopathology)
                - **Fusion level-0:** Clinical branch inside multimodal stacks only
                - **Level-1:** Logistic Regression meta-learner on stacked OOF probabilities
                - **Histopathology:** Frozen ResNet50 embeddings (2048-D), mean-pooled per patient
                """
            )
            st.markdown("**Data sources**")
            st.markdown(
                """
                - [TCGA LUAD](https://portal.gdc.cancer.gov/projects/TCGA-LUAD) via GDC
                - [cBioPortal TCGA Pan-Cancer Atlas](https://www.cbioportal.org/)
                """

            )
            st.markdown("**Developer**")
            st.markdown("Soumya Ranjan Mohanty")

    with st.container(border=True):
        st.markdown("**References**")
        st.markdown(
            """
            - TCGA Research Network — comprehensive molecular profiling of lung adenocarcinoma
            - Late-fusion multimodal stacking for survival and staging prediction
            """
        )

    render_disclaimer()


render()
