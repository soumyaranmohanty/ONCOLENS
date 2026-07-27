"""About page."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.config import APP_VERSION, DISCLAIMER, MODELS, MODEL_DESCRIPTIONS, MODELS_OVERVIEW


def render() -> None:
    st.title("About ONCOLENS")

    st.markdown(
        f"""
        **Version:** {APP_VERSION}

        ONCOLENS is a multimodal machine learning platform for TCGA Lung Adenocarcinoma (LUAD)
        that predicts overall survival status, progression-free survival status, and cancer stage.

        {MODELS_OVERVIEW}

        ### Prediction Backends
        """
    )
    for name in MODELS:
        st.markdown(f"- **{name}:** {MODEL_DESCRIPTIONS[name]}")

    st.markdown(
        """
        ### Architecture
        - **Standalone level-0:** Random Forest + Logistic Regression per modality (Expression, Mutation, Histopathology)
        - **Fusion level-0:** Same per branch, plus Clinical inside multimodal stacks only
        - **Level-1:** Logistic Regression meta-learner on stacked OOF probabilities
        - **Histopathology:** Frozen ResNet50 embeddings (2048-D), mean-pooled per patient

        ### Data Source
        - [TCGA LUAD](https://portal.gdc.cancer.gov/projects/TCGA-LUAD) via GDC
        - [cBioPortal TCGA Pan-Cancer Atlas](https://www.cbioportal.org/)

        ### References
        - TCGA Research Network. Comprehensive molecular profiling of lung adenocarcinoma.
        - Late-fusion multimodal stacking for survival and staging prediction.

        ### Developer
        Soumya Ranjan Mohanty

        """
        + DISCLAIMER
    )


render()
