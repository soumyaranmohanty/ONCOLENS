"""Research & Explainability page."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.components.charts import render_auc_comparison
from app.components.metrics import render_disclaimer
from app.config import TARGETS
from app.services.explain import (
    load_modality_improvement_table,
    modality_auc_comparison,
    overlapping_important_genes,
    top_important_genes_expression,
    top_important_genes_mutation,
    top_multimodal_branch_genes,
)


def _render_gene_table(df, empty_msg: str) -> None:
    if df.empty:
        st.info(empty_msg)
    else:
        st.dataframe(df, hide_index=True, use_container_width=True)


def render() -> None:
    st.title("Research & Explainability")
    st.markdown(
        "Cohort-level model research: **Random Forest feature importances** from standalone and "
        "3-Modality level-0 models, plus cross-model gene overlap and backend AUC comparison."
    )
    st.caption(
        "All gene rankings are global (training-cohort importances), not per-patient. "
        "Patient-specific analysis is on **Patient Prediction**."
    )

    target = st.selectbox("Target", TARGETS, key="explain_target")
    top_n = st.slider("Top N genes per model", min_value=5, max_value=30, value=15, key="explain_top_n")

    st.markdown("### Standalone Expression model")
    st.caption(
        "From `Data/feature_store/expression/{target}/model.joblib` — the same standalone "
        "Expression Random Forest used for inference (`feature_importances_` on the selected gene panel)."
    )
    _render_gene_table(
        top_important_genes_expression(target, top_n),
        "Feature importance unavailable for this target.",
    )

    st.markdown("### Standalone Mutation model")
    st.caption(
        "From the standalone Mutation Random Forest (`feature_importances_`). "
        "Ranks genes by predictive importance, not cohort mutation frequency. "
        "Non-gene columns (`TMB`, index) are excluded from this table."
    )
    _render_gene_table(
        top_important_genes_mutation(target, top_n),
        "Mutation model feature importance unavailable for this target.",
    )

    st.markdown("### 3-Modality · Expression branch")
    st.caption(
        "Level-0 Expression Random Forest **inside the multimodal stack** "
        "(`multimodal_model/{target}/multimodal_stack.joblib`). "
        "Trained on the 3-modality cohort; gene panel may differ from the standalone Expression model."
    )
    _render_gene_table(
        top_multimodal_branch_genes(target, "Expression", top_n),
        "3-Modality bundle or Expression branch unavailable for this target.",
    )

    st.markdown("### 3-Modality · Mutation branch")
    st.caption(
        "Level-0 Mutation Random Forest inside the 3-Modality stack. "
        "Clinical branch features are staging/clinical fields, not genes — omitted here."
    )
    _render_gene_table(
        top_multimodal_branch_genes(target, "Mutation", top_n),
        "3-Modality bundle or Mutation branch unavailable for this target.",
    )

    st.markdown("### Cross-model gene overlap")
    overlap_mode = st.radio(
        "Show genes appearing in",
        options=["At least 2 sources", "All 4 sources (strict intersection)"],
        horizontal=True,
        key="overlap_mode",
    )
    min_models = 4 if overlap_mode.startswith("All") else 2
    st.caption(
        f"Compares top-{top_n} genes from: standalone Expression, standalone Mutation, "
        "3-Modality Expression branch, and 3-Modality Mutation branch."
    )
    overlap = overlapping_important_genes(target, top_n, min_models=min_models)
    _render_gene_table(
        overlap,
        f"No genes appear in the top-{top_n} list of {min_models} or more sources for this target.",
    )

    st.markdown("### Modality Performance Comparison")
    st.caption(
        "Test-set AUC by backend from feature-store registries. Clinical contributes only "
        "inside 3-Modality and 4-Modality fusion stacks."
    )
    auc_df = modality_auc_comparison()
    render_auc_comparison(auc_df)

    st.markdown("### 3-Modality vs 4-Modality")
    improvement = load_modality_improvement_table()
    if improvement is not None:
        st.dataframe(improvement, hide_index=True, use_container_width=True)
    else:
        st.info("Pending multimodal v4 training — modality improvement table not yet generated.")

    st.markdown("### Histopathology")
    st.write(
        "Batch-0 provisional model: 2048-D ResNet50 embeddings, mean-pooled per patient. "
        "Full cohort retrain pending GDC download completion."
    )

    render_disclaimer()


render()
