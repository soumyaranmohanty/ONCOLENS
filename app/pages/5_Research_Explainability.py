"""Research & Explainability page."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.components.charts import render_auc_comparison
from app.components.metrics import render_disclaimer
from app.components.patient_context import render_patient_context_banner
from app.config import TARGETS
from app.services.explain import (
    frequently_mutated_genes,
    load_modality_improvement_table,
    modality_auc_comparison,
    top_important_genes,
)


def render() -> None:
    st.title("Research & Explainability")
    st.markdown(
        "Cohort-level analysis and comparison across the five prediction backends "
        "(Expression, Mutation, Histopathology, 3-Modality, 4-Modality)."
    )

    render_patient_context_banner(cohort_level=True)

    target = st.selectbox("Target", TARGETS, key="explain_target")

    st.markdown("### Top Important Genes (Expression)")
    genes = top_important_genes(target)
    if genes.empty:
        st.info("Feature importance unavailable for this target.")
    else:
        st.dataframe(genes, hide_index=True, use_container_width=True)

    st.markdown("### Frequently Mutated Genes")
    mut_prev = frequently_mutated_genes()
    if mut_prev.empty:
        st.info("Mutation prevalence data unavailable.")
    else:
        st.dataframe(mut_prev, hide_index=True, use_container_width=True)

    st.markdown("### Modality Performance Comparison")
    st.caption(
        "Test-set AUC by backend. Clinical contributes only inside 3-Modality and 4-Modality stacks."
    )
    auc_df = modality_auc_comparison()
    render_auc_comparison(auc_df)

    st.markdown("### 3-Modality vs 4-Modality")
    improvement = load_modality_improvement_table()
    if improvement is not None:
        st.dataframe(improvement, hide_index=True, use_container_width=True)
    else:
        st.info("Pending multimodalv3 — modality improvement table not yet generated.")

    st.markdown("### Histopathology")
    st.write(
        "Batch-0 provisional model: 2048-D ResNet50 embeddings, mean-pooled per patient. "
        "Full cohort retrain pending GDC download completion."
    )

    render_disclaimer()


render()
