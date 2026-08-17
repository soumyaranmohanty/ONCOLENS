"""Research & Explainability page."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.components.charts import (
    render_auc_comparison,
    render_gene_importance_bar,
    render_overlap_summary,
)
from app.components.layout import render_disclaimer, section
from app.config import TARGETS
from app.services.explain import (
    load_modality_improvement_table,
    modality_auc_comparison,
    overlapping_important_genes,
    top_important_genes_expression,
    top_important_genes_mutation,
    top_multimodal_branch_genes,
)


def _gene_section(df, title: str, empty_msg: str) -> None:
    if df.empty:
        st.info(empty_msg)
        return
    render_gene_importance_bar(df, title)
    with st.expander("View table"):
        st.dataframe(df, hide_index=True, use_container_width=True)


def render() -> None:
    section(
        "Research & Explainability",
        "Cohort-level feature importances, cross-model gene overlap, and backend performance.",
    )

    c1, c2 = st.columns([1, 2])
    with c1:
        target = st.selectbox("Target", TARGETS, key="explain_target")
    with c2:
        top_n = st.slider("Top N genes per model", min_value=5, max_value=30, value=15, key="explain_top_n")

    tab_expr, tab_mut, tab_3mod, tab_overlap, tab_perf = st.tabs(
        ["Expression", "Mutation", "3-Mod branches", "Overlap", "Performance"]
    )

    with tab_expr:
        st.caption("Standalone Expression Random Forest — global `feature_importances_`.")
        _gene_section(
            top_important_genes_expression(target, top_n),
            "Standalone Expression — top genes",
            "Feature importance unavailable for this target.",
        )

    with tab_mut:
        st.caption("Standalone Mutation Random Forest (genes only; TMB excluded).")
        _gene_section(
            top_important_genes_mutation(target, top_n),
            "Standalone Mutation — top genes",
            "Mutation model feature importance unavailable.",
        )

    with tab_3mod:
        st.caption("Level-0 Random Forest branches inside the 3-Modality stack.")
        c_a, c_b = st.columns(2)
        with c_a:
            _gene_section(
                top_multimodal_branch_genes(target, "Expression", top_n),
                "3-Mod · Expression branch",
                "3-Mod Expression branch unavailable.",
            )
        with c_b:
            _gene_section(
                top_multimodal_branch_genes(target, "Mutation", top_n),
                "3-Mod · Mutation branch",
                "3-Mod Mutation branch unavailable.",
            )

    with tab_overlap:
        overlap_mode = st.radio(
            "Show genes in",
            ["At least 2 sources", "All 4 sources"],
            horizontal=True,
            key="overlap_mode",
        )
        min_models = 4 if overlap_mode.startswith("All") else 2
        overlap = overlapping_important_genes(target, top_n, min_models=min_models)
        if overlap.empty:
            st.info(f"No genes in top-{top_n} of {min_models}+ sources for this target.")
        else:
            render_overlap_summary(overlap)
            display = overlap.copy()
            display["n_sources"] = display["n_sources"].apply(
                lambda n: f"{n} sources"
            )
            st.dataframe(display, hide_index=True, use_container_width=True)

    with tab_perf:
        st.caption("Test-set AUC from feature-store registries.")
        auc_df = modality_auc_comparison()
        render_auc_comparison(auc_df)
        improvement = load_modality_improvement_table()
        if improvement is not None:
            st.dataframe(improvement, hide_index=True, use_container_width=True)
        else:
            st.info("3-Mod vs 4-Mod improvement table not yet generated.")

        st.markdown("**Histopathology**")
        st.caption(
            "124 patients with histopathology embeddings; 120 overlap the 3-mod cohort (450 patients)."
        )

    render_disclaimer()


render()
