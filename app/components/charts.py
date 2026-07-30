"""Reusable chart components."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.components.theme import MODEL_COLORS, PRIMARY, apply_plotly_layout
from app.config import POSITIVE_CLASS_LABEL, TARGET_TITLES


def render_roc_curve(roc_data: dict[str, Any], title: str = "ROC Curve") -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=roc_data["fpr"],
            y=roc_data["tpr"],
            mode="lines",
            name=f"AUC = {roc_data['auc']:.3f}",
            line=dict(color=PRIMARY, width=2.5),
            fill="tozeroy",
            fillcolor="rgba(13, 148, 136, 0.12)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="#94a3b8", width=1),
            name="Random",
        )
    )
    apply_plotly_layout(fig, title)
    fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(fig, use_container_width=True)


def render_confusion_matrix(
    cm: np.ndarray,
    labels: list[str] | None = None,
    title: str = "Confusion Matrix",
) -> None:
    if labels is None:
        labels = [str(i) for i in range(cm.shape[0])]
    fig = px.imshow(
        cm,
        text_auto=True,
        x=labels,
        y=labels,
        color_continuous_scale=[[0, "#f0fdfa"], [0.5, "#5eead4"], [1, "#0d9488"]],
        aspect="auto",
    )
    apply_plotly_layout(fig, title)
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)


def render_comparison_bars(results: list[dict[str, Any]], target: str) -> None:
    rows = []
    y_label = "Probability"
    if target in POSITIVE_CLASS_LABEL:
        y_label = f"P({POSITIVE_CLASS_LABEL[target]})"

    for r in results:
        if not r.get("available", True):
            rows.append({"model": r["model"], "probability": 0, "status": "unavailable"})
        elif target == "Stage":
            rows.append({"model": r["model"], "probability": r["confidence"], "status": "ok"})
        else:
            rows.append({"model": r["model"], "probability": r["probability"], "status": "ok"})

    df = pd.DataFrame(rows)
    color_map = {m: MODEL_COLORS.get(m, "#64748b") for m in df["model"].unique()}
    title = f"Model Comparison — {TARGET_TITLES.get(target, target)}"
    fig = px.bar(
        df,
        x="model",
        y="probability",
        color="model",
        color_discrete_map=color_map,
        title=title,
        labels={"probability": y_label, "model": "Model"},
        pattern_shape="status",
        pattern_shape_map={"unavailable": "/", "ok": ""},
    )
    apply_plotly_layout(fig)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def render_auc_comparison(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No registry data available.")
        return
    plot_df = df.dropna(subset=["test_auc"]).copy()
    color_map = {m: MODEL_COLORS.get(m, "#64748b") for m in plot_df["model"].unique()}
    fig = px.bar(
        plot_df,
        x="target",
        y="test_auc",
        color="model",
        color_discrete_map=color_map,
        barmode="group",
        title="Test AUC by Model and Target",
        labels={"test_auc": "Test AUC", "target": "Target", "model": "Model"},
    )
    apply_plotly_layout(fig)
    fig.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)


def render_gene_importance_bar(df: pd.DataFrame, title: str, value_col: str = "importance") -> None:
    if df.empty:
        st.info("No data to chart.")
        return
    plot_df = df.head(15).sort_values(value_col, ascending=True)
    gene_col = "gene" if "gene" in plot_df.columns else plot_df.columns[0]
    fig = px.bar(
        plot_df,
        x=value_col,
        y=gene_col,
        orientation="h",
        title=title,
        color_discrete_sequence=[PRIMARY],
        labels={value_col: value_col.replace("_", " ").title(), gene_col: "Gene"},
    )
    apply_plotly_layout(fig)
    fig.update_layout(height=max(320, len(plot_df) * 22))
    st.plotly_chart(fig, use_container_width=True)


def render_overlap_summary(df: pd.DataFrame) -> None:
    if df.empty or "n_sources" not in df.columns:
        return
    counts = df.groupby("n_sources").size().reset_index(name="gene_count")
    fig = px.bar(
        counts,
        x="n_sources",
        y="gene_count",
        title="Genes by number of overlapping sources",
        color_discrete_sequence=[PRIMARY],
        labels={"n_sources": "Sources", "gene_count": "Genes"},
    )
    apply_plotly_layout(fig)
    st.plotly_chart(fig, use_container_width=True)
