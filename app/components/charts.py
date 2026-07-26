"""Reusable chart components."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_roc_curve(roc_data: dict[str, Any], title: str = "ROC Curve") -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=roc_data["fpr"],
            y=roc_data["tpr"],
            mode="lines",
            name=f"AUC={roc_data['auc']:.3f}",
        )
    )
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Random"))
    fig.update_layout(title=title, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(fig, use_container_width=True)


def render_confusion_matrix(cm: np.ndarray, labels: list[str] | None = None, title: str = "Confusion Matrix") -> None:
    if labels is None:
        labels = [str(i) for i in range(cm.shape[0])]
    fig = px.imshow(
        cm,
        text_auto=True,
        x=labels,
        y=labels,
        color_continuous_scale="Blues",
        title=title,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_comparison_bars(results: list[dict[str, Any]], target: str) -> None:
    rows = []
    for r in results:
        if not r.get("available", True):
            rows.append({"model": r["model"], "probability": 0, "status": "unavailable"})
        elif target == "Stage":
            rows.append({"model": r["model"], "probability": r["confidence"], "status": "ok"})
        else:
            rows.append({"model": r["model"], "probability": r["probability"], "status": "ok"})
    import pandas as pd

    df = pd.DataFrame(rows)
    fig = px.bar(df, x="model", y="probability", color="status", title=f"Model Comparison — {target}")
    st.plotly_chart(fig, use_container_width=True)


def render_auc_comparison(df) -> None:
    if df.empty:
        st.info("No registry data available.")
        return
    plot_df = df.dropna(subset=["test_auc"])
    fig = px.bar(
        plot_df,
        x="target",
        y="test_auc",
        color="model",
        barmode="group",
        title="Test AUC by Model and Target",
    )
    st.plotly_chart(fig, use_container_width=True)
