"""Model Analytics page."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.components.charts import render_confusion_matrix, render_roc_curve
from app.components.layout import coming_soon_badge, render_disclaimer, section
from app.components.metrics import render_metric_cards
from app.components.model_help import render_model_caption, render_models_overview
from app.config import MODELS, TARGETS, human_class_label
from app.inference.loaders import four_mod_available
from app.services.analytics import compute_metrics


def render() -> None:
    section(
        "Model Analytics",
        "Dataset info, test metrics, and evaluation charts for each prediction backend.",
    )

    c1, c2 = st.columns(2)
    with c1:
        model = st.selectbox("Model", MODELS, key="analytics_model")
    with c2:
        target = st.selectbox("Target", TARGETS, key="analytics_target")
    render_model_caption(model)

    if model == "4-Modality" and not four_mod_available(target):
        st.warning("4-Modality artifacts are not available for this target yet.")

    metrics = compute_metrics(model, target)
    if not metrics.get("available"):
        st.error("No test predictions or metadata found for this model/target combination.")
        render_disclaimer()
        return

    meta = metrics["metadata"]
    section("Dataset")
    k1, k2, k3 = st.columns(3)
    k1.metric("Train patients", meta.get("train_patients", "N/A"))
    k2.metric("Test patients", meta.get("test_patients", "N/A"))
    k3.metric("Model type", meta.get("model_type", meta.get("level1_model_type", "N/A")))

    section("Metrics")
    render_metric_cards(metrics["scores"], emphasize="roc_auc")

    labels = None
    if target == "Stage":
        labels = [human_class_label(target, i) for i in range(1, 5)]
    elif target in ("OS_STATUS", "PFS_STATUS"):
        labels = [human_class_label(target, i) for i in (0, 1)]

    section("Evaluation charts")
    chart_left, chart_right = st.columns(2)
    with chart_left:
        with st.container(border=True):
            render_confusion_matrix(metrics["confusion_matrix"], labels=labels)
    with chart_right:
        with st.container(border=True):
            if metrics.get("roc_data"):
                render_roc_curve(metrics["roc_data"])
            else:
                st.caption("ROC curve not applicable for multiclass Stage target.")

    with st.expander("Metadata"):
        st.json(meta)

    section("Explainability")
    coming_soon_badge("SHAP feature attribution — coming soon")

    render_disclaimer()


render()
