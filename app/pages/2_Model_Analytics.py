"""Model Analytics page."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.components.charts import render_confusion_matrix, render_roc_curve
from app.components.metrics import render_disclaimer, render_metric_cards
from app.config import MODELS, TARGETS
from app.inference.loaders import four_mod_available
from app.services.analytics import compute_metrics


def render() -> None:
    st.title("Model Analytics")
    st.markdown("Explore dataset info, metrics, and evaluation charts per model.")

    model_options = [m for m in MODELS if m != "4-Modality" or True]
    model = st.selectbox("Model", model_options, key="analytics_model")
    target = st.selectbox("Target", TARGETS, key="analytics_target")

    if model == "4-Modality" and not four_mod_available(target):
        st.warning("4-Modality artifacts not available yet. Metrics will be unavailable.")

    metrics = compute_metrics(model, target)
    if not metrics.get("available"):
        st.error("No test predictions or metadata found for this model/target combination.")
        render_disclaimer()
        return

    meta = metrics["metadata"]
    st.markdown("### Dataset Information")
    c1, c2, c3 = st.columns(3)
    c1.metric("Train patients", meta.get("train_patients", "N/A"))
    c2.metric("Test patients", meta.get("test_patients", "N/A"))
    c3.metric("Model type", meta.get("model_type", meta.get("level1_model_type", "N/A")))

    st.markdown("### Metrics")
    render_metric_cards(metrics["scores"])

    st.markdown("### Confusion Matrix")
    labels = None
    if target == "Stage":
        labels = ["1", "2", "3", "4"]
    elif target in ("OS_STATUS", "PFS_STATUS"):
        labels = ["0", "1"]
    render_confusion_matrix(metrics["confusion_matrix"], labels=labels)

    if metrics.get("roc_data"):
        st.markdown("### ROC Curve")
        render_roc_curve(metrics["roc_data"])

    with st.expander("Metadata"):
        st.json(meta)

    st.markdown("### Future: SHAP Explainability")
    st.info("SHAP-based feature attribution — coming soon.")

    render_disclaimer()


render()
