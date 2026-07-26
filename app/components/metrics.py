"""Metric display components."""

from __future__ import annotations

from typing import Any

import streamlit as st


RISK_COLORS = {"Low": "green", "Medium": "orange", "High": "red"}


def render_prediction_card(result: dict[str, Any]) -> None:
    if not result.get("available", True):
        st.warning(f"**{result['model']}**: {result.get('reason', 'Unavailable')}")
        return

    color = RISK_COLORS.get(result.get("risk_band", ""), "gray")
    st.markdown(
        f"**{result['model']}** — {result['predicted_label_name']} "
        f"(confidence: {result['confidence']:.1%}, risk: :{color}[{result['risk_band']}])"
    )
    if result.get("probabilities"):
        with st.expander("Class probabilities"):
            for cls, prob in result["probabilities"].items():
                st.write(f"Class {cls}: {prob:.1%}")


def render_metric_cards(scores: dict[str, float]) -> None:
    cols = st.columns(5)
    for col, (name, val) in zip(cols, scores.items()):
        col.metric(name.replace("_", " ").title(), f"{val:.3f}")


def render_disclaimer() -> None:
    from app.config import DISCLAIMER

    st.caption(DISCLAIMER)
