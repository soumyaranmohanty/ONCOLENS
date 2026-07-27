"""Metric display components."""

from __future__ import annotations

from typing import Any

import streamlit as st

from app.config import (
    POSITIVE_CLASS_LABEL,
    TARGET_TITLES,
    build_interpretation,
    class_probability_caption,
)


RISK_COLORS = {"Low": "green", "Medium": "orange", "High": "red"}


def render_prediction_card(result: dict[str, Any]) -> None:
    if not result.get("available", True):
        pid = result.get("patient_id", "—")
        st.warning(f"**Patient `{pid}`** · **{result['model']}**: {result.get('reason', 'Unavailable')}")
        return

    pid = result.get("patient_id", "—")
    target = result.get("target", "")
    color = RISK_COLORS.get(result.get("risk_band", ""), "gray")
    target_title = TARGET_TITLES.get(target, target)

    st.markdown(
        f"**Patient `{pid}`** · **{result['model']}** · {target_title}  \n"
        f"**Prediction:** {result['predicted_label_name']} "
        f"(confidence: {result['confidence']:.1%}, risk: :{color}[{result['risk_band']}])"
    )

    probs = result.get("probabilities", {})
    if probs:
        interpretation = build_interpretation(
            target,
            result.get("predicted_label", 0),
            probs,
        )
        st.info(interpretation)

        with st.expander("Outcome probabilities", expanded=False):
            for cls, prob in probs.items():
                st.markdown(f"- {class_probability_caption(target, cls, prob)}")

    prob_label = result.get("probability_label")
    if prob_label and target in POSITIVE_CLASS_LABEL:
        st.caption(
            f"Primary risk score: **{result['probability']:.1%}** chance of "
            f"**{prob_label}** ({POSITIVE_CLASS_LABEL[target]} event)."
        )


def render_metric_cards(scores: dict[str, float]) -> None:
    cols = st.columns(5)
    for col, (name, val) in zip(cols, scores.items()):
        col.metric(name.replace("_", " ").title(), f"{val:.3f}")


def render_disclaimer() -> None:
    from app.config import DISCLAIMER

    st.caption(DISCLAIMER)
