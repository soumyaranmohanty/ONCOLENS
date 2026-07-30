"""Metric display components."""

from __future__ import annotations

from typing import Any

import streamlit as st

from app.components.cards import prediction_result_card
from app.components.layout import render_disclaimer


def render_prediction_card(result: dict[str, Any]) -> None:
    prediction_result_card(result)


def render_metric_cards(scores: dict[str, float], emphasize: str | None = "roc_auc") -> None:
    keys = list(scores.keys())
    cols = st.columns(len(keys))
    for col, name in zip(cols, keys):
        val = scores[name]
        label = name.replace("_", " ").title()
        if name == emphasize:
            col.markdown(
                f'<p class="oncolens-pred-label">{label}</p>'
                f'<p class="oncolens-pred-metric">{val:.3f}</p>',
                unsafe_allow_html=True,
            )
        else:
            col.metric(label, f"{val:.3f}")


__all__ = ["render_prediction_card", "render_metric_cards", "render_disclaimer"]
