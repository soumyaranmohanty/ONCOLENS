"""Visual card components for ONCOLENS."""

from __future__ import annotations

from typing import Any

import streamlit as st

from app.components.theme import risk_badge_class
from app.config import (
    POSITIVE_CLASS_LABEL,
    TARGET_TITLES,
    build_interpretation,
    class_probability_caption,
)


def risk_badge_html(risk_band: str) -> str:
    cls = risk_badge_class(risk_band)
    return f'<span class="oncolens-badge {cls}">{risk_band} risk</span>'


def patient_summary_card(
    patient_id: str,
    source: str,
    modalities: list[str],
) -> None:
    chips = []
    for mod in modalities:
        chip_cls = "oncolens-chip oncolens-chip--clinical" if mod == "Clinical" else "oncolens-chip"
        title = f"{mod} (fusion input)" if mod == "Clinical" else mod
        chips.append(f'<span class="{chip_cls}">{title}</span>')
    chips_html = "".join(chips) if chips else '<span class="oncolens-chip">No data</span>'
    st.markdown(
        f"""
        <div class="oncolens-card">
          <div class="oncolens-card-title">Active patient</div>
          <div class="oncolens-patient-id">{patient_id}</div>
          <p class="oncolens-card-sub">Source: {source}</p>
          <div class="oncolens-chips">{chips_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def prediction_result_card(result: dict[str, Any]) -> None:
    if not result.get("available", True):
        pid = result.get("patient_id", "—")
        st.warning(f"**{result['model']}** — unavailable: {result.get('reason', 'N/A')}")
        return

    pid = result.get("patient_id", "—")
    target = result.get("target", "")
    target_title = TARGET_TITLES.get(target, target)
    risk_html = risk_badge_html(result.get("risk_band", "Unknown"))
    pred_name = result.get("predicted_label_name", "N/A")
    conf = result.get("confidence", 0)

    with st.container(border=True):
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            st.markdown(f"**{result['model']}** · {target_title}")
            st.markdown(f"Patient `{pid}`")
            st.markdown(f"**Outcome:** {pred_name}", unsafe_allow_html=True)
        with c2:
            st.markdown('<p class="oncolens-pred-label">Confidence</p>', unsafe_allow_html=True)
            st.markdown(
                f'<p class="oncolens-pred-metric">{conf:.1%}</p>',
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown('<p class="oncolens-pred-label">Risk</p>', unsafe_allow_html=True)
            st.markdown(risk_html, unsafe_allow_html=True)

        probs = result.get("probabilities", {})
        if probs:
            st.info(
                build_interpretation(
                    target,
                    result.get("predicted_label", 0),
                    probs,
                )
            )
            with st.expander("Outcome probabilities", expanded=False):
                for cls, prob in probs.items():
                    st.markdown(f"- {class_probability_caption(target, cls, prob)}")

        prob_label = result.get("probability_label")
        if prob_label and target in POSITIVE_CLASS_LABEL:
            st.caption(
                f"Primary score: **{result['probability']:.1%}** chance of "
                f"**{POSITIVE_CLASS_LABEL[target]}**"
            )


def report_section_card(title: str, items: list[str] | str) -> None:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        if isinstance(items, str):
            st.write(items)
        else:
            for item in items:
                st.markdown(f"- {item}")


def quick_link_card(title: str, description: str, page_path: str, icon: str = "") -> None:
    label = f"{icon} {title}".strip()
    st.page_link(page_path, label=label)
    st.caption(description)


def compact_patient_chip(patient_id: str | None, last_pred_line: str | None = None) -> None:
    if not patient_id:
        return
    sub = f" · {last_pred_line}" if last_pred_line else ""
    st.markdown(
        f'<span class="oncolens-badge oncolens-badge--primary">Patient: {patient_id}{sub}</span>',
        unsafe_allow_html=True,
    )
