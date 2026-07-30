"""Reusable layout helpers for ONCOLENS pages."""

from __future__ import annotations

from typing import Any

import streamlit as st

from app.config import DISCLAIMER


def page_hero(title: str, subtitle: str, badges: list[str] | None = None) -> None:
    badge_html = ""
    if badges:
        spans = "".join(f'<span class="oncolens-badge oncolens-badge--hero">{b}</span>' for b in badges)
        badge_html = f'<div class="oncolens-hero-badges">{spans}</div>'
    st.markdown(
        f"""
        <div class="oncolens-hero">
          <h1>{title}</h1>
          <p>{subtitle}</p>
          {badge_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str, subtitle: str | None = None) -> None:
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)


def kpi_row(items: list[tuple[str, str, str | None]]) -> None:
    """Each item: (label, value, delta optional)."""
    cols = st.columns(len(items))
    for col, (label, value, delta) in zip(cols, items):
        col.metric(label, value, delta=delta, delta_color="off" if delta else "normal")


def empty_state(message: str, hint: str | None = None) -> None:
    hint_html = f'<p style="font-size:0.8rem;margin-top:0.5rem;">{hint}</p>' if hint else ""
    st.markdown(
        f'<div class="oncolens-empty">{message}{hint_html}</div>',
        unsafe_allow_html=True,
    )


def workflow_steps(steps: list[tuple[str, str]], active_index: int) -> None:
    """Horizontal step indicator. status: pending | active | done inferred from index."""
    parts = []
    for i, (num, label) in enumerate(steps):
        if i < active_index:
            cls = "oncolens-step oncolens-step--done"
        elif i == active_index:
            cls = "oncolens-step oncolens-step--active"
        else:
            cls = "oncolens-step"
        parts.append(f'<div class="{cls}"><strong>{num}</strong> {label}</div>')
    st.markdown(f'<div class="oncolens-steps">{"".join(parts)}</div>', unsafe_allow_html=True)


def pipeline_steps(labels: list[str]) -> None:
    items = "".join(
        f'<div class="oncolens-workflow-item"><span class="oncolens-workflow-num">{i + 1}</span>{lbl}</div>'
        for i, lbl in enumerate(labels)
    )
    st.markdown(f'<div class="oncolens-workflow-row">{items}</div>', unsafe_allow_html=True)


def coming_soon_badge(label: str = "Coming soon") -> None:
    st.markdown(
        f'<span class="oncolens-badge oncolens-badge--soon">{label}</span>',
        unsafe_allow_html=True,
    )


def ai_status_banner(available: bool) -> None:
    if available:
        st.markdown(
            '<span class="oncolens-badge oncolens-badge--ai-on">AI connected</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="oncolens-badge oncolens-badge--ai-off">Template / offline mode</span>',
            unsafe_allow_html=True,
        )


def render_disclaimer() -> None:
    st.markdown(f'<div class="oncolens-disclaimer">{DISCLAIMER}</div>', unsafe_allow_html=True)


def report_preview(text: str) -> None:
    escaped = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    st.markdown(f'<div class="oncolens-report-preview">{escaped}</div>', unsafe_allow_html=True)
