"""ONCOLENS design tokens, global CSS injection, and Plotly theme."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

# Color tokens
PRIMARY = "#0d9488"
PRIMARY_DARK = "#0f766e"
SURFACE = "#ffffff"
BACKGROUND = "#f1f5f9"
TEXT = "#0f172a"
MUTED = "#64748b"
BORDER = "#e2e8f0"
RISK_LOW = "#16a34a"
RISK_MEDIUM = "#ea580c"
RISK_HIGH = "#dc2626"

MODEL_COLORS = {
    "Expression": "#0d9488",
    "Mutation": "#4f46e5",
    "Histopathology": "#9333ea",
    "3-Modality": "#0284c7",
    "4-Modality": "#d97706",
    "Compare All": "#64748b",
}

PLOTLY_LAYOUT = {
    "font": {"family": "sans-serif", "color": TEXT, "size": 12},
    "paper_bgcolor": SURFACE,
    "plot_bgcolor": SURFACE,
    "margin": {"l": 48, "r": 24, "t": 48, "b": 48},
    "colorway": list(MODEL_COLORS.values()),
    "xaxis": {
        "gridcolor": BORDER,
        "linecolor": BORDER,
        "zerolinecolor": BORDER,
    },
    "yaxis": {
        "gridcolor": BORDER,
        "linecolor": BORDER,
        "zerolinecolor": BORDER,
    },
    "legend": {"bgcolor": "rgba(255,255,255,0.8)", "bordercolor": BORDER, "borderwidth": 1},
}


def inject_global_css() -> None:
    css_path = Path(__file__).resolve().parent.parent / "static" / "theme.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def risk_badge_class(risk_band: str) -> str:
    mapping = {
        "Low": "oncolens-badge--risk-low",
        "Medium": "oncolens-badge--risk-medium",
        "High": "oncolens-badge--risk-high",
    }
    return mapping.get(risk_band, "oncolens-badge--muted")


def apply_plotly_layout(fig, title: str | None = None):
    layout = dict(PLOTLY_LAYOUT)
    if title:
        layout["title"] = {"text": title, "font": {"size": 14, "color": TEXT}}
    fig.update_layout(**layout)
    return fig
