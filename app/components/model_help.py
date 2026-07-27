"""Shared model architecture help for UI pages."""

from __future__ import annotations

import streamlit as st

from app.config import MODEL_DESCRIPTIONS, MODELS, MODELS_OVERVIEW


def render_models_overview(expanded: bool = False) -> None:
    with st.expander("Model architecture overview", expanded=expanded):
        st.markdown(MODELS_OVERVIEW)
        st.markdown("**Backend descriptions**")
        for name in MODELS:
            st.markdown(f"- **{name}:** {MODEL_DESCRIPTIONS[name]}")
        st.markdown(f"- **Compare All:** {MODEL_DESCRIPTIONS['Compare All']}")


def render_model_caption(model_name: str) -> None:
    desc = MODEL_DESCRIPTIONS.get(model_name)
    if desc:
        st.caption(desc)
