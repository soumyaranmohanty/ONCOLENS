"""CSV upload form helpers."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def read_upload(label: str, key: str) -> pd.DataFrame | None:
    file = st.file_uploader(label, type=["csv"], key=key)
    if file is None:
        return None
    return pd.read_csv(file)


def parse_uploads(
    expression_file,
    mutation_file,
    clinical_file,
    histo_file,
) -> dict[str, pd.DataFrame | None]:
    out: dict[str, pd.DataFrame | None] = {
        "Expression": None,
        "Mutation": None,
        "Clinical": None,
        "Histopathology": None,
    }
    if expression_file is not None:
        out["Expression"] = pd.read_csv(expression_file)
    if mutation_file is not None:
        out["Mutation"] = pd.read_csv(mutation_file)
    if clinical_file is not None:
        out["Clinical"] = pd.read_csv(clinical_file)
    if histo_file is not None:
        from app.services.histopathology_embed import embedding_from_upload

        out["Histopathology"] = embedding_from_upload(histo_file, "UPLOAD")
    return out
