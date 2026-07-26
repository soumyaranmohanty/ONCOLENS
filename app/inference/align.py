"""Align raw modality DataFrames to model-expected feature order."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.config import CLINICAL_FEATS_PER_TARGET


def _ensure_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.name in (None, "patient_id", "PATIENT_ID"):
        if "patient_id" in df.columns:
            return df.set_index("patient_id")
        if "PATIENT_ID" in df.columns:
            return df.set_index("PATIENT_ID")
    return df


def align_expression(
    raw: pd.DataFrame,
    feature_names: list[str],
    preprocessing: dict[str, Any] | None = None,
    from_feature_store: bool = False,
) -> pd.DataFrame:
    df = _ensure_index(raw.copy())
    if from_feature_store:
        aligned = df.reindex(columns=feature_names, fill_value=0.0)
    else:
        numeric = df.select_dtypes(include=[np.number])
        if preprocessing and preprocessing.get("log1p"):
            numeric = np.log1p(numeric.clip(lower=0))
        aligned = numeric.reindex(columns=feature_names, fill_value=0.0)
    return aligned.astype(float)


def align_mutation(raw: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    df = _ensure_index(raw.copy())
    aligned = df.reindex(columns=feature_names, fill_value=0.0)
    return aligned.astype(float)


def align_clinical(
    raw: pd.DataFrame,
    feature_names: list[str],
    preprocessing: dict[str, Any] | None = None,
    target: str = "OS_STATUS",
) -> pd.DataFrame:
    df = _ensure_index(raw.copy())
    feats = feature_names or CLINICAL_FEATS_PER_TARGET[target]
    aligned = df.reindex(columns=feats).copy()
    if preprocessing and preprocessing.get("replace_minus_one_with_nan"):
        aligned = aligned.replace(-1, np.nan)
    medians = (preprocessing or {}).get("median_imputation_values")
    if medians:
        for col, val in medians.items():
            if col in aligned.columns:
                aligned[col] = aligned[col].fillna(val)
    aligned = aligned.fillna(aligned.median(numeric_only=True))
    aligned = aligned.fillna(0.0)
    return aligned.astype(float)


def align_histopathology(raw: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    df = _ensure_index(raw.copy())
    if not feature_names:
        embed_cols = [c for c in df.columns if str(c).startswith("embed_")]
        if not embed_cols:
            embed_cols = [c for c in df.columns if c not in ("patient_id", "PATIENT_ID")]
        feature_names = embed_cols
    aligned = df.reindex(columns=feature_names, fill_value=0.0)
    return aligned.astype(float)


def align_modality(
    name: str,
    raw_df: pd.DataFrame,
    level0_entry: dict[str, Any] | None,
    target: str,
    from_feature_store: bool = False,
) -> pd.DataFrame:
    feature_names = (level0_entry or {}).get("feature_names", [])
    preprocessing = (level0_entry or {}).get("preprocessing")

    if name == "Expression":
        return align_expression(raw_df, feature_names, preprocessing, from_feature_store)
    if name == "Mutation":
        return align_mutation(raw_df, feature_names)
    if name == "Clinical":
        return align_clinical(raw_df, feature_names, preprocessing, target)
    if name == "Histopathology":
        return align_histopathology(raw_df, feature_names)
    raise ValueError(f"Unknown modality: {name}")


def align_single_modality(
    modality: str,
    raw_df: pd.DataFrame,
    feature_columns: list[str],
    target: str,
    from_feature_store: bool = True,
) -> pd.DataFrame:
    if modality == "Expression":
        return align_expression(raw_df, feature_columns, from_feature_store=from_feature_store)
    if modality == "Mutation":
        return align_mutation(raw_df, feature_columns)
    if modality == "Histopathology":
        return align_histopathology(raw_df, feature_columns)
    if modality == "Clinical":
        return align_clinical(raw_df, feature_columns, target=target)
    raise ValueError(f"Unknown modality: {modality}")
