"""Build patient dict from uploads or demo selection."""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.config import CLINICAL_FEATS_PER_TARGET
from app.inference.tables import get_demo_patient_row


def _normalize_patient_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ("patient_id", "PATIENT_ID", "Patient_ID"):
        if col in df.columns:
            return df.set_index(col)
    if df.shape[0] == 1 and df.index.name is None:
        return df
    return df


def make_uploaded_patient(
    patient_id: str,
    expression: pd.DataFrame | None = None,
    mutation: pd.DataFrame | None = None,
    clinical: pd.DataFrame | None = None,
    histopathology: pd.DataFrame | None = None,
) -> dict[str, Any]:
    patient: dict[str, Any] = {"patient_id": patient_id, "from_feature_store": False}

    if clinical is not None:
        patient["Clinical"] = _normalize_patient_id(clinical)
    else:
        feats = CLINICAL_FEATS_PER_TARGET["OS_STATUS"]
        patient["Clinical"] = pd.DataFrame([{}], index=[patient_id]).reindex(columns=feats)

    if expression is not None:
        patient["Expression"] = _normalize_patient_id(expression)
    if mutation is not None:
        patient["Mutation"] = _normalize_patient_id(mutation)
    if histopathology is not None:
        patient["Histopathology"] = _normalize_patient_id(histopathology)

    return patient


def make_demo_patient(patient_id: str, target: str) -> dict[str, Any]:
    rows = get_demo_patient_row(patient_id, target)
    return {
        "patient_id": patient_id,
        "from_feature_store": True,
        **rows,
    }
