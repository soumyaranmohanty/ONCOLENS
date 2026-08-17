"""Load processed feature tables and compute cohort intersections."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from app.config import CLINICAL_PATH, FEATURE_STORE, SINGLE_MOD_STORE, TARGETS


def _read_features(modality: str, target: str) -> pd.DataFrame | None:
    path = SINGLE_MOD_STORE[modality] / target / "features.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    id_col = None
    for col in ("PATIENT_ID", "patient_id", "Patient_ID"):
        if col in df.columns:
            id_col = col
            break
    if id_col is None:
        if "Unnamed: 0" in df.columns:
            id_col = "Unnamed: 0"
        elif df.columns[0] not in [c for c in df.columns if str(c).startswith("embed_")]:
            df = df.rename(columns={df.columns[0]: "PATIENT_ID"})
            id_col = "PATIENT_ID"
    if id_col:
        df = df.rename(columns={id_col: "PATIENT_ID"})
        df = df.set_index("PATIENT_ID")
    drop_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def load_clinical() -> pd.DataFrame:
    clinical = pd.read_csv(CLINICAL_PATH)
    clinical = clinical.rename(columns={"PATIENT_ID": "patient_id"})
    clinical = clinical.set_index("patient_id")
    clinical["Stage"] = clinical["AJCC_PATHOLOGIC_TUMOR_STAGE"].apply(
        lambda x: int(x) if pd.notna(x) else float("nan")
    )
    return clinical


def load_processed_tables() -> dict[str, Any]:
    clinical = load_clinical()
    tables: dict[str, Any] = {"clinical": clinical}

    per_target: dict[str, dict[str, pd.DataFrame | None]] = {}
    intersections: dict[str, dict[str, set[str]]] = {}

    for target in TARGETS:
        expr = _read_features("Expression", target)
        mut = _read_features("Mutation", target)
        histo = _read_features("Histopathology", target)
        per_target[target] = {"Expression": expr, "Mutation": mut, "Histopathology": histo}

        sets = [set(clinical.index.astype(str))]
        names = ["Clinical"]
        for name, df in (("Expression", expr), ("Mutation", mut), ("Histopathology", histo)):
            if df is not None:
                sets.append(set(df.index.astype(str)))
                names.append(name)

        three_way = sets[0].copy()
        for s in sets[1:3] if len(sets) >= 3 else []:
            three_way &= s

        four_way = three_way.copy()
        if histo is not None:
            four_way &= set(histo.index.astype(str))

        intersections[target] = {
            "three_way": three_way,
            "four_way": four_way,
            "modalities": names,
        }

    tables["per_target"] = per_target
    tables["intersections"] = intersections
    return tables


def _test_patient_ids(target: str) -> set[str]:
    meta_path = FEATURE_STORE / "multimodal_model" / target / "metadata.json"
    if not meta_path.exists():
        return set()
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    return set(meta.get("test_patient_ids", []))


def demo_patient_ids(
    target: str,
    four_mod: bool = False,
    require_modality: str | None = None,
) -> list[str]:
    tables = load_processed_tables()
    key = "four_way" if four_mod else "three_way"
    ids = tables["intersections"][target][key]
    histo = tables["per_target"][target]["Histopathology"]
    if four_mod and histo is not None:
        ids = ids & set(histo.index.astype(str))
    ids &= _test_patient_ids(target)
    if require_modality:
        mod_df = tables["per_target"][target].get(require_modality)
        if mod_df is not None:
            ids &= set(mod_df.index.astype(str))
    return sorted(ids)


def get_demo_patient_row(patient_id: str, target: str) -> dict[str, pd.DataFrame]:
    tables = load_processed_tables()
    clinical = tables["clinical"]
    rows: dict[str, pd.DataFrame] = {}

    if patient_id in clinical.index:
        rows["Clinical"] = clinical.loc[[patient_id]]

    for modality in ("Expression", "Mutation", "Histopathology"):
        df = tables["per_target"][target].get(modality)
        if df is not None and patient_id in df.index:
            rows[modality] = df.loc[[patient_id]]

    return rows
