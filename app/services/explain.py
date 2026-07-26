"""Explainability helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.config import FEATURE_STORE, SINGLE_MOD_STORE
from app.inference.loaders import all_registries_summary, load_single_model


def top_important_genes(target: str, top_n: int = 15) -> pd.DataFrame:
    try:
        model, _ = load_single_model(target, "Expression")
    except FileNotFoundError:
        return pd.DataFrame()

    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame()

    path = SINGLE_MOD_STORE["Expression"] / target / "features.csv"
    if not path.exists():
        return pd.DataFrame()
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    id_col = "PATIENT_ID" if "PATIENT_ID" in cols else cols[0]
    feature_names = [c for c in cols if c != id_col]

    importances = model.feature_importances_
    if len(feature_names) != len(importances):
        feature_names = [f"feat_{i}" for i in range(len(importances))]

    df = pd.DataFrame({"gene": feature_names, "importance": importances})
    return df.sort_values("importance", ascending=False).head(top_n)


def frequently_mutated_genes(top_n: int = 15) -> pd.DataFrame:
    path = SINGLE_MOD_STORE["Mutation"] / "OS_STATUS" / "features.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0)
    gene_cols = [c for c in df.columns if c not in ("TMB", "index")]
    prevalence = df[gene_cols].mean().sort_values(ascending=False)
    out = prevalence.head(top_n).reset_index()
    out.columns = ["gene", "prevalence"]
    return out


def modality_auc_comparison() -> pd.DataFrame:
    return pd.DataFrame(all_registries_summary())


def load_modality_improvement_table() -> pd.DataFrame | None:
    path = FEATURE_STORE / "comparative_evaluation" / "modality_improvement_table.csv"
    if path.exists():
        return pd.read_csv(path)
    return None
