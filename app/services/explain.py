"""Explainability helpers — feature importance and cross-model gene overlap."""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.config import FEATURE_STORE, SINGLE_MOD_STORE
from app.inference.loaders import all_registries_summary, load_multimodal_bundle, load_single_model

_NON_GENE_FEATURES = frozenset({"index", "TMB", "Unnamed: 0"})


def _read_feature_names(modality: str, target: str) -> list[str]:
    path = SINGLE_MOD_STORE[modality] / target / "features.csv"
    if not path.exists():
        return []
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    id_col = next((c for c in ("PATIENT_ID", "patient_id", "Patient_ID") if c in cols), cols[0])
    return [c for c in cols if c != id_col]


def _is_gene_feature(name: str) -> bool:
    return name not in _NON_GENE_FEATURES and not str(name).startswith("Unnamed")


def _importance_from_model(
    model: Any,
    feature_names: list[str],
    top_n: int,
    *,
    genes_only: bool = True,
) -> pd.DataFrame:
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame()

    importances = model.feature_importances_
    if len(feature_names) != len(importances):
        feature_names = [f"feat_{i}" for i in range(len(importances))]

    df = pd.DataFrame({"gene": feature_names, "importance": importances})
    if genes_only:
        df = df[df["gene"].map(_is_gene_feature)]
    if df.empty:
        return df
    return df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)


def top_important_genes_expression(target: str, top_n: int = 15) -> pd.DataFrame:
    """Top genes from the standalone Expression model (Random Forest feature importances)."""
    try:
        model, _ = load_single_model(target, "Expression")
    except FileNotFoundError:
        return pd.DataFrame()
    names = _read_feature_names("Expression", target)
    return _importance_from_model(model, names, top_n)


def top_important_genes_mutation(target: str, top_n: int = 15) -> pd.DataFrame:
    """Top genes from the standalone Mutation model (Random Forest feature importances)."""
    try:
        model, _ = load_single_model(target, "Mutation")
    except FileNotFoundError:
        return pd.DataFrame()
    names = _read_feature_names("Mutation", target)
    return _importance_from_model(model, names, top_n)


def top_multimodal_branch_genes(
    target: str,
    branch: str,
    top_n: int = 15,
) -> pd.DataFrame:
    """Top genes from a 3-Modality level-0 Random Forest branch (Expression or Mutation)."""
    bundle = load_multimodal_bundle(target, four_mod=False)
    if not bundle or branch not in bundle.get("level0", {}):
        return pd.DataFrame()

    level0 = bundle["level0"][branch]
    rf = level0.get("rf")
    names = level0.get("feature_names", [])
    if rf is None or not names:
        return pd.DataFrame()
    return _importance_from_model(rf, list(names), top_n)


def _top_gene_sets(target: str, top_n: int) -> dict[str, set[str]]:
    sources = {
        "Expression (standalone)": top_important_genes_expression(target, top_n),
        "Mutation (standalone)": top_important_genes_mutation(target, top_n),
        "3-Modality · Expression branch": top_multimodal_branch_genes(target, "Expression", top_n),
        "3-Modality · Mutation branch": top_multimodal_branch_genes(target, "Mutation", top_n),
    }
    return {
        label: set(df["gene"].astype(str).tolist())
        for label, df in sources.items()
        if not df.empty
    }


def overlapping_important_genes(
    target: str,
    top_n: int = 15,
    min_models: int = 2,
) -> pd.DataFrame:
    """
    Genes in the top-N list of at least `min_models` importance sources.
    Sources: standalone Expression/Mutation + 3-Modality Expression/Mutation branches.
    """
    gene_sets = _top_gene_sets(target, top_n)
    if len(gene_sets) < min_models:
        return pd.DataFrame()

    all_genes: set[str] = set()
    for genes in gene_sets.values():
        all_genes |= genes

    rows: list[dict[str, Any]] = []
    for gene in sorted(all_genes):
        hits = [label for label, genes in gene_sets.items() if gene in genes]
        if len(hits) >= min_models:
            rows.append(
                {
                    "gene": gene,
                    "n_sources": len(hits),
                    "sources": ", ".join(hits),
                }
            )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(["n_sources", "gene"], ascending=[False, True])
        .reset_index(drop=True)
    )


def top_important_genes(target: str, top_n: int = 15) -> pd.DataFrame:
    """Backward-compatible alias for standalone Expression importances."""
    return top_important_genes_expression(target, top_n)


def modality_auc_comparison() -> pd.DataFrame:
    return pd.DataFrame(all_registries_summary())


def load_modality_improvement_table() -> pd.DataFrame | None:
    path = FEATURE_STORE / "comparative_evaluation" / "modality_improvement_table.csv"
    if path.exists():
        return pd.read_csv(path)
    return None
