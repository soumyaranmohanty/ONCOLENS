"""Build level-1 meta-features from level-0 modality models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_meta_features_from_level0(
    level0: dict[str, Any],
    modality_order: list[str],
    is_multiclass: bool,
    X_by_modality: dict[str, pd.DataFrame],
) -> np.ndarray:
    modality_proba = []
    for mname in modality_order:
        rf_model = level0[mname]["rf"]
        lr_model = level0[mname]["lr"]
        feature_names = level0[mname]["feature_names"]
        X = X_by_modality[mname][feature_names]
        proba = (rf_model.predict_proba(X) + lr_model.predict_proba(X)) / 2
        modality_proba.append(proba if is_multiclass else proba[:, 1])

    if is_multiclass:
        return np.hstack(modality_proba)
    return np.column_stack(modality_proba)


def build_meta_features(
    bundle: dict[str, Any],
    X_by_modality: dict[str, pd.DataFrame],
) -> np.ndarray:
    is_multiclass = bundle["metadata"]["is_multiclass"]
    modality_order = bundle["level1"]["feature_order"]
    return build_meta_features_from_level0(
        bundle["level0"],
        modality_order,
        is_multiclass,
        X_by_modality,
    )
