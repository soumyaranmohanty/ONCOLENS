"""Prediction functions for single and multimodal models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.config import RISK_THRESHOLDS, SINGLE_MOD_STORE, TARGET_LABEL_MAPS
from app.inference.align import align_modality, align_single_modality
from app.inference.loaders import (
    detect_modality_count,
    four_mod_available,
    load_multimodal_bundle,
    load_single_model,
)
from app.inference.meta import build_meta_features


def _risk_band(prob_positive: float) -> str:
    if prob_positive < RISK_THRESHOLDS["low"]:
        return "Low"
    if prob_positive < RISK_THRESHOLDS["medium"]:
        return "Medium"
    return "High"


def _format_binary_result(
    target: str,
    pred_label: int,
    proba: np.ndarray,
    classes: np.ndarray,
    model_name: str,
    patient_id: str,
) -> dict[str, Any]:
    label_map = TARGET_LABEL_MAPS[target]
    pos_idx = 1 if len(classes) > 1 else 0
    pos_prob = float(proba[pos_idx]) if len(proba) > 1 else float(proba[0])
    return {
        "patient_id": patient_id,
        "model": model_name,
        "target": target,
        "predicted_label": int(pred_label),
        "predicted_label_name": label_map.get(int(pred_label), str(pred_label)),
        "probability": pos_prob,
        "probabilities": {str(int(c)): float(p) for c, p in zip(classes, proba)},
        "confidence": float(max(proba)),
        "risk_band": _risk_band(pos_prob),
        "available": True,
    }


def _format_multiclass_result(
    target: str,
    pred_label: int,
    proba: np.ndarray,
    classes: np.ndarray,
    model_name: str,
    patient_id: str,
) -> dict[str, Any]:
    label_map = TARGET_LABEL_MAPS[target]
    conf = float(max(proba))
    return {
        "patient_id": patient_id,
        "model": model_name,
        "target": target,
        "predicted_label": int(pred_label),
        "predicted_label_name": label_map.get(int(pred_label), f"Stage {pred_label}"),
        "probability": conf,
        "probabilities": {str(int(c)): float(p) for c, p in zip(classes, proba)},
        "confidence": conf,
        "risk_band": label_map.get(int(pred_label), str(pred_label)),
        "available": True,
    }


def _get_feature_columns(modality: str, target: str) -> list[str]:
    path = SINGLE_MOD_STORE[modality] / target / "features.csv"
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    skip = {"PATIENT_ID", "patient_id", "Patient_ID", "Unnamed: 0"}
    return [c for c in cols if c not in skip and not str(c).startswith("Unnamed")]


def _unavailable_result(model_name: str, target: str, patient_id: str, reason: str) -> dict[str, Any]:
    return {
        "patient_id": patient_id,
        "model": model_name,
        "target": target,
        "available": False,
        "reason": reason,
    }


def predict_single(
    modality: str,
    target: str,
    patient: dict[str, Any],
) -> dict[str, Any]:
    patient_id = patient["patient_id"]
    if modality not in patient:
        return _unavailable_result(modality, target, patient_id, f"Missing {modality} data")

    try:
        model, metadata = load_single_model(target, modality)
    except FileNotFoundError as exc:
        return _unavailable_result(modality, target, patient_id, str(exc))

    raw = patient[modality]
    feature_cols = _get_feature_columns(modality, target)
    from_store = patient.get("from_feature_store", False)

    X = align_single_modality(
        modality,
        raw,
        feature_cols,
        target,
        from_feature_store=from_store,
    )

    proba = model.predict_proba(X)[0]
    pred = model.predict(X)[0]
    classes = model.classes_

    if target == "Stage":
        return _format_multiclass_result(target, int(pred), proba, classes, modality, patient_id)
    return _format_binary_result(target, int(pred), proba, classes, modality, patient_id)


def predict_multimodal(
    target: str,
    patient: dict[str, Any],
    four_mod: bool = False,
) -> dict[str, Any]:
    patient_id = patient["patient_id"]
    model_name = "4-Modality" if four_mod else "3-Modality"

    if four_mod and not four_mod_available(target):
        return _unavailable_result(
            model_name,
            target,
            patient_id,
            "4-modality bundle not available. Train multimodalv3 to enable.",
        )

    bundle = load_multimodal_bundle(target, four_mod=four_mod)
    if bundle is None:
        return _unavailable_result(model_name, target, patient_id, "Multimodal bundle not found")

    modality_order = bundle["level1"]["feature_order"]
    from_store = patient.get("from_feature_store", False)
    X_by_modality: dict[str, pd.DataFrame] = {}

    for mname in modality_order:
        if mname not in patient:
            if mname == "Histopathology":
                return _unavailable_result(
                    model_name,
                    target,
                    patient_id,
                    "Histopathology features required for 4-modality prediction.",
                )
            return _unavailable_result(model_name, target, patient_id, f"Missing {mname} data")
        X_by_modality[mname] = align_modality(
            mname,
            patient[mname],
            bundle["level0"].get(mname),
            target,
            from_feature_store=from_store,
        )

    meta_X = build_meta_features(bundle, X_by_modality)
    meta_model = bundle["level1"]["meta_model"]
    proba = meta_model.predict_proba(meta_X)[0]
    pred = meta_model.predict(meta_X)[0]
    classes = meta_model.classes_
    is_multiclass = bundle["metadata"]["is_multiclass"]

    modality_contributions = {}
    for mname in modality_order:
        rf = bundle["level0"][mname]["rf"]
        lr = bundle["level0"][mname]["lr"]
        fn = bundle["level0"][mname]["feature_names"]
        mproba = (rf.predict_proba(X_by_modality[mname][fn]) + lr.predict_proba(X_by_modality[mname][fn])) / 2
        if is_multiclass:
            modality_contributions[mname] = {str(int(c)): float(p) for c, p in zip(rf.classes_, mproba[0])}
        else:
            modality_contributions[mname] = float(mproba[0, 1])

    if is_multiclass:
        result = _format_multiclass_result(target, int(pred), proba, classes, model_name, patient_id)
    else:
        result = _format_binary_result(target, int(pred), proba, classes, model_name, patient_id)

    result["modality_contributions"] = modality_contributions
    result["modality_count"] = detect_modality_count(bundle)
    return result


def compare_all(target: str, patient: dict[str, Any]) -> list[dict[str, Any]]:
    results = []
    for modality in ("Expression", "Mutation", "Histopathology"):
        results.append(predict_single(modality, target, patient))
    results.append(predict_multimodal(target, patient, four_mod=False))
    results.append(predict_multimodal(target, patient, four_mod=True))
    return results
