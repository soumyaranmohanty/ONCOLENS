"""Load registries, model bundles, and single-modality artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from app.config import (
    FEATURE_STORE,
    MULTIMODAL_3_DIR,
    MULTIMODAL_4_DIR,
    REGISTRY_FILES,
    SINGLE_MOD_STORE,
)


def resolve_artifact_path(stored_path: str | None, fallback: Path) -> Path:
    if stored_path:
        candidate = Path(stored_path)
        if candidate.exists():
            return candidate
    return fallback


def load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_registry(model_name: str) -> dict[str, Any]:
    path = REGISTRY_FILES.get(model_name)
    if path is None or not path.exists():
        return {}
    return load_json(path)


def detect_modality_count(bundle: dict[str, Any]) -> int:
    order = bundle["level1"]["feature_order"]
    return 4 if "Histopathology" in order else 3


def multimodal_bundle_path(target: str, four_mod: bool = False) -> Path:
    base = MULTIMODAL_4_DIR if four_mod else MULTIMODAL_3_DIR
    return base / target / "multimodal_stack.joblib"


def load_multimodal_bundle(target: str, four_mod: bool = False) -> dict[str, Any] | None:
    fallback = multimodal_bundle_path(target, four_mod=four_mod)
    if not fallback.exists():
        return None

    registry_key = "4-Modality" if four_mod else "3-Modality"
    registry = load_registry(registry_key)
    entry = registry.get(target, {})
    path = resolve_artifact_path(entry.get("artifact_path"), fallback)
    if not path.exists():
        return None
    return joblib.load(path)


def four_mod_available(target: str) -> bool:
    bundle = load_multimodal_bundle(target, four_mod=True)
    return bundle is not None and detect_modality_count(bundle) == 4


def load_single_model(target: str, modality: str) -> tuple[Any, dict[str, Any]]:
    store = SINGLE_MOD_STORE[modality]
    model_path = store / target / "model.joblib"
    meta_path = store / target / "metadata.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = joblib.load(model_path)
    metadata = load_json(meta_path) if meta_path.exists() else {}
    return model, metadata


def load_features_manifest(target: str, four_mod: bool = False) -> dict[str, Any]:
    base = MULTIMODAL_4_DIR if four_mod else MULTIMODAL_3_DIR
    path = base / target / "features_manifest.json"
    if path.exists():
        return load_json(path)
    return {}


def load_test_predictions(model_name: str, target: str) -> Path | None:
    if model_name in ("3-Modality", "4-Modality"):
        four_mod = model_name == "4-Modality"
        base = MULTIMODAL_4_DIR if four_mod else MULTIMODAL_3_DIR
        path = base / target / "test_predictions.csv"
    elif model_name in SINGLE_MOD_STORE:
        path = SINGLE_MOD_STORE[model_name] / target / "test_predictions.csv"
    else:
        return None
    return path if path.exists() else None


def load_metadata(model_name: str, target: str) -> dict[str, Any]:
    if model_name in ("3-Modality", "4-Modality"):
        four_mod = model_name == "4-Modality"
        base = MULTIMODAL_4_DIR if four_mod else MULTIMODAL_3_DIR
        path = base / target / "metadata.json"
    elif model_name in SINGLE_MOD_STORE:
        path = SINGLE_MOD_STORE[model_name] / target / "metadata.json"
    else:
        return {}
    return load_json(path) if path.exists() else {}


def all_registries_summary() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_name, reg_path in REGISTRY_FILES.items():
        if not reg_path.exists():
            if model_name == "4-Modality":
                continue
            rows.append({"model": model_name, "target": "-", "test_auc": None, "note": "registry missing"})
            continue
        registry = load_json(reg_path)
        for target in ("OS_STATUS", "PFS_STATUS", "Stage"):
            entry = registry.get(target)
            if not entry:
                continue
            auc = entry.get("test_auc") or entry.get("stacked_heldout_test_auc")
            rows.append(
                {
                    "model": model_name,
                    "target": target,
                    "test_auc": auc,
                    "train_patients": entry.get("train_patients"),
                    "test_patients": entry.get("test_patients"),
                    "total_patients": entry.get("total_patients"),
                    "model_type": entry.get("model_type"),
                }
            )
    return rows
