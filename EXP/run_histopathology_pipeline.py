#!/usr/bin/env python3
"""Run histopathology_v1 notebook pipeline steps 7–13 with logging."""

from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = REPO_ROOT / "EXP"
sys.path.insert(0, str(EXP_DIR))

_brew_lib = Path("/opt/homebrew/lib")
if _brew_lib.is_dir():
    os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = (
        f"{_brew_lib}:{os.environ.get('DYLD_FALLBACK_LIBRARY_PATH', '')}"
    )

from histopathology_utils import (  # noqa: E402
    aggregate_patient_features,
    build_resnet50_encoder,
    cache_slide_embeddings,
)

SLIDE_MANIFEST_PATH = REPO_ROOT / "Data/raw_data/histopathology/slide_manifest.csv"
TILE_CACHE_DIR = REPO_ROOT / "Data/feature_store/histopathology/tile_embeddings_cache"
FEATURE_STORE_ROOT = REPO_ROOT / "Data/feature_store/histopathology"
CLINICAL_PATH = REPO_ROOT / "Data/processed_data/mutation_data_processed/selected_clinical.csv"
REGISTRY_PATH = REPO_ROOT / "Data/feature_store/registry_histopathology.json"
LOG_PATH = REPO_ROOT / "Data/feature_store/histopathology/pipeline.log"

TILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    line = msg if msg.endswith("\n") else msg + "\n"
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line)


def run_embedding_pipeline(slide_manifest: pd.DataFrame) -> None:
    encoder, device = build_resnet50_encoder()
    log(f"STEP 7 — Embedding on device: {device}")
    total = len(slide_manifest)
    for i, row in slide_manifest.iterrows():
        slide_id = Path(row["slide_id"]).stem
        log(f"  [{i + 1}/{total}] {row['slide_id']}")
        cache_slide_embeddings(slide_id, row["path"], TILE_CACHE_DIR, encoder, device)


def load_clinical() -> pd.DataFrame:
    clinical = pd.read_csv(CLINICAL_PATH, index_col="PATIENT_ID")
    clinical["Stage"] = clinical["AJCC_PATHOLOGIC_TUMOR_STAGE"].apply(
        lambda x: int(x) if pd.notna(x) else np.nan
    )
    clinical["OS_STATUS"] = clinical["OS_STATUS"].astype(int)
    clinical["PFS_STATUS"] = clinical["PFS_STATUS"].astype(int)
    return clinical


def _cv_n_splits(y_train: pd.Series, max_splits: int = 5) -> int:
    min_class = int(y_train.value_counts().min())
    if min_class < 2:
        return 0
    return min(max_splits, min_class)


def train_and_eval(X_train, y_train, X_test, y_test, label, scoring="roc_auc"):
    n_splits = _cv_n_splits(y_train)
    cv = (
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        if n_splits >= 2
        else None
    )
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1
        ),
        "Logistic (LASSO)": LogisticRegression(
            solver="saga",
            l1_ratio=1,
            C=0.1,
            random_state=42,
            class_weight="balanced",
            max_iter=5000,
        ),
    }

    multiclass = scoring == "roc_auc_ovr_weighted"
    if multiclass:
        labels = np.sort(y_train.unique())
        cv_scorer = make_scorer(
            roc_auc_score,
            response_method="predict_proba",
            multi_class="ovr",
            average="weighted",
            labels=labels,
        )

    log(f"\n--- {label} ---")
    if multiclass:
        log(f"  Stage classes (train): {y_train.value_counts().sort_index().to_dict()}")
    if cv is None:
        log("  CV skipped: at least one class has fewer than 2 training samples")

    res = {}
    for name, model in models.items():
        if cv is not None:
            cv_scoring = cv_scorer if multiclass else scoring
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=cv_scoring)
        else:
            cv_scores = np.array([np.nan])

        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)
        if multiclass:
            test_auc = roc_auc_score(
                y_test, y_prob, multi_class="ovr", average="weighted", labels=labels
            )
        else:
            test_auc = roc_auc_score(y_test, y_prob[:, 1])

        res[name] = {"cv_scores": cv_scores, "test_auc": test_auc, "model": model}
        if cv is None:
            cv_summary = "n/a (CV skipped)"
        else:
            cv_summary = f"{np.nanmean(cv_scores):.3f} +/- {np.nanstd(cv_scores):.3f}"
        log(f"  {name:<25} {cv_summary:<22}   test={test_auc:.3f}")
    return res


def save_target_to_feature_store(
    target, model_name, model, X_all_target, y_all, X_test, y_test, meta_extra
):
    store_dir = FEATURE_STORE_ROOT / target
    store_dir.mkdir(parents=True, exist_ok=True)

    X_all_target.to_csv(store_dir / "features.csv")
    joblib.dump(model, store_dir / "model.joblib")

    proba = model.predict_proba(X_test)
    pred = model.predict(X_test)
    if target == "Stage":
        out = pd.DataFrame(
            {"patient_id": X_test.index, "true_label": y_test.values, "predicted_label": pred}
        )
        for i, cls in enumerate(model.classes_):
            out[f"probability_class_{int(cls)}"] = proba[:, i]
    else:
        out = pd.DataFrame(
            {
                "patient_id": X_test.index,
                "true_label": y_test.values,
                "predicted_label": pred,
                "predicted_probability": proba[:, 1],
            }
        )
    out.to_csv(store_dir / "test_predictions.csv", index=False)

    meta = {
        "target": target,
        "model_type": model_name,
        "n_features": X_all_target.shape[1],
        "created": str(date.today()),
        **meta_extra,
    }
    with open(store_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def main() -> None:
    LOG_PATH.write_text("", encoding="utf-8")
    log("=" * 60)
    log("Histopathology pipeline — steps 7–13")
    log("=" * 60)

    slide_manifest = pd.read_csv(SLIDE_MANIFEST_PATH)
    log(f"Slides: {len(slide_manifest)}")

    run_embedding_pipeline(slide_manifest)
    features_all = aggregate_patient_features(slide_manifest, TILE_CACHE_DIR)
    log(f"STEP 7 done — patient matrix: {features_all.shape}")

    log("\nSTEP 9 — Merge clinical labels + train/test split")
    clinical = load_clinical()
    ml = features_all.join(clinical, how="inner").dropna(
        subset=["OS_STATUS", "PFS_STATUS", "Stage"]
    )
    embed_cols = list(features_all.columns)
    train_idx, test_idx = train_test_split(
        ml.index, test_size=0.2, stratify=ml["OS_STATUS"], random_state=42
    )
    X_all = ml[embed_cols]
    X_train = ml.loc[train_idx, embed_cols]
    X_test = ml.loc[test_idx, embed_cols]
    log(f"Cohort: {len(ml)} patients (train={len(train_idx)}, test={len(test_idx)})")

    log("\nSTEP 11 — Train RF/LR per target")
    os_results = train_and_eval(
        X_train,
        ml.loc[train_idx, "OS_STATUS"],
        X_test,
        ml.loc[test_idx, "OS_STATUS"],
        "OS_STATUS — histopathology",
    )
    pfs_results = train_and_eval(
        X_train,
        ml.loc[train_idx, "PFS_STATUS"],
        X_test,
        ml.loc[test_idx, "PFS_STATUS"],
        "PFS_STATUS — histopathology",
    )
    stage_results = train_and_eval(
        X_train,
        ml.loc[train_idx, "Stage"],
        X_test,
        ml.loc[test_idx, "Stage"],
        "Stage — histopathology",
        scoring="roc_auc_ovr_weighted",
    )

    log("\nSTEP 13 — Save feature store")
    registry = {}
    for target, results in {
        "OS_STATUS": os_results,
        "PFS_STATUS": pfs_results,
        "Stage": stage_results,
    }.items():
        best_name = max(results, key=lambda k: results[k]["test_auc"])
        best = results[best_name]
        meta = save_target_to_feature_store(
            target,
            best_name,
            best["model"],
            X_all,
            ml[target],
            X_test,
            ml.loc[test_idx, target],
            meta_extra={
                "cv_auc_mean": round(float(best["cv_scores"].mean()), 4),
                "cv_auc_std": round(float(best["cv_scores"].std()), 4),
                "test_auc": round(float(best["test_auc"]), 4),
                "train_patients": len(train_idx),
                "test_patients": len(test_idx),
                "total_patients": len(ml),
            },
        )
        registry[target] = meta

    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    log(f"Saved: {FEATURE_STORE_ROOT}")
    log(f"Registry: {REGISTRY_PATH}")
    log("PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
