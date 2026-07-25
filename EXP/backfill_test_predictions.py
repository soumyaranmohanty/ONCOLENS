"""
Backfill test_predictions.csv for the Expression and Mutation modalities.

Reproduces the exact 80/20 stratified split (stratify=OS_STATUS, random_state=42,
test_size=0.2) used in gene_expression_v2.ipynb / new_mutation_data.ipynb, refits
the selected model type on the train split only, and writes honest held-out
predictions in the same schema as the Multimodal model's test_predictions.csv:

  binary targets (OS_STATUS, PFS_STATUS): patient_id, true_label, predicted_label, predicted_probability
  multiclass target (Stage):              patient_id, true_label, predicted_label, probability_class_1..4

Run once from the EXP/ directory: python backfill_test_predictions.py
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

CLINICAL_PATH = "../Data/processed_data/mutation_data_processed/selected_clinical.csv"

# (feature_store_root, RF n_estimators, LR C) per modality, matching each
# notebook's train_and_eval() hyperparameters exactly.
MODALITIES = {
    "expression": {"root": "../Data/feature_store/expression", "rf_n_estimators": 300, "lr_c": 0.1},
    "mutation": {"root": "../Data/feature_store/mutation", "rf_n_estimators": 500, "lr_c": 0.5},
}

TARGETS = ["OS_STATUS", "PFS_STATUS", "Stage"]


def load_clinical():
    clinical = pd.read_csv(CLINICAL_PATH, index_col="PATIENT_ID")
    clinical["Stage"] = clinical["AJCC_PATHOLOGIC_TUMOR_STAGE"].apply(
        lambda x: int(x) if pd.notna(x) else np.nan
    )
    clinical["OS_STATUS"] = clinical["OS_STATUS"].astype(int)
    clinical["PFS_STATUS"] = clinical["PFS_STATUS"].astype(int)
    return clinical


def build_model(model_type, n_estimators, c):
    if model_type == "Random Forest":
        return RandomForestClassifier(
            n_estimators=n_estimators, random_state=42, class_weight="balanced", n_jobs=-1
        )
    if model_type == "Logistic (LASSO)":
        return LogisticRegression(
            solver="saga", l1_ratio=1, C=c, random_state=42, class_weight="balanced", max_iter=5000
        )
    raise ValueError(f"Unknown model_type: {model_type}")


def backfill_modality(name, cfg, clinical):
    print(f"\n=== {name} ===")
    for target in TARGETS:
        store_dir = os.path.join(cfg["root"], target)
        features_path = os.path.join(store_dir, "features.csv")
        meta_path = os.path.join(store_dir, "metadata.json")

        features = pd.read_csv(features_path, index_col=0)
        with open(meta_path) as f:
            meta = json.load(f)

        patients = np.array(features.index.tolist())
        labels = clinical.loc[patients]

        # Same split for all 3 targets within a modality: stratify on OS_STATUS,
        # random_state=42, test_size=0.2 (matches both source notebooks).
        train_idx, test_idx = train_test_split(
            patients, test_size=0.2, stratify=labels["OS_STATUS"].to_numpy(), random_state=42
        )

        X_train, X_test = features.loc[train_idx], features.loc[test_idx]
        y_train, y_test = labels.loc[train_idx, target], labels.loc[test_idx, target]

        assert len(test_idx) == meta["test_patients"], (
            f"{name}/{target}: reproduced test split size {len(test_idx)} != "
            f"recorded test_patients {meta['test_patients']} — split does not match original"
        )

        model = build_model(meta["model_type"], cfg["rf_n_estimators"], cfg["lr_c"])
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        pred = model.predict(X_test)

        if target == "Stage":
            classes = model.classes_
            out = pd.DataFrame(
                {
                    "patient_id": test_idx,
                    "true_label": y_test.values,
                    "predicted_label": pred,
                }
            )
            for i, cls in enumerate(classes):
                out[f"probability_class_{int(cls)}"] = proba[:, i]
        else:
            out = pd.DataFrame(
                {
                    "patient_id": test_idx,
                    "true_label": y_test.values,
                    "predicted_label": pred,
                    "predicted_probability": proba[:, 1],
                }
            )

        out_path = os.path.join(store_dir, "test_predictions.csv")
        out.to_csv(out_path, index=False)
        print(f"  {target}: wrote {len(out)} rows -> {out_path}")


if __name__ == "__main__":
    clinical = load_clinical()
    for name, cfg in MODALITIES.items():
        backfill_modality(name, cfg, clinical)
