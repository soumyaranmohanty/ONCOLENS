"""Metrics and evaluation from test_predictions.csv."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from app.inference.loaders import load_metadata, load_test_predictions


def load_predictions_df(model_name: str, target: str) -> pd.DataFrame | None:
    path = load_test_predictions(model_name, target)
    if path is None:
        return None
    return pd.read_csv(path)


def _binary_scores(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }


def _multiclass_scores(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba, multi_class="ovr", average="weighted")),
    }


def compute_metrics(model_name: str, target: str) -> dict[str, Any]:
    df = load_predictions_df(model_name, target)
    metadata = load_metadata(model_name, target)
    if df is None or df.empty:
        return {"available": False}

    y_true = df["true_label"].to_numpy()
    y_pred = df["predicted_label"].to_numpy()

    if target == "Stage":
        prob_cols = [c for c in df.columns if c.startswith("probability_class_")]
        proba = df[prob_cols].to_numpy()
        scores = _multiclass_scores(y_true, y_pred, proba)
        roc_data = None
    else:
        y_score = df["predicted_probability"].to_numpy()
        scores = _binary_scores(y_true, y_pred, y_score)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        roc_data = {"fpr": fpr, "tpr": tpr, "auc": scores["roc_auc"], "precision": prec, "recall": rec}

    cm = confusion_matrix(y_true, y_pred)

    return {
        "available": True,
        "scores": scores,
        "confusion_matrix": cm,
        "roc_data": roc_data,
        "metadata": metadata,
        "n_samples": len(df),
    }
