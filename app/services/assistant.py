"""Rule-based virtual assistant (Level 1)."""

from __future__ import annotations

from typing import Any

FAQ = {
    "tp53": (
        "TP53 is a tumor suppressor gene. Mutations in TP53 are common in LUAD and are "
        "associated with more aggressive disease and poorer prognosis."
    ),
    "pfs": (
        "PFS (Progression-Free Survival) measures time without disease progression after treatment. "
        "OS (Overall Survival) measures time until death from any cause."
    ),
    "roc": (
        "ROC-AUC measures how well a model separates classes across thresholds. "
        "AUC=0.5 is random; higher values indicate better discrimination."
    ),
    "stage": (
        "AJCC pathologic stage (I–IV) describes extent of disease based on tumor size, "
        "node involvement, and metastasis."
    ),
    "modalities": (
        "ONCOLENS combines Gene Expression, Mutation, Clinical, and Histopathology modalities "
        "using late-fusion stacking for survival and staging prediction."
    ),
}


def answer_question(message: str, context: dict[str, Any] | None = None) -> str:
    msg = message.lower().strip()
    context = context or {}

    if "explain this prediction" in msg or "explain prediction" in msg:
        return _explain_prediction(context.get("last_prediction"), context.get("patient_id"))

    if "tp53" in msg:
        return FAQ["tp53"]
    if "pfs" in msg or "progression" in msg:
        return FAQ["pfs"]
    if "roc" in msg or "auc" in msg:
        return FAQ["roc"]
    if "stage" in msg:
        return FAQ["stage"]
    if "modality" in msg or "multimodal" in msg:
        return FAQ["modalities"]
    if "treatment" in msg:
        return (
            "Treatment recommendations in ONCOLENS are educational only. "
            "Discuss all options with a qualified oncologist."
        )

    return (
        "I can help explain TP53 mutations, PFS vs OS, ROC curves, staging, modalities, "
        "or your latest prediction. Try: 'Explain this prediction' or 'What is PFS?'"
    )


def _explain_prediction(pred: dict[str, Any] | None, patient_id: str | None = None) -> str:
    if not pred:
        msg = "No prediction in context. Run a prediction on the Patient Prediction page first."
        if patient_id:
            msg = f"No prediction in context for patient `{patient_id}`. Run a prediction on the Patient Prediction page first."
        return msg
    if not pred.get("available", True):
        pid = pred.get("patient_id") or patient_id or "unknown"
        return f"Patient `{pid}` — last prediction unavailable: {pred.get('reason', 'unknown')}"
    pid = pred.get("patient_id") or patient_id or "unknown"
    return (
        f"Patient: `{pid}`\n"
        f"Model: {pred['model']} | Target: {pred['target']}\n"
        f"Prediction: {pred['predicted_label_name']} (confidence {pred['confidence']:.2f})\n"
        f"Risk band: {pred['risk_band']} | Probability: {pred.get('probability', 0):.1%}"
    )
