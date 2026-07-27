"""Tools for the ONCOLENS Virtual Assistant ReAct agent."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool

from app.config import DISCLAIMER, MODEL_DESCRIPTIONS, MODELS, MODELS_OVERVIEW, TARGET_TITLES, TARGETS
from app.inference.predict import compare_all
from app.services.report import _expression_highlights, _mutation_flags
from llm.agents.tools import _format_predictions, _modalities

ONCOLENS_GLOSSARY: dict[str, str] = {
    "tp53": (
        "TP53 is a tumor suppressor gene. Mutations in TP53 are common in LUAD and are "
        "associated with more aggressive disease and poorer prognosis."
    ),
    "pfs": (
        "PFS (Progression-Free Survival) measures time without disease progression after treatment. "
        "OS (Overall Survival) measures time until death from any cause."
    ),
    "os": (
        "OS (Overall Survival) measures time until death from any cause during follow-up. "
        "ONCOLENS predicts OS_STATUS as Alive vs Deceased."
    ),
    "roc": (
        "ROC-AUC measures how well a model separates classes across thresholds. "
        "AUC=0.5 is random; higher values indicate better discrimination."
    ),
    "auc": (
        "AUC (Area Under the ROC Curve) summarizes model discrimination. "
        "Higher AUC indicates better separation between outcome classes."
    ),
    "stage": (
        "AJCC pathologic stage (I–IV) describes extent of LUAD based on tumor size, "
        "node involvement, and metastasis."
    ),
    "modalities": (
        "ONCOLENS combines Gene Expression, Mutation, Clinical, and Histopathology modalities "
        "using late-fusion stacking for survival and staging prediction."
    ),
    "luad": (
        "Lung Adenocarcinoma (LUAD) is the focus cancer type in ONCOLENS, using TCGA cohort data."
    ),
    "tmb": (
        "Tumor Mutational Burden (TMB) counts somatic mutations per megabase. "
        "High TMB (≥10 in ONCOLENS) may support immunotherapy discussion."
    ),
    "egfr": "EGFR mutations in LUAD may indicate eligibility for EGFR tyrosine kinase inhibitors.",
    "kras": "KRAS mutations in LUAD affect targeted therapy options; G12C may enable KRAS inhibitors.",
}


def build_assistant_tools(
    patient: dict[str, Any] | None,
    last_prediction: dict[str, Any] | None,
    *,
    default_target: str = "OS_STATUS",
    prediction_history: list[dict[str, Any]] | None = None,
) -> list[StructuredTool]:
    """Create assistant-scoped tools for a single chat turn."""

    history = prediction_history or []

    def get_oncolens_overview() -> str:
        """Return ONCOLENS project scope, prediction targets, and model backends."""
        model_lines = "\n".join(f"- {name}: {MODEL_DESCRIPTIONS[name]}" for name in MODELS)
        target_lines = "\n".join(
            f"- {t}: {TARGET_TITLES.get(t, t)}" for t in TARGETS
        )
        return (
            "ONCOLENS is a research decision-support tool for TCGA lung adenocarcinoma (LUAD).\n"
            f"{MODELS_OVERVIEW}\n\n"
            f"Prediction targets:\n{target_lines}\n\n"
            f"Model backends:\n{model_lines}\n\n"
            f"Disclaimer: {DISCLAIMER}"
        )

    def get_oncolens_glossary(topic: str = "") -> str:
        """Return educational definitions for oncology/ONCOLENS terms (e.g. TP53, PFS, ROC, stage)."""
        key = topic.strip().lower()
        if not key:
            entries = "\n".join(f"- {k}: {v[:120]}..." if len(v) > 120 else f"- {k}: {v}"
                                for k, v in ONCOLENS_GLOSSARY.items())
            return f"Available glossary topics:\n{entries}"
        for term, definition in ONCOLENS_GLOSSARY.items():
            if term in key or key in term:
                return definition
        return (
            f"No exact glossary match for '{topic}'. "
            f"Available topics: {', '.join(sorted(ONCOLENS_GLOSSARY))}."
        )

    def get_active_patient_summary() -> str:
        """Return the active session patient ID and loaded data modalities."""
        if patient is None:
            return "No active patient loaded. Load a patient on the Patient Prediction page."
        pid = patient.get("patient_id", "Unknown")
        mods = _modalities(patient)
        source = "TCGA Demo" if patient.get("from_feature_store") else "Uploaded CSV"
        return (
            f"Patient ID: {pid}\n"
            f"Data source: {source}\n"
            f"Modalities: {', '.join(mods) if mods else 'None'}"
        )

    def get_last_session_prediction() -> str:
        """Return the most recent prediction from the current session."""
        if last_prediction is None:
            return "No prediction in session. Run a prediction on the Patient Prediction page first."
        if not last_prediction.get("available", True):
            return (
                f"Last prediction unavailable — model: {last_prediction.get('model')}, "
                f"reason: {last_prediction.get('reason', 'N/A')}"
            )
        prob = last_prediction.get("probability")
        prob_str = f"{prob:.1%}" if isinstance(prob, (int, float)) else "N/A"
        return (
            f"Patient: {last_prediction.get('patient_id', 'Unknown')}\n"
            f"Model: {last_prediction.get('model')} | Target: {last_prediction.get('target')}\n"
            f"Prediction: {last_prediction.get('predicted_label_name')} "
            f"(confidence {last_prediction.get('confidence', 0):.2f})\n"
            f"Risk band: {last_prediction.get('risk_band')} | Probability: {prob_str}"
        )

    def get_patient_biomarkers() -> str:
        """Return mutation flags, TMB, and top expression highlights for the active patient."""
        if patient is None:
            return "No active patient loaded — biomarkers unavailable."
        flags = _mutation_flags(patient)
        expr = _expression_highlights(patient)
        lines = ["Mutation / biomarker flags:"]
        lines.extend(f"  • {f}" for f in flags)
        lines.append("\nTop expression highlights (log1p RSEM):")
        lines.extend(f"  • {h}" for h in expr)
        histo = patient.get("Histopathology")
        if histo is not None:
            row = histo.iloc[0]
            norm = float((row**2).sum() ** 0.5)
            lines.append(f"\nHistopathology: 2048-D embedding available (L2 norm={norm:.3f}).")
        else:
            lines.append("\nHistopathology: not available.")
        return "\n".join(lines)

    def get_patient_all_predictions() -> str:
        """Run all five ONCOLENS model backends for the active patient and return results."""
        if patient is None:
            return "No active patient loaded — cannot run predictions."
        try:
            predictions = compare_all(default_target, patient)
        except Exception as exc:
            return f"Prediction run failed: {exc}"
        header = f"All-model predictions (target: {TARGET_TITLES.get(default_target, default_target)}):\n"
        return header + _format_predictions(predictions)

    def get_session_prediction_history() -> str:
        """Return recent prediction history entries from the current session."""
        if not history:
            return "No prediction history in this session."
        lines = []
        for entry in history[:10]:
            lines.append(
                f"- {entry.get('timestamp', '?')}: patient {entry.get('patient_id')} | "
                f"{entry.get('model')} / {entry.get('target')} → {entry.get('prediction')} "
                f"(risk {entry.get('risk_band')}, conf {entry.get('confidence', 0):.2f})"
            )
        return "\n".join(lines)

    return [
        StructuredTool.from_function(get_oncolens_overview),
        StructuredTool.from_function(get_oncolens_glossary),
        StructuredTool.from_function(get_active_patient_summary),
        StructuredTool.from_function(get_last_session_prediction),
        StructuredTool.from_function(get_patient_biomarkers),
        StructuredTool.from_function(get_patient_all_predictions),
        StructuredTool.from_function(get_session_prediction_history),
    ]
