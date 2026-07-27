"""LangChain tools bound to a patient context for ReAct agents."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import StructuredTool

from app.config import TARGET_TITLES
from app.services.report import _expression_highlights, _mutation_flags
from app.services.treatment import generate_treatment_recommendations


def _modalities(patient: dict[str, Any]) -> list[str]:
    skip = {"patient_id", "from_feature_store"}
    return sorted(k for k in patient if k not in skip)


def _format_predictions(predictions: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for pred in predictions:
        target = pred.get("target", "")
        title = TARGET_TITLES.get(target, target)
        if not pred.get("available", True):
            lines.append(
                f"- {pred.get('model')} ({title}): unavailable — {pred.get('reason', 'N/A')}"
            )
            continue
        prob = pred.get("probability")
        prob_str = f"{prob:.1%}" if isinstance(prob, (int, float)) else "N/A"
        lines.append(
            f"- {pred.get('model')} ({title}): {pred.get('predicted_label_name')} "
            f"(confidence={pred.get('confidence', 0):.2f}, risk={pred.get('risk_band')}, "
            f"P(positive/class)={prob_str})"
        )
    return "\n".join(lines) or "No predictions provided."


def build_patient_tools(
    patient: dict[str, Any],
    predictions: list[dict[str, Any]],
    *,
    target: str,
) -> list[StructuredTool]:
    """Create patient-scoped tools for a single agent invocation."""

    pid = patient.get("patient_id", "Unknown")
    mods = _modalities(patient)
    pred_text = _format_predictions(predictions)
    rule_based_rec = generate_treatment_recommendations(patient, predictions)

    def get_patient_summary() -> str:
        """Return patient ID, target outcome, and loaded data modalities."""
        return (
            f"Patient ID: {pid}\n"
            f"Analysis target: {TARGET_TITLES.get(target, target)}\n"
            f"Modalities: {', '.join(mods) if mods else 'None'}"
        )

    def get_expression_highlights() -> str:
        """Return top log1p(RSEM) expression values from the model gene panel."""
        highlights = _expression_highlights(patient)
        return "\n".join(f"- {h}" for h in highlights)

    def get_mutation_flags() -> str:
        """Return driver mutation flags and TMB status from the mutation panel."""
        flags = _mutation_flags(patient)
        return "\n".join(f"- {f}" for f in flags)

    def get_histopathology_summary() -> str:
        """Return histopathology embedding availability and summary stats."""
        histo = patient.get("Histopathology")
        if histo is None:
            return "Histopathology data not available for this patient."
        row = histo.iloc[0]
        norm = float((row**2).sum() ** 0.5)
        return (
            f"Precomputed ResNet50 slide embedding available (2048-D, L2 norm={norm:.3f}). "
            "No image-level interpretation performed."
        )

    def get_model_predictions() -> str:
        """Return predictions from all ONCOLENS model backends for this patient."""
        return pred_text

    def get_rule_based_treatment_baseline() -> str:
        """Return deterministic rule-based treatment suggestions as structured JSON context."""
        return json.dumps(rule_based_rec, indent=2)

    return [
        StructuredTool.from_function(get_patient_summary),
        StructuredTool.from_function(get_expression_highlights),
        StructuredTool.from_function(get_mutation_flags),
        StructuredTool.from_function(get_histopathology_summary),
        StructuredTool.from_function(get_model_predictions),
        StructuredTool.from_function(get_rule_based_treatment_baseline),
    ]
