"""Rule-based treatment recommendations (educational).

Deterministic fallback until the LLM treatment agent is wired in
(see app/pages/4_Treatment_Recommendation.py). The agent should consume the
same inputs: patient dict, compare_all() predictions, and optionally this
function's output as structured context for the prompt.
"""

from __future__ import annotations

from typing import Any


def generate_treatment_recommendations(
    patient: dict[str, Any],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    mut_flags = _detect_mutations(patient)
    os_risk = _get_risk(predictions, "OS_STATUS")
    pfs_risk = _get_risk(predictions, "PFS_STATUS")
    stage_pred = _get_stage(predictions)

    options: list[str] = []
    drug_classes: list[str] = []
    monitoring: list[str] = []
    questions: list[str] = [
        "What is the recommended surveillance schedule for my stage and risk profile?",
        "Are there clinical trials I may be eligible for?",
    ]

    if mut_flags.get("EGFR"):
        options.append("Discuss EGFR-targeted therapy eligibility.")
        drug_classes.append("EGFR tyrosine kinase inhibitors")
        questions.append("Should I receive EGFR mutation testing confirmation?")

    if mut_flags.get("KRAS"):
        options.append("Discuss KRAS-directed therapy options where applicable.")
        drug_classes.append("KRAS G12C inhibitors (if G12C confirmed)")

    if mut_flags.get("TP53") and os_risk == "High":
        options.append("Discuss platinum-based chemotherapy options with your oncologist.")

    if mut_flags.get("high_tmb"):
        options.append("Discuss immunotherapy eligibility given elevated TMB.")
        drug_classes.append("Immune checkpoint inhibitors")

    if stage_pred in ("Stage I", "Stage II") and os_risk == "Low":
        options.append("Surgical resection may be primary treatment; confirm with care team.")
        monitoring.append("Periodic CT imaging per NCCN guidelines.")

    if os_risk == "High":
        monitoring.append("Consider more frequent follow-up imaging and labs.")
        options.append("Multidisciplinary tumor board review recommended.")

    if pfs_risk == "High":
        monitoring.append("Monitor for early progression signs closely.")

    if not options:
        options.append("Standard-of-care LUAD treatment per NCCN guidelines should be discussed.")

    lifestyle = [
        "Smoking cessation if applicable.",
        "Maintain nutrition and physical activity as tolerated.",
        "Report new symptoms promptly.",
    ]

    return {
        "risk_assessment": f"OS risk: {os_risk or 'N/A'}; PFS risk: {pfs_risk or 'N/A'}",
        "likely_diagnosis": stage_pred or "LUAD — stage prediction unavailable",
        "treatment_options": options,
        "drug_classes": drug_classes or ["Chemotherapy, targeted therapy, immunotherapy per biomarkers"],
        "lifestyle_advice": lifestyle,
        "monitoring": monitoring or ["Routine oncology follow-up"],
        "questions_for_oncologist": questions,
    }


def _detect_mutations(patient: dict[str, Any]) -> dict[str, Any]:
    flags: dict[str, Any] = {}
    mut = patient.get("Mutation")
    if mut is None:
        return flags
    row = mut.iloc[0]
    for gene in ("TP53", "EGFR", "KRAS", "STK11", "ALK"):
        if gene in row.index and row[gene] == 1:
            flags[gene] = True
    if "TMB" in row.index and row["TMB"] >= 10:
        flags["high_tmb"] = True
    return flags


def _get_risk(predictions: list[dict[str, Any]], target: str) -> str | None:
    for p in predictions:
        if p.get("target") == target and p.get("available", True):
            return p.get("risk_band")
    return None


def _get_stage(predictions: list[dict[str, Any]]) -> str | None:
    for p in predictions:
        if p.get("target") == "Stage" and p.get("available", True):
            return p.get("predicted_label_name")
    return None
