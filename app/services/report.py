"""Rule-based clinical report generation."""

from __future__ import annotations

from io import BytesIO
from typing import Any

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from app.config import DISCLAIMER, TARGET_LABEL_MAPS


def _mutation_flags(patient: dict[str, Any]) -> list[str]:
    flags = []
    mut = patient.get("Mutation")
    if mut is None:
        return ["Mutation data not provided."]
    row = mut.iloc[0]
    for gene in ("TP53", "EGFR", "KRAS", "STK11", "ALK"):
        if gene in row.index and row[gene] == 1:
            flags.append(f"{gene} mutation detected.")
    if "TMB" in row.index and row["TMB"] >= 10:
        flags.append(f"High tumor mutational burden (TMB={row['TMB']:.1f}).")
    return flags or ["No flagged driver mutations in panel."]


def _expression_highlights(patient: dict[str, Any], top_n: int = 5) -> list[str]:
    expr = patient.get("Expression")
    if expr is None:
        return ["Expression data not provided."]
    row = expr.iloc[0].sort_values(ascending=False)
    return [f"{gene}: log1p(RSEM)={val:.2f}" for gene, val in row.head(top_n).items()]


def generate_report_text(
    patient: dict[str, Any],
    predictions: list[dict[str, Any]],
) -> str:
    pid = patient.get("patient_id", "Unknown")
    lines = [
        "ONCOLENS Clinical Report (template-based)",
        "=" * 40,
        f"Patient ID: {pid}",
        "",
        "Patient Summary",
        "-" * 20,
        f"Modalities available: {', '.join(k for k in patient if k not in ('patient_id', 'from_feature_store'))}",
        "",
        "Gene Expression (selected panel)",
        "-" * 20,
        "  Top genes by log1p(RSEM) value in the model's selected gene panel",
        "  (not model feature importance — values are log-transformed expression).",
    ]
    lines.extend(f"  • {h}" for h in _expression_highlights(patient))
    lines.extend(["", "Mutation Analysis", "-" * 20])
    lines.extend(f"  • {f}" for f in _mutation_flags(patient))

    if "Histopathology" in patient:
        histo = patient["Histopathology"].iloc[0]
        norm = float((histo ** 2).sum() ** 0.5)
        lines.extend(
            [
                "",
                "Histopathology Findings",
                "-" * 20,
                f"  • Precomputed embedding available (L2 norm={norm:.3f}).",
                "  • No image-level interpretation performed.",
            ]
        )
    else:
        lines.extend(["", "Histopathology Findings", "-" * 20, "  • Not available for this patient."])

    lines.extend(["", "Predictions", "-" * 20])
    for pred in predictions:
        if not pred.get("available", True):
            lines.append(f"  • {pred['model']} ({pred['target']}): unavailable — {pred.get('reason', 'N/A')}")
            continue
        lines.append(
            f"  • {pred['model']} ({pred['target']}): {pred['predicted_label_name']} "
            f"(confidence={pred['confidence']:.2f}, risk={pred['risk_band']})"
        )

    lines.extend(
        [
            "",
            "Clinical Interpretation",
            "-" * 20,
            _interpretation_block(predictions),
            "",
            "Risk Summary",
            "-" * 20,
        ]
    )
    for pred in predictions:
        if pred.get("available", True) and pred["target"] in ("OS_STATUS", "PFS_STATUS"):
            lines.append(f"  • {pred['target']}: {pred['risk_band']} risk ({pred['probability']:.1%})")

    lines.extend(["", "Disclaimer", "-" * 20, DISCLAIMER])
    return "\n".join(lines)


def _interpretation_block(predictions: list[dict[str, Any]]) -> str:
    os_preds = [p for p in predictions if p.get("target") == "OS_STATUS" and p.get("available")]
    if not os_preds:
        return "Insufficient prediction data for automated interpretation."
    best = os_preds[0]
    band = best.get("risk_band", "Unknown")
    if band == "High":
        return (
            "Elevated predicted risk suggests closer monitoring and discussion of "
            "aggressive treatment options with the oncology care team."
        )
    if band == "Medium":
        return "Moderate predicted risk warrants standard follow-up and individualized treatment planning."
    return "Lower predicted risk may support less intensive surveillance, subject to clinical judgment."


def report_to_pdf(text: str) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    for line in text.split("\n"):
        if line.startswith("=") or line.startswith("-"):
            story.append(Spacer(1, 6))
            continue
        story.append(Paragraph(line.replace("&", "&amp;"), styles["Normal"]))
        story.append(Spacer(1, 4))
    doc.build(story)
    return buffer.getvalue()
