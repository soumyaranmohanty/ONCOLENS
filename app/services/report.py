"""Rule-based clinical report generation and PDF export."""

from __future__ import annotations

import re
from datetime import date
from io import BytesIO
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer

from app.config import DISCLAIMER

_PREAMBLE_PATTERNS = (
    r"^#+\s*",
    r"(?i)^clinical\s+(research\s+)?report",
    r"(?i)^oncolens\s+clinical",
    r"(?i)^date of analysis:",
    r"(?i)^patient id:",
    r"(?i)^report date:",
    r"^=+$",
    r"^-+$",
)


def _normalize_line_for_match(line: str) -> str:
    """Strip markdown decoration so metadata patterns match reliably."""
    cleaned = re.sub(r"[*#_>`]", " ", line.strip())
    return re.sub(r"\s+", " ", cleaned).strip()


def _is_metadata_line(line: str) -> bool:
    """True for title/date/patient-id lines that belong only in the report header."""
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped) >= 3 and all(ch in "-=_" for ch in stripped):
        return True

    normalized = _normalize_line_for_match(stripped)
    lower = normalized.lower()
    if lower.startswith(("date of analysis:", "report date:", "analysis date:")):
        return True
    if lower.startswith("patient id:"):
        return True
    if re.match(r"(?i)^oncolens clinical", normalized):
        return True
    if re.match(r"(?i)^clinical (research )?report", normalized):
        return True
    return False


def remove_llm_metadata_lines(text: str) -> str:
    """Drop fabricated title/date/patient-id lines anywhere in the report body."""
    kept: list[str] = []
    for line in text.splitlines():
        if _is_metadata_line(line):
            continue
        kept.append(line)
    cleaned = "\n".join(kept).strip()
    return f"{cleaned}\n" if cleaned else ""


def infer_report_mode(text: str) -> str:
    """Infer template vs AI report from existing content."""
    return "template" if "template-based" in text.lower() else "ai"


def normalize_clinical_report(
    text: str,
    patient_id: str,
    *,
    mode: str | None = None,
) -> str:
    """Ensure exactly one canonical header with today's analysis date."""
    report_mode = mode or infer_report_mode(text)
    body = remove_llm_metadata_lines(strip_llm_report_preamble(text))
    return build_report_header(patient_id, mode=report_mode) + body.lstrip("\n")


def analysis_date_label(when: date | None = None) -> str:
    """Human-readable analysis date for report headers."""
    return (when or date.today()).strftime("%B %d, %Y")


def build_report_header(patient_id: str, *, mode: str = "template") -> str:
    """Standard report header with patient ID and today's analysis date."""
    if mode == "ai":
        title = "ONCOLENS Clinical Research Report"
    else:
        title = "ONCOLENS Clinical Report (template-based)"
    return "\n".join(
        [
            title,
            f"Patient ID: {patient_id}",
            f"Date of Analysis: {analysis_date_label()}",
            "",
        ]
    )


def strip_llm_report_preamble(text: str) -> str:
    """Remove LLM-invented title/date lines from the start of a narrative report."""
    lines = text.splitlines()
    while lines:
        stripped = lines[0].strip()
        if not stripped:
            lines.pop(0)
            continue
        if any(re.match(pattern, stripped) for pattern in _PREAMBLE_PATTERNS):
            lines.pop(0)
            continue
        break
    return "\n".join(lines).lstrip("\n")


def finalize_llm_report(text: str, patient_id: str) -> str:
    """Attach a canonical header and drop any duplicate LLM metadata."""
    return normalize_clinical_report(text, patient_id, mode="ai")


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
        build_report_header(pid, mode="template").rstrip(),
        "Patient Summary",
        "-" * 20,
        f"Modalities available: {', '.join(k for k in patient if k not in ('patient_id', 'from_feature_store'))}",
        "",
        "Gene Expression (selected panel)",
        "-" * 20,
        "Top genes by log1p(RSEM) value in the model's selected gene panel",
        "(not model feature importance — values are log-transformed expression).",
    ]
    lines.extend(f"  • {h}" for h in _expression_highlights(patient))
    lines.extend(["", "Mutation Analysis", "-" * 20])
    lines.extend(f"  • {f}" for f in _mutation_flags(patient))

    if "Histopathology" in patient:
        histo = patient["Histopathology"].iloc[0]
        norm = float((histo**2).sum() ** 0.5)
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


def _escape_xml(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _inline_markdown_to_reportlab(text: str) -> str:
    escaped = _escape_xml(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", escaped)
    return escaped


def _is_separator_line(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and len(stripped) >= 3 and all(ch in "-=_" for ch in stripped)


def _is_bullet_line(line: str) -> tuple[bool, str]:
    stripped = line.strip()
    for prefix in ("•", "-", "*"):
        if stripped.startswith(prefix):
            return True, stripped[1:].strip()
    match = re.match(r"^\d+\.\s+(.*)$", stripped)
    if match:
        return True, match.group(1)
    return False, stripped


def _append_section_header(story: list[Any], styles: dict[str, ParagraphStyle], text: str) -> None:
    story.append(Paragraph(_inline_markdown_to_reportlab(text), styles["section"]))


def _maybe_close_report_header(
    story: list[Any],
    *,
    title_rendered: bool,
    header_complete: bool,
) -> bool:
    if title_rendered and not header_complete:
        story.append(Spacer(1, 6))
        story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#cbd5e1")))
        story.append(Spacer(1, 10))
        return True
    return header_complete


def _pdf_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            textColor=colors.HexColor("#0f172a"),
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "meta": ParagraphStyle(
            "ReportMeta",
            parent=base["Normal"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#64748b"),
            alignment=TA_CENTER,
            spaceAfter=3,
        ),
        "section": ParagraphStyle(
            "ReportSection",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#1e40af"),
            spaceBefore=14,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "ReportBody",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#334155"),
            alignment=TA_LEFT,
            spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "ReportBullet",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#334155"),
            leftIndent=16,
            bulletIndent=0,
            spaceAfter=4,
        ),
        "disclaimer": ParagraphStyle(
            "ReportDisclaimer",
            parent=base["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#64748b"),
            spaceBefore=8,
            spaceAfter=4,
        ),
    }


def report_to_pdf(text: str, *, patient_id: str | None = None) -> bytes:
    """Render a clinical report as a styled PDF."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title=f"ONCOLENS Clinical Report — {patient_id or 'patient'}",
    )
    styles = _pdf_styles()
    story: list[Any] = []
    lines = text.splitlines()
    title_rendered = False
    header_complete = False
    i = 0

    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()

        if _is_separator_line(raw):
            i += 1
            continue

        if not stripped:
            story.append(Spacer(1, 6))
            i += 1
            continue

        lower = stripped.lower()
        if lower.startswith("date of analysis:") or lower.startswith("patient id:"):
            story.append(Paragraph(_inline_markdown_to_reportlab(stripped), styles["meta"]))
            i += 1
            continue

        bold_section = re.fullmatch(r"\*\*(.+?)\*\*", stripped)
        if bold_section:
            header_complete = _maybe_close_report_header(
                story, title_rendered=title_rendered, header_complete=header_complete
            )
            _append_section_header(story, styles, bold_section.group(1))
            i += 1
            continue

        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            content = stripped.lstrip("#").strip()
            style = styles["title"] if level == 1 and not title_rendered else styles["section"]
            story.append(Paragraph(_inline_markdown_to_reportlab(content), style))
            title_rendered = True
            i += 1
            continue

        if not title_rendered and (
            "clinical report" in lower or "clinical research report" in lower or "oncolens" in lower
        ):
            story.append(Paragraph(_inline_markdown_to_reportlab(stripped), styles["title"]))
            title_rendered = True
            i += 1
            continue

        next_line = lines[i + 1] if i + 1 < len(lines) else ""
        if _is_separator_line(next_line):
            header_complete = _maybe_close_report_header(
                story, title_rendered=title_rendered, header_complete=header_complete
            )
            _append_section_header(story, styles, stripped)
            i += 2
            continue

        is_bullet, bullet_text = _is_bullet_line(raw)
        if is_bullet and not re.match(r"^\d+\.\s+", stripped):
            story.append(
                Paragraph(
                    f"• {_inline_markdown_to_reportlab(bullet_text)}",
                    styles["bullet"],
                )
            )
            i += 1
            continue

        if re.match(r"^\d+\.\s+", stripped):
            header_complete = _maybe_close_report_header(
                story, title_rendered=title_rendered, header_complete=header_complete
            )
            _append_section_header(story, styles, stripped)
            i += 1
            continue

        if stripped == "Disclaimer":
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0")))
            story.append(Spacer(1, 8))
            story.append(Paragraph(_inline_markdown_to_reportlab(stripped), styles["section"]))
            i += 1
            continue

        if lower.startswith("disclaimer") or stripped == DISCLAIMER:
            story.append(Paragraph(_inline_markdown_to_reportlab(stripped), styles["disclaimer"]))
            i += 1
            continue

        story.append(Paragraph(_inline_markdown_to_reportlab(stripped), styles["body"]))
        i += 1

    doc.build(story)
    return buffer.getvalue()
