"""
Fetch a gdc-client manifest for TCGA-LUAD diagnostic whole-slide images.

Queries the public GDC Files API (no credentials required for open-access slides)
and writes a tab-separated manifest compatible with:

    gdc-client download -m Data/raw_data/histopathology/gdc_manifest.txt \\
        -d Data/raw_data/histopathology

Run from repo root:
    uv run python EXP/fetch_gdc_manifest.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import requests

GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
RAW_DIR = Path("Data/raw_data/histopathology")
MANIFEST_PATH = RAW_DIR / "gdc_manifest.txt"

# Diagnostic slides only: barcode suffix -DX1 / -DX2 (exclude -TS frozen sections).
DIAGNOSTIC_SUFFIXES = ("-DX1", "-DX2")


def _is_diagnostic_slide(filename: str) -> bool:
    upper = filename.upper()
    return any(suffix in upper for suffix in DIAGNOSTIC_SUFFIXES)


def fetch_luad_diagnostic_slides() -> list[dict]:
    filters = {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "cases.project.project_id",
                    "value": ["TCGA-LUAD"],
                },
            },
            {
                "op": "in",
                "content": {"field": "data_type", "value": ["Slide Image"]},
            },
            {
                "op": "in",
                "content": {"field": "access", "value": ["open"]},
            },
        ],
    }

    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,md5sum,file_size,state",
        "format": "JSON",
        "size": "10000",
    }

    response = requests.get(GDC_FILES_ENDPOINT, params=params, timeout=120)
    response.raise_for_status()
    hits = response.json()["data"]["hits"]

    diagnostic = [h for h in hits if _is_diagnostic_slide(h["file_name"])]
    return diagnostic


def write_manifest(files: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["id\tfilename\tmd5\tsize\tstate"]
    for f in files:
        lines.append(
            "\t".join(
                [
                    f["file_id"],
                    f["file_name"],
                    f["md5sum"],
                    str(f["file_size"]),
                    f.get("state", "released"),
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    print("Querying GDC Files API for TCGA-LUAD diagnostic slides …")
    files = fetch_luad_diagnostic_slides()
    print(f"Found {len(files)} diagnostic slide files.")

    if not files:
        raise SystemExit("No files returned — check GDC API filters.")

    write_manifest(files, MANIFEST_PATH)
    total_gb = sum(int(f["file_size"]) for f in files) / (1024**3)
    print(f"Wrote manifest → {MANIFEST_PATH}")
    print(f"Estimated download size: {total_gb:.1f} GB")
    print(
        "\nNext: install gdc-client, then run:\n"
        f"  gdc-client download -m {MANIFEST_PATH} -d {RAW_DIR}"
    )


if __name__ == "__main__":
    main()
