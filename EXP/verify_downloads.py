"""
Verify downloaded GDC slide files against gdc_manifest.txt.

Checks file existence, size, MD5, and optionally OpenSlide readability.

Run from repo root:
    uv run python EXP/verify_downloads.py --batch-size 50 --batch-index 0
    uv run python EXP/verify_downloads.py --all
    uv run python EXP/verify_downloads.py --batch-size 50 --batch-index 0 --openslide-check
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from download_gdc_slides import (
    MANIFEST_PATH,
    RAW_DIR,
    dest_path,
    load_status,
    parse_manifest,
    save_status,
    select_batch_rows,
    update_status,
    utc_now,
    verify_file,
)

REPORT_PATH = RAW_DIR / "verification_report.csv"


def openslide_ok(path: Path) -> tuple[bool, str | None]:
    try:
        import openslide

        slide = openslide.OpenSlide(str(path))
        _ = slide.dimensions
        slide.close()
        return True, None
    except Exception as exc:
        return False, str(exc)


def write_report(rows: list[dict]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with REPORT_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def verify_rows(
    manifest_rows: list[dict],
    batch_index: int,
    verify_md5: bool,
    openslide_check: bool,
    openslide_sample: int,
) -> tuple[int, int, list[dict]]:
    status = load_status()
    report: list[dict] = []
    passed = failed = 0

    openslide_targets: set[str] = set()
    if openslide_check and manifest_rows:
        n = min(openslide_sample, len(manifest_rows))
        openslide_targets = {r["id"] for r in random.sample(manifest_rows, n)}

    for row in manifest_rows:
        file_id = row["id"]
        dest = dest_path(row)
        expected_size = int(row["size"])
        expected_md5 = row.get("md5")

        ok, local_size, md5_ok = verify_file(dest, expected_size, expected_md5, verify_md5)
        os_ok, os_error = (None, None)
        if file_id in openslide_targets and dest.exists():
            os_ok, os_error = openslide_ok(dest)

        all_ok = ok and (os_ok is not False)
        if all_ok:
            update_status(status, file_id, row, batch_index, state="complete", local_size=local_size, md5_ok=md5_ok)
            passed += 1
        else:
            update_status(
                status,
                file_id,
                row,
                batch_index,
                state="failed",
                local_size=local_size,
                md5_ok=md5_ok,
                error=os_error or "verification failed",
            )
            failed += 1

        report.append(
            {
                "file_id": file_id,
                "filename": row["filename"],
                "batch_index": batch_index,
                "exists": dest.exists(),
                "size_ok": local_size == expected_size if dest.exists() else False,
                "local_size": local_size,
                "expected_size": expected_size,
                "md5_ok": md5_ok if verify_md5 else "",
                "openslide_ok": os_ok if os_ok is not None else "",
                "openslide_error": os_error or "",
                "passed": all_ok,
                "verified_at": utc_now(),
            }
        )

    write_report(report)
    return passed, failed, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify GDC slide downloads")
    parser.add_argument("--batch-size", type=int, default=50, help="Files per batch (default 50)")
    parser.add_argument("--batch-index", type=int, default=0, help="0-based batch index")
    parser.add_argument("--all", action="store_true", help="Verify entire manifest")
    parser.add_argument("--no-verify-md5", action="store_true", help="Skip MD5 check")
    parser.add_argument("--openslide-check", action="store_true", help="Open a random sample with OpenSlide")
    parser.add_argument("--openslide-sample", type=int, default=3, help="Number of slides to OpenSlide-check per batch")
    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        raise SystemExit(f"Manifest not found: {MANIFEST_PATH}")

    all_rows = parse_manifest(MANIFEST_PATH)
    verify_md5 = not args.no_verify_md5

    if args.all:
        batch_index = -1
        rows = all_rows
    else:
        rows, batch_index = select_batch_rows(all_rows, args.batch_size, args.batch_index)

    print(f"Verifying {len(rows)} files (batch_index={batch_index}) …")
    passed, failed, _ = verify_rows(rows, batch_index, verify_md5, args.openslide_check, args.openslide_sample)

    print(f"Passed: {passed}, Failed: {failed}")
    print(f"Report written to {REPORT_PATH}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
