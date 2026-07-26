"""
Download TCGA open-access slide files listed in gdc_manifest.txt via the GDC Data API.

Supports batched downloads with persistent progress tracking and MD5 verification.

Run from repo root:
    uv run python EXP/download_gdc_slides.py --batch-size 50 --batch-index 0
    uv run python EXP/download_gdc_slides.py --batch-size 50 --batch-index -1   # auto next batch
    uv run python EXP/download_gdc_slides.py --batch-size 50 --batch-index 0 --verify-only
    uv run python EXP/download_gdc_slides.py --limit 3   # legacy smoke test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

RAW_DIR = Path("Data/raw_data/histopathology")
MANIFEST_PATH = RAW_DIR / "gdc_manifest.txt"
STATUS_PATH = RAW_DIR / "download_status.json"
LOG_PATH = RAW_DIR / "download.log"
GDC_DATA_URL = "https://api.gdc.cancer.gov/data"


def parse_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        header = fh.readline().strip().split("\t")
        for line in fh:
            parts = line.strip().split("\t")
            rows.append(dict(zip(header, parts)))
    return rows


def md5_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def dest_path(row: dict) -> Path:
    return RAW_DIR / row["id"] / row["filename"]


def load_status() -> dict:
    if not STATUS_PATH.exists():
        return {}
    with STATUS_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def save_status(status: dict) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATUS_PATH.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(status, fh, indent=2)
    tmp.replace(STATUS_PATH)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def update_status(
    status: dict,
    file_id: str,
    row: dict,
    batch_index: int,
    *,
    state: str,
    local_size: int | None = None,
    md5_ok: bool | None = None,
    error: str | None = None,
) -> None:
    entry = status.get(file_id, {})
    entry.update(
        {
            "filename": row["filename"],
            "batch_index": batch_index,
            "status": state,
            "expected_size": int(row["size"]),
            "local_size": local_size,
            "md5_ok": md5_ok,
            "updated_at": utc_now(),
        }
    )
    if error:
        entry["error"] = error
    elif "error" in entry and state == "complete":
        entry.pop("error", None)
    status[file_id] = entry
    save_status(status)


def is_complete(dest: Path, expected_size: int, expected_md5: str | None, verify_md5: bool) -> bool:
    if not dest.exists():
        return False
    local_size = dest.stat().st_size
    if local_size != expected_size:
        return False
    if verify_md5 and expected_md5 and md5_file(dest) != expected_md5:
        return False
    return True


def verify_file(dest: Path, expected_size: int, expected_md5: str | None, verify_md5: bool) -> tuple[bool, int, bool | None]:
    if not dest.exists():
        return False, 0, None
    local_size = dest.stat().st_size
    if local_size != expected_size:
        return False, local_size, None
    md5_ok = None
    if verify_md5 and expected_md5:
        md5_ok = md5_file(dest) == expected_md5
        return md5_ok, local_size, md5_ok
    return True, local_size, md5_ok


def download_file(file_id: str, dest: Path, expected_size: int, expected_md5: str | None, verify_md5: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        local_size = dest.stat().st_size
        if local_size == expected_size:
            if verify_md5 and expected_md5 and md5_file(dest) != expected_md5:
                print(f"  md5 mismatch — re-downloading {dest.name}")
                dest.unlink()
            else:
                print(f"  skip (complete): {dest.name}")
                return
        else:
            print(f"  deleting partial ({local_size}/{expected_size} bytes): {dest.name}")
            dest.unlink()

    url = f"{GDC_DATA_URL}/{file_id}"
    with requests.get(url, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        with dest.open("wb") as out:
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    out.write(chunk)

    local_size = dest.stat().st_size
    if local_size != expected_size:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"Size mismatch for {dest.name}: got {local_size}, expected {expected_size}")

    if verify_md5 and expected_md5 and md5_file(dest) != expected_md5:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"MD5 verification failed for {dest.name}")


def select_batch_rows(all_rows: list[dict], batch_size: int, batch_index: int) -> tuple[list[dict], int]:
    if batch_index < 0:
        status = load_status()
        n_batches = math.ceil(len(all_rows) / batch_size) if batch_size else 1
        for idx in range(n_batches):
            start = idx * batch_size
            batch_rows = all_rows[start : start + batch_size]
            for row in batch_rows:
                file_id = row["id"]
                dest = dest_path(row)
                if not is_complete(dest, int(row["size"]), row.get("md5"), verify_md5=True):
                    st = status.get(file_id, {}).get("status")
                    if st != "complete":
                        return batch_rows, idx
        last_idx = max(n_batches - 1, 0)
        start = last_idx * batch_size
        return all_rows[start : start + batch_size], last_idx

    start = batch_index * batch_size
    end = start + batch_size
    return all_rows[start:end], batch_index


def append_log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(message.rstrip() + "\n")


def run_verify_batch(
    rows: list[dict],
    batch_index: int,
    verify_md5: bool,
    status: dict,
) -> tuple[int, int, int, list[str]]:
    complete = failed = 0
    failed_ids: list[str] = []

    for row in rows:
        file_id = row["id"]
        dest = dest_path(row)
        expected_size = int(row["size"])
        expected_md5 = row.get("md5")

        ok, local_size, md5_ok = verify_file(dest, expected_size, expected_md5, verify_md5)
        if ok:
            update_status(status, file_id, row, batch_index, state="complete", local_size=local_size, md5_ok=md5_ok)
            complete += 1
            print(f"  OK: {row['filename']}")
        else:
            update_status(
                status,
                file_id,
                row,
                batch_index,
                state="failed",
                local_size=local_size,
                md5_ok=md5_ok,
                error="verification failed",
            )
            failed += 1
            failed_ids.append(file_id)
            print(f"  FAIL: {row['filename']} (size={local_size}, expected={expected_size})")

    return complete, failed, 0, failed_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GDC slides from manifest")
    parser.add_argument("--limit", type=int, default=0, help="Legacy: max files from start of selection (0 = all in batch)")
    parser.add_argument("--batch-size", type=int, default=50, help="Files per batch (default 50)")
    parser.add_argument(
        "--batch-index",
        type=int,
        default=-1,
        help="0-based batch index (-1 = auto-select first batch with pending/failed files)",
    )
    parser.add_argument("--verify-only", action="store_true", help="Verify files without downloading")
    parser.add_argument("--no-verify-md5", action="store_true", help="Skip MD5 check (smoke tests only)")
    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        raise SystemExit(f"Manifest not found: {MANIFEST_PATH}. Run fetch_gdc_manifest.py first.")

    all_rows = parse_manifest(MANIFEST_PATH)
    verify_md5 = not args.no_verify_md5

    if args.limit and args.batch_index == -1 and not args.verify_only:
        rows = all_rows[: args.limit]
        batch_index = 0
    else:
        rows, batch_index = select_batch_rows(all_rows, args.batch_size, args.batch_index)
        if args.limit:
            rows = rows[: args.limit]

    if not rows:
        raise SystemExit("No files in selected batch.")

    status = load_status()
    batch_bytes = sum(int(r["size"]) for r in rows)
    total = len(rows)

    header = (
        f"\n=== Batch {batch_index} ({total} files, {batch_bytes / (1024**3):.1f} GB) "
        f"{'VERIFY' if args.verify_only else 'DOWNLOAD'} @ {utc_now()} ==="
    )
    print(header)
    append_log(header)

    complete = failed = skipped = 0
    failed_ids: list[str] = []

    if args.verify_only:
        complete, failed, skipped, failed_ids = run_verify_batch(rows, batch_index, verify_md5, status)
    else:
        for i, row in enumerate(rows, 1):
            file_id = row["id"]
            filename = row["filename"]
            expected_size = int(row["size"])
            expected_md5 = row.get("md5") if verify_md5 else None
            dest = dest_path(row)

            if is_complete(dest, expected_size, row.get("md5"), verify_md5):
                local_size = dest.stat().st_size
                md5_ok = True if verify_md5 and row.get("md5") else None
                update_status(status, file_id, row, batch_index, state="complete", local_size=local_size, md5_ok=md5_ok)
                skipped += 1
                print(f"[{i}/{total}] skip (complete): {filename}")
                continue

            print(f"[{i}/{total}] {filename}")
            update_status(status, file_id, row, batch_index, state="downloading")
            t0 = time.time()
            try:
                download_file(file_id, dest, expected_size, expected_md5, verify_md5)
                local_size = dest.stat().st_size
                md5_ok = True if verify_md5 and row.get("md5") else None
                update_status(status, file_id, row, batch_index, state="complete", local_size=local_size, md5_ok=md5_ok)
                complete += 1
                elapsed = time.time() - t0
                print(f"  done in {elapsed:.0f}s")
            except Exception as exc:
                update_status(
                    status,
                    file_id,
                    row,
                    batch_index,
                    state="failed",
                    local_size=dest.stat().st_size if dest.exists() else 0,
                    error=str(exc),
                )
                failed += 1
                failed_ids.append(file_id)
                print(f"  ERROR: {exc}")

    total_complete = sum(1 for r in all_rows if status.get(r["id"], {}).get("status") == "complete")
    summary = (
        f"Batch {batch_index} ({total} files): complete={complete}, failed={failed}, skipped={skipped}\n"
        f"Failed IDs: {failed_ids or 'none'}\n"
        f"Estimated batch size: {batch_bytes / (1024**3):.1f} GB\n"
        f"Total progress: {total_complete}/{len(all_rows)} ({100 * total_complete / len(all_rows):.1f}%)"
    )
    print(summary)
    append_log(summary)


if __name__ == "__main__":
    main()
