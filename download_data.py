#!/usr/bin/env python3
"""Download the public Kaggle dataset locally without committing it to Git."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

SLUG = "kylefengkfeng209/college-majors-2026-earnings-debt-jobs-ai"
CSV_NAME = "college_majors_2026.csv"
ROOT = Path(__file__).resolve().parent
DEST = ROOT / CSV_NAME
MANIFEST = ROOT / "data" / "source_manifest.json"
EXPECTED_ROWS = 227_980
EXPECTED_COLUMNS = 72


def check_csv(path: Path) -> tuple[int, int, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            hasher.update(block)
    with path.open("r", newline="", encoding="utf-8-sig") as fh:
        records = csv.reader(fh)
        header = next(records)
        if len(header) != EXPECTED_COLUMNS or header[0] != "program_id":
            raise ValueError(
                f"Unexpected schema: {len(header)} columns, first is {header[0]!r}"
            )
        count = sum(1 for row in records)
    if count != EXPECTED_ROWS:
        raise ValueError(
            f"Unexpected dataset version: {count:,} rows (expected {EXPECTED_ROWS:,}). "
            "Review the Kaggle version and project baseline before proceeding."
        )
    return count, len(header), hasher.hexdigest()


def main() -> None:
    if DEST.exists():
        rows, cols, sha = check_csv(DEST)
        print(f"Data already exists: {DEST} ({rows:,} x {cols}; SHA256={sha})")
    else:
        try:
            import kagglehub
        except ImportError as exc:
            raise RuntimeError("Run python setup.py to install dependencies first") from exc
        print(f"Downloading public Kaggle dataset {SLUG} ...", flush=True)
        downloaded = Path(kagglehub.dataset_download(SLUG))
        matches = list(downloaded.rglob(CSV_NAME))
        if downloaded.is_file() and downloaded.name == CSV_NAME:
            matches = [downloaded]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one {CSV_NAME} in {downloaded}; found {len(matches)}. "
                "Check dataset version or place the CSV in the project root."
            )
        temporary = DEST.with_suffix(".csv.partial")
        try:
            shutil.copyfile(matches[0], temporary)
            rows, cols, sha = check_csv(temporary)
            os.replace(temporary, DEST)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Downloaded and verified: {rows:,} x {cols}; SHA256={sha}")

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps(
            {
                "kaggle_dataset": SLUG,
                "source": f"https://www.kaggle.com/datasets/{SLUG}",
                "license": "CC BY-SA 4.0 (as listed by Kaggle)",
                "csv_file": CSV_NAME,
                "rows": rows,
                "columns": cols,
                "sha256": sha,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Local metadata: {MANIFEST} (ignored by Git)")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        sys.exit(f"Data download/validation failed: {exc}")
