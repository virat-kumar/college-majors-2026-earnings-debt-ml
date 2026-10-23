#!/usr/bin/env python3
"""Fetch and verify the public, versioned College Majors CSV snapshot.

The original notebook expects college_majors_2026.csv next to itself.
The 120 MiB file is a GitHub Release asset, not a regular Git blob.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SLUG = "kylefengkfeng209/college-majors-2026-earnings-debt-jobs-ai"
REPOSITORY = "virat-kumar/college-majors-2026-earnings-debt-ml"
RELEASE_TAG = "data-v1.0.0"
CSV_NAME = "college_majors_2026.csv"
RELEASE_URL = (
    f"https://github.com/{REPOSITORY}/releases/download/{RELEASE_TAG}/{CSV_NAME}"
)
EXPECTED_ROWS = 227_980
EXPECTED_COLUMNS = 72
EXPECTED_SHA256 = "58f1c10e79793f33049f3b381cd81d4d0c190ca1052e427c29779d0c89c5ceca"

ROOT = Path(__file__).resolve().parent
DEST = ROOT / CSV_NAME
MANIFEST = ROOT / "data" / "source_manifest.json"


def check_csv(path: Path) -> tuple[int, int, str]:
    """Verify the exact published data snapshot; never silently accept changed rows."""
    if not path.is_file():
        raise FileNotFoundError(path)

    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            hasher.update(block)
    actual_hash = hasher.hexdigest()
    if actual_hash != EXPECTED_SHA256:
        raise ValueError(
            f"Unexpected data SHA256 {actual_hash}; expected {EXPECTED_SHA256}. "
            "Do not overwrite or redistribute a changed dataset without review."
        )

    with path.open("r", newline="", encoding="utf-8-sig") as fh:
        records = csv.reader(fh)
        header = next(records)
        if len(header) != EXPECTED_COLUMNS or header[0] != "program_id":
            raise ValueError(
                f"Unexpected schema: {len(header)} columns, first is {header[0]!r}"
            )
        count = sum(1 for _ in records)
    if count != EXPECTED_ROWS:
        raise ValueError(
            f"Unexpected dataset version: {count:,} rows "
            f"(expected {EXPECTED_ROWS:,})."
        )
    return count, len(header), actual_hash


def download_release(temporary: Path) -> None:
    """Fetch the exact published public asset without Git LFS or Kaggle credentials."""
    request = Request(
        RELEASE_URL,
        headers={
            "User-Agent": "college-majors-2026-earnings-debt-ml/1.0",
            "Accept": "application/octet-stream",
        },
    )
    print(f"Downloading versioned dataset from {RELEASE_URL}", flush=True)
    with urlopen(request, timeout=120) as source, temporary.open("wb") as out:
        shutil.copyfileobj(source, out, length=4 * 1024 * 1024)


def download_kaggle(temporary: Path) -> None:
    """Fallback to Kaggle's public API if the GitHub release is unavailable."""
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("Install requirements first: python setup.py") from exc
    print(f"GitHub unavailable; downloading public Kaggle dataset {SLUG}...", flush=True)
    downloaded = Path(kagglehub.dataset_download(SLUG))
    matches = list(downloaded.rglob(CSV_NAME))
    if downloaded.is_file() and downloaded.name == CSV_NAME:
        matches = [downloaded]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {CSV_NAME} in {downloaded}, got {len(matches)}."
        )
    shutil.copyfile(matches[0], temporary)


def main() -> None:
    source = "existing locally verified copy"

    if not DEST.exists():
        temporary = DEST.with_suffix(".csv.partial")
        errors: list[str] = []
        for label, download in (
            ("GitHub Release", download_release),
            ("Kaggle fallback", download_kaggle),
        ):
            try:
                temporary.unlink(missing_ok=True)
                download(temporary)
                rows, cols, sha = check_csv(temporary)
                os.replace(temporary, DEST)
                source = label
                break
            except (OSError, HTTPError, URLError, TimeoutError, ValueError, RuntimeError) as exc:
                errors.append(f"{label}: {exc}")
                print(f"{label} could not provide the verified snapshot: {exc}", flush=True)
            finally:
                temporary.unlink(missing_ok=True)
        else:
            raise RuntimeError(
                "Unable to retrieve the exact dataset version.\n"
                + "\n".join(errors)
                + "\nSee DATA.md for manual download instructions."
            )
    else:
        rows, cols, sha = check_csv(DEST)

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps(
            {
                "source": f"https://www.kaggle.com/datasets/{SLUG}",
                "license": "CC BY-SA 4.0 (as listed by Kaggle)",
                "distribution": RELEASE_URL,
                "retrieved_via": source,
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
    print(
        f"PASS: {DEST.name} ({rows:,} rows x {cols} columns; SHA256={sha}).\n"
        f"Source metadata: {MANIFEST} (ignored by Git).",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        sys.exit(f"Data setup failed: {exc}")
