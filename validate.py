#!/usr/bin/env python3
"""Lightweight project smoke tests; optionally execute the complete notebook."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
NB = ROOT / "College_Majors_EDA_Colab.ipynb"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-data", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    assert NB.is_file(), f"Missing original notebook at {NB}"
    notebook = json.loads(NB.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 50, "Original notebook cell count changed."
    for name in [
        "numpy", "pandas", "scipy", "sklearn", "matplotlib",
        "IPython", "ipykernel", "jupyterlab", "nbconvert", "kagglehub",
    ]:
        importlib.import_module(name)
    print(f"PASS: notebook parsed ({len(notebook['cells'])} cells) and runtime packages import.")

    if args.require_data or (ROOT / "college_majors_2026.csv").exists():
        from download_data import DEST, check_csv
        rows, cols, sha = check_csv(DEST)
        print(f"PASS: verified local private dataset {rows:,} x {cols}, SHA256={sha}")
    else:
        print("INFO: dataset not present; metadata-only smoke check (CI mode).")

    if args.execute:
        from nbconvert.preprocessors import ExecutePreprocessor
        import nbformat
        import copy

        if not (ROOT / "college_majors_2026.csv").is_file():
            sys.exit("--execute requires the dataset. Run python setup.py first.")
        original = nbformat.read(str(NB), as_version=4)
        executed = copy.deepcopy(original)
        executor = ExecutePreprocessor(timeout=1800, kernel_name="python3")
        print("Executing original notebook in a copy ...", flush=True)
        executor.preprocess(executed, {"metadata": {"path": str(ROOT)}})
        out = ROOT / ".artifacts" / "College_Majors_EDA_Colab.executed.ipynb"
        out.parent.mkdir(exist_ok=True)
        nbformat.write(executed, str(out))
        print(f"PASS: original notebook executed. Output copy: {out} (ignored by Git)")
    print("Validation complete.")


if __name__ == "__main__":
    main()
