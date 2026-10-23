#!/usr/bin/env python3
"""Cross-platform, end-to-end Group 2 experiment runner using project .venv.

    python setup.py
    python run_project.py
    python run_project.py --verify-only
    python run_project.py --render-pdf    # optional Pandoc + XeLaTeX
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run(*args: str) -> None:
    command = [str(VENV_PYTHON), *args]
    print("\n" + "=" * 72, flush=True)
    print("Executing:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Check saved work without retraining models.")
    parser.add_argument("--render-pdf", action="store_true", help="Regenerate PDF from Markdown using installed Pandoc/XeLaTeX.")
    args = parser.parse_args()
    if not VENV_PYTHON.is_file():
        sys.exit("Run 'python setup.py' first to prepare the Python environment and verified CSV.")
    if not args.verify_only:
        for module in (
            "modeling.eda_audit",
            "modeling.extract_eda_figures",
            "modeling.model_selection",
            "modeling.model_fine_tuning",
            "modeling.model_explainability",
        ):
            run("-m", module)
    if args.render_pdf:
        run(str(ROOT / "submission" / "render_report.py"))
    run("-m", "unittest", "discover", "-s", "tests", "-v")
    run(str(ROOT / "verify_project.py"))
    print("\nPROJECT VALIDATION PASSED. Final submission: submission/Group_2_Final_Project_Report.pdf", flush=True)


if __name__ == "__main__":
    main()
