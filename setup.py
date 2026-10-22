#!/usr/bin/env python3
"""One-command Windows/Linux bootstrap: python setup.py."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
PYTHON = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run(command: list[str]) -> None:
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Install and smoke-test without downloading Kaggle data (used by CI).",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Use an already-created .venv without reinstalling dependencies.",
    )
    args = parser.parse_args()

    if not (3, 10) <= sys.version_info[:2] <= (3, 13):
        sys.exit(
            "Python 3.10-3.13 is supported. Install one of these versions, "
            "then run 'python setup.py' again."
        )

    if not PYTHON.exists():
        print("Creating isolated environment:", VENV, flush=True)
        try:
            venv.EnvBuilder(with_pip=True).create(VENV)
        except Exception as exc:
            sys.exit(
                f"Could not create virtual environment: {exc}\n"
                "On Ubuntu/Debian, install python3-venv for your Python version."
            )

    if not args.skip_install:
        run([str(PYTHON), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")])

    if not args.skip_data:
        run([str(PYTHON), str(ROOT / "download_data.py")])

    validate = [str(PYTHON), str(ROOT / "validate.py")]
    if not args.skip_data:
        validate.append("--require-data")
    run(validate)

    print(
        "\nSetup complete. Start JupyterLab with:\n"
        "    python launch.py\n"
        "Open College_Majors_EDA_Colab.ipynb from the project root.\n"
        "Both commands work on Windows and Linux; no activation is required.",
        flush=True,
    )


if __name__ == "__main__":
    main()
