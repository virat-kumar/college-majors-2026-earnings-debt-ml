#!/usr/bin/env python3
"""Launch the project's isolated JupyterLab on Windows or Linux."""
from __future__ import annotations
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
python = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
if not python.is_file():
    sys.exit("Missing .venv. Run 'python setup.py' first.")
raise SystemExit(subprocess.call([str(python), "-m", "jupyter", "lab"], cwd=root))
