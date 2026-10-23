"""Render the authoritative Markdown report to PDF via Pandoc and XeLaTeX.

Usage: python submission/render_report.py
Requires Pandoc and XeLaTeX (not required to *read* committed PDF).
"""
from __future__ import annotations
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "Group_2_Final_Project_Report.md"
DESTINATION = HERE / "Group_2_Final_Project_Report.pdf"


def main():
    for binary in ("pandoc", "xelatex"):
        if shutil.which(binary) is None:
            raise SystemExit(f"Cannot render PDF: install {binary} and retry.")
    cmd = [
        "pandoc", SOURCE.name,
        "--from=markdown+raw_tex+tex_math_dollars",
        "--to=pdf",
        "--pdf-engine=xelatex",
        "--toc",
        "--toc-depth=2",
        "--number-sections",
        "--resource-path=.:../reports",
        "--include-in-header=report_style.tex",
        "--output", DESTINATION.name,
    ]
    subprocess.run(cmd, check=True, cwd=HERE)
    if not DESTINATION.is_file() or DESTINATION.stat().st_size < 100_000:
        raise AssertionError("PDF render did not produce a complete report.")
    print(f"Rendered {DESTINATION} ({DESTINATION.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
