"""Preserve the original teammate's charts as PNGs, without changing the notebook."""
from __future__ import annotations
import base64
import hashlib
import json
from pathlib import Path
from modeling.common import ROOT, EXPECTED_NOTEBOOK_SHA256

FIGURES = {
    8: "eda_missing_by_variable_group.png",
    19: "eda_earnings_raw_vs_log.png",
    21: "eda_earnings_by_credential.png",
    23: "eda_earnings_by_major_family.png",
    26: "eda_institution_type_and_region.png",
    28: "eda_numeric_relationships.png",
    29: "eda_spearman_correlations.png",
    35: "eda_earnings_trajectory.png",
    37: "eda_debt_risk_thresholds.png",
    38: "eda_risk_by_credential_and_school.png",
    39: "eda_risk_by_major.png",
    44: "eda_ai_exposure_and_job_growth.png",
}


def main():
    path = ROOT / "College_Majors_EDA_Colab.ipynb"
    if hashlib.sha256(path.read_bytes()).hexdigest() != EXPECTED_NOTEBOOK_SHA256:
        raise RuntimeError("Original notebook source changed; refusing to extract.")
    notebook = json.loads(path.read_text())
    out = ROOT / "reports" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, name in FIGURES.items():
        outputs = notebook["cells"][index].get("outputs", [])
        encoded = [
            o["data"]["image/png"] for o in outputs if "image/png" in o.get("data", {})
        ]
        if len(encoded) != 1:
            raise RuntimeError(f"Expected exactly 1 original figure in source cell {index}")
        image = base64.b64decode("".join(encoded[0]))
        if not image.startswith(b"\x89PNG\r\n\x1a\n"):
            raise RuntimeError("Unexpected image format in notebook")
        (out / name).write_bytes(image)
        manifest.append({
            "cell_index_zero_based": index,
            "notebook": path.name,
            "figure": f"reports/figures/{name}",
            "sha256": hashlib.sha256(image).hexdigest(),
            "bytes": len(image),
        })
    (ROOT / "reports" / "eda_figure_manifest.json").write_text(
        json.dumps({
            "source_notebook_sha256": EXPECTED_NOTEBOOK_SHA256,
            "figures": manifest,
            "description": "Exact embedded PNG output bytes from the published original EDA notebook; not regenerated or modified.",
        }, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Extracted {len(manifest)} exact original EDA output PNGs. Notebook unchanged.")


if __name__ == "__main__":
    main()
