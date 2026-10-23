#!/usr/bin/env python3
"""Independent integrity and report audit, without retraining or changing data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from modeling.common import ROOT, EXPECTED_NOTEBOOK_SHA256, FAMILIES


def require(condition: bool, description: str) -> None:
    if not condition:
        raise AssertionError(description)
    print("PASS:", description, flush=True)


def sha256_file(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    report=ROOT/"reports"
    submission=ROOT/"submission"
    data=ROOT/"data"
    require(sha256_file(ROOT/"College_Majors_EDA_Colab.ipynb")==EXPECTED_NOTEBOOK_SHA256,
            "Original 50-cell EDA notebook byte hash unchanged")
    require(sha256_file(ROOT/"college_majors_2026.csv")=="58f1c10e79793f33049f3b381cd81d4d0c190ca1052e427c29779d0c89c5ceca",
            "Exact versioned Kaggle source data SHA256")
    require(sha256_file(data/"split_ids.csv")=="e5215d887840ae25e0c953793842b1886670f51b879783904988a1348f7bfe6c",
            "Canonical original group split SHA256")
    require(not list((ROOT/".github"/"workflows").glob("*")) if (ROOT/".github"/"workflows").exists() else True,
            "No GitHub Actions workflow files")
    eda=json.loads((report/"eda_audit.json").read_text())
    require((eda["raw_n"],eda["foreign_removed"],eda["shared_campus_rows_removed"],eda["cleaned_rows"])==
            (227980,4662,10611,212707),"Independently audited cleaning matches original EDA")
    require(eda["non_shared_rows_removed"]==0 and eda["groups_with_conflicting_reported_earnings"]==0,
            "Original shared-campus compression removed no conflicting earnings")
    require((eda["reported_earnings"],eda["bachelor_reported_earnings"])==(51818,25283),
            "Outcome reporting and bachelor's counts source verified")

    selection=pd.read_csv(report/"model_comparison.csv")
    folds=pd.read_csv(report/"model_selection_folds.csv")
    models=set(selection.model)
    require(models==set(FAMILIES)|{"dummy"} and len(folds)==25 and
            all(set(g.fold)==set(range(5)) for _,g in folds.groupby("model")),
            "Exactly four model families + mean control on identical five folds")
    for row in selection.itertuples():
        actual=float(folds.loc[folds.model==row.model,"mae_usd"].mean())
        require(abs(actual-row.cv_mae_usd)<.02,f"{row.model} CV MAE from independent fold mean")
    chosen=str(selection.loc[selection.model.ne("dummy")].sort_values("cv_mae_usd").iloc[0].model)
    require(chosen=="hist_gradient_boosting","Selected family is lowest CV MAE among four")
    meta=json.loads((report/"model_evaluation.json").read_text())
    require(meta["selected_family"]==chosen and meta["train_programs"]==19512 and meta["test_programs"]==5771,
            "Tuning/holdout refer to selected family and exact original population")
    require(len(pd.read_csv(report/"tuning_candidates.csv"))==12 and meta["tuning_grouped_folds"]==5,
            "12 tuned candidates, five institution-disjoint validation folds")

    pred=pd.read_csv(report/"heldout_predictions.csv")
    require(len(pred)==5771 and pred.program_id.is_unique and pred.opeid6.nunique()==379,
            "Exactly 5771 program-level heldout predictions from 379 unseen institutions")
    actual=pred["actual_earnings_4yr_usd"].to_numpy()
    for model,col in [("tuned_winner","tuned_prediction_usd"),("untuned_winner","original_prediction_usd")]:
        estimated=pred[col].to_numpy()
        result=meta["holdout"][model]
        require(abs(mean_absolute_error(actual,estimated)-result["mae_usd"])<.02 and
                abs(np.sqrt(mean_squared_error(actual,estimated))-result["rmse_usd"])<.02 and
                abs(r2_score(actual,estimated)-result["r2"])<.0002,
                f"Final {model} MAE/RMSE/R2 independently recomputed from public CSV")
    require(meta["holdout"]["tuned_winner"]["mae_usd"]<meta["holdout"]["dummy_mean"]["mae_usd"],
            "Model beats mean-only final holdout baseline")
    explanation=json.loads((report/"explainability_summary.json").read_text())
    interval=explanation["bootstrap_tuned_minus_untuned"]
    require(interval["lower_95_usd"]<0<interval["upper_95_usd"],
            "Report's claim of inconclusive small tuning gain agrees with institution bootstrap")
    require(len(pd.read_csv(report/"feature_importance.csv"))==14,
            "Permutation importance covers all fourteen original feature names")

    manifest=json.loads((report/"eda_figure_manifest.json").read_text())
    require(manifest["source_notebook_sha256"]==EXPECTED_NOTEBOOK_SHA256 and len(manifest["figures"])==12,
            "Twelve original untouched notebook PNG outputs traced to source cells")
    for entry in manifest["figures"]:
        png=ROOT/entry["figure"]
        require(png.is_file() and sha256_file(png)==entry["sha256"],
                "Exact original embedded EDA output: "+png.name)
    pdf=submission/"Group_2_Final_Project_Report.pdf"
    md=submission/"Group_2_Final_Project_Report.md"
    require(md.is_file() and pdf.is_file() and pdf.stat().st_size>200000 and
            pdf.read_bytes()[:5]==b"%PDF-", "Final Markdown and true non-empty searchable PDF exist")
    source=md.read_text(encoding="utf-8")
    image_links=re.findall(r"!\[[^\]]*\]\(([^)]+)\)",source)
    require(len(image_links)==15 and all((submission/path).is_file() for path in image_links),
            "Final report cites fifteen existing EDA/model plots")
    require(all(key in source for key in
                ["19,512","5,771","6,956.68","6,993.01","6,855.28","56.89%","10,611",
                 "Business case","permutation importance","Ridge","Random Forest"]),
            "Direct submission report includes all checked key stages and results")
    if shutil.which("pdfinfo") and shutil.which("pdftotext"):
        info=subprocess.run(["pdfinfo",str(pdf)],capture_output=True,text=True,check=True).stdout
        text=subprocess.run(["pdftotext","-layout",str(pdf),"-"],capture_output=True,text=True,check=True).stdout
        match=re.search(r"^Pages:\s+(\d+)\s*$",info,re.M)
        require(match is not None and int(match.group(1))>=20, "PDF physically multi-page and readable")
        fig_nums=set(int(i) for i in re.findall(r"Figure\s+(\d+)\.",text))
        require(fig_nums==set(range(1,16)),"All fifteen numbered figures appear in PDF text")
        require("Executive summary" in text and "Conclusion" in text and "Business case" in text,
                "PDF extraction includes complete front matter, business section and ending")
    require((ROOT/"README.md").read_text().count("submission/Group_2_Final_Project_Report.pdf")>=1,
            "README directly navigates to final submission PDF")
    print("\nALL PROJECT AUDIT CHECKS PASSED.",flush=True)


if __name__ == "__main__":
    main()
