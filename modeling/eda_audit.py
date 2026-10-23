"""Independent dataset/cleaning audit mirroring the protected source EDA notebook.

This does not modify the original notebook, target partition, or raw CSV.
Usage: python -m modeling.eda_audit
"""
from __future__ import annotations
import json
import hashlib
import numpy as np
import pandas as pd
from modeling.common import ROOT, TARGET, EXPECTED_NOTEBOOK_SHA256

KEY = ["opeid6", "cip_code_4digit", "credential_level"]
COLUMNS = [
    "program_id", "institution_control", "opeid6", "cip_code_4digit",
    "credential_level", "unitid", "is_main_campus",
    "outcomes_shared_across_campuses", TARGET, "median_debt_usd",
    "payment_to_income_pct_1yr",
]


def main():
    raw= pd.read_csv(ROOT/"college_majors_2026.csv",usecols=COLUMNS,low_memory=False)
    n_raw=len(raw)
    assert raw.program_id.is_unique
    foreign=raw.institution_control.eq("Foreign")
    domest=raw.loc[~foreign].copy()
    sorted_df=domest.sort_values(["is_main_campus","unitid"],ascending=[False,True],na_position="last")
    duplicate_flag=sorted_df.duplicated(KEY,keep="first")
    duplicates=sorted_df.loc[duplicate_flag]
    if not duplicates.outcomes_shared_across_campuses.eq(1).all():
        raise AssertionError("EDA removed non-shared outcomes; audit discrepancy!")
    if domest.groupby(KEY)[TARGET].nunique(dropna=True).gt(1).any():
        raise AssertionError("Grouped campus earnings disagree; cannot deduplicate.")
    base=sorted_df.loc[~duplicate_flag].sort_index()
    split=pd.read_csv(ROOT/"data"/"split_ids.csv")
    assert len(base)==len(split)==212707
    assert set(base.program_id)==set(split.program_id)
    target_reported=base[TARGET].notna().sum()
    bachelor_reported=(base.credential_level.eq(3)&base[TARGET].notna()).sum()
    counts=[
        ("Raw observations",n_raw,"Original Kaggle snapshot"),
        ("After dropping exact duplicates",n_raw,"No exact duplicate raw rows per EDA"),
        ("After removing foreign institutions",len(domest),"Restrict to U.S. scope"),
        ("After collapsing repeated shared-campus outcomes",len(base),"Only flag=1 rows removed; verified no conflicting group earnings"),
    ]
    scope=[
        ("Cleaned U.S. program records",len(base)),
        ("Programs reporting 4-year earnings",int(target_reported)),
        ("Bachelor's programs with 4-year earnings",int(bachelor_reported)),
        ("Programs reporting debt-to-income",int(base.payment_to_income_pct_1yr.notna().sum())),
    ]
    out=ROOT/"reports"
    out.mkdir(exist_ok=True)
    pd.DataFrame(counts,columns=["stage","n_records","audit_reason"]).to_csv(out/"eda_cleaning_audit.csv",index=False)
    pd.DataFrame(scope,columns=["population","n_records"]).to_csv(out/"eda_scope_audit.csv",index=False)
    observed=base.loc[base[TARGET].gt(0),TARGET]
    summary={
        "raw_n":n_raw,
        "raw_unique_program_ids":int(raw.program_id.nunique()),
        "foreign_removed":int(foreign.sum()),
        "shared_campus_rows_removed":int(duplicate_flag.sum()),
        "non_shared_rows_removed":int(duplicates.outcomes_shared_across_campuses.ne(1).sum()),
        "groups_with_conflicting_reported_earnings":int(domest.groupby(KEY)[TARGET].nunique(dropna=True).gt(1).sum()),
        "cleaned_rows":len(base),
        "reported_earnings":int(target_reported),
        "reported_earnings_pct_of_cleaned":round(100*target_reported/len(base),2),
        "bachelor_reported_earnings":int(bachelor_reported),
        "debt_income_reported":int(base.payment_to_income_pct_1yr.notna().sum()),
        "earnings_raw_skewness":round(float(observed.skew()),3),
        "earnings_log_skewness":round(float(np.log(observed).skew()),3),
        "raw_dataset_sha256":hashlib.sha256((ROOT/"college_majors_2026.csv").read_bytes()).hexdigest(),
        "original_notebook_sha256":hashlib.sha256((ROOT/"College_Majors_EDA_Colab.ipynb").read_bytes()).hexdigest(),
        "reference_split_program_ids_match_cleaned":True,
    }
    assert summary["original_notebook_sha256"]==EXPECTED_NOTEBOOK_SHA256
    (out/"eda_audit.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
