# Exploratory Data Analysis — Group 2

*Source: the provided “EDA Summary: College Majors 2026 (Earnings, Debt, Jobs, AI)”, two-page group summary by Manoj Elangovan Jayanthi. The original notebook contains the associated figures, tables and analytical code. This Markdown guide summarizes the supplied report rather than replacing the authored notebook.*

## 1. Data audit and cleaning

The Kaggle dataset combines **College Scorecard**, the **NCES CIP-to-SOC crosswalk**, **BLS job projections** and **O*NET**. Each row represents a unique program (institution × study field × credential):

| Stage | Program records |
| --- | ---: |
| Raw dataset (72 variables) | 227,980 |
| Foreign institutions removed | 4,662 |
| Duplicate branch-campus copies removed | 10,611 |
| **Remaining cleaned programs** | **212,707** |
| Programs reporting 4-year earnings | 51,818 |
| Bachelor's programs reporting 4-year earnings | 25,283 |
| Programs reporting debt-to-income | 37,789 |

These outcome fields are largely **privacy-suppressed** for smaller cohorts; missingness is not random. The notebook identifies roughly 30 leakage-prone outcome/derived fields, including 1-/5-year earnings, debt-to-earnings, `national_median_earnings_4yr_usd` and `earnings_vs_national_pct`.

The EDA uses an **80/20 split by institution (`opeid6`)** and five grouped CV folds. All exploratory comparisons use **training rows only**; `split_ids.csv` is exported locally for the team.

## 2. Regression: median earnings four years after completion

The reported median is **$59,074**, with a maximum of **$336,392**. The earnings distribution is right-skewed (skewness ≈1.80, approaching 0.05 after log transformation).

Median earnings by credential rise from approximately **$41k certificates** and **$49k associate degrees** to **$59k bachelor's**, **$74k master's**, **$99k doctorates**, and **$113k first-professional degrees**.

For bachelor's programs, the reported typical comparison is approximately **$94k Engineering**, **$87k Computer Science**, **$50k Psychology**, and **$43k Visual and Performing Arts**. Even within CS bachelor's programs the middle 80% spans approximately **$73k–$144k**. School type, region and field are associated with outcomes; online delivery did not show a meaningful difference in this EDA.

Strong numeric correlates of 4-year earnings (Spearman rank correlation) include technology use (~0.45), occupation wage (~0.40), job growth (~0.40), AI-software share (~0.35) and average SAT (~0.34). **Association is not causality.** Missing SAT, tuition and admission-rate data may themselves be informative; the pipeline retains missingness indicators.

Outliers are retained. A **mean-only regression baseline** has **grouped-CV MAE $15,799** for bachelor's programs.

## 3. Classification: program debt risk

The report defines `payment_to_income_pct_1yr` as **annual loan payment as a percentage of first-year earnings**. Different cutoffs change the class distribution:

| High-risk condition | Training-program share |
| --- | ---: |
| Above 8% | 35.7% |
| Above 10% | 20.6% |
| Above 12% | 11.9% |
| Above 20% | 3.0% |

The 8% cutoff labels more than a third of the 29,192 relevant training programs as high-risk; the minority class is **not** tiny at that threshold. Risk rises substantially with credential level and varies by field. Bachelor's-only classification appears driven mostly by **low early-career earnings**, whereas all-credential classification shows debt-related differences. The team needs to decide whether classification adds distinct predictive value.

## 4. Proposed preprocessing

The report proposes a scikit-learn `ColumnTransformer` with **14 predictors**: nine numeric institutional/job/AI variables, HBCU indicator and four categoricals (institution control, field family, region, distance education). It uses median imputation + missing indicators, categorical one-hot encoding, and standardization **for linear models only**. Fitting happens inside each grouped CV fold, preventing preprocessing leakage.

## 5. Outstanding decisions

- Restrict regression to bachelor's programs or use all credential levels?
- Keep debt-risk classification, and if so at which cutoff?
- Confirm leakage-safe predictor columns; is median debt appropriate for earnings regression?
- Model raw earnings or `log(earnings)` and how to report evaluation metrics?
- Update proposal assumptions (dataset has **72 variables**; 8% risk class ≈36% of relevant programs).

## 6. Limitations

The findings are restricted to programs reporting outcomes and should not be generalized to suppressed cohorts. Small program cohorts produce noisy medians; cohort-size weights have not been applied. Earnings are not cost-of-living-adjusted; interactions and detailed CIP codes remain for the modeling phase. Decisions to remove foreign institutions and main/branch duplication warrant group review.

*See the **original Colab notebook** for the exact methodology, tables and figures. The numbers above are transcribed from the supplied two-page EDA summary and are not newly re-estimated.*
