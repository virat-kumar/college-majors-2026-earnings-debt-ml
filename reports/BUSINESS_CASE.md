# Business Case: An Evidence-Aware College Program Earnings Comparison Tool

**BUAN 6341 | Applied Machine Learning | Group 2 | The University of Texas at Dallas**

## Executive summary

We propose a **decision-support prototype** for students, university advisors and academic program leaders who want a clearer evidence-based comparison of college fields and institutions. Our study uses the College Majors 2026 public program data to predict **median earnings four years after completing a bachelor's program**. It does not predict a particular student's pay, prove the benefit of a degree, or provide an individualized investment return.

We tested four regression families on institution-disjoint validation folds and selected histogram gradient boosting by its lowest validation mean absolute error (MAE), **$6,881**. After a bounded five-fold hyperparameter search, our tuned predictor achieved **$6,957 MAE and R-squared 0.779** on 5,771 reported-outcome bachelor programs at 379 **unseen institutions**. The comparable mean-only benchmark had **$16,138 MAE**: an improvement of **56.9%** in average absolute prediction error. Tuning itself improved the heldout model by only **$36 MAE**; we have not shown that small benefit is reliably different from zero.

**Business proposition:** Help people *frame better questions* about study fields, institutional characteristics and career-market context. We recommend a **limited, human-supervised pilot**, with education about prediction error and missing data, rather than a public automated ranking or admissions decision system. The success measure would be better informed comparison and transparent error reporting, **not** an unverified promise of higher future earnings.

## 1. Business problem and decision opportunity

Prospective students often compare colleges using headline starting salaries, reputation or sticker tuition. These proxies can omit the major, program outcomes, regional wage markets, occupational demand and reporting uncertainty. University career-services teams also need an organized, responsible way to identify which program-level outcomes deserve further discussion.

We therefore ask: **Can a model trained on publicly reported program outcomes provide useful directional benchmarks for bachelor's-degree earnings at new institutions, without overstating certainty?** Our analysis is a first technical step. We cannot evaluate program profitability, individual satisfaction or tuition-adjusted return on investment without more data.

## 2. Intended users and proposed workflow

| Stakeholder | Decision supported | Responsible presentation |
|:---|:---|:---|
| Student or family | Compare major families and program outcome ranges | Show verified observed medians where available; show model estimate with typical dollar error; do not rank students |
| Career or academic advisor | Guide a conversation about majors, region and linked occupations | Put model alongside public program disclosures, advising context and non-economic preferences |
| Program director | See whether observed earnings materially differ from model benchmarks | Aggregate patterns, error analysis and coverage; never infer causal success/failure from predictions |
| Institutional research analyst | Decide whether data/reporting is adequate for expansion | Track suppression, institution coverage, drift and subgroup differences before any deployment |

**Suggested interaction:** (1) choose a reported bachelor's program; (2) show actual official program-level outcomes when released, instead of replacing them; (3) offer an estimated benchmark only when current verified inputs match our training schema; (4) show the error range and a disclosure that this is not individual advice; (5) link to source/limitations; (6) allow user feedback. Predictions should be hidden or flagged where data is missing or out of distribution.

## 3. What we learned that changes decisions

- **Major choice is an important predictor.** In the frozen final model, permuting CIP field family increased holdout-sample MAE by about **$4,578**, the strongest measured input. This supports comparing fields explicitly. It does **not** prove that switching majors causes the average earnings difference.
- **Occupation indicators add meaningful context.** Permuting linked occupational growth, AI-software share and occupation median wage increased MAE by approximately **$3,036, $2,352 and $1,607**, respectively, on the fixed 2,000-program diagnostic sample. We should label them as proxies for labor-market composition, not an endorsement that deploying AI tools raises salaries.
- **One-size-fits-all error estimates are misleading.** Our reported 5,771-program test MAE is **$6,957**, but the most common fields vary: about **$4,393 for Education**, **$5,547 for Engineering**, **$7,568 for Business/Management** and **$8,657 for Health Professions**. The highest observed earnings quintile averaged **$10,595 MAE**; we recommend warning against treating an elite program's point estimate as precise.
- **Most programs do not report the target.** Four-year earnings are only observed for **51,818 of 212,707** cleaned programs. We selected the 25,283 observed bachelor's outcomes for our primary task. Privacy suppression means our predictions and error assessment may not represent smaller, newer or suppressed programs.

![Predictive features, not causal drivers.](figures/permutation_importance.png)

![Test-set errors by frequently observed field family.](figures/errors_by_major.png)

## 4. Recommended product and adoption scope

We recommend an **internal advising/research dashboard pilot** with two display layers:

1. **Observed reporting layer:** source, program definition, median reported four-year earnings, institution/major, credential and reporting availability. This layer remains the source of truth.
2. **Predictive context layer:** estimated program median, model version, 14-input schema, indicative expected absolute error, applicable population and a link to our technical model report. Label missing or unsupported cases rather than silently extrapolating.

We should not expose a "best college" score, affordability ranking, automatic program closure recommendation, or student-level eligibility determination. Tuition and earnings are not sufficient to estimate debt burden or net present value. The separate debt-risk target was explored in EDA, **not included as a trained classifier** in this project's final modeling deliverable.

## 5. Proposed pilot and measurable success checks

| Phase | Proposed action | Objective completion evidence |
|---|---|---|
| 1. Data governance | Confirm latest College Scorecard release, suppression rules, field coding and permissible reuse | Source inventory, versioned data checksums and leakage review |
| 2. Offline validation | Reproduce this study; evaluate a new time cohort or independent university sample | Locked external test MAE/RMSE, coverage and institution-level uncertainty |
| 3. Advisor-only prototype | Present observed data and predicted benchmarks with error warnings | Advisor feedback, usability completion, documented cases of confusion |
| 4. Fairness and drift review | Compare error by field, region, institutional control and missing-data groups | Subgroup error dashboard; explicit action thresholds before release |
| 5. Limited student-facing trial | If earlier gates pass, compare informed-decision surveys and source comprehension | Pre-registered feedback metrics and controlled evaluation |

**Proposed evaluation metrics, not achieved business results:**

- **Technical:** coverage of eligible programs, independent cohort MAE and RMSE, absolute-error quantiles and subgroup calibration.
- **User:** percentage of people who understand what an estimate represents; successful comparison completion; advisor feedback quality.
- **Operational:** data freshness, source attribution completeness, missing-input frequency and model/version traceability.
- **Safety:** misleading ranking/individual outcome complaints; differential error across vulnerable or low-coverage segments.

We would use **stop-and-review** rules if new holdout error becomes materially worse than the established baseline, if missingness/coverage shifts sharply, if users mistake the estimate for a guaranteed salary, or if observed errors are unacceptably unequal across segments. Numeric thresholds should be pre-registered by the actual pilot owner, not retrofitted after seeing outcomes.

## 6. Responsible interpretation and economics

Our project establishes **prediction improvement over a constant benchmark**, not realized earnings increases, tuition savings, successful program decisions or a measured financial return. We do not convert the model's $9,181 MAE reduction into claimed dollars saved for students. Doing so would confuse predictive error with an economic benefit.

Potential advantages include clearer advising conversations, a unified view of open data and an auditable risk/uncertainty discussion. Potential implementation costs include data stewardship, validation, security/privacy review, accessibility, error monitoring, advisor training and source updates. A responsible future business case would estimate those costs and measure observed user decision quality in a pilot before scaling.

### Bias, privacy and uncertainty safeguards

The source rows are institution-program aggregates, not student records. That reduces direct personal-data risk but does not eliminate population or fairness concerns. Reporting suppression is not random, institutional characteristics reflect structural inequities, and our HBCU feature may act as a proxy for historical and demographic context. We should review whether the feature is justified before any real-world advisory use. We will not interpret model importance as causal policy guidance.

**We recommend proceeding to a controlled prototype only**, accompanied by verified public outcomes, uncertainty communication and independent validation. The current model is appropriate as a classroom research demonstration, not a high-stakes decision engine.

## 7. Reproducibility and evidence

- **Original EDA:** [College_Majors_EDA_Colab.ipynb](../College_Majors_EDA_Colab.ipynb), unchanged; the team's earlier notebook explains the cleaning and originally defined institution split.
- **Data:** [DATA.md](../DATA.md), [data/split_ids.csv](../data/split_ids.csv), original 126.3 MB dataset in public GitHub Release `data-v1.0.0`.
- **Four-model experiment:** [model_selection.py](../modeling/model_selection.py), [model_comparison.csv](model_comparison.csv) and [model_selection_folds.csv](model_selection_folds.csv).
- **Tuning and final check:** [model_fine_tuning.py](../modeling/model_fine_tuning.py), [tuning_candidates.csv](tuning_candidates.csv) and [model_evaluation.json](model_evaluation.json).
- **Explanation:** [MODEL_EXPLANATION.md](MODEL_EXPLANATION.md), [feature_importance.csv](feature_importance.csv) and [error_by_segment.csv](error_by_segment.csv).

**Attribution:** Data are credited to the public College Majors 2026: Earnings, Debt, Jobs, AI Kaggle publisher, with the source listing's CC BY-SA 4.0 terms. The original EDA notebook credits Manoj Elangovan Jayanthi. We describe the modeling and discussion as the work of BUAN 6341 Group 2 and preserve the original source notebook.
