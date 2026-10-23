# Model Explanation and Independent Evaluation

**BUAN 6341 | Applied Machine Learning | Group 2, UT Dallas**

We model reported median earnings **four years after program completion**, in U.S. dollars, for bachelor's-degree programs. This is a **program-level**, not individual-student, prediction. Our initial question was whether accessible institutional characteristics, field of study and occupation-market indicators could improve on the EDA notebook's mean-only baseline while keeping schools in the final holdout entirely separate from schools used for fitting.

## 1. Task definition and methodological decisions

We restricted the regression task to bachelor's programs reporting the four-year earnings target, following our original EDA's recommended scope. From the raw 227,980 records, the original cleaning retained 212,707 unique U.S. program records; 25,283 bachelor's programs reported earnings. We reused the **unchanged, committed** `data/split_ids.csv` created by the original notebook, rather than inventing a new split: 19,512 train programs from 1,420 schools and 5,771 held-out programs from 379 other schools. Exactly zero institutions overlap. Five existing school-disjoint training folds supported selection and tuning.

We chose 14 EDA-motivated inputs: 9 numeric features (institution admission rate, SAT average, undergraduate enrollment, in/out-of-state tuition, program awards, projected occupational growth, occupation median wage, AI-software occupation share); 1 binary feature (HBCU indicator); and 4 categorical features (institution control, CIP field family, region, distance education). All inputs flow through an estimator-specific `Pipeline` and `ColumnTransformer`. Numeric missing values are median-imputed with missing-value indicator flags, categorical missingness is imputed then one-hot encoded, and rare categorical levels are handled by `OneHotEncoder(min_frequency=20)`. Ridge uses standardized numeric features; tree models do **not** need numeric scaling. Pipeline fitting occurs within each training fold.

We excluded target-derived earnings, national earnings benchmarks calculated across both partitions, other graduation outcome columns and identifiers. In particular, `earnings_vs_national_pct` is calculated from the response and **cannot** serve as an input. The occupation wage and growth proxies are aggregate sectoral descriptors, not individual earnings. Their measurement dates and representativeness deserve further validation before operational deployment.

The EDA found that untransformed earnings are right-skewed and explored their logarithm. We **did not** log-transform the final response: every candidate directly predicts raw dollar earnings, making MAE interpretable as average absolute dollar error. We retained authentic outliers; MAE is relatively robust compared with squared-error metrics. The EDA's log-scale figure is descriptive, not evidence that this final fit used a log target.

## 2. Four-family model comparison

We trained exactly four distinct families and a mean-only reference. All used the **same five preset school-disjoint validation folds**, the **same 14 raw feature columns**, the same bachelor-only sample and MAE as the primary choice criterion. RMSE and R-squared are secondary diagnostics.

| Model family | Five-fold CV MAE | CV RMSE | CV R-squared |
|:---|---:|---:|---:|
| **Histogram gradient boosting** | **$6,880.79** | $9,903.75 | 0.7701 |
| Random forest | $6,918.22 | $10,074.52 | 0.7624 |
| Decision tree | $8,247.56 | $11,924.36 | 0.6675 |
| Ridge regression | $8,964.19 | $12,790.52 | 0.6165 |
| Mean-only reference | $15,806.18 | $20,697.67 | -0.0012 |

![Four models plus the baseline on identical institutional folds.](figures/model_comparison.png)

**Why we picked boosting.** Unlike linear Ridge, decision trees can express nonlinearities. Bagged forests reduce single-tree variance, while histogram boosting adds many sequential, corrective small trees to minimize remaining prediction errors. Boosting had the **lowest preset mean CV MAE**, which was the sole selection criterion, and a slightly stronger CV R-squared. However, it beat random forest by **only $37.43** MAE, so we do not claim reliable statistical superiority of this family over the forest.

## 3. Hyperparameter tuning

After fixing the winning model family, we ran **12 predetermined candidate settings x 5 preassigned institutional folds = 60 validation fits**, using GridSearchCV on **training institutions only**. The candidate list includes the untouched default selection configuration plus 11 reproducibly sampled combinations. The search considers learning rate, boosting iterations, tree leaf count, minimum leaf sample size and L2 regularization. The candidate list and scores are recorded in [tuning_candidates.csv](tuning_candidates.csv).

| Boosting hyperparameter | Initial family setting | Best searched setting |
|---|---:|---:|
| Learning rate | 0.070 | **0.095** |
| Boosting iterations | 170 | **170** |
| Maximum leaves per tree | 31 | **47** |
| Minimum samples per leaf | 30 | **15** |
| L2 regularization | 1.0 | **4.0** |

| Five-fold training-only validation | MAE (USD) |
|---|---:|
| Selected family, before tuning | $6,880.79 |
| Selected family, after tuning | **$6,855.28** |
| Improvement | **$25.51 (0.37%)** |

![Grouped CV errors for the top ten tuning candidates.](figures/tuning_candidates.png)

The small CV improvement is meaningful as an observed arithmetic difference but is not sufficient evidence that tuning made the predictor robustly better in every possible institution.

## 4. Final independent school-level holdout

Only after the hyperparameter search ended did we evaluate the reserved **5,771 programs from 379 unseen schools**, fitting the tuned and original candidate on the full training partition. The mean-only control uses the same train split. This is a **single final holdout**, not a tuning set.

| Final holdout model | MAE (USD) | RMSE (USD) | R-squared |
|---|---:|---:|---:|
| **Tuned histogram gradient boosting** | **$6,956.68** | **$10,111.64** | **0.7793** |
| Untuned histogram gradient boosting | $6,993.01 | $10,173.43 | 0.7766 |
| Mean-only prediction | $16,137.72 | $21,586.67 | -0.0057 |

Our tuned predictor reduced holdout MAE by **$9,181.04 (56.89%)** against the mean-only benchmark, but by only **$36.34 (0.52%)** relative to the otherwise comparable untuned boosting model. Its median absolute error was **$5,069.47** and the 90th-percentile absolute error was **$14,800.24**. In practical terms, the latter means about one in ten evaluated programs had an absolute prediction error above roughly $14,800; this is not a confidence interval for an individual student's salary.

We checked uncertainty via a **paired 2,000-resample bootstrap that resamples held-out institutions**, not individual rows. For *tuned minus untuned* MAE the descriptive 95% bootstrap interval was **-$107.73 to +$40.11**, which includes zero. We **cannot establish a reliable tuning advantage** from this holdout. For *tuned minus mean-only*, the interval was **-$9,799.47 to -$8,634.72**, indicating a large improvement over that simple baseline in this sample.

![Predicted versus actual program median earnings in the untouched test institutions.](figures/actual_vs_predicted.png)

![Distribution of final model residuals, showing the middle 98% for readability.](figures/residual_distribution.png)

## 5. Explainability: what the fitted model uses

We froze the final predictor and then computed **raw-feature permutation importance** on a fixed random sample of **2,000 held-out programs**, with three seeded permutations per input. Importance is the increase in dollar MAE caused by shuffling one field while holding all other inputs fixed. A large increase means the *fitted model relies on that input for prediction*; it does **not** mean manipulating the field causes earnings to change.

| Input | Increase in MAE when shuffled |
|---|---:|
| CIP field family | $4,578 |
| Occupational growth projection | $3,036 |
| AI software occupation share | $2,352 |
| Occupation median wage | $1,607 |
| Institution region | $987 |
| Out-of-state tuition | $943 |
| Average SAT | $920 |

![Feature-level permutation importance computed after model selection.](figures/permutation_importance.png)

Major family dominates this descriptive importance ranking, consistent with the original EDA's large field-level earnings differences. Industry growth and AI occupation usage are also strong *predictive proxies*. Their relative permutation effects can be distorted by correlated inputs; neither importance nor correlations can isolate causal effects of an AI curriculum, SAT admissions or tuition changes.

## 6. Where the predictions work less well

The holdout has substantial variation across institutions and fields. For the most common major families, example error slices were:

| Field of study | Tested programs | Held-out MAE |
|---|---:|---:|
| Engineering | 361 | $5,547 |
| Education | 281 | $4,393 |
| Social sciences | 509 | $6,241 |
| Business/management/marketing | 890 | $7,568 |
| Health professions | 541 | $8,657 |
| Visual and performing arts | 366 | $6,009 |

![Held-out dollar errors for frequently observed fields.](figures/errors_by_major.png)

Average error is higher for the highest actual-earnings quintile: approximately **$10,595**, versus about **$4,814** in the second quintile. Our model also tends to predict **above** actual earnings in the lowest quintile (actual mean about $42,095, prediction mean about $47,478) and **below** actual earnings in the highest quintile (actual mean about $98,931, prediction mean about $92,317). These are **post-hoc diagnostics on the final test**; we did not change features or hyperparameters after inspecting them.

![Mean true and predicted values by actual-outcome quintile.](figures/earnings_quintile_calibration.png)

Potential fairness and reliability issues include different errors by institution control (public, nonprofit, for-profit), region, and study family, as well as uncertain completeness of suppressed outcomes. We must not equate an observed group-level difference with student-level discrimination or a causal factor without further study.

## 7. Limits and appropriate use

- **Reported programs only.** Only 25,283 of the bachelor programs in the cleaned source have the four-year earnings label. Privacy suppression can bias the observed sample. Our test performance does not apply automatically to excluded programs or individual graduates.
- **No program-level causal effect.** The fitted model gives predictions/associations, not returns to enrolling, changing majors, altering tuition or adopting AI technologies.
- **No temporal or external validation.** A random unseen-institution holdout tests transfer across institutions **within this snapshot**; it is not proof about future cohorts, new degree offerings or distribution shifts.
- **Comparability and data quality.** Earnings are unadjusted for regional living costs, household needs, enrollment selection, program resources and student background. Group-level reporting variability may affect target precision.
- **Small tuning advantage.** The grouped bootstrap interval for the $36 tuning gain crosses zero; the conservative interpretation is that tuned and untuned boosting perform similarly on this holdout.
- **Out-of-domain caution.** We should abstain for missing, structurally unseen majors or unverified school data; use actual official public reports alongside model estimates.

**Artifacts and audit trail:** [model_selection_folds.csv](model_selection_folds.csv), [model_comparison.csv](model_comparison.csv), [model_selection.json](model_selection.json), [tuning_candidates.csv](tuning_candidates.csv), [model_evaluation.json](model_evaluation.json), [feature_importance.csv](feature_importance.csv), [error_by_segment.csv](error_by_segment.csv), and [heldout_predictions.csv](heldout_predictions.csv) are generated by the separate executable scripts in `modeling/`. The **original EDA notebook is unchanged**.
