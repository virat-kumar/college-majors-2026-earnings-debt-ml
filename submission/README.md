# Group 2 | Complete academic submission

This folder contains the **final, team-ready applied machine-learning report**, not the earlier two-page EDA-only working note.

## Primary deliverables

| File | How to use it |
| --- | --- |
| **[Group_2_Final_Project_Report.pdf](Group_2_Final_Project_Report.pdf)** | **Submit or circulate for review.** A 26-page, searchable, academically styled PDF with all final experiment results and 16 embedded visuals. |
| **[Group_2_Final_Project_Report.docx](Group_2_Final_Project_Report.docx)** | **Edit collaboratively in Word or Microsoft 365.** A 21-page polished document with a separate cover/contents, tables, numbered figure captions and all 16 embedded visuals. |
| [Group_2_Final_Project_Report.md](Group_2_Final_Project_Report.md) | Authoritative editable Markdown source for PDF and Word generation. |

## The report covers

- Original 227,980-row / 72-column source, a justified cleaning audit, missingness and exploratory findings, and extracted original EDA charts.
- Defined bachelor's earnings regression task, 14 leakage-audited predictors, preprocessing and frozen institution-level holdout.
- **Four executed model families** (Ridge, Decision Tree, Random Forest, Histogram Gradient Boosting) compared on the same five group-disjoint validation folds.
- A **training-only search across 12 boosting configurations**, showing before/after results without test leakage.
- **Unseen-institution final test results**, including MAE, RMSE, R-squared, comparison to a mean-only baseline, uncertainty and error segments.
- Model reliance and interpretation visuals, risk/limitations, a human-supervised advising business case, a workflow diagram, references and a responsible next-steps decision record.

The held-out tuned model achieves **USD 6,956.68 MAE** and **R² = 0.7793** on 5,771 bachelor's programs from 379 previously unseen institutions. These are **program-median prediction** results, not individual salary projections. No debt-risk classifier was trained; that task remains exploratory.

## Reproduction

From the repository root, use **python setup.py** and **python run_project.py** for the data-science workflow; the existing notebook is never modified.

The PDF is generated from the Markdown using Pandoc and XeLaTeX:

~~~sh
python submission/render_report.py
~~~

The editable Word report uses Pandoc and the Python **python-docx** package:

~~~sh
python -m pip install python-docx
python submission/render_word.py
~~~

The workflow figure can be regenerated with Matplotlib via **python submission/make_workflow_diagram.py**. The finished PDF and DOCX are already committed; regenerating either is **not required** to read, edit or submit them.

## Team review boundary

The report contains model-selection results already supported by executable code and saved evidence. The early [EDA_SUMMARY.md](../reports/EDA_SUMMARY.md) predates those model experiments and **is not the final report**. Publication of this report does not itself mean the model has been deployed, externally validated, or proven to improve student outcomes.
