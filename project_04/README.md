# LearnDepth — Logistic Regression: 10 Problems

A complete, reproducible Logistic Regression project covering all 10 problems
from the LearnDepth assignment brief, built as plain Python scripts (no
Jupyter required — designed to run straight from VS Code's terminal).

Each problem takes a different business dataset (1,000 rows, 6 numeric
features, balanced binary target) through the same rigorous pipeline:

**Inspect → Preprocess & Split → Baseline → Train (GridSearchCV-tuned) →
Cross-Validate → Evaluate (Accuracy/Precision/Recall/F1/ROC-AUC) →
Interpret coefficients → Write report**

## Project Structure

```
learndepth_logistic_regression/
├── datasets/                          10 input CSVs (provided data)
├── src/
│   ├── engine.py                      shared pipeline — all modelling logic lives here
│   ├── problem_01_customer_churn_risk.py
│   ├── problem_02_loan_approval_screening.py
│   ├── problem_03_employee_attrition.py
│   ├── problem_04_ecommerce_purchase_intent.py
│   ├── problem_05_insurance_claim_flagging.py
│   ├── problem_06_equipment_failure_warning.py
│   ├── problem_07_student_placement_eligibility.py
│   ├── problem_08_email_priority_classification.py
│   ├── problem_09_delivery_delay_risk.py
│   ├── problem_10_retail_return_risk.py
│   └── run_all.py                     runs all 10 + builds the summary report
├── outputs/                           generated plots + metrics.json per problem
│   └── 01_customer_churn_risk/
│       ├── correlation_heatmap.png
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       ├── coefficient_importance.png
│       └── metrics.json
│   ... (one folder per problem)
├── reports/                           the written analysis — start here
│   ├── 01_customer_churn_risk.md
│   ├── ... (one per problem)
│   ├── SUMMARY_REPORT.md              comparison table across all 10 models
│   └── summary_metrics.json
├── requirements.txt
└── README.md
```

All 10 problems have **already been run** — the `outputs/` and `reports/`
folders are pre-populated with real results. You don't have to run anything
to see the analysis; open any file in `reports/` to read it.

## How to run it yourself in VS Code

1. Open the `learndepth_logistic_regression` folder in VS Code.
2. Open a terminal (`` Ctrl+` ``) and create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS/Linux
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run a single problem:
   ```bash
   cd src
   python problem_01_customer_churn_risk.py
   ```
   This regenerates that problem's plots (`outputs/01_.../`) and report (`reports/01_....md`).

5. Or run all 10 problems + the summary report in one go:
   ```bash
   cd src
   python run_all.py
   ```

No Jupyter needed — just plain `.py` files you can run and debug directly
in VS Code (set breakpoints, use "Run Python File" or the built-in debugger).

## What each problem covers

| # | Problem | Business Question |
|---|---|---|
| 01 | Customer Churn Risk | Will this subscriber churn next billing cycle? |
| 02 | Loan Approval Screening | Should this loan application be approved? |
| 03 | Employee Attrition | Will this employee leave within 6 months? |
| 04 | E-commerce Purchase Intent | Will this browsing session end in a purchase? |
| 05 | Insurance Claim Flagging | Should this claim be flagged for manual review? |
| 06 | Equipment Failure Warning | Will this machine fail soon? |
| 07 | Student Placement Eligibility | Is this student eligible for placement? |
| 08 | Email Priority Classification | Should this email be marked high priority? |
| 09 | Delivery Delay Risk | Will this order arrive late? |
| 10 | Retail Return Risk | Will this order be returned? |

See `reports/SUMMARY_REPORT.md` for the full results comparison across all 10.

## Methodology notes

- **No data leakage:** train/test split happens before scaling; `StandardScaler`
  is fit only on the training set.
- **Hyperparameter tuning:** `GridSearchCV` searches over L1/L2 penalty and 8
  values of `C`, scored on ROC-AUC via 5-fold stratified cross-validation.
- **Generalization check:** cross-validation is reported separately from the
  final held-out test-set score, so you can see both the model's stability
  across folds and its performance on truly unseen data.
- **Interpretability:** coefficients are computed on standardized features and
  converted to odds ratios, so each report explains *which features drive the
  prediction and by how much* — not just the accuracy number.
- **Reproducibility:** `RANDOM_STATE = 42` is fixed throughout, so re-running
  `run_all.py` reproduces the exact same numbers shown in these reports.
