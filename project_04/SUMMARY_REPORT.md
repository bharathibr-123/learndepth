# Summary Report — LearnDepth Logistic Regression (10 Problems)

Comparison of all 10 Logistic Regression models, each trained with the same
reproducible pipeline (see `src/engine.py`): stratified 80/20 split, scaling
fit on train only, GridSearchCV hyperparameter tuning, 5-fold stratified
cross-validation, and held-out test-set evaluation.

## Results Table

| # | Problem | Accuracy | Precision | Recall | F1 | ROC-AUC | Baseline Acc. |
|---|---|---|---|---|---|---|---|
| 01 | Customer Churn Risk | 69.00% | 0.698 | 0.670 | 0.684 | 0.748 | 50.0% |
| 02 | Loan Approval Screening | 70.50% | 0.688 | 0.750 | 0.718 | 0.782 | 50.0% |
| 03 | Employee Attrition | 66.00% | 0.660 | 0.660 | 0.660 | 0.737 | 50.0% |
| 04 | E-commerce Purchase Intent | 72.50% | 0.737 | 0.700 | 0.718 | 0.800 | 50.0% |
| 05 | Insurance Claim Flagging | 62.50% | 0.632 | 0.600 | 0.615 | 0.666 | 50.0% |
| 06 | Equipment Failure Warning | 70.50% | 0.725 | 0.660 | 0.691 | 0.776 | 50.0% |
| 07 | Student Placement Eligibility | 70.00% | 0.689 | 0.730 | 0.709 | 0.787 | 50.0% |
| 08 | Email Priority Classification | 66.00% | 0.651 | 0.690 | 0.670 | 0.734 | 50.0% |
| 09 | Delivery Delay Risk | 62.50% | 0.613 | 0.680 | 0.645 | 0.715 | 50.0% |
| 10 | Retail Return Risk | 67.00% | 0.655 | 0.720 | 0.686 | 0.731 | 50.0% |

**Average across all 10 problems:** Accuracy = 67.65%, ROC-AUC = 0.748

- **Strongest model:** Problem 04 — E-commerce Purchase Intent (ROC-AUC 0.800)
- **Weakest model:** Problem 05 — Insurance Claim Flagging (ROC-AUC 0.666)

## Individual Reports

- [Problem 01 — Customer Churn Risk](01_customer_churn_risk.md)
- [Problem 02 — Loan Approval Screening](02_loan_approval_screening.md)
- [Problem 03 — Employee Attrition](03_employee_attrition.md)
- [Problem 04 — E-commerce Purchase Intent](04_ecommerce_purchase_intent.md)
- [Problem 05 — Insurance Claim Flagging](05_insurance_claim_flagging.md)
- [Problem 06 — Equipment Failure Warning](06_equipment_failure_warning.md)
- [Problem 07 — Student Placement Eligibility](07_student_placement_eligibility.md)
- [Problem 08 — Email Priority Classification](08_email_priority_classification.md)
- [Problem 09 — Delivery Delay Risk](09_delivery_delay_risk.md)
- [Problem 10 — Retail Return Risk](10_retail_return_risk.md)

## Overall Observations

- All 10 datasets were clean (no missing values, no duplicates) and perfectly class-balanced (500/500),
  which is why baseline (majority-class) accuracy sits at 50% across every problem — any lift above
  that is attributable entirely to the features and the model, not to class-imbalance artifacts.
- Logistic Regression performed consistently well across all 10 business domains (churn, credit,
  HR, e-commerce, insurance, manufacturing, education, communications, logistics, retail), which
  suggests the underlying signal-to-noise ratio in these synthetic datasets is high and mostly linear
  — real-world versions of these problems would likely need richer feature engineering and possibly
  non-linear models to match this performance.
- The same reproducible pipeline (`engine.py`) was applied to every problem, so differences in score
  reflect genuine differences in how separable each dataset's classes are — not inconsistent methodology.
