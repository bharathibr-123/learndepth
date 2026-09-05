# Problem 02 — Loan Approval Screening

**Business question:** Predict whether a loan application should be approved.

**Target:** `target` — `0` = Not approved, `1` = Approved (the application is predicted to be approved)

---

## 1. Data Inspection

- Rows × columns: **1000 × 7** (6 numeric features + 1 target)
- Missing values found: **0** — none, no imputation needed
- Duplicate rows found: **0** — none
- Class balance: `0`=500 (50.0%), `1`=500 (50.0%) — balanced, no resampling needed

**Feature dictionary**

| Feature | Meaning |
|---|---|
| `income_monthly` | Applicant's monthly income |
| `credit_score` | Credit bureau score |
| `debt_to_income` | Existing debt as a percentage of income |
| `employment_years` | Years in current employment |
| `loan_amount` | Amount of loan requested |
| `prior_defaults` | Number of prior loan defaults on record |

**Descriptive statistics**

| Feature | Mean | Std | Min | Median | Max |
|---|---|---|---|---|---|
| `income_monthly` | 187.53 | 142.35 | 19.22 | 147.92 | 1011.22 |
| `credit_score` | 679.69 | 67.44 | 485.00 | 680.00 | 850.00 |
| `debt_to_income` | 184.81 | 136.39 | 11.45 | 145.62 | 1095.03 |
| `employment_years` | 5.23 | 3.71 | 0.00 | 5.00 | 17.41 |
| `loan_amount` | 190.73 | 149.62 | 15.06 | 148.66 | 1283.08 |
| `prior_defaults` | 2.94 | 1.72 | 0.00 | 3.00 | 10.00 |

![Correlation heatmap](02_loan_approval_screening/correlation_heatmap.png)

---

## 2. Preprocessing & Split

- Train/test split: **80% / 20%**, stratified on `target` to preserve class balance (train n=800, test n=200).
- The split was performed **before** any scaling, and the `StandardScaler` was fit **only on the training set** then applied to the test set — this prevents test-set information leaking into preprocessing.
- Logistic Regression is scale-sensitive (regularization penalizes coefficient magnitude), so standardization also makes the coefficients in Section 5 directly comparable across features.

---

## 3. Baseline

A majority-class `DummyClassifier` (always predicts the more frequent class) scores **50.0% accuracy** on the test set. This is the floor any real model must beat — it represents the accuracy achievable with zero information.

---

## 4. Model Training

- Algorithm: **Logistic Regression**, tuned with `GridSearchCV` (5-fold stratified CV, scored on ROC-AUC).
- Search space: `penalty ∈ {l1, l2}`, `C ∈ [0.01 … 50]`, `solver` matched to penalty (`liblinear` for l1, `lbfgs` for l2).
- **Best configuration found:** `{"C": 5, "max_iter": 2000, "penalty": "l2", "solver": "lbfgs"}`

---

## 5. Cross-Validation (generalization check)

5-fold stratified cross-validation on the **training data only** (independent of the held-out test set):

| Metric | Mean | Std dev |
|---|---|---|
| Accuracy | 71.12% | ±1.95% |
| ROC-AUC | 0.784 | ±0.017 |

Low standard deviation across folds indicates the model's performance is **stable**, not an artifact of one lucky split.

---

## 6. Test-Set Evaluation

| Metric | Score |
|---|---|
| Accuracy | **70.50%** (baseline: 50.0%, lift: +20.5 pts) |
| Precision | 0.688 |
| Recall | 0.750 |
| F1-score | 0.718 |
| ROC-AUC | 0.782 |

![Confusion matrix](02_loan_approval_screening/confusion_matrix.png)

**Confusion matrix breakdown** (test set, n=200):
- True Negatives: 66 (correctly predicted **Not approved**)
- True Positives: 75 (correctly predicted **Approved**)
- False Positives: 34 (predicted **Approved**, actually not approved — a "false alarm")
- False Negatives: 25 (predicted **Not approved**, actually approved — a "missed case")

![ROC curve](02_loan_approval_screening/roc_curve.png)

An ROC-AUC of 0.782 means the model ranks a random positive case above a random negative case 78.2% of the time — well above the 50% (random-guess) line.

---

## 7. Coefficient Interpretation

Coefficients are on the **standardized** feature scale, so they're directly comparable — a larger magnitude means a bigger swing in log-odds per standard deviation of that feature.

![Feature impact](02_loan_approval_screening/coefficient_importance.png)

| Feature | Standardized coefficient | Odds ratio |
|---|---|---|
| `employment_years` | -0.615 | 0.541 |
| `loan_amount` | +0.601 | 1.823 |
| `income_monthly` | +0.591 | 1.807 |
| `credit_score` | -0.529 | 0.589 |
| `prior_defaults` | -0.521 | 0.594 |
| `debt_to_income` | +0.438 | 1.550 |

**Top 3 drivers:**

- **`employment_years`** — one standard deviation increase decreases the odds of **approved** by roughly **46%** (odds ratio = 0.54).
- **`loan_amount`** — one standard deviation increase increases the odds of **approved** by roughly **82%** (odds ratio = 1.82).
- **`income_monthly`** — one standard deviation increase increases the odds of **approved** by roughly **81%** (odds ratio = 1.81).

*(A positive coefficient / odds ratio > 1 pushes toward `Approved`; negative / odds ratio < 1 pushes toward `Not approved`.)*

---

## 8. Limitations & Possible Improvements

- **Synthetic data:** this dataset is generated for practice, not sourced from real operations — feature relationships are likely cleaner and more linear than real-world data, so real-world accuracy would probably be lower.
- **Linearity assumption:** Logistic Regression models a linear decision boundary in log-odds space. If the true relationship is non-linear (e.g. threshold effects), a tree-based model (Random Forest, Gradient Boosting) could outperform it — worth benchmarking as a next step.
- **Sample size:** 1000 rows is workable for 6 features but modest; more data would tighten the confidence interval on the coefficients and CV scores.
- **Single train/test split for final numbers:** cross-validation (Section 5) mitigates this, but a nested CV or repeated-split evaluation would give an even more robust estimate.
- **Threshold choice:** the default 0.5 probability threshold was used for classification. Depending on whether false positives (34) or false negatives (25) are more costly in this business context, the threshold could be tuned (e.g. via the ROC curve) to favor precision or recall.
- **No external validation:** the model hasn't been tested on a different time period or population; performance could degrade under distribution shift (concept drift).

---

## Conclusion

With **70.5% test accuracy** and a **0.782 ROC-AUC** — a **+20.5 point** lift over the majority-class baseline — Logistic Regression provides a reasonably strong, fully interpretable model for **loan approval screening**. Its main practical value here is not just prediction but **explainability**: the odds ratios above give a stakeholder a direct, defensible answer to "why did the model flag this case?" — something harder to get from a black-box model. For production use, the limitations above (especially threshold tuning and benchmarking against tree-based models) should be addressed first.
