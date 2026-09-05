# Problem 03 — Employee Attrition

**Business question:** Predict whether an employee is likely to leave within six months.

**Target:** `target` — `0` = Stays, `1` = Leaves (employee is predicted to leave within 6 months)

---

## 1. Data Inspection

- Rows × columns: **1000 × 7** (6 numeric features + 1 target)
- Missing values found: **0** — none, no imputation needed
- Duplicate rows found: **0** — none
- Class balance: `0`=500 (50.0%), `1`=500 (50.0%) — balanced, no resampling needed

**Feature dictionary**

| Feature | Meaning |
|---|---|
| `age` | Employee age in years |
| `monthly_income` | Monthly salary |
| `years_at_company` | Tenure at the company, in years |
| `job_satisfaction` | Self-reported job satisfaction score |
| `overtime_hours` | Average overtime hours per month |
| `promotion_years` | Years since the last promotion |

**Descriptive statistics**

| Feature | Mean | Std | Min | Median | Max |
|---|---|---|---|---|---|
| `age` | 41.23 | 13.81 | 18.00 | 41.00 | 65.00 |
| `monthly_income` | 194.32 | 177.70 | 17.98 | 146.13 | 2161.23 |
| `years_at_company` | 5.25 | 3.60 | 0.00 | 5.07 | 20.69 |
| `job_satisfaction` | 3.03 | 1.75 | 0.00 | 3.00 | 10.00 |
| `overtime_hours` | 10.53 | 7.07 | 1.00 | 9.50 | 31.96 |
| `promotion_years` | 4.95 | 3.57 | 0.00 | 4.65 | 16.50 |

![Correlation heatmap](03_employee_attrition/correlation_heatmap.png)

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
- **Best configuration found:** `{"C": 0.01, "max_iter": 2000, "penalty": "l2", "solver": "lbfgs"}`

---

## 5. Cross-Validation (generalization check)

5-fold stratified cross-validation on the **training data only** (independent of the held-out test set):

| Metric | Mean | Std dev |
|---|---|---|
| Accuracy | 70.38% | ±4.93% |
| ROC-AUC | 0.781 | ±0.045 |

Low standard deviation across folds indicates the model's performance is **stable**, not an artifact of one lucky split.

---

## 6. Test-Set Evaluation

| Metric | Score |
|---|---|
| Accuracy | **66.00%** (baseline: 50.0%, lift: +16.0 pts) |
| Precision | 0.660 |
| Recall | 0.660 |
| F1-score | 0.660 |
| ROC-AUC | 0.737 |

![Confusion matrix](03_employee_attrition/confusion_matrix.png)

**Confusion matrix breakdown** (test set, n=200):
- True Negatives: 66 (correctly predicted **Stays**)
- True Positives: 66 (correctly predicted **Leaves**)
- False Positives: 34 (predicted **Leaves**, actually stays — a "false alarm")
- False Negatives: 34 (predicted **Stays**, actually leaves — a "missed case")

![ROC curve](03_employee_attrition/roc_curve.png)

An ROC-AUC of 0.737 means the model ranks a random positive case above a random negative case 73.7% of the time — well above the 50% (random-guess) line.

---

## 7. Coefficient Interpretation

Coefficients are on the **standardized** feature scale, so they're directly comparable — a larger magnitude means a bigger swing in log-odds per standard deviation of that feature.

![Feature impact](03_employee_attrition/coefficient_importance.png)

| Feature | Standardized coefficient | Odds ratio |
|---|---|---|
| `job_satisfaction` | +0.346 | 1.414 |
| `overtime_hours` | -0.328 | 0.720 |
| `promotion_years` | +0.311 | 1.365 |
| `monthly_income` | +0.277 | 1.319 |
| `years_at_company` | -0.235 | 0.791 |
| `age` | -0.235 | 0.791 |

**Top 3 drivers:**

- **`job_satisfaction`** — one standard deviation increase increases the odds of **leaves** by roughly **41%** (odds ratio = 1.41).
- **`overtime_hours`** — one standard deviation increase decreases the odds of **leaves** by roughly **28%** (odds ratio = 0.72).
- **`promotion_years`** — one standard deviation increase increases the odds of **leaves** by roughly **37%** (odds ratio = 1.37).

*(A positive coefficient / odds ratio > 1 pushes toward `Leaves`; negative / odds ratio < 1 pushes toward `Stays`.)*

---

## 8. Limitations & Possible Improvements

- **Synthetic data:** this dataset is generated for practice, not sourced from real operations — feature relationships are likely cleaner and more linear than real-world data, so real-world accuracy would probably be lower.
- **Linearity assumption:** Logistic Regression models a linear decision boundary in log-odds space. If the true relationship is non-linear (e.g. threshold effects), a tree-based model (Random Forest, Gradient Boosting) could outperform it — worth benchmarking as a next step.
- **Sample size:** 1000 rows is workable for 6 features but modest; more data would tighten the confidence interval on the coefficients and CV scores.
- **Single train/test split for final numbers:** cross-validation (Section 5) mitigates this, but a nested CV or repeated-split evaluation would give an even more robust estimate.
- **Threshold choice:** the default 0.5 probability threshold was used for classification. Depending on whether false positives (34) or false negatives (34) are more costly in this business context, the threshold could be tuned (e.g. via the ROC curve) to favor precision or recall.
- **No external validation:** the model hasn't been tested on a different time period or population; performance could degrade under distribution shift (concept drift).

---

## Conclusion

With **66.0% test accuracy** and a **0.737 ROC-AUC** — a **+16.0 point** lift over the majority-class baseline — Logistic Regression provides a reasonably strong, fully interpretable model for **employee attrition**. Its main practical value here is not just prediction but **explainability**: the odds ratios above give a stakeholder a direct, defensible answer to "why did the model flag this case?" — something harder to get from a black-box model. For production use, the limitations above (especially threshold tuning and benchmarking against tree-based models) should be addressed first.
