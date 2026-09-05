# Problem 07 — Student Placement Eligibility

**Business question:** Predict whether a student is likely to meet a placement eligibility criterion.

**Target:** `target` — `0` = Not eligible, `1` = Eligible (the student is predicted to meet placement eligibility)

---

## 1. Data Inspection

- Rows × columns: **1000 × 7** (6 numeric features + 1 target)
- Missing values found: **0** — none, no imputation needed
- Duplicate rows found: **0** — none
- Class balance: `0`=500 (50.0%), `1`=500 (50.0%) — balanced, no resampling needed

**Feature dictionary**

| Feature | Meaning |
|---|---|
| `cgpa` | Cumulative GPA |
| `attendance_pct` | Attendance percentage |
| `coding_score` | Coding/technical assessment score |
| `projects_completed` | Number of academic/personal projects completed |
| `internship_months` | Months of internship experience |
| `backlogs` | Number of outstanding academic backlogs |

**Descriptive statistics**

| Feature | Mean | Std | Min | Median | Max |
|---|---|---|---|---|---|
| `cgpa` | 7.26 | 1.26 | 4.00 | 7.27 | 10.00 |
| `attendance_pct` | 50.86 | 20.16 | 0.00 | 50.60 | 100.00 |
| `coding_score` | 3.00 | 1.71 | 0.00 | 3.00 | 10.00 |
| `projects_completed` | 2.98 | 1.75 | 0.00 | 3.00 | 9.00 |
| `internship_months` | 30.54 | 17.20 | 1.00 | 30.00 | 60.00 |
| `backlogs` | 2.99 | 1.72 | 0.00 | 3.00 | 9.00 |

![Correlation heatmap](07_student_placement_eligibility/correlation_heatmap.png)

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
- **Best configuration found:** `{"C": 0.1, "max_iter": 2000, "penalty": "l2", "solver": "lbfgs"}`

---

## 5. Cross-Validation (generalization check)

5-fold stratified cross-validation on the **training data only** (independent of the held-out test set):

| Metric | Mean | Std dev |
|---|---|---|
| Accuracy | 70.00% | ±3.71% |
| ROC-AUC | 0.776 | ±0.035 |

Low standard deviation across folds indicates the model's performance is **stable**, not an artifact of one lucky split.

---

## 6. Test-Set Evaluation

| Metric | Score |
|---|---|
| Accuracy | **70.00%** (baseline: 50.0%, lift: +20.0 pts) |
| Precision | 0.689 |
| Recall | 0.730 |
| F1-score | 0.709 |
| ROC-AUC | 0.787 |

![Confusion matrix](07_student_placement_eligibility/confusion_matrix.png)

**Confusion matrix breakdown** (test set, n=200):
- True Negatives: 67 (correctly predicted **Not eligible**)
- True Positives: 73 (correctly predicted **Eligible**)
- False Positives: 33 (predicted **Eligible**, actually not eligible — a "false alarm")
- False Negatives: 27 (predicted **Not eligible**, actually eligible — a "missed case")

![ROC curve](07_student_placement_eligibility/roc_curve.png)

An ROC-AUC of 0.787 means the model ranks a random positive case above a random negative case 78.7% of the time — well above the 50% (random-guess) line.

---

## 7. Coefficient Interpretation

Coefficients are on the **standardized** feature scale, so they're directly comparable — a larger magnitude means a bigger swing in log-odds per standard deviation of that feature.

![Feature impact](07_student_placement_eligibility/coefficient_importance.png)

| Feature | Standardized coefficient | Odds ratio |
|---|---|---|
| `backlogs` | +0.566 | 1.762 |
| `projects_completed` | +0.507 | 1.660 |
| `attendance_pct` | +0.506 | 1.659 |
| `coding_score` | -0.475 | 0.622 |
| `cgpa` | -0.421 | 0.657 |
| `internship_months` | -0.360 | 0.698 |

**Top 3 drivers:**

- **`backlogs`** — one standard deviation increase increases the odds of **eligible** by roughly **76%** (odds ratio = 1.76).
- **`projects_completed`** — one standard deviation increase increases the odds of **eligible** by roughly **66%** (odds ratio = 1.66).
- **`attendance_pct`** — one standard deviation increase increases the odds of **eligible** by roughly **66%** (odds ratio = 1.66).

*(A positive coefficient / odds ratio > 1 pushes toward `Eligible`; negative / odds ratio < 1 pushes toward `Not eligible`.)*

---

## 8. Limitations & Possible Improvements

- **Synthetic data:** this dataset is generated for practice, not sourced from real operations — feature relationships are likely cleaner and more linear than real-world data, so real-world accuracy would probably be lower.
- **Linearity assumption:** Logistic Regression models a linear decision boundary in log-odds space. If the true relationship is non-linear (e.g. threshold effects), a tree-based model (Random Forest, Gradient Boosting) could outperform it — worth benchmarking as a next step.
- **Sample size:** 1000 rows is workable for 6 features but modest; more data would tighten the confidence interval on the coefficients and CV scores.
- **Single train/test split for final numbers:** cross-validation (Section 5) mitigates this, but a nested CV or repeated-split evaluation would give an even more robust estimate.
- **Threshold choice:** the default 0.5 probability threshold was used for classification. Depending on whether false positives (33) or false negatives (27) are more costly in this business context, the threshold could be tuned (e.g. via the ROC curve) to favor precision or recall.
- **No external validation:** the model hasn't been tested on a different time period or population; performance could degrade under distribution shift (concept drift).

---

## Conclusion

With **70.0% test accuracy** and a **0.787 ROC-AUC** — a **+20.0 point** lift over the majority-class baseline — Logistic Regression provides a reasonably strong, fully interpretable model for **student placement eligibility**. Its main practical value here is not just prediction but **explainability**: the odds ratios above give a stakeholder a direct, defensible answer to "why did the model flag this case?" — something harder to get from a black-box model. For production use, the limitations above (especially threshold tuning and benchmarking against tree-based models) should be addressed first.
