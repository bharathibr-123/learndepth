# Problem 06 — Equipment Failure Warning

**Business question:** Predict whether an industrial machine will fail within the next operating window.

**Target:** `target` — `0` = No failure expected, `1` = Failure expected (the machine is predicted to fail soon)

---

## 1. Data Inspection

- Rows × columns: **1000 × 7** (6 numeric features + 1 target)
- Missing values found: **0** — none, no imputation needed
- Duplicate rows found: **0** — none
- Class balance: `0`=500 (50.0%), `1`=500 (50.0%) — balanced, no resampling needed

**Feature dictionary**

| Feature | Meaning |
|---|---|
| `temperature` | Operating temperature reading |
| `vibration` | Vibration sensor reading |
| `pressure` | Pressure sensor reading |
| `runtime_hours` | Cumulative runtime hours since last service |
| `maintenance_gap_days` | Days since last maintenance |
| `power_draw` | Power draw reading |

**Descriptive statistics**

| Feature | Mean | Std | Min | Median | Max |
|---|---|---|---|---|---|
| `temperature` | 70.18 | 12.14 | 30.32 | 70.47 | 109.32 |
| `vibration` | 3.98 | 1.68 | 0.10 | 3.94 | 8.37 |
| `pressure` | 100.05 | 17.91 | 40.19 | 100.48 | 158.57 |
| `runtime_hours` | 10.83 | 7.15 | 1.00 | 10.52 | 34.86 |
| `maintenance_gap_days` | 30.27 | 18.10 | 0.00 | 30.00 | 60.00 |
| `power_draw` | 186.45 | 150.15 | 20.73 | 144.84 | 1940.66 |

![Correlation heatmap](06_equipment_failure_warning/correlation_heatmap.png)

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
| Accuracy | 66.62% | ±1.96% |
| ROC-AUC | 0.755 | ±0.020 |

Low standard deviation across folds indicates the model's performance is **stable**, not an artifact of one lucky split.

---

## 6. Test-Set Evaluation

| Metric | Score |
|---|---|
| Accuracy | **70.50%** (baseline: 50.0%, lift: +20.5 pts) |
| Precision | 0.725 |
| Recall | 0.660 |
| F1-score | 0.691 |
| ROC-AUC | 0.776 |

![Confusion matrix](06_equipment_failure_warning/confusion_matrix.png)

**Confusion matrix breakdown** (test set, n=200):
- True Negatives: 75 (correctly predicted **No failure expected**)
- True Positives: 66 (correctly predicted **Failure expected**)
- False Positives: 25 (predicted **Failure expected**, actually no failure expected — a "false alarm")
- False Negatives: 34 (predicted **No failure expected**, actually failure expected — a "missed case")

![ROC curve](06_equipment_failure_warning/roc_curve.png)

An ROC-AUC of 0.776 means the model ranks a random positive case above a random negative case 77.6% of the time — well above the 50% (random-guess) line.

---

## 7. Coefficient Interpretation

Coefficients are on the **standardized** feature scale, so they're directly comparable — a larger magnitude means a bigger swing in log-odds per standard deviation of that feature.

![Feature impact](06_equipment_failure_warning/coefficient_importance.png)

| Feature | Standardized coefficient | Odds ratio |
|---|---|---|
| `temperature` | +0.314 | 1.368 |
| `pressure` | +0.309 | 1.362 |
| `maintenance_gap_days` | +0.252 | 1.287 |
| `power_draw` | -0.239 | 0.787 |
| `runtime_hours` | -0.216 | 0.805 |
| `vibration` | -0.203 | 0.817 |

**Top 3 drivers:**

- **`temperature`** — one standard deviation increase increases the odds of **failure expected** by roughly **37%** (odds ratio = 1.37).
- **`pressure`** — one standard deviation increase increases the odds of **failure expected** by roughly **36%** (odds ratio = 1.36).
- **`maintenance_gap_days`** — one standard deviation increase increases the odds of **failure expected** by roughly **29%** (odds ratio = 1.29).

*(A positive coefficient / odds ratio > 1 pushes toward `Failure expected`; negative / odds ratio < 1 pushes toward `No failure expected`.)*

---

## 8. Limitations & Possible Improvements

- **Synthetic data:** this dataset is generated for practice, not sourced from real operations — feature relationships are likely cleaner and more linear than real-world data, so real-world accuracy would probably be lower.
- **Linearity assumption:** Logistic Regression models a linear decision boundary in log-odds space. If the true relationship is non-linear (e.g. threshold effects), a tree-based model (Random Forest, Gradient Boosting) could outperform it — worth benchmarking as a next step.
- **Sample size:** 1000 rows is workable for 6 features but modest; more data would tighten the confidence interval on the coefficients and CV scores.
- **Single train/test split for final numbers:** cross-validation (Section 5) mitigates this, but a nested CV or repeated-split evaluation would give an even more robust estimate.
- **Threshold choice:** the default 0.5 probability threshold was used for classification. Depending on whether false positives (25) or false negatives (34) are more costly in this business context, the threshold could be tuned (e.g. via the ROC curve) to favor precision or recall.
- **No external validation:** the model hasn't been tested on a different time period or population; performance could degrade under distribution shift (concept drift).

---

## Conclusion

With **70.5% test accuracy** and a **0.776 ROC-AUC** — a **+20.5 point** lift over the majority-class baseline — Logistic Regression provides a reasonably strong, fully interpretable model for **equipment failure warning**. Its main practical value here is not just prediction but **explainability**: the odds ratios above give a stakeholder a direct, defensible answer to "why did the model flag this case?" — something harder to get from a black-box model. For production use, the limitations above (especially threshold tuning and benchmarking against tree-based models) should be addressed first.
