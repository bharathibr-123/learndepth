# Problem 01 — Customer Churn Risk

**Business question:** Predict whether a subscription customer will churn in the next billing cycle.

**Target:** `target` — `0` = Did not churn, `1` = Churned (customer is predicted to cancel their subscription)

---

## 1. Data Inspection

- Rows × columns: **1000 × 7** (6 numeric features + 1 target)
- Missing values found: **0** — none, no imputation needed
- Duplicate rows found: **0** — none
- Class balance: `0`=500 (50.0%), `1`=500 (50.0%) — balanced, no resampling needed

**Feature dictionary**

| Feature | Meaning |
|---|---|
| `tenure_months` | How many months the customer has been subscribed |
| `monthly_charges` | Monthly bill amount (currency units) |
| `support_tickets` | Number of support tickets raised recently |
| `avg_session_minutes` | Average minutes per usage session |
| `late_payments` | Count of late/missed payments |
| `contract_months` | Length of the current contract, in months |

**Descriptive statistics**

| Feature | Mean | Std | Min | Median | Max |
|---|---|---|---|---|---|
| `tenure_months` | 30.12 | 17.22 | 1.00 | 30.00 | 60.00 |
| `monthly_charges` | 182.74 | 144.74 | 16.81 | 144.69 | 1530.68 |
| `support_tickets` | 3.13 | 1.77 | 0.00 | 3.00 | 11.00 |
| `avg_session_minutes` | 35.51 | 19.95 | 2.00 | 35.08 | 103.35 |
| `late_payments` | 3.08 | 1.76 | 0.00 | 3.00 | 10.00 |
| `contract_months` | 30.23 | 17.25 | 1.00 | 31.00 | 60.00 |

![Correlation heatmap](01_customer_churn_risk/correlation_heatmap.png)

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
- **Best configuration found:** `{"C": 0.5, "max_iter": 2000, "penalty": "l1", "solver": "liblinear"}`

---

## 5. Cross-Validation (generalization check)

5-fold stratified cross-validation on the **training data only** (independent of the held-out test set):

| Metric | Mean | Std dev |
|---|---|---|
| Accuracy | 69.75% | ±2.87% |
| ROC-AUC | 0.765 | ±0.030 |

Low standard deviation across folds indicates the model's performance is **stable**, not an artifact of one lucky split.

---

## 6. Test-Set Evaluation

| Metric | Score |
|---|---|
| Accuracy | **69.00%** (baseline: 50.0%, lift: +19.0 pts) |
| Precision | 0.698 |
| Recall | 0.670 |
| F1-score | 0.684 |
| ROC-AUC | 0.748 |

![Confusion matrix](01_customer_churn_risk/confusion_matrix.png)

**Confusion matrix breakdown** (test set, n=200):
- True Negatives: 71 (correctly predicted **Did not churn**)
- True Positives: 67 (correctly predicted **Churned**)
- False Positives: 29 (predicted **Churned**, actually did not churn — a "false alarm")
- False Negatives: 33 (predicted **Did not churn**, actually churned — a "missed case")

![ROC curve](01_customer_churn_risk/roc_curve.png)

An ROC-AUC of 0.748 means the model ranks a random positive case above a random negative case 74.8% of the time — well above the 50% (random-guess) line.

---

## 7. Coefficient Interpretation

Coefficients are on the **standardized** feature scale, so they're directly comparable — a larger magnitude means a bigger swing in log-odds per standard deviation of that feature.

![Feature impact](01_customer_churn_risk/coefficient_importance.png)

| Feature | Standardized coefficient | Odds ratio |
|---|---|---|
| `monthly_charges` | +0.627 | 1.872 |
| `contract_months` | +0.522 | 1.686 |
| `avg_session_minutes` | +0.493 | 1.637 |
| `support_tickets` | -0.434 | 0.648 |
| `late_payments` | -0.413 | 0.662 |
| `tenure_months` | -0.338 | 0.713 |

**Top 3 drivers:**

- **`monthly_charges`** — one standard deviation increase increases the odds of **churned** by roughly **87%** (odds ratio = 1.87).
- **`contract_months`** — one standard deviation increase increases the odds of **churned** by roughly **69%** (odds ratio = 1.69).
- **`avg_session_minutes`** — one standard deviation increase increases the odds of **churned** by roughly **64%** (odds ratio = 1.64).

*(A positive coefficient / odds ratio > 1 pushes toward `Churned`; negative / odds ratio < 1 pushes toward `Did not churn`.)*

---

## 8. Limitations & Possible Improvements

- **Synthetic data:** this dataset is generated for practice, not sourced from real operations — feature relationships are likely cleaner and more linear than real-world data, so real-world accuracy would probably be lower.
- **Linearity assumption:** Logistic Regression models a linear decision boundary in log-odds space. If the true relationship is non-linear (e.g. threshold effects), a tree-based model (Random Forest, Gradient Boosting) could outperform it — worth benchmarking as a next step.
- **Sample size:** 1000 rows is workable for 6 features but modest; more data would tighten the confidence interval on the coefficients and CV scores.
- **Single train/test split for final numbers:** cross-validation (Section 5) mitigates this, but a nested CV or repeated-split evaluation would give an even more robust estimate.
- **Threshold choice:** the default 0.5 probability threshold was used for classification. Depending on whether false positives (29) or false negatives (33) are more costly in this business context, the threshold could be tuned (e.g. via the ROC curve) to favor precision or recall.
- **No external validation:** the model hasn't been tested on a different time period or population; performance could degrade under distribution shift (concept drift).

---

## Conclusion

With **69.0% test accuracy** and a **0.748 ROC-AUC** — a **+19.0 point** lift over the majority-class baseline — Logistic Regression provides a reasonably strong, fully interpretable model for **customer churn risk**. Its main practical value here is not just prediction but **explainability**: the odds ratios above give a stakeholder a direct, defensible answer to "why did the model flag this case?" — something harder to get from a black-box model. For production use, the limitations above (especially threshold tuning and benchmarking against tree-based models) should be addressed first.
