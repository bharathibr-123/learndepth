# Problem 08 — Email Priority Classification

**Business question:** Predict whether an incoming email should be marked high priority.

**Target:** `target` — `0` = Normal priority, `1` = High priority (the email is predicted to be high priority)

---

## 1. Data Inspection

- Rows × columns: **1000 × 7** (6 numeric features + 1 target)
- Missing values found: **0** — none, no imputation needed
- Duplicate rows found: **0** — none
- Class balance: `0`=500 (50.0%), `1`=500 (50.0%) — balanced, no resampling needed

**Feature dictionary**

| Feature | Meaning |
|---|---|
| `sender_frequency` | How often this sender has emailed before |
| `keyword_score` | Score based on urgency-related keywords found |
| `thread_length` | Number of messages in the email thread |
| `response_deadline_hours` | Hours until an implied/explicit response deadline |
| `attachment_count` | Number of attachments |
| `previous_priority_rate` | Historical rate this sender's emails were marked high priority |

**Descriptive statistics**

| Feature | Mean | Std | Min | Median | Max |
|---|---|---|---|---|---|
| `sender_frequency` | 3.01 | 1.80 | 0.00 | 3.00 | 10.00 |
| `keyword_score` | 49.83 | 19.24 | 0.00 | 49.26 | 100.00 |
| `thread_length` | 2.96 | 1.71 | 0.00 | 3.00 | 10.00 |
| `response_deadline_hours` | 10.61 | 6.97 | 1.00 | 9.95 | 37.37 |
| `attachment_count` | 2.95 | 1.77 | 0.00 | 3.00 | 10.00 |
| `previous_priority_rate` | 49.64 | 19.14 | 0.00 | 49.50 | 100.00 |

![Correlation heatmap](08_email_priority_classification/correlation_heatmap.png)

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
| Accuracy | 71.25% | ±3.69% |
| ROC-AUC | 0.789 | ±0.044 |

Low standard deviation across folds indicates the model's performance is **stable**, not an artifact of one lucky split.

---

## 6. Test-Set Evaluation

| Metric | Score |
|---|---|
| Accuracy | **66.00%** (baseline: 50.0%, lift: +16.0 pts) |
| Precision | 0.651 |
| Recall | 0.690 |
| F1-score | 0.670 |
| ROC-AUC | 0.734 |

![Confusion matrix](08_email_priority_classification/confusion_matrix.png)

**Confusion matrix breakdown** (test set, n=200):
- True Negatives: 63 (correctly predicted **Normal priority**)
- True Positives: 69 (correctly predicted **High priority**)
- False Positives: 37 (predicted **High priority**, actually normal priority — a "false alarm")
- False Negatives: 31 (predicted **Normal priority**, actually high priority — a "missed case")

![ROC curve](08_email_priority_classification/roc_curve.png)

An ROC-AUC of 0.734 means the model ranks a random positive case above a random negative case 73.4% of the time — well above the 50% (random-guess) line.

---

## 7. Coefficient Interpretation

Coefficients are on the **standardized** feature scale, so they're directly comparable — a larger magnitude means a bigger swing in log-odds per standard deviation of that feature.

![Feature impact](08_email_priority_classification/coefficient_importance.png)

| Feature | Standardized coefficient | Odds ratio |
|---|---|---|
| `attachment_count` | +0.377 | 1.458 |
| `sender_frequency` | +0.360 | 1.433 |
| `thread_length` | +0.334 | 1.397 |
| `response_deadline_hours` | -0.269 | 0.764 |
| `keyword_score` | -0.224 | 0.799 |
| `previous_priority_rate` | -0.202 | 0.817 |

**Top 3 drivers:**

- **`attachment_count`** — one standard deviation increase increases the odds of **high priority** by roughly **46%** (odds ratio = 1.46).
- **`sender_frequency`** — one standard deviation increase increases the odds of **high priority** by roughly **43%** (odds ratio = 1.43).
- **`thread_length`** — one standard deviation increase increases the odds of **high priority** by roughly **40%** (odds ratio = 1.40).

*(A positive coefficient / odds ratio > 1 pushes toward `High priority`; negative / odds ratio < 1 pushes toward `Normal priority`.)*

---

## 8. Limitations & Possible Improvements

- **Synthetic data:** this dataset is generated for practice, not sourced from real operations — feature relationships are likely cleaner and more linear than real-world data, so real-world accuracy would probably be lower.
- **Linearity assumption:** Logistic Regression models a linear decision boundary in log-odds space. If the true relationship is non-linear (e.g. threshold effects), a tree-based model (Random Forest, Gradient Boosting) could outperform it — worth benchmarking as a next step.
- **Sample size:** 1000 rows is workable for 6 features but modest; more data would tighten the confidence interval on the coefficients and CV scores.
- **Single train/test split for final numbers:** cross-validation (Section 5) mitigates this, but a nested CV or repeated-split evaluation would give an even more robust estimate.
- **Threshold choice:** the default 0.5 probability threshold was used for classification. Depending on whether false positives (37) or false negatives (31) are more costly in this business context, the threshold could be tuned (e.g. via the ROC curve) to favor precision or recall.
- **No external validation:** the model hasn't been tested on a different time period or population; performance could degrade under distribution shift (concept drift).

---

## Conclusion

With **66.0% test accuracy** and a **0.734 ROC-AUC** — a **+16.0 point** lift over the majority-class baseline — Logistic Regression provides a reasonably strong, fully interpretable model for **email priority classification**. Its main practical value here is not just prediction but **explainability**: the odds ratios above give a stakeholder a direct, defensible answer to "why did the model flag this case?" — something harder to get from a black-box model. For production use, the limitations above (especially threshold tuning and benchmarking against tree-based models) should be addressed first.
