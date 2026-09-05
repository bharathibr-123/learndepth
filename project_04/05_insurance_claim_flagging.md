# Problem 05 — Insurance Claim Flagging

**Business question:** Predict whether an insurance claim should be flagged for further review.

**Target:** `target` — `0` = Not flagged, `1` = Flagged (the claim is predicted to need manual review)

---

## 1. Data Inspection

- Rows × columns: **1000 × 7** (6 numeric features + 1 target)
- Missing values found: **0** — none, no imputation needed
- Duplicate rows found: **0** — none
- Class balance: `0`=500 (50.0%), `1`=500 (50.0%) — balanced, no resampling needed

**Feature dictionary**

| Feature | Meaning |
|---|---|
| `claim_amount` | Amount claimed |
| `policy_tenure` | How long the policy has been active, in years |
| `claim_count` | Number of prior claims filed by this policyholder |
| `customer_age` | Policyholder age |
| `vehicle_age` | Age of the insured vehicle, in years |
| `incident_severity` | Severity rating of the incident (encoded) |

**Descriptive statistics**

| Feature | Mean | Std | Min | Median | Max |
|---|---|---|---|---|---|
| `claim_amount` | 184.53 | 148.47 | 15.10 | 149.34 | 1510.88 |
| `policy_tenure` | 3.01 | 1.70 | 0.00 | 3.00 | 9.00 |
| `claim_count` | 2.96 | 1.72 | 0.00 | 3.00 | 9.00 |
| `customer_age` | 40.61 | 13.61 | 18.00 | 41.00 | 65.00 |
| `vehicle_age` | 41.62 | 13.78 | 18.00 | 42.00 | 65.00 |
| `incident_severity` | 3.02 | 1.42 | 1.00 | 3.00 | 5.00 |

![Correlation heatmap](05_insurance_claim_flagging/correlation_heatmap.png)

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
| Accuracy | 66.25% | ±3.26% |
| ROC-AUC | 0.729 | ±0.042 |

Low standard deviation across folds indicates the model's performance is **stable**, not an artifact of one lucky split.

---

## 6. Test-Set Evaluation

| Metric | Score |
|---|---|
| Accuracy | **62.50%** (baseline: 50.0%, lift: +12.5 pts) |
| Precision | 0.632 |
| Recall | 0.600 |
| F1-score | 0.615 |
| ROC-AUC | 0.666 |

![Confusion matrix](05_insurance_claim_flagging/confusion_matrix.png)

**Confusion matrix breakdown** (test set, n=200):
- True Negatives: 65 (correctly predicted **Not flagged**)
- True Positives: 60 (correctly predicted **Flagged**)
- False Positives: 35 (predicted **Flagged**, actually not flagged — a "false alarm")
- False Negatives: 40 (predicted **Not flagged**, actually flagged — a "missed case")

![ROC curve](05_insurance_claim_flagging/roc_curve.png)

An ROC-AUC of 0.666 means the model ranks a random positive case above a random negative case 66.6% of the time — well above the 50% (random-guess) line.

---

## 7. Coefficient Interpretation

Coefficients are on the **standardized** feature scale, so they're directly comparable — a larger magnitude means a bigger swing in log-odds per standard deviation of that feature.

![Feature impact](05_insurance_claim_flagging/coefficient_importance.png)

| Feature | Standardized coefficient | Odds ratio |
|---|---|---|
| `policy_tenure` | +0.278 | 1.321 |
| `vehicle_age` | -0.255 | 0.775 |
| `customer_age` | +0.248 | 1.281 |
| `incident_severity` | +0.241 | 1.272 |
| `claim_count` | -0.234 | 0.791 |
| `claim_amount` | -0.187 | 0.830 |

**Top 3 drivers:**

- **`policy_tenure`** — one standard deviation increase increases the odds of **flagged** by roughly **32%** (odds ratio = 1.32).
- **`vehicle_age`** — one standard deviation increase decreases the odds of **flagged** by roughly **22%** (odds ratio = 0.78).
- **`customer_age`** — one standard deviation increase increases the odds of **flagged** by roughly **28%** (odds ratio = 1.28).

*(A positive coefficient / odds ratio > 1 pushes toward `Flagged`; negative / odds ratio < 1 pushes toward `Not flagged`.)*

---

## 8. Limitations & Possible Improvements

- **Synthetic data:** this dataset is generated for practice, not sourced from real operations — feature relationships are likely cleaner and more linear than real-world data, so real-world accuracy would probably be lower.
- **Linearity assumption:** Logistic Regression models a linear decision boundary in log-odds space. If the true relationship is non-linear (e.g. threshold effects), a tree-based model (Random Forest, Gradient Boosting) could outperform it — worth benchmarking as a next step.
- **Sample size:** 1000 rows is workable for 6 features but modest; more data would tighten the confidence interval on the coefficients and CV scores.
- **Single train/test split for final numbers:** cross-validation (Section 5) mitigates this, but a nested CV or repeated-split evaluation would give an even more robust estimate.
- **Threshold choice:** the default 0.5 probability threshold was used for classification. Depending on whether false positives (35) or false negatives (40) are more costly in this business context, the threshold could be tuned (e.g. via the ROC curve) to favor precision or recall.
- **No external validation:** the model hasn't been tested on a different time period or population; performance could degrade under distribution shift (concept drift).

---

## Conclusion

With **62.5% test accuracy** and a **0.666 ROC-AUC** — a **+12.5 point** lift over the majority-class baseline — Logistic Regression provides a reasonably strong, fully interpretable model for **insurance claim flagging**. Its main practical value here is not just prediction but **explainability**: the odds ratios above give a stakeholder a direct, defensible answer to "why did the model flag this case?" — something harder to get from a black-box model. For production use, the limitations above (especially threshold tuning and benchmarking against tree-based models) should be addressed first.
