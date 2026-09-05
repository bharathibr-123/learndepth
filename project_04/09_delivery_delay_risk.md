# Problem 09 — Delivery Delay Risk

**Business question:** Predict whether an order will be delivered later than its promised window.

**Target:** `target` — `0` = On time, `1` = Delayed (the order is predicted to arrive late)

---

## 1. Data Inspection

- Rows × columns: **1000 × 7** (6 numeric features + 1 target)
- Missing values found: **0** — none, no imputation needed
- Duplicate rows found: **0** — none
- Class balance: `0`=500 (50.0%), `1`=500 (50.0%) — balanced, no resampling needed

**Feature dictionary**

| Feature | Meaning |
|---|---|
| `distance_km` | Delivery distance in kilometers |
| `warehouse_load` | Current load/utilization at the dispatching warehouse |
| `order_hour` | Hour of day the order was placed |
| `items_count` | Number of items in the order |
| `weather_risk` | Weather-related risk score for the delivery route |
| `carrier_delay_rate` | Historical delay rate for the assigned carrier |

**Descriptive statistics**

| Feature | Mean | Std | Min | Median | Max |
|---|---|---|---|---|---|
| `distance_km` | 29.58 | 21.94 | 3.41 | 23.76 | 162.00 |
| `warehouse_load` | 3.08 | 1.75 | 0.00 | 3.00 | 12.00 |
| `order_hour` | 2.91 | 1.71 | 0.00 | 3.00 | 10.00 |
| `items_count` | 2.96 | 1.75 | 0.00 | 3.00 | 9.00 |
| `weather_risk` | 2.95 | 1.44 | 1.00 | 3.00 | 5.00 |
| `carrier_delay_rate` | 50.07 | 19.97 | 0.00 | 50.47 | 100.00 |

![Correlation heatmap](09_delivery_delay_risk/correlation_heatmap.png)

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
- **Best configuration found:** `{"C": 0.5, "max_iter": 2000, "penalty": "l2", "solver": "lbfgs"}`

---

## 5. Cross-Validation (generalization check)

5-fold stratified cross-validation on the **training data only** (independent of the held-out test set):

| Metric | Mean | Std dev |
|---|---|---|
| Accuracy | 68.50% | ±2.55% |
| ROC-AUC | 0.749 | ±0.037 |

Low standard deviation across folds indicates the model's performance is **stable**, not an artifact of one lucky split.

---

## 6. Test-Set Evaluation

| Metric | Score |
|---|---|
| Accuracy | **62.50%** (baseline: 50.0%, lift: +12.5 pts) |
| Precision | 0.613 |
| Recall | 0.680 |
| F1-score | 0.645 |
| ROC-AUC | 0.715 |

![Confusion matrix](09_delivery_delay_risk/confusion_matrix.png)

**Confusion matrix breakdown** (test set, n=200):
- True Negatives: 57 (correctly predicted **On time**)
- True Positives: 68 (correctly predicted **Delayed**)
- False Positives: 43 (predicted **Delayed**, actually on time — a "false alarm")
- False Negatives: 32 (predicted **On time**, actually delayed — a "missed case")

![ROC curve](09_delivery_delay_risk/roc_curve.png)

An ROC-AUC of 0.715 means the model ranks a random positive case above a random negative case 71.5% of the time — well above the 50% (random-guess) line.

---

## 7. Coefficient Interpretation

Coefficients are on the **standardized** feature scale, so they're directly comparable — a larger magnitude means a bigger swing in log-odds per standard deviation of that feature.

![Feature impact](09_delivery_delay_risk/coefficient_importance.png)

| Feature | Standardized coefficient | Odds ratio |
|---|---|---|
| `carrier_delay_rate` | +0.592 | 1.807 |
| `warehouse_load` | +0.521 | 1.683 |
| `items_count` | +0.518 | 1.679 |
| `order_hour` | -0.342 | 0.710 |
| `distance_km` | -0.295 | 0.745 |
| `weather_risk` | -0.243 | 0.784 |

**Top 3 drivers:**

- **`carrier_delay_rate`** — one standard deviation increase increases the odds of **delayed** by roughly **81%** (odds ratio = 1.81).
- **`warehouse_load`** — one standard deviation increase increases the odds of **delayed** by roughly **68%** (odds ratio = 1.68).
- **`items_count`** — one standard deviation increase increases the odds of **delayed** by roughly **68%** (odds ratio = 1.68).

*(A positive coefficient / odds ratio > 1 pushes toward `Delayed`; negative / odds ratio < 1 pushes toward `On time`.)*

---

## 8. Limitations & Possible Improvements

- **Synthetic data:** this dataset is generated for practice, not sourced from real operations — feature relationships are likely cleaner and more linear than real-world data, so real-world accuracy would probably be lower.
- **Linearity assumption:** Logistic Regression models a linear decision boundary in log-odds space. If the true relationship is non-linear (e.g. threshold effects), a tree-based model (Random Forest, Gradient Boosting) could outperform it — worth benchmarking as a next step.
- **Sample size:** 1000 rows is workable for 6 features but modest; more data would tighten the confidence interval on the coefficients and CV scores.
- **Single train/test split for final numbers:** cross-validation (Section 5) mitigates this, but a nested CV or repeated-split evaluation would give an even more robust estimate.
- **Threshold choice:** the default 0.5 probability threshold was used for classification. Depending on whether false positives (43) or false negatives (32) are more costly in this business context, the threshold could be tuned (e.g. via the ROC curve) to favor precision or recall.
- **No external validation:** the model hasn't been tested on a different time period or population; performance could degrade under distribution shift (concept drift).

---

## Conclusion

With **62.5% test accuracy** and a **0.715 ROC-AUC** — a **+12.5 point** lift over the majority-class baseline — Logistic Regression provides a reasonably strong, fully interpretable model for **delivery delay risk**. Its main practical value here is not just prediction but **explainability**: the odds ratios above give a stakeholder a direct, defensible answer to "why did the model flag this case?" — something harder to get from a black-box model. For production use, the limitations above (especially threshold tuning and benchmarking against tree-based models) should be addressed first.
