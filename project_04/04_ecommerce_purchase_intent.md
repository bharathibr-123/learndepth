# Problem 04 — E-commerce Purchase Intent

**Business question:** Predict whether a website session will result in a purchase.

**Target:** `target` — `0` = No purchase, `1` = Purchase (the session is predicted to end in a purchase)

---

## 1. Data Inspection

- Rows × columns: **1000 × 7** (6 numeric features + 1 target)
- Missing values found: **0** — none, no imputation needed
- Duplicate rows found: **0** — none
- Class balance: `0`=500 (50.0%), `1`=500 (50.0%) — balanced, no resampling needed

**Feature dictionary**

| Feature | Meaning |
|---|---|
| `pages_viewed` | Number of pages viewed in the session |
| `session_minutes` | Session duration in minutes |
| `products_viewed` | Number of distinct products viewed |
| `cart_additions` | Number of items added to cart |
| `discount_seen` | Whether/how much a discount was shown (encoded) |
| `previous_orders` | Number of prior completed orders by this customer |

**Descriptive statistics**

| Feature | Mean | Std | Min | Median | Max |
|---|---|---|---|---|---|
| `pages_viewed` | 40.79 | 13.90 | 18.00 | 40.00 | 65.00 |
| `session_minutes` | 36.41 | 18.67 | 2.00 | 36.70 | 99.35 |
| `products_viewed` | 2.92 | 1.74 | 0.00 | 3.00 | 10.00 |
| `cart_additions` | 2.94 | 1.72 | 0.00 | 3.00 | 10.00 |
| `discount_seen` | 2.91 | 1.68 | 0.00 | 3.00 | 9.00 |
| `previous_orders` | 3.02 | 1.73 | 0.00 | 3.00 | 9.00 |

![Correlation heatmap](04_ecommerce_purchase_intent/correlation_heatmap.png)

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
| Accuracy | 71.37% | ±3.07% |
| ROC-AUC | 0.781 | ±0.014 |

Low standard deviation across folds indicates the model's performance is **stable**, not an artifact of one lucky split.

---

## 6. Test-Set Evaluation

| Metric | Score |
|---|---|
| Accuracy | **72.50%** (baseline: 50.0%, lift: +22.5 pts) |
| Precision | 0.737 |
| Recall | 0.700 |
| F1-score | 0.718 |
| ROC-AUC | 0.800 |

![Confusion matrix](04_ecommerce_purchase_intent/confusion_matrix.png)

**Confusion matrix breakdown** (test set, n=200):
- True Negatives: 75 (correctly predicted **No purchase**)
- True Positives: 70 (correctly predicted **Purchase**)
- False Positives: 25 (predicted **Purchase**, actually no purchase — a "false alarm")
- False Negatives: 30 (predicted **No purchase**, actually purchase — a "missed case")

![ROC curve](04_ecommerce_purchase_intent/roc_curve.png)

An ROC-AUC of 0.800 means the model ranks a random positive case above a random negative case 80.0% of the time — well above the 50% (random-guess) line.

---

## 7. Coefficient Interpretation

Coefficients are on the **standardized** feature scale, so they're directly comparable — a larger magnitude means a bigger swing in log-odds per standard deviation of that feature.

![Feature impact](04_ecommerce_purchase_intent/coefficient_importance.png)

| Feature | Standardized coefficient | Odds ratio |
|---|---|---|
| `discount_seen` | +0.362 | 1.437 |
| `pages_viewed` | +0.350 | 1.419 |
| `products_viewed` | +0.348 | 1.416 |
| `previous_orders` | -0.220 | 0.802 |
| `session_minutes` | -0.196 | 0.822 |
| `cart_additions` | -0.190 | 0.827 |

**Top 3 drivers:**

- **`discount_seen`** — one standard deviation increase increases the odds of **purchase** by roughly **44%** (odds ratio = 1.44).
- **`pages_viewed`** — one standard deviation increase increases the odds of **purchase** by roughly **42%** (odds ratio = 1.42).
- **`products_viewed`** — one standard deviation increase increases the odds of **purchase** by roughly **42%** (odds ratio = 1.42).

*(A positive coefficient / odds ratio > 1 pushes toward `Purchase`; negative / odds ratio < 1 pushes toward `No purchase`.)*

---

## 8. Limitations & Possible Improvements

- **Synthetic data:** this dataset is generated for practice, not sourced from real operations — feature relationships are likely cleaner and more linear than real-world data, so real-world accuracy would probably be lower.
- **Linearity assumption:** Logistic Regression models a linear decision boundary in log-odds space. If the true relationship is non-linear (e.g. threshold effects), a tree-based model (Random Forest, Gradient Boosting) could outperform it — worth benchmarking as a next step.
- **Sample size:** 1000 rows is workable for 6 features but modest; more data would tighten the confidence interval on the coefficients and CV scores.
- **Single train/test split for final numbers:** cross-validation (Section 5) mitigates this, but a nested CV or repeated-split evaluation would give an even more robust estimate.
- **Threshold choice:** the default 0.5 probability threshold was used for classification. Depending on whether false positives (25) or false negatives (30) are more costly in this business context, the threshold could be tuned (e.g. via the ROC curve) to favor precision or recall.
- **No external validation:** the model hasn't been tested on a different time period or population; performance could degrade under distribution shift (concept drift).

---

## Conclusion

With **72.5% test accuracy** and a **0.800 ROC-AUC** — a **+22.5 point** lift over the majority-class baseline — Logistic Regression provides a reasonably strong, fully interpretable model for **e-commerce purchase intent**. Its main practical value here is not just prediction but **explainability**: the odds ratios above give a stakeholder a direct, defensible answer to "why did the model flag this case?" — something harder to get from a black-box model. For production use, the limitations above (especially threshold tuning and benchmarking against tree-based models) should be addressed first.
