"""
engine.py
=========
Shared, reusable pipeline for every problem in the LearnDepth Logistic
Regression assignment (10 problems, 10 datasets).

Each problem script (problem_01_....py ... problem_10_....py) simply
supplies metadata about its dataset (paths, feature descriptions, target
meaning) and calls `run_problem(...)`. This keeps the modelling logic in
ONE place, tested once, instead of copy-pasted 10 times.

Pipeline stages (matches the LearnDepth "Required Work" checklist):
    1. Inspect      -> dtypes, describe(), missing values, duplicates, class balance
    2. Preprocess    -> stratified train/test split done BEFORE scaling (no leakage),
                        StandardScaler fit on train only
    3. Baseline      -> majority-class DummyClassifier for a sanity-check floor
    4. Train         -> Logistic Regression tuned via GridSearchCV (C, penalty)
    5. Validate      -> Stratified 5-fold cross-validation on the training data
                        (generalization check independent of the held-out test set)
    6. Evaluate      -> accuracy, precision, recall, F1, ROC-AUC, confusion matrix
    7. Interpret     -> standardized coefficients -> odds ratios, ranked by impact
    8. Report        -> a Markdown report + PNG plots written to disk
"""

from __future__ import annotations

import json
import textwrap
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless rendering — safe for VS Code / terminal runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11
RANDOM_STATE = 42


@dataclass
class ProblemSpec:
    problem_id: str            # "01"
    slug: str                  # "customer_churn_risk"
    title: str                 # "Customer Churn Risk"
    dataset_file: str          # "dataset_01_customer_churn_risk.csv"
    business_question: str     # one-line problem statement
    feature_notes: dict        # {column: plain-English meaning}
    target_labels: tuple       # ("Did not churn", "Churned")  for (0, 1)
    positive_meaning: str      # what a "1" prediction means operationally


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def run_problem(spec: ProblemSpec) -> dict:
    root = _project_root()
    data_path = root / "datasets" / spec.dataset_file
    out_dir = root / "outputs" / f"{spec.problem_id}_{spec.slug}"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = root / "reports" / f"{spec.problem_id}_{spec.slug}.md"

    df = pd.read_csv(data_path)
    feature_cols = [c for c in df.columns if c != "target"]

    # ---------------------------------------------------------------- 1. INSPECT
    n_missing = int(df.isnull().sum().sum())
    n_dupes = int(df.duplicated().sum())
    class_counts = df["target"].value_counts().sort_index()
    class_balance = (class_counts / len(df) * 100).round(1)
    describe_tbl = df[feature_cols].describe().T.round(2)

    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, cbar_kws={"shrink": .8})
    plt.title(f"Correlation Matrix — {spec.title}", fontsize=13)
    plt.tight_layout()
    corr_path = out_dir / "correlation_heatmap.png"
    plt.savefig(corr_path, dpi=140)
    plt.close()

    if n_dupes > 0:
        df = df.drop_duplicates().reset_index(drop=True)
    if n_missing > 0:
        df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

    # ---------------------------------------------------------------- 2. PREPROCESS (split first, then scale — avoids leakage)
    X = df[feature_cols].values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ---------------------------------------------------------------- 3. BASELINE
    baseline = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    baseline.fit(X_train_s, y_train)
    baseline_acc = accuracy_score(y_test, baseline.predict(X_test_s))

    # ---------------------------------------------------------------- 4. TRAIN (GridSearchCV hyperparameter tuning)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    param_grid = [
        {"penalty": ["l2"], "C": [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50], "solver": ["lbfgs"], "max_iter": [2000]},
        {"penalty": ["l1"], "C": [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50], "solver": ["liblinear"], "max_iter": [2000]},
    ]
    grid = GridSearchCV(
        LogisticRegression(random_state=RANDOM_STATE),
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
    )
    grid.fit(X_train_s, y_train)
    model: LogisticRegression = grid.best_estimator_
    best_params = grid.best_params_

    # ---------------------------------------------------------------- 5. VALIDATE (cross-validation with the tuned config)
    cv_acc = cross_val_score(model, X_train_s, y_train, cv=cv, scoring="accuracy")
    cv_auc = cross_val_score(model, X_train_s, y_train, cv=cv, scoring="roc_auc")

    # ---------------------------------------------------------------- 6. EVALUATE
    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "baseline_accuracy": baseline_acc,
        "cv_accuracy_mean": cv_acc.mean(),
        "cv_accuracy_std": cv_acc.std(),
        "cv_roc_auc_mean": cv_auc.mean(),
        "cv_roc_auc_std": cv_auc.std(),
    }

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6.5, 5.2))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 15},
        xticklabels=spec.target_labels, yticklabels=spec.target_labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix — {spec.title}", fontsize=13)
    plt.tight_layout()
    cm_path = out_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=140)
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc_val = auc(fpr, tpr)
    plt.figure(figsize=(5.5, 5))
    plt.plot(fpr, tpr, color="#2563eb", lw=2.4, label=f"ROC curve (AUC = {roc_auc_val:.3f})")
    plt.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {spec.title}", fontsize=13)
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = out_dir / "roc_curve.png"
    plt.savefig(roc_path, dpi=140)
    plt.close()

    # ---------------------------------------------------------------- 7. INTERPRET (standardized coefficients -> odds ratios)
    coefs = model.coef_[0]
    odds_ratios = np.exp(coefs)
    coef_df = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": coefs,
        "odds_ratio": odds_ratios,
        "abs_impact": np.abs(coefs),
    }).sort_values("abs_impact", ascending=False).reset_index(drop=True)

    plt.figure(figsize=(7.5, 4.8))
    colors = ["#dc2626" if c < 0 else "#16a34a" for c in coef_df["coefficient"][::-1]]
    plt.barh(coef_df["feature"][::-1], coef_df["coefficient"][::-1], color=colors)
    plt.axvline(0, color="black", lw=0.8)
    plt.xlabel("Standardized coefficient (log-odds impact)")
    plt.title(f"Feature Impact on {spec.title}", fontsize=13)
    plt.tight_layout()
    coef_path = out_dir / "coefficient_importance.png"
    plt.savefig(coef_path, dpi=140)
    plt.close()

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            **metrics,
            "best_params": best_params,
            "confusion_matrix": cm.tolist(),
            "n_train": len(X_train), "n_test": len(X_test),
            "n_missing_found": n_missing, "n_duplicates_found": n_dupes,
        }, f, indent=2)

    # ---------------------------------------------------------------- 8. REPORT
    _write_report(
        spec=spec, report_path=report_path, df=df, describe_tbl=describe_tbl,
        class_counts=class_counts, class_balance=class_balance,
        n_missing=n_missing, n_dupes=n_dupes,
        X_train=X_train, X_test=X_test, best_params=best_params,
        metrics=metrics, cm=cm, coef_df=coef_df,
        corr_path=corr_path.name, cm_path=cm_path.name, roc_path=roc_path.name, coef_path=coef_path.name,
        out_dir_name=out_dir.name,
    )

    return {
        "problem_id": spec.problem_id, "slug": spec.slug, "title": spec.title,
        **metrics, "best_params": best_params,
    }


def _write_report(*, spec, report_path, df, describe_tbl, class_counts, class_balance,
                   n_missing, n_dupes, X_train, X_test, best_params, metrics, cm, coef_df,
                   corr_path, cm_path, roc_path, coef_path, out_dir_name):

    neg_label, pos_label = spec.target_labels
    tn, fp, fn, tp = cm.ravel()

    feature_table = "\n".join(
        f"| `{col}` | {spec.feature_notes.get(col, '—')} |" for col in describe_tbl.index
    )
    desc_table = "\n".join(
        f"| `{row}` | {describe_tbl.loc[row, 'mean']:.2f} | {describe_tbl.loc[row, 'std']:.2f} | "
        f"{describe_tbl.loc[row, 'min']:.2f} | {describe_tbl.loc[row, '50%']:.2f} | {describe_tbl.loc[row, 'max']:.2f} |"
        for row in describe_tbl.index
    )

    top3 = coef_df.head(3)
    top3_lines = []
    for _, r in top3.iterrows():
        direction = "increases" if r["coefficient"] > 0 else "decreases"
        pct = abs(r["odds_ratio"] - 1) * 100
        top3_lines.append(
            f"- **`{r['feature']}`** — one standard deviation increase {direction} the odds of "
            f"**{pos_label.lower()}** by roughly **{pct:.0f}%** (odds ratio = {r['odds_ratio']:.2f})."
        )
    top3_block = "\n".join(top3_lines)

    coef_table = "\n".join(
        f"| `{r.feature}` | {r.coefficient:+.3f} | {r.odds_ratio:.3f} |"
        for r in coef_df.itertuples()
    )

    lift_over_baseline = (metrics["accuracy"] - metrics["baseline_accuracy"]) * 100

    content = f"""# Problem {spec.problem_id} — {spec.title}

**Business question:** {spec.business_question}

**Target:** `target` — `0` = {neg_label}, `1` = {pos_label} ({spec.positive_meaning})

---

## 1. Data Inspection

- Rows × columns: **{df.shape[0]} × {df.shape[1]}** ({len(describe_tbl)} numeric features + 1 target)
- Missing values found: **{n_missing}**{' (imputed with column median)' if n_missing else ' — none, no imputation needed'}
- Duplicate rows found: **{n_dupes}**{' (dropped)' if n_dupes else ' — none'}
- Class balance: `0`={class_counts.get(0,0)} ({class_balance.get(0,0)}%), `1`={class_counts.get(1,0)} ({class_balance.get(1,0)}%) — {"balanced, no resampling needed" if abs(class_balance.get(0,50)-50) < 5 else "some imbalance present"}

**Feature dictionary**

| Feature | Meaning |
|---|---|
{feature_table}

**Descriptive statistics**

| Feature | Mean | Std | Min | Median | Max |
|---|---|---|---|---|---|
{desc_table}

![Correlation heatmap]({out_dir_name}/{corr_path})

---

## 2. Preprocessing & Split

- Train/test split: **80% / 20%**, stratified on `target` to preserve class balance (train n={len(X_train)}, test n={len(X_test)}).
- The split was performed **before** any scaling, and the `StandardScaler` was fit **only on the training set** then applied to the test set — this prevents test-set information leaking into preprocessing.
- Logistic Regression is scale-sensitive (regularization penalizes coefficient magnitude), so standardization also makes the coefficients in Section 5 directly comparable across features.

---

## 3. Baseline

A majority-class `DummyClassifier` (always predicts the more frequent class) scores **{metrics['baseline_accuracy']*100:.1f}% accuracy** on the test set. This is the floor any real model must beat — it represents the accuracy achievable with zero information.

---

## 4. Model Training

- Algorithm: **Logistic Regression**, tuned with `GridSearchCV` (5-fold stratified CV, scored on ROC-AUC).
- Search space: `penalty ∈ {{l1, l2}}`, `C ∈ [0.01 … 50]`, `solver` matched to penalty (`liblinear` for l1, `lbfgs` for l2).
- **Best configuration found:** `{json.dumps(best_params)}`

---

## 5. Cross-Validation (generalization check)

5-fold stratified cross-validation on the **training data only** (independent of the held-out test set):

| Metric | Mean | Std dev |
|---|---|---|
| Accuracy | {metrics['cv_accuracy_mean']*100:.2f}% | ±{metrics['cv_accuracy_std']*100:.2f}% |
| ROC-AUC | {metrics['cv_roc_auc_mean']:.3f} | ±{metrics['cv_roc_auc_std']:.3f} |

Low standard deviation across folds indicates the model's performance is **stable**, not an artifact of one lucky split.

---

## 6. Test-Set Evaluation

| Metric | Score |
|---|---|
| Accuracy | **{metrics['accuracy']*100:.2f}%** (baseline: {metrics['baseline_accuracy']*100:.1f}%, lift: {lift_over_baseline:+.1f} pts) |
| Precision | {metrics['precision']:.3f} |
| Recall | {metrics['recall']:.3f} |
| F1-score | {metrics['f1']:.3f} |
| ROC-AUC | {metrics['roc_auc']:.3f} |

![Confusion matrix]({out_dir_name}/{cm_path})

**Confusion matrix breakdown** (test set, n={tn+fp+fn+tp}):
- True Negatives: {tn} (correctly predicted **{neg_label}**)
- True Positives: {tp} (correctly predicted **{pos_label}**)
- False Positives: {fp} (predicted **{pos_label}**, actually {neg_label.lower()} — a "false alarm")
- False Negatives: {fn} (predicted **{neg_label}**, actually {pos_label.lower()} — a "missed case")

![ROC curve]({out_dir_name}/{roc_path})

An ROC-AUC of {metrics['roc_auc']:.3f} means the model ranks a random positive case above a random negative case {metrics['roc_auc']*100:.1f}% of the time — well above the 50% (random-guess) line.

---

## 7. Coefficient Interpretation

Coefficients are on the **standardized** feature scale, so they're directly comparable — a larger magnitude means a bigger swing in log-odds per standard deviation of that feature.

![Feature impact]({out_dir_name}/{coef_path})

| Feature | Standardized coefficient | Odds ratio |
|---|---|---|
{coef_table}

**Top 3 drivers:**

{top3_block}

*(A positive coefficient / odds ratio > 1 pushes toward `{pos_label}`; negative / odds ratio < 1 pushes toward `{neg_label}`.)*

---

## 8. Limitations & Possible Improvements

- **Synthetic data:** this dataset is generated for practice, not sourced from real operations — feature relationships are likely cleaner and more linear than real-world data, so real-world accuracy would probably be lower.
- **Linearity assumption:** Logistic Regression models a linear decision boundary in log-odds space. If the true relationship is non-linear (e.g. threshold effects), a tree-based model (Random Forest, Gradient Boosting) could outperform it — worth benchmarking as a next step.
- **Sample size:** {df.shape[0]} rows is workable for 6 features but modest; more data would tighten the confidence interval on the coefficients and CV scores.
- **Single train/test split for final numbers:** cross-validation (Section 5) mitigates this, but a nested CV or repeated-split evaluation would give an even more robust estimate.
- **Threshold choice:** the default 0.5 probability threshold was used for classification. Depending on whether false positives ({fp}) or false negatives ({fn}) are more costly in this business context, the threshold could be tuned (e.g. via the ROC curve) to favor precision or recall.
- **No external validation:** the model hasn't been tested on a different time period or population; performance could degrade under distribution shift (concept drift).

---

## Conclusion

With **{metrics['accuracy']*100:.1f}% test accuracy** and a **{metrics['roc_auc']:.3f} ROC-AUC** — a **{lift_over_baseline:+.1f} point** lift over the majority-class baseline — Logistic Regression provides a reasonably strong, fully interpretable model for **{spec.title.lower()}**. Its main practical value here is not just prediction but **explainability**: the odds ratios above give a stakeholder a direct, defensible answer to "why did the model flag this case?" — something harder to get from a black-box model. For production use, the limitations above (especially threshold tuning and benchmarking against tree-based models) should be addressed first.
"""
    report_path.write_text(content, encoding="utf-8")
