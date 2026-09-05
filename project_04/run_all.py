"""
run_all.py
==========
Runs all 10 Logistic Regression problems end-to-end and writes a
comparison summary report across all of them.

Usage (from the project root, or from inside src/):
    python src/run_all.py
"""

import importlib
import json
from pathlib import Path

PROBLEM_MODULES = [
    "problem_01_customer_churn_risk",
    "problem_02_loan_approval_screening",
    "problem_03_employee_attrition",
    "problem_04_ecommerce_purchase_intent",
    "problem_05_insurance_claim_flagging",
    "problem_06_equipment_failure_warning",
    "problem_07_student_placement_eligibility",
    "problem_08_email_priority_classification",
    "problem_09_delivery_delay_risk",
    "problem_10_retail_return_risk",
]


def main():
    from engine import run_problem

    results = []
    for mod_name in PROBLEM_MODULES:
        mod = importlib.import_module(mod_name)
        print(f"\n=== Running {mod_name} ===")
        result = run_problem(mod.SPEC)
        print(f"  Accuracy={result['accuracy']*100:.2f}%  ROC-AUC={result['roc_auc']:.3f}  F1={result['f1']:.3f}")
        results.append(result)

    _write_summary(results)
    print("\nAll 10 problems complete. See reports/SUMMARY_REPORT.md")


def _write_summary(results):
    root = Path(__file__).resolve().parent.parent
    rows = "\n".join(
        f"| {r['problem_id']} | {r['title']} | {r['accuracy']*100:.2f}% | {r['precision']:.3f} | "
        f"{r['recall']:.3f} | {r['f1']:.3f} | {r['roc_auc']:.3f} | {r['baseline_accuracy']*100:.1f}% |"
        for r in results
    )
    avg_acc = sum(r["accuracy"] for r in results) / len(results) * 100
    avg_auc = sum(r["roc_auc"] for r in results) / len(results)
    best = max(results, key=lambda r: r["roc_auc"])
    worst = min(results, key=lambda r: r["roc_auc"])

    content = f"""# Summary Report — LearnDepth Logistic Regression (10 Problems)

Comparison of all 10 Logistic Regression models, each trained with the same
reproducible pipeline (see `src/engine.py`): stratified 80/20 split, scaling
fit on train only, GridSearchCV hyperparameter tuning, 5-fold stratified
cross-validation, and held-out test-set evaluation.

## Results Table

| # | Problem | Accuracy | Precision | Recall | F1 | ROC-AUC | Baseline Acc. |
|---|---|---|---|---|---|---|---|
{rows}

**Average across all 10 problems:** Accuracy = {avg_acc:.2f}%, ROC-AUC = {avg_auc:.3f}

- **Strongest model:** Problem {best['problem_id']} — {best['title']} (ROC-AUC {best['roc_auc']:.3f})
- **Weakest model:** Problem {worst['problem_id']} — {worst['title']} (ROC-AUC {worst['roc_auc']:.3f})

## Individual Reports

{chr(10).join(f"- [Problem {r['problem_id']} — {r['title']}]({r['problem_id']}_{r['slug']}.md)" for r in results)}

## Overall Observations

- All 10 datasets were clean (no missing values, no duplicates) and perfectly class-balanced (500/500),
  which is why baseline (majority-class) accuracy sits at 50% across every problem — any lift above
  that is attributable entirely to the features and the model, not to class-imbalance artifacts.
- Logistic Regression performed consistently well across all 10 business domains (churn, credit,
  HR, e-commerce, insurance, manufacturing, education, communications, logistics, retail), which
  suggests the underlying signal-to-noise ratio in these synthetic datasets is high and mostly linear
  — real-world versions of these problems would likely need richer feature engineering and possibly
  non-linear models to match this performance.
- The same reproducible pipeline (`engine.py`) was applied to every problem, so differences in score
  reflect genuine differences in how separable each dataset's classes are — not inconsistent methodology.
"""
    (root / "reports" / "SUMMARY_REPORT.md").write_text(content, encoding="utf-8")

    with open(root / "reports" / "summary_metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
