"""
Problem 10 — Retail Return Risk
===============================

Predict whether a retail order is likely to be returned after delivery.

Run directly (no Jupyter needed):
    python src/problem_10_retail_return_risk.py

Produces:
    outputs/10_retail_return_risk/   (plots + metrics.json)
    reports/10_retail_return_risk.md (full written analysis)
"""

from engine import ProblemSpec, run_problem

SPEC = ProblemSpec(
    problem_id="10",
    slug="retail_return_risk",
    title="Retail Return Risk",
    dataset_file="dataset_10_retail_return_risk.csv",
    business_question="Predict whether a retail order is likely to be returned after delivery.",
    feature_notes={
    "order_value": "Total order value",
    "item_count": "Number of items in the order",
    "discount_pct": "Discount percentage applied to the order",
    "customer_return_rate": "Customer's historical return rate",
    "delivery_days": "Days taken to deliver the order",
    "category_risk": "Risk score of the product category for returns"
},
    target_labels=('Not returned', 'Returned'),
    positive_meaning="the order is predicted to be returned",
)

if __name__ == "__main__":
    result = run_problem(SPEC)
    print(f"[Problem 10] Retail Return Risk")
    print(f"  Accuracy : {result['accuracy']*100:.2f}%")
    print(f"  ROC-AUC  : {result['roc_auc']:.3f}")
    print(f"  F1-score : {result['f1']:.3f}")
    print(f"  Report   -> reports/10_retail_return_risk.md")
