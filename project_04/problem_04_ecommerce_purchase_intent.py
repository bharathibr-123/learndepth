"""
Problem 04 — E-commerce Purchase Intent
=======================================

Predict whether a website session will result in a purchase.

Run directly (no Jupyter needed):
    python src/problem_04_ecommerce_purchase_intent.py

Produces:
    outputs/04_ecommerce_purchase_intent/   (plots + metrics.json)
    reports/04_ecommerce_purchase_intent.md (full written analysis)
"""

from engine import ProblemSpec, run_problem

SPEC = ProblemSpec(
    problem_id="04",
    slug="ecommerce_purchase_intent",
    title="E-commerce Purchase Intent",
    dataset_file="dataset_04_ecommerce_purchase_intent.csv",
    business_question="Predict whether a website session will result in a purchase.",
    feature_notes={
    "pages_viewed": "Number of pages viewed in the session",
    "session_minutes": "Session duration in minutes",
    "products_viewed": "Number of distinct products viewed",
    "cart_additions": "Number of items added to cart",
    "discount_seen": "Whether/how much a discount was shown (encoded)",
    "previous_orders": "Number of prior completed orders by this customer"
},
    target_labels=('No purchase', 'Purchase'),
    positive_meaning="the session is predicted to end in a purchase",
)

if __name__ == "__main__":
    result = run_problem(SPEC)
    print(f"[Problem 04] E-commerce Purchase Intent")
    print(f"  Accuracy : {result['accuracy']*100:.2f}%")
    print(f"  ROC-AUC  : {result['roc_auc']:.3f}")
    print(f"  F1-score : {result['f1']:.3f}")
    print(f"  Report   -> reports/04_ecommerce_purchase_intent.md")
