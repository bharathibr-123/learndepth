"""
Problem 01 — Customer Churn Risk
================================

Predict whether a subscription customer will churn in the next billing cycle.

Run directly (no Jupyter needed):
    python src/problem_01_customer_churn_risk.py

Produces:
    outputs/01_customer_churn_risk/   (plots + metrics.json)
    reports/01_customer_churn_risk.md (full written analysis)
"""

from engine import ProblemSpec, run_problem

SPEC = ProblemSpec(
    problem_id="01",
    slug="customer_churn_risk",
    title="Customer Churn Risk",
    dataset_file="dataset_01_customer_churn_risk.csv",
    business_question="Predict whether a subscription customer will churn in the next billing cycle.",
    feature_notes={
    "tenure_months": "How many months the customer has been subscribed",
    "monthly_charges": "Monthly bill amount (currency units)",
    "support_tickets": "Number of support tickets raised recently",
    "avg_session_minutes": "Average minutes per usage session",
    "late_payments": "Count of late/missed payments",
    "contract_months": "Length of the current contract, in months"
},
    target_labels=('Did not churn', 'Churned'),
    positive_meaning="customer is predicted to cancel their subscription",
)

if __name__ == "__main__":
    result = run_problem(SPEC)
    print(f"[Problem 01] Customer Churn Risk")
    print(f"  Accuracy : {result['accuracy']*100:.2f}%")
    print(f"  ROC-AUC  : {result['roc_auc']:.3f}")
    print(f"  F1-score : {result['f1']:.3f}")
    print(f"  Report   -> reports/01_customer_churn_risk.md")
