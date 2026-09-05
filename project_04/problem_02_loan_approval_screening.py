"""
Problem 02 — Loan Approval Screening
====================================

Predict whether a loan application should be approved.

Run directly (no Jupyter needed):
    python src/problem_02_loan_approval_screening.py

Produces:
    outputs/02_loan_approval_screening/   (plots + metrics.json)
    reports/02_loan_approval_screening.md (full written analysis)
"""

from engine import ProblemSpec, run_problem

SPEC = ProblemSpec(
    problem_id="02",
    slug="loan_approval_screening",
    title="Loan Approval Screening",
    dataset_file="dataset_02_loan_approval_screening.csv",
    business_question="Predict whether a loan application should be approved.",
    feature_notes={
    "income_monthly": "Applicant's monthly income",
    "credit_score": "Credit bureau score",
    "debt_to_income": "Existing debt as a percentage of income",
    "employment_years": "Years in current employment",
    "loan_amount": "Amount of loan requested",
    "prior_defaults": "Number of prior loan defaults on record"
},
    target_labels=('Not approved', 'Approved'),
    positive_meaning="the application is predicted to be approved",
)

if __name__ == "__main__":
    result = run_problem(SPEC)
    print(f"[Problem 02] Loan Approval Screening")
    print(f"  Accuracy : {result['accuracy']*100:.2f}%")
    print(f"  ROC-AUC  : {result['roc_auc']:.3f}")
    print(f"  F1-score : {result['f1']:.3f}")
    print(f"  Report   -> reports/02_loan_approval_screening.md")
