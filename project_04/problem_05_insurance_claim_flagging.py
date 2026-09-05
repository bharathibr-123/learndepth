"""
Problem 05 — Insurance Claim Flagging
=====================================

Predict whether an insurance claim should be flagged for further review.

Run directly (no Jupyter needed):
    python src/problem_05_insurance_claim_flagging.py

Produces:
    outputs/05_insurance_claim_flagging/   (plots + metrics.json)
    reports/05_insurance_claim_flagging.md (full written analysis)
"""

from engine import ProblemSpec, run_problem

SPEC = ProblemSpec(
    problem_id="05",
    slug="insurance_claim_flagging",
    title="Insurance Claim Flagging",
    dataset_file="dataset_05_insurance_claim_flagging.csv",
    business_question="Predict whether an insurance claim should be flagged for further review.",
    feature_notes={
    "claim_amount": "Amount claimed",
    "policy_tenure": "How long the policy has been active, in years",
    "claim_count": "Number of prior claims filed by this policyholder",
    "customer_age": "Policyholder age",
    "vehicle_age": "Age of the insured vehicle, in years",
    "incident_severity": "Severity rating of the incident (encoded)"
},
    target_labels=('Not flagged', 'Flagged'),
    positive_meaning="the claim is predicted to need manual review",
)

if __name__ == "__main__":
    result = run_problem(SPEC)
    print(f"[Problem 05] Insurance Claim Flagging")
    print(f"  Accuracy : {result['accuracy']*100:.2f}%")
    print(f"  ROC-AUC  : {result['roc_auc']:.3f}")
    print(f"  F1-score : {result['f1']:.3f}")
    print(f"  Report   -> reports/05_insurance_claim_flagging.md")
