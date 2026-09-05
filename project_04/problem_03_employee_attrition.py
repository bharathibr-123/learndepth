"""
Problem 03 — Employee Attrition
===============================

Predict whether an employee is likely to leave within six months.

Run directly (no Jupyter needed):
    python src/problem_03_employee_attrition.py

Produces:
    outputs/03_employee_attrition/   (plots + metrics.json)
    reports/03_employee_attrition.md (full written analysis)
"""

from engine import ProblemSpec, run_problem

SPEC = ProblemSpec(
    problem_id="03",
    slug="employee_attrition",
    title="Employee Attrition",
    dataset_file="dataset_03_employee_attrition.csv",
    business_question="Predict whether an employee is likely to leave within six months.",
    feature_notes={
    "age": "Employee age in years",
    "monthly_income": "Monthly salary",
    "years_at_company": "Tenure at the company, in years",
    "job_satisfaction": "Self-reported job satisfaction score",
    "overtime_hours": "Average overtime hours per month",
    "promotion_years": "Years since the last promotion"
},
    target_labels=('Stays', 'Leaves'),
    positive_meaning="employee is predicted to leave within 6 months",
)

if __name__ == "__main__":
    result = run_problem(SPEC)
    print(f"[Problem 03] Employee Attrition")
    print(f"  Accuracy : {result['accuracy']*100:.2f}%")
    print(f"  ROC-AUC  : {result['roc_auc']:.3f}")
    print(f"  F1-score : {result['f1']:.3f}")
    print(f"  Report   -> reports/03_employee_attrition.md")
