"""
Problem 07 — Student Placement Eligibility
==========================================

Predict whether a student is likely to meet a placement eligibility criterion.

Run directly (no Jupyter needed):
    python src/problem_07_student_placement_eligibility.py

Produces:
    outputs/07_student_placement_eligibility/   (plots + metrics.json)
    reports/07_student_placement_eligibility.md (full written analysis)
"""

from engine import ProblemSpec, run_problem

SPEC = ProblemSpec(
    problem_id="07",
    slug="student_placement_eligibility",
    title="Student Placement Eligibility",
    dataset_file="dataset_07_student_placement_eligibility.csv",
    business_question="Predict whether a student is likely to meet a placement eligibility criterion.",
    feature_notes={
    "cgpa": "Cumulative GPA",
    "attendance_pct": "Attendance percentage",
    "coding_score": "Coding/technical assessment score",
    "projects_completed": "Number of academic/personal projects completed",
    "internship_months": "Months of internship experience",
    "backlogs": "Number of outstanding academic backlogs"
},
    target_labels=('Not eligible', 'Eligible'),
    positive_meaning="the student is predicted to meet placement eligibility",
)

if __name__ == "__main__":
    result = run_problem(SPEC)
    print(f"[Problem 07] Student Placement Eligibility")
    print(f"  Accuracy : {result['accuracy']*100:.2f}%")
    print(f"  ROC-AUC  : {result['roc_auc']:.3f}")
    print(f"  F1-score : {result['f1']:.3f}")
    print(f"  Report   -> reports/07_student_placement_eligibility.md")
