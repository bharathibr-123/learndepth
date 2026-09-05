"""
Problem 06 — Equipment Failure Warning
======================================

Predict whether an industrial machine will fail within the next operating window.

Run directly (no Jupyter needed):
    python src/problem_06_equipment_failure_warning.py

Produces:
    outputs/06_equipment_failure_warning/   (plots + metrics.json)
    reports/06_equipment_failure_warning.md (full written analysis)
"""

from engine import ProblemSpec, run_problem

SPEC = ProblemSpec(
    problem_id="06",
    slug="equipment_failure_warning",
    title="Equipment Failure Warning",
    dataset_file="dataset_06_equipment_failure_warning.csv",
    business_question="Predict whether an industrial machine will fail within the next operating window.",
    feature_notes={
    "temperature": "Operating temperature reading",
    "vibration": "Vibration sensor reading",
    "pressure": "Pressure sensor reading",
    "runtime_hours": "Cumulative runtime hours since last service",
    "maintenance_gap_days": "Days since last maintenance",
    "power_draw": "Power draw reading"
},
    target_labels=('No failure expected', 'Failure expected'),
    positive_meaning="the machine is predicted to fail soon",
)

if __name__ == "__main__":
    result = run_problem(SPEC)
    print(f"[Problem 06] Equipment Failure Warning")
    print(f"  Accuracy : {result['accuracy']*100:.2f}%")
    print(f"  ROC-AUC  : {result['roc_auc']:.3f}")
    print(f"  F1-score : {result['f1']:.3f}")
    print(f"  Report   -> reports/06_equipment_failure_warning.md")
