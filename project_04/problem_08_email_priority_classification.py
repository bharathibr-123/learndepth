"""
Problem 08 — Email Priority Classification
==========================================

Predict whether an incoming email should be marked high priority.

Run directly (no Jupyter needed):
    python src/problem_08_email_priority_classification.py

Produces:
    outputs/08_email_priority_classification/   (plots + metrics.json)
    reports/08_email_priority_classification.md (full written analysis)
"""

from engine import ProblemSpec, run_problem

SPEC = ProblemSpec(
    problem_id="08",
    slug="email_priority_classification",
    title="Email Priority Classification",
    dataset_file="dataset_08_email_priority_classification.csv",
    business_question="Predict whether an incoming email should be marked high priority.",
    feature_notes={
    "sender_frequency": "How often this sender has emailed before",
    "keyword_score": "Score based on urgency-related keywords found",
    "thread_length": "Number of messages in the email thread",
    "response_deadline_hours": "Hours until an implied/explicit response deadline",
    "attachment_count": "Number of attachments",
    "previous_priority_rate": "Historical rate this sender's emails were marked high priority"
},
    target_labels=('Normal priority', 'High priority'),
    positive_meaning="the email is predicted to be high priority",
)

if __name__ == "__main__":
    result = run_problem(SPEC)
    print(f"[Problem 08] Email Priority Classification")
    print(f"  Accuracy : {result['accuracy']*100:.2f}%")
    print(f"  ROC-AUC  : {result['roc_auc']:.3f}")
    print(f"  F1-score : {result['f1']:.3f}")
    print(f"  Report   -> reports/08_email_priority_classification.md")
