"""
Problem 09 — Delivery Delay Risk
================================

Predict whether an order will be delivered later than its promised window.

Run directly (no Jupyter needed):
    python src/problem_09_delivery_delay_risk.py

Produces:
    outputs/09_delivery_delay_risk/   (plots + metrics.json)
    reports/09_delivery_delay_risk.md (full written analysis)
"""

from engine import ProblemSpec, run_problem

SPEC = ProblemSpec(
    problem_id="09",
    slug="delivery_delay_risk",
    title="Delivery Delay Risk",
    dataset_file="dataset_09_delivery_delay_risk.csv",
    business_question="Predict whether an order will be delivered later than its promised window.",
    feature_notes={
    "distance_km": "Delivery distance in kilometers",
    "warehouse_load": "Current load/utilization at the dispatching warehouse",
    "order_hour": "Hour of day the order was placed",
    "items_count": "Number of items in the order",
    "weather_risk": "Weather-related risk score for the delivery route",
    "carrier_delay_rate": "Historical delay rate for the assigned carrier"
},
    target_labels=('On time', 'Delayed'),
    positive_meaning="the order is predicted to arrive late",
)

if __name__ == "__main__":
    result = run_problem(SPEC)
    print(f"[Problem 09] Delivery Delay Risk")
    print(f"  Accuracy : {result['accuracy']*100:.2f}%")
    print(f"  ROC-AUC  : {result['roc_auc']:.3f}")
    print(f"  F1-score : {result['f1']:.3f}")
    print(f"  Report   -> reports/09_delivery_delay_risk.md")
