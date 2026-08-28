"""
House Price Prediction — Streamlit Demo App
Loads the model trained in notebooks/house_price_eda_fe.ipynb and predicts
price (in lakh) from user-entered house details, with a live explainability
breakdown (SHAP) and a market-position gauge.

Run with:
    streamlit run app/streamlit_app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap
import plotly.graph_objects as go

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #0e1117 0%, #12151c 100%); }
    div[data-testid="stForm"] {
        border: 1px solid #2a2f3a;
        border-radius: 14px;
        padding: 1.5rem;
        background-color: #161a23;
    }
    .price-banner {
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        background: linear-gradient(90deg, #14532d 0%, #166534 100%);
        border: 1px solid #22c55e55;
        margin-top: 1rem;
    }
    .price-banner h2 { color: #4ade80; margin: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))
    scaled_columns = joblib.load(os.path.join(MODELS_DIR, "scaled_columns.pkl"))
    background = joblib.load(os.path.join(MODELS_DIR, "shap_background.pkl"))
    price_distribution = joblib.load(os.path.join(MODELS_DIR, "price_distribution.pkl"))
    explainer = shap.Explainer(model, background)
    return model, scaler, feature_columns, scaled_columns, explainer, price_distribution


def engineer_features(area_sqft, bedrooms, age_years, distance_city_km):
    """Recreates the exact feature engineering pipeline used in the notebook,
    for a single input row."""
    row = {
        "area_sqft": area_sqft,
        "bedrooms": bedrooms,
        "age_years": age_years,
        "distance_city_km": distance_city_km,
    }
    row["area_per_bedroom"] = area_sqft / (bedrooms if bedrooms != 0 else 1)
    row["is_far_from_city"] = int(distance_city_km > 15)
    row["bedrooms_area_interaction"] = bedrooms * area_sqft

    if age_years <= 5:
        age_cat = "New"
    elif age_years <= 20:
        age_cat = "Mid-age"
    else:
        age_cat = "Old"

    if distance_city_km <= 5:
        dist_bucket = "Close"
    elif distance_city_km <= 15:
        dist_bucket = "Moderate"
    else:
        dist_bucket = "Far"

    row["age_category_Mid-age"] = int(age_cat == "Mid-age")
    row["age_category_Old"] = int(age_cat == "Old")
    row["distance_bucket_Moderate"] = int(dist_bucket == "Moderate")
    row["distance_bucket_Far"] = int(dist_bucket == "Far")

    return row


FRIENDLY_NAMES = {
    "area_sqft": "Area (sqft)",
    "bedrooms": "Bedrooms",
    "age_years": "Property age",
    "distance_city_km": "Distance from city",
    "area_per_bedroom": "Area per bedroom",
    "is_far_from_city": "Far from city (flag)",
    "bedrooms_area_interaction": "Bedrooms x area",
    "age_category_Mid-age": "Age bucket: Mid-age",
    "age_category_Old": "Age bucket: Old",
    "distance_bucket_Moderate": "Distance bucket: Moderate",
    "distance_bucket_Far": "Distance bucket: Far",
}


def make_gauge(predicted_price, price_distribution):
    percentile = float((price_distribution < predicted_price).mean() * 100)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=percentile,
            number={"suffix": "th percentile"},
            title={"text": "Market Position"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#4ade80"},
                "steps": [
                    {"range": [0, 33], "color": "#1f2937"},
                    {"range": [33, 66], "color": "#374151"},
                    {"range": [66, 100], "color": "#4b5563"},
                ],
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)", font_color="white")
    return fig, percentile


def make_explanation_chart(shap_row, feature_names):
    contributions = pd.Series(shap_row, index=feature_names)
    contributions = contributions[contributions.abs() > 0.01].sort_values()
    labels = [FRIENDLY_NAMES.get(f, f) for f in contributions.index]
    colors = ["#ef4444" if v < 0 else "#22c55e" for v in contributions.values]

    fig = go.Figure(
        go.Bar(
            x=contributions.values,
            y=labels,
            orientation="h",
            marker_color=colors,
        )
    )
    fig.update_layout(
        title="Why this price? (impact in lakh)",
        height=max(280, 40 * len(contributions)),
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        xaxis_title="Contribution to predicted price (₹ lakh)",
    )
    return fig


def main():
    st.title("🏠 House Price Predictor")
    st.caption(
        "Demo app for the House Price Prediction EDA + Feature Engineering project. "
        "Enter house details below to get a predicted price with a live explanation."
    )

    model, scaler, feature_columns, scaled_columns, explainer, price_distribution = load_artifacts()

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            area_sqft = st.number_input("Area (sqft)", min_value=200.0, max_value=6000.0, value=1500.0, step=50.0)
            bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3, step=1)
        with col2:
            age_years = st.number_input("Age of property (years)", min_value=0.0, max_value=60.0, value=10.0, step=1.0)
            distance_city_km = st.number_input("Distance from city center (km)", min_value=0.1, max_value=50.0, value=10.0, step=0.5)

        submitted = st.form_submit_button("Predict Price")

    if submitted:
        row = engineer_features(area_sqft, float(bedrooms), age_years, distance_city_km)
        X = pd.DataFrame([row])

        for col in feature_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_columns]
        X[scaled_columns] = scaler.transform(X[scaled_columns])

        prediction = max(model.predict(X)[0], 0)

        st.markdown(
            f'<div class="price-banner"><h2>Predicted Price: ₹ {prediction:.2f} lakh</h2></div>',
            unsafe_allow_html=True,
        )

        gauge_col, explain_col = st.columns([1, 1.4])

        with gauge_col:
            gauge_fig, percentile = make_gauge(prediction, price_distribution)
            st.plotly_chart(gauge_fig, use_container_width=True)
            st.caption(f"This house is priced higher than **{percentile:.0f}%** of homes in the dataset.")

        with explain_col:
            shap_values = explainer(X)
            explain_fig = make_explanation_chart(shap_values.values[0], feature_columns)
            st.plotly_chart(explain_fig, use_container_width=True)
            st.caption("Green bars push the price up, red bars pull it down, relative to the average predicted house.")

        with st.expander("See engineered features used for this prediction"):
            st.dataframe(pd.DataFrame([row]))

    st.divider()
    st.caption(
        "Model trained on `data/02_house_price.csv` with Linear Regression / "
        "tuned Random Forest / XGBoost compared via GridSearchCV — best model auto-selected. "
        "Explanations powered by SHAP. See `notebooks/house_price_eda_fe.ipynb` for the full pipeline."
    )


if __name__ == "__main__":
    main()
