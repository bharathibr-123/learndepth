# 🔥 Employee Burnout Risk Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange?style=for-the-badge&logo=scikit-learn)
![Plotly](https://img.shields.io/badge/Plotly-5.15%2B-3F4F75?style=for-the-badge&logo=plotly)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**An industry-level HR Analytics solution that predicts employee burnout risk using Linear Regression, enabling proactive workforce management and data-driven HR interventions.**

[View Notebook](notebooks/Employee_Burnout_Prediction.ipynb) • [Run Pipeline](#usage) • [Business Insights](#business-insights) • [Dashboard](#dashboard)

</div>

---

## 📌 Project Overview

Employee burnout costs organizations an estimated **$125–190 billion** in healthcare spending annually (Harvard Business Review). This project delivers a production-ready predictive analytics system that:

- **Predicts** individual employee burnout risk scores (0–100 scale)
- **Identifies** the strongest workplace and lifestyle drivers of burnout
- **Simulates** the quantified impact of HR interventions (sleep, workload, exercise, stress)
- **Surfaces** actionable, data-backed recommendations for HR leadership
- **Presents** findings through an executive-ready interactive dashboard

---

## 🏆 Key Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **MAE** | 3.69 | Average prediction error of ±3.7 burnout points |
| **RMSE** | 5.17 | Root mean squared error on held-out test data |
| **R² Score** | 0.747 | Model explains **74.7%** of burnout variance |
| **Adjusted R²** | 0.727 | Accounts for all 15 features, no inflation |
| **CV R² (10-fold)** | 0.638 ± 0.037 | Stable generalisation across data splits |

---

## 📂 Project Structure

```
employee_burnout_project/
│
├── 📓 notebooks/
│   └── Employee_Burnout_Prediction.ipynb     # Full analysis notebook (10 phases)
│
├── 🐍 src/
│   └── burnout_pipeline.py                   # Production-grade CLI pipeline
│
├── 📊 data/
│   └── employee_burnout_dataset_1000_records.csv
│
├── 📈 outputs/
│   ├── 01_distributions.png                  # Feature distribution analysis
│   ├── 02_boxplots.png                        # Outlier detection
│   ├── 03_correlation.png                     # Correlation heatmap
│   ├── 04_violin_plots.png                    # Risk category distributions
│   ├── 05_scatter_regression.png             # Feature vs burnout relationships
│   ├── 06_vif.png                             # Multicollinearity check
│   ├── 07_feature_importance.png             # MI + Pearson importance
│   ├── 08_coefficients.png                   # Model coefficients
│   ├── 09_assumptions.png                    # LR assumption diagnostics
│   ├── 10_evaluation.png                     # Actual vs Predicted plots
│   ├── 11_cv_scores.png                      # 10-fold CV results
│   ├── 12_scenarios.png                      # HR intervention simulations
│   └── executive_dashboard.html             # 🖥️ Interactive Plotly dashboard
│
├── 📋 reports/
│   └── Executive_Report.docx                 # Consulting-grade report
│
├── requirements.txt
└── README.md
```

---

## 🔬 Technical Architecture

```
Raw Data (CSV)
      │
      ▼
┌─────────────────┐
│  Data Ingestion │  → Shape validation, type checks, null detection
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   EDA Engine    │  → 10 visualisation types, correlation analysis, VIF
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  Preprocessing Pipeline     │
│  • Winsorisation (3×IQR)    │
│  • Feature Engineering (×3) │
│  • Stratified Train/Test    │
│  • StandardScaler           │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Linear Regression Model    │
│  + OLS (statsmodels)        │
│  + Assumption Testing       │
│  + 10-Fold CV               │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Evaluation & Diagnostics   │
│  MAE / MSE / RMSE / R² /    │
│  Adj-R² / Shapiro-Wilk /    │
│  Durbin-Watson              │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Business Intelligence      │
│  • Scenario Simulation      │
│  • Risk Profiling           │
│  • HR Recommendations       │
│  • Executive Dashboard      │
└─────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.10+
```

### Installation
```bash
# Clone the repository
git clone https://github.com/your-username/employee-burnout-prediction.git
cd employee-burnout-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Usage

**Option 1 — Jupyter Notebook (Recommended for full analysis)**
```bash
jupyter notebook notebooks/Employee_Burnout_Prediction.ipynb
```

**Option 2 — Production CLI Pipeline**
```bash
# Full pipeline on training data
python src/burnout_pipeline.py --data data/employee_burnout_dataset_1000_records.csv

# Score new employees
python src/burnout_pipeline.py \
  --data data/employee_burnout_dataset_1000_records.csv \
  --predict data/new_employees.csv
```

**Option 3 — Open Interactive Dashboard**
```bash
# After running the notebook or pipeline:
open outputs/executive_dashboard.html
```

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Records | 1,000 employees |
| Features | 12 independent variables |
| Target | `burnout_risk_score` (0–100) |
| Missing values | None |
| Duplicates | None |

### Features

| Feature | Type | Description |
|---------|------|-------------|
| `age` | int | Employee age |
| `years_experience` | int | Total work experience |
| `weekly_work_hours` | int | Average hours worked per week |
| `meetings_per_week` | int | Number of meetings attended |
| `emails_sent_per_day` | int | Daily email volume |
| `projects_handled` | int | Active projects count |
| `remote_days_per_month` | int | Days working remotely |
| `sleep_hours` | float | Average nightly sleep |
| `stress_level` | int | Self-reported stress (1–10) |
| `exercise_hours_week` | float | Weekly exercise hours |
| `sick_leaves_year` | int | Annual sick days taken |
| `productivity_score` | int | Self-assessed productivity (0–100) |

---

## 💼 Business Insights

### 🔴 Top Burnout Drivers (increase risk)
1. **Stress Level** — strongest predictor; every unit increase ≈ +1.5 burnout points
2. **Weekly Work Hours** — overwork compounds rapidly above 50h/week
3. **Emails Sent Per Day** — proxy for digital overload and lack of focus time
4. **Workload Index** (engineered) — combined work volume pressure

### 🟢 Protective Factors (reduce risk)
1. **Productivity Score** — engaged, effective employees show lower burnout
2. **Exercise Hours/Week** — physical activity is a strong buffer
3. **Sleep Hours** — each additional hour provides measurable protection
4. **Work-Life Balance** (engineered) — composite wellness signal

### 🏢 Employee Risk Segmentation
| Risk Level | Score Range | Recommended Action |
|------------|-------------|-------------------|
| 🟢 Low | 0–10 | Maintain, monitor quarterly |
| 🟡 Moderate | 10–25 | Preventive check-in, workload review |
| 🟠 High | 25–45 | Manager intervention, stress audit |
| 🔴 Critical | >45 | Immediate HR escalation, EAP referral |

---

## 🔬 Scenario Simulations

| Intervention | Burnout Change | Business Recommendation |
|--------------|----------------|------------------------|
| Work hours −10% | ~−8–12% | Cap weekly hours at 45 |
| Sleep +1 hr/day | ~−4–7% | Launch sleep hygiene campaigns |
| Exercise +3 hrs/week | ~−5–8% | Subsidise gym memberships |
| Stress −20% | ~−10–15% | Deploy mindfulness + EAP programs |
| Productivity +10% | ~−6–9% | Invest in tooling & focus time |

---

## 📑 Phases Covered

| Phase | Description |
|-------|-------------|
| Phase 1 | Advanced EDA (10 visualisation types, VIF, MI) |
| Phase 2 | Preprocessing (Winsorisation, Feature Engineering, Scaling) |
| Phase 3 | Linear Regression + OLS + Assumption Testing |
| Phase 4 | Full Evaluation (MAE, RMSE, R², CV, Diagnostics) |
| Phase 5 | Business Insights + Risk Profiling |
| Phase 6 | Scenario Simulation (5 HR interventions) |
| Phase 7 | Interactive Plotly Executive Dashboard |
| Phase 8 | Professional Report (Executive-level) |
| Phase 9 | 10-Slide Presentation |
| Phase 10 | GitHub-Ready Project (this README) |

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.10+ |
| Data Manipulation | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-Learn |
| Statistical Modelling | Statsmodels |
| Notebook | Jupyter |
| Dashboard | Plotly (HTML export) |

---

## 📈 Model Performance Dashboard

```
╔════════════════════════════════════════╗
║  EVALUATION SUMMARY                    ║
╠════════════════════════════════════════╣
║  MAE             :  3.6943             ║
║  MSE             : 26.6814             ║
║  RMSE            :  5.1654             ║
║  R² Score        :  0.7472  ✅          ║
║  Adjusted R²     :  0.7266             ║
║  CV R² (10-fold) :  0.638 ± 0.037      ║
╚════════════════════════════════════════╝
```

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Data Science Team**  
*HR Analytics | Machine Learning | People Analytics*

---

<div align="center">

⭐ If this project helped you, please give it a star!

*"The best time to identify burnout is before it happens."*

</div>
