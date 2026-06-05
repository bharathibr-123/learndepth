"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          EMPLOYEE BURNOUT RISK PREDICTION SYSTEM                            ║
║          Production-Grade Linear Regression Pipeline                        ║
║          Author  : Data Science Team                                        ║
║          Version : 1.0.0                                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage:
    python burnout_pipeline.py --data ../data/employee_burnout_dataset_1000_records.csv
    python burnout_pipeline.py --data ../data/employee_burnout_dataset_1000_records.csv --predict new_employees.csv
"""

import argparse
import logging
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

# ─── Logging Setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('../outputs/pipeline.log', mode='w')
    ]
)
log = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────
TARGET          = 'burnout_risk_score'
DROP_COLS       = ['employee_id']
PALETTE         = ['#2E86AB','#E84855','#3BB273','#F4A261','#9B5DE5',
                   '#00BBF9','#F15BB5','#FEE440','#00F5D4','#FB5607']
RANDOM_STATE    = 42
TEST_SIZE       = 0.20
WINSOR_FACTOR   = 3.0
OUTPUT_DIR      = '../outputs'

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
plt.rcParams.update({'figure.dpi': 110})


# ════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ════════════════════════════════════════════════════════════════════════════
def load_data(path: str) -> pd.DataFrame:
    """Load and perform initial validation of the dataset."""
    log.info(f"Loading dataset from: {path}")
    df = pd.read_csv(path)
    log.info(f"Dataset loaded — shape: {df.shape}")
    assert TARGET in df.columns, f"Target column '{TARGET}' not found."
    log.info(f"Missing values: {df.isnull().sum().sum()} | Duplicates: {df.duplicated().sum()}")
    return df


# ════════════════════════════════════════════════════════════════════════════
# 2.  EDA
# ════════════════════════════════════════════════════════════════════════════
def run_eda(df: pd.DataFrame) -> dict:
    """Perform EDA and generate professional visualisations."""
    log.info("Starting Exploratory Data Analysis …")
    numeric_df = df.drop(DROP_COLS, axis=1, errors='ignore')

    # Statistical summary
    stats_df = numeric_df.describe().T
    stats_df['skewness'] = numeric_df.skew()
    stats_df['kurtosis'] = numeric_df.kurt()
    log.info(f"\n{stats_df[['mean','std','min','max','skewness','kurtosis']].to_string()}")

    # Correlation with target
    corr_target = numeric_df.corr()[TARGET].drop(TARGET).sort_values(ascending=False)
    log.info(f"\nCorrelation with {TARGET}:\n{corr_target.to_string()}")

    # ── Plot 1: Correlation Heatmap ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 10))
    mask = np.triu(np.ones_like(numeric_df.corr(), dtype=bool))
    sns.heatmap(numeric_df.corr(), mask=mask, annot=True, fmt='.2f',
                cmap='RdYlGn', center=0, linewidths=0.4, ax=ax,
                cbar_kws={'shrink': 0.8})
    ax.set_title('Feature Correlation Heatmap', fontweight='bold', fontsize=14)
    plt.tight_layout()
    _save('corr_heatmap.png')
    plt.close()

    # ── Plot 2: Burnout distribution ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df[TARGET], bins=40, color='#2E86AB', alpha=0.75, edgecolor='white', density=True)
    df[TARGET].plot.kde(ax=axes[0], color='navy', linewidth=2)
    axes[0].set_title('Burnout Risk Score Distribution', fontweight='bold')
    axes[0].set_xlabel('Burnout Risk Score')
    axes[1].boxplot(df[TARGET], patch_artist=True,
                    boxprops=dict(facecolor='#2E86AB', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2))
    axes[1].set_title('Burnout Risk Score — Box Plot', fontweight='bold')
    plt.tight_layout()
    _save('burnout_distribution.png')
    plt.close()

    log.info("EDA visualisations saved.")
    return {'correlation_with_target': corr_target, 'stats': stats_df}


# ════════════════════════════════════════════════════════════════════════════
# 3.  PREPROCESSING
# ════════════════════════════════════════════════════════════════════════════
def preprocess(df: pd.DataFrame):
    """Full preprocessing pipeline: clean → engineer → split → scale."""
    log.info("Starting Preprocessing Pipeline …")

    # Drop non-predictive columns
    df_proc = df.drop(DROP_COLS, axis=1, errors='ignore').copy()
    log.info(f"Dropped: {DROP_COLS}")

    # Winsorisation (3×IQR)
    winsor_cols = ['emails_sent_per_day','sick_leaves_year','weekly_work_hours',
                   TARGET,'stress_level']
    for col in winsor_cols:
        q1, q3 = df_proc[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        df_proc[col] = df_proc[col].clip(q1 - WINSOR_FACTOR*iqr, q3 + WINSOR_FACTOR*iqr)
    log.info(f"Winsorisation applied to: {winsor_cols}")

    # Feature engineering
    df_proc['work_life_balance'] = (df_proc['sleep_hours'] +
                                    df_proc['exercise_hours_week'] -
                                    df_proc['weekly_work_hours'] / 10)
    df_proc['workload_index']    = (df_proc['weekly_work_hours'] * df_proc['projects_handled'] +
                                    df_proc['emails_sent_per_day'] / 10) / 3
    df_proc['wellness_score']    = (df_proc['sleep_hours'] * 2 +
                                    df_proc['exercise_hours_week'] +
                                    df_proc['productivity_score'] / 10) / 4
    log.info("Feature engineering: work_life_balance, workload_index, wellness_score added.")

    # Split
    feature_cols = [c for c in df_proc.columns if c != TARGET]
    X, y = df_proc[feature_cols], df_proc[TARGET]
    risk_bins = pd.qcut(y, q=4, labels=False, duplicates='drop')
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=risk_bins)
    log.info(f"Train: {X_tr.shape[0]} | Test: {X_te.shape[0]}")

    # Scale
    scaler = StandardScaler()
    X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr), columns=feature_cols)
    X_te_s = pd.DataFrame(scaler.transform(X_te),     columns=feature_cols)
    log.info("StandardScaler applied.")

    return X_tr_s, X_te_s, y_tr, y_te, scaler, feature_cols, df_proc


# ════════════════════════════════════════════════════════════════════════════
# 4.  MODEL
# ════════════════════════════════════════════════════════════════════════════
def train_model(X_tr, y_tr, feature_cols):
    """Train Linear Regression and return model + coefficient summary."""
    log.info("Training Linear Regression model …")
    model = LinearRegression()
    model.fit(X_tr, y_tr)

    coef_df = pd.DataFrame({'Feature': feature_cols, 'Coefficient': model.coef_})
    coef_df['Abs'] = coef_df['Coefficient'].abs()
    coef_df = coef_df.sort_values('Abs', ascending=False)
    log.info(f"Intercept: {model.intercept_:.4f}")
    log.info(f"\n{coef_df[['Feature','Coefficient']].to_string(index=False)}")

    # OLS for statistical significance
    X_ols = sm.add_constant(X_tr.reset_index(drop=True))
    ols   = sm.OLS(y_tr.reset_index(drop=True), X_ols).fit()
    log.info(f"\nOLS Summary:\n{ols.summary()}")

    return model, coef_df, ols


# ════════════════════════════════════════════════════════════════════════════
# 5.  EVALUATION
# ════════════════════════════════════════════════════════════════════════════
def evaluate(model, X_tr, X_te, y_tr, y_te):
    """Calculate all evaluation metrics and generate diagnostic plots."""
    log.info("Evaluating model …")

    y_pred_tr = model.predict(X_tr)
    y_pred_te = np.clip(model.predict(X_te), 0, 100)
    residuals = y_te.values - y_pred_te

    n, k    = len(y_te), X_te.shape[1]
    mae     = mean_absolute_error(y_te, y_pred_te)
    mse     = mean_squared_error(y_te, y_pred_te)
    rmse    = np.sqrt(mse)
    r2      = r2_score(y_te, y_pred_te)
    adj_r2  = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    r2_tr   = r2_score(y_tr, y_pred_tr)

    cv      = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    cv_r2   = cross_val_score(LinearRegression(), X_tr, y_tr, cv=cv, scoring='r2')

    # Assumption tests
    stat_sw, p_sw = stats.shapiro(residuals[:200])
    dw_stat       = durbin_watson(residuals)

    metrics = dict(MAE=mae, MSE=mse, RMSE=rmse, R2=r2, Adj_R2=adj_r2,
                   Train_R2=r2_tr, CV_R2_mean=cv_r2.mean(), CV_R2_std=cv_r2.std(),
                   Shapiro_W=stat_sw, Shapiro_p=p_sw, Durbin_Watson=dw_stat)

    log.info("\n" + "="*50)
    log.info("  EVALUATION RESULTS")
    log.info("="*50)
    for k_m, v in metrics.items():
        log.info(f"  {k_m:<20}: {v:.4f}")
    log.info("="*50)

    # ── Actual vs Predicted ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    axes[0].scatter(y_te, y_pred_te, alpha=0.55, color='#2E86AB', s=22)
    lims = [min(y_te.min(), y_pred_te.min())-1, max(y_te.max(), y_pred_te.max())+1]
    axes[0].plot(lims, lims, 'r--', lw=2)
    axes[0].set(xlabel='Actual', ylabel='Predicted',
                title=f'Actual vs Predicted\nR² = {r2:.4f}')
    axes[1].scatter(y_pred_te, residuals, alpha=0.5, color='#9B5DE5', s=18)
    axes[1].axhline(0, color='red', lw=1.5, ls='--')
    axes[1].set(xlabel='Predicted', ylabel='Residuals', title='Residual Plot')
    axes[2].hist(residuals, bins=35, color='#F4A261', alpha=0.75, density=True, edgecolor='white')
    xr = np.linspace(residuals.min(), residuals.max(), 200)
    axes[2].plot(xr, stats.norm.pdf(xr, residuals.mean(), residuals.std()), 'navy', lw=2)
    axes[2].set(xlabel='Residual', title='Error Distribution')
    plt.tight_layout()
    _save('model_evaluation.png')
    plt.close()

    return metrics, y_pred_te, residuals, cv_r2


# ════════════════════════════════════════════════════════════════════════════
# 6.  SCENARIO SIMULATION
# ════════════════════════════════════════════════════════════════════════════
def scenario_simulation(df_proc, model, scaler, feature_cols):
    """Simulate HR interventions and quantify expected burnout reduction."""
    log.info("Running Scenario Simulations …")

    X_base     = df_proc[feature_cols]
    base_pred  = np.clip(model.predict(scaler.transform(X_base)), 0, 100)
    base_avg   = base_pred.mean()

    scenarios = [
        ('S1: Work Hours −10%',    'weekly_work_hours',   'multiply', 0.90),
        ('S2: Sleep +1h/day',      'sleep_hours',         'add',      1.00),
        ('S3: Exercise +3h/week',  'exercise_hours_week', 'add',      3.00),
        ('S4: Stress −20%',        'stress_level',        'multiply', 0.80),
        ('S5: Productivity +10%',  'productivity_score',  'multiply', 1.10),
    ]

    results = []
    for name, col, op, val in scenarios:
        X_sim = X_base.copy()
        if op == 'multiply':
            X_sim[col] *= val
        else:
            X_sim[col] = (X_sim[col] + val).clip(upper=X_base[col].max() + val + 5)
        sim_pred = np.clip(model.predict(scaler.transform(X_sim)), 0, 100)
        sim_avg  = sim_pred.mean()
        delta    = sim_avg - base_avg
        pct      = delta / base_avg * 100
        results.append({'Scenario': name, 'Baseline': base_avg, 'Scenario Avg': sim_avg,
                         'Delta': delta, 'Pct Change': pct})
        log.info(f"  {name:<30} → {sim_avg:.4f}  ({pct:+.2f}%)")

    res_df = pd.DataFrame(results)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    colors  = ['#3BB273' if v < 0 else '#E84855' for v in res_df['Pct Change']]
    bars    = ax.barh(res_df['Scenario'], res_df['Pct Change'], color=colors, edgecolor='white')
    ax.axvline(0, color='black', lw=1.5)
    for bar, val in zip(bars, res_df['Pct Change']):
        ax.text(val + (0.1 if val >= 0 else -0.1), bar.get_y() + bar.get_height()/2,
                f'{val:+.2f}%', va='center', ha='left' if val >= 0 else 'right', fontweight='bold')
    ax.set(title='Burnout Reduction by HR Intervention Scenario\n(Green = improvement)',
           xlabel='% Change in Average Burnout Score')
    plt.tight_layout()
    _save('scenario_simulation.png')
    plt.close()

    return res_df


# ════════════════════════════════════════════════════════════════════════════
# 7.  PREDICT NEW EMPLOYEES  (optional)
# ════════════════════════════════════════════════════════════════════════════
def predict_new(model, scaler, feature_cols, path: str):
    """Score a new batch of employee records."""
    log.info(f"Scoring new employees from: {path}")
    df_new = pd.read_csv(path)
    # Re-engineer features if needed
    if 'work_life_balance' not in df_new.columns:
        df_new['work_life_balance'] = (df_new['sleep_hours'] +
                                       df_new['exercise_hours_week'] -
                                       df_new['weekly_work_hours'] / 10)
        df_new['workload_index']    = (df_new['weekly_work_hours'] * df_new['projects_handled'] +
                                       df_new['emails_sent_per_day'] / 10) / 3
        df_new['wellness_score']    = (df_new['sleep_hours'] * 2 +
                                       df_new['exercise_hours_week'] +
                                       df_new['productivity_score'] / 10) / 4
    X_new = df_new[feature_cols]
    preds = np.clip(model.predict(scaler.transform(X_new)), 0, 100)
    df_new['predicted_burnout_risk'] = preds
    df_new['risk_category'] = pd.cut(preds,
        bins=[-0.001, 10, 25, 45, 100],
        labels=['Low','Moderate','High','Critical'])
    out_path = os.path.join(OUTPUT_DIR, 'predictions.csv')
    df_new.to_csv(out_path, index=False)
    log.info(f"Predictions saved to: {out_path}")
    return df_new


# ════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ════════════════════════════════════════════════════════════════════════════
def _save(fname):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=120, bbox_inches='tight')


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='Employee Burnout Risk Prediction Pipeline')
    parser.add_argument('--data',    required=True,  help='Path to training CSV')
    parser.add_argument('--predict', required=False, help='Path to new employees CSV (optional)')
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("  EMPLOYEE BURNOUT RISK PREDICTION SYSTEM  v1.0.0")
    log.info("=" * 60)

    df               = load_data(args.data)
    eda_results      = run_eda(df)
    X_tr, X_te, y_tr, y_te, scaler, feature_cols, df_proc = preprocess(df)
    model, coef_df, ols_model = train_model(X_tr, y_tr, feature_cols)
    metrics, y_pred, residuals, cv_r2 = evaluate(model, X_tr, X_te, y_tr, y_te)
    scenario_df      = scenario_simulation(df_proc, model, scaler, feature_cols)

    if args.predict:
        predict_new(model, scaler, feature_cols, args.predict)

    log.info("=" * 60)
    log.info("  PIPELINE COMPLETE")
    log.info(f"  MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}  R²={metrics['R2']:.4f}")
    log.info("=" * 60)
    return model, scaler, feature_cols, metrics


if __name__ == '__main__':
    main()
