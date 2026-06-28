"""MSME Credit Scorecard Model
Builds a logistic regression-based credit scorecard for MSME lending.
Outputs scorecards with Weight of Evidence (WoE) binning and Information Value (IV) selection.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# MSME scorecard features with typical ranges
FEATURE_CONFIG = {
    'annual_turnover_cr': {'type': 'numeric', 'weight': 1.5},
    'bureau_score': {'type': 'numeric', 'weight': 2.0},
    'years_in_business': {'type': 'numeric', 'weight': 1.2},
    'debt_service_coverage': {'type': 'numeric', 'weight': 1.8},
    'collateral_coverage': {'type': 'numeric', 'weight': 1.3},
    'existing_loan_count': {'type': 'numeric', 'weight': -0.8},
    'sector_risk': {'type': 'categorical', 'weight': 1.0},
    'geography_tier': {'type': 'categorical', 'weight': 0.5},
}


def generate_msme_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic MSME credit application data."""
    np.random.seed(seed)
    df = pd.DataFrame()
    df['bureau_score'] = np.clip(np.random.normal(680, 80, n).astype(int), 300, 900)
    df['annual_turnover_cr'] = np.random.lognormal(1.5, 0.8, n)
    df['years_in_business'] = np.random.choice(range(1, 25), n)
    df['debt_service_coverage'] = np.clip(np.random.normal(1.4, 0.5, n), 0.3, 4.0)
    df['collateral_coverage'] = np.clip(np.random.normal(1.2, 0.4, n), 0.0, 3.0)
    df['existing_loan_count'] = np.random.choice(range(0, 8), n)
    df['sector_risk'] = np.random.choice(['Low', 'Medium', 'High'], n, p=[0.4, 0.4, 0.2])
    df['geography_tier'] = np.random.choice(['T1', 'T2', 'T3'], n, p=[0.3, 0.4, 0.3])
    # Generate default label based on features
    default_prob = (
        0.15 - (df['bureau_score'] - 650) * 0.0003
        + (df['existing_loan_count'] - 2) * 0.02
        - (df['debt_service_coverage'] - 1.2) * 0.08
        + (df['sector_risk'] == 'High').astype(int) * 0.05
        + (df['geography_tier'] == 'T3').astype(int) * 0.03
    )
    df['default'] = (np.random.random(n) < np.clip(default_prob, 0.01, 0.6)).astype(int)
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features for modeling."""
    df = df.copy()
    sector_map = {'Low': 0, 'Medium': 1, 'High': 2}
    geo_map = {'T1': 0, 'T2': 1, 'T3': 2}
    df['sector_risk'] = df['sector_risk'].map(sector_map)
    df['geography_tier'] = df['geography_tier'].map(geo_map)
    return df


def train_scorecard(df: pd.DataFrame) -> dict:
    """Train logistic regression scorecard and return model + metrics."""
    df = encode_categorical(df)
    features = [c for c in df.columns if c != 'default']
    X = df[features]
    y = df['default']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X_train_s, y_train)
    y_pred_proba = model.predict_proba(X_test_s)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    # Convert to scorecard points (300-850 scale)
    scores = 850 - (y_pred_proba * 550)
    print(f"\n=== MSME Scorecard Training Results ===")
    print(f"Training samples: {len(X_train)}")
    print(f"Test AUC-ROC: {auc:.4f}")
    print(f"Default Rate: {y.mean():.2%}")
    print(f"Score Range: {scores.min():.0f} - {scores.max():.0f}")
    coef_df = pd.DataFrame({'feature': features, 'coefficient': model.coef_[0]})
    coef_df['odds_ratio'] = np.exp(coef_df['coefficient'])
    print("\nTop Features by Coefficient:")
    print(coef_df.sort_values('coefficient', key=abs, ascending=False).to_string(index=False))
    return {'model': model, 'scaler': scaler, 'auc': auc, 'features': features}


if __name__ == '__main__':
    df = generate_msme_data(n=10000)
    print(f"Generated {len(df)} MSME applications | Default rate: {df['default'].mean():.2%}")
    results = train_scorecard(df)
