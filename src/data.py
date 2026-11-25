"""Data loading for the UCI Heart Disease (Cleveland) dataset.

Order of attempts:
 1. data/heart.csv (if present; UCI coding, target column named 'target')
 2. Download from UCI via `ucimlrepo` and cache to data/heart.csv
 3. Synthetic demo data (offline fallback ONLY - not for real conclusions)
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "heart.csv"

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "target"


def _synthetic(n=600, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(29, 78, n),
        "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n),
        "trestbps": rng.normal(131, 17, n).round(),
        "chol": rng.normal(246, 51, n).round(),
        "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n),
        "thalach": rng.normal(150, 22, n).round(),
        "exang": rng.integers(0, 2, n),
        "oldpeak": rng.gamma(1.2, 0.9, n).round(1),
        "slope": rng.integers(1, 4, n),
        "ca": rng.integers(0, 4, n),
        "thal": rng.choice([3, 6, 7], n),
    })
    z = (0.04 * (df.age - 54) + 0.8 * (df.cp == 4) + 0.9 * df.exang + 0.6 * df.oldpeak
         + 0.7 * df.ca + 0.8 * (df.thal == 7) + 0.5 * df.sex - 0.03 * (df.thalach - 150) - 1.2)
    df[TARGET] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return df


def load_data():
    """Returns (DataFrame, source_string)."""
    if CSV_PATH.exists():
        return pd.read_csv(CSV_PATH), "local data/heart.csv"
    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=45)  # Heart Disease
        df = ds.data.features.copy()
        df[TARGET] = (ds.data.targets["num"] > 0).astype(int)  # 0 = no disease, 1 = disease
        df = df[FEATURES + [TARGET]]
        CSV_PATH.parent.mkdir(exist_ok=True)
        df.to_csv(CSV_PATH, index=False)
        return df, "UCI repository (downloaded and cached)"
    except Exception as e:
        print(f"[WARN] Could not download UCI data ({e}). Using SYNTHETIC demo data.")
        return _synthetic(), "SYNTHETIC demo data (offline fallback)"


def risk_category(p):
    """Map predicted probability of disease to a risk band."""
    if p < 0.33:
        return "Low"
    if p < 0.66:
        return "Moderate"
    return "High"
