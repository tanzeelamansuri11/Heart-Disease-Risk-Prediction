"""Streamlit app: Early Detection of Heart Disease Risk."""
import json
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path
from src.data import FEATURES, risk_category

ROOT = Path(__file__).parent
st.set_page_config(page_title="Heart Disease Risk", page_icon="❤️", layout="wide")
st.title("❤️ Early Detection of Heart Disease Risk")
st.caption("Educational ML demo - not a medical diagnosis. Consult a clinician for health decisions.")

models_path = ROOT / "models" / "all_models.joblib"
if not models_path.exists():
    st.error("No trained models found. Run `python -m src.train` first.")
    st.stop()


@st.cache_resource
def load():
    return joblib.load(models_path), json.load(open(ROOT / "models" / "meta.json"))


models, meta = load()
if "SYNTHETIC" in meta["source"]:
    st.warning("Models were trained on SYNTHETIC demo data. Retrain with real data for meaningful results.")

tab1, tab2 = st.tabs(["Predict risk", "Model comparison"])

with tab1:
    model_name = st.sidebar.selectbox("Model", list(models), index=list(models).index(meta["best"]))
    st.sidebar.header("Patient attributes")
    d = {
        "age": st.sidebar.slider("Age", 20, 90, 54),
        "sex": st.sidebar.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male", index=1),
        "cp": st.sidebar.selectbox("Chest pain type", [1, 2, 3, 4], index=3,
                                   format_func={1: "Typical angina", 2: "Atypical angina",
                                                3: "Non-anginal pain", 4: "Asymptomatic"}.get),
        "trestbps": st.sidebar.slider("Resting BP (mm Hg)", 80, 220, 130),
        "chol": st.sidebar.slider("Cholesterol (mg/dl)", 100, 600, 245),
        "fbs": st.sidebar.selectbox("Fasting blood sugar > 120 mg/dl", [0, 1],
                                    format_func=lambda x: "Yes" if x else "No"),
        "restecg": st.sidebar.selectbox("Resting ECG", [0, 1, 2],
                                        format_func={0: "Normal", 1: "ST-T abnormality", 2: "LV hypertrophy"}.get),
        "thalach": st.sidebar.slider("Max heart rate achieved", 60, 220, 150),
        "exang": st.sidebar.selectbox("Exercise-induced angina", [0, 1],
                                      format_func=lambda x: "Yes" if x else "No"),
        "oldpeak": st.sidebar.slider("ST depression (oldpeak)", 0.0, 7.0, 1.0, 0.1),
        "slope": st.sidebar.selectbox("ST slope", [1, 2, 3],
                                      format_func={1: "Upsloping", 2: "Flat", 3: "Downsloping"}.get),
        "ca": st.sidebar.selectbox("Major vessels colored (0-3)", [0, 1, 2, 3]),
        "thal": st.sidebar.selectbox("Thalassemia", [3, 6, 7],
                                     format_func={3: "Normal", 6: "Fixed defect", 7: "Reversible defect"}.get),
    }
    X = pd.DataFrame([d])[FEATURES]
    p = float(models[model_name].predict_proba(X)[0, 1])
    cat = risk_category(p)
    c1, c2 = st.columns(2)
    c1.metric("Probability of heart disease", f"{p:.1%}")
    c2.metric("Risk category", cat)
    st.progress(min(p, 1.0))
    {"Low": st.success, "Moderate": st.warning, "High": st.error}[cat](f"{cat} risk according to {model_name}")
    st.subheader("All models on this patient")
    st.dataframe(pd.DataFrame({n: [float(m.predict_proba(X)[0, 1])] for n, m in models.items()},
                              index=["P(disease)"]).T.style.format("{:.1%}"))

with tab2:
    st.write(f"Data source: **{meta['source']}**")
    metrics = pd.read_csv(ROOT / "reports" / "metrics.csv", index_col=0)
    st.dataframe(metrics.style.format("{:.3f}").highlight_max(axis=0, color="#cfe8cf"))
    a, b = st.columns(2)
    a.image(str(ROOT / "reports" / "roc_curves.png"))
    b.image(str(ROOT / "reports" / "feature_importance.png"))
