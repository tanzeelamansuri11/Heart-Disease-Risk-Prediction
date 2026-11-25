"""Train and compare Logistic Regression, Random Forest, AdaBoost and a Voting Classifier."""
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, ConfusionMatrixDisplay, f1_score,
                             precision_score, recall_score, roc_auc_score, RocCurveDisplay)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from src.data import CATEGORICAL, FEATURES, NUMERIC, ROOT, TARGET, load_data
except ImportError:  # running as `python src/train.py`
    from data import CATEGORICAL, FEATURES, NUMERIC, ROOT, TARGET, load_data

SEED = 42


def preprocessor():
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, NUMERIC), ("cat", cat, CATEGORICAL)])


def build_models():
    lr = LogisticRegression(max_iter=1000, random_state=SEED)
    rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=SEED)
    ada = AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=SEED)
    vote = VotingClassifier([("lr", lr), ("rf", rf), ("ada", ada)], voting="soft")
    clfs = {"Logistic Regression": lr, "Random Forest": rf, "AdaBoost": ada, "Voting Classifier": vote}
    return {n: Pipeline([("prep", preprocessor()), ("clf", clone(c))]) for n, c in clfs.items()}


def main():
    df, source = load_data()
    print(f"Data source: {source} | rows: {len(df)} | disease prevalence: {df[TARGET].mean():.2%}")
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    rows, fitted = [], {}
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    for name, pipe in build_models().items():
        cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc").mean()
        pipe.fit(X_tr, y_tr)
        pred, proba = pipe.predict(X_te), pipe.predict_proba(X_te)[:, 1]
        rows.append({"Model": name, "CV ROC-AUC": cv_auc,
                     "Accuracy": accuracy_score(y_te, pred),
                     "Precision": precision_score(y_te, pred),
                     "Recall": recall_score(y_te, pred),
                     "F1": f1_score(y_te, pred),
                     "ROC-AUC": roc_auc_score(y_te, proba)})
        fitted[name] = pipe
        RocCurveDisplay.from_predictions(y_te, proba, name=name, ax=ax_roc)
        fig, ax = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay.from_predictions(y_te, pred, ax=ax, colorbar=False)
        ax.set_title(name)
        fig.tight_layout()
        fig.savefig(ROOT / "reports" / f"cm_{name.replace(' ', '_').lower()}.png", dpi=120)
        plt.close(fig)
    ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax_roc.set_title("ROC curves (test set)")
    fig_roc.tight_layout()
    fig_roc.savefig(ROOT / "reports" / "roc_curves.png", dpi=120)

    res = pd.DataFrame(rows).set_index("Model").round(4).sort_values("ROC-AUC", ascending=False)
    print("\n", res.to_string())
    res.to_csv(ROOT / "reports" / "metrics.csv")
    best = res.index[0]
    print(f"\nBest model by test ROC-AUC: {best}")

    rf = fitted["Random Forest"]
    names = rf.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=names).sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(6, 4))
    imp.plot.barh(ax=ax)
    ax.set_title("Top Random Forest feature importances")
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "feature_importance.png", dpi=120)

    joblib.dump(fitted, ROOT / "models" / "all_models.joblib")
    with open(ROOT / "models" / "meta.json", "w") as f:
        json.dump({"best": best, "source": source}, f)
    print("Saved models/ and reports/")


if __name__ == "__main__":
    main()
