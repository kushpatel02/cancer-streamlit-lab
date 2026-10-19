import json
from pathlib import Path

import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data import CLASSES, FEATURES, load_data, split_data

MODEL_DIR = Path(__file__).resolve().parents[1] / "model"


def train():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)),
    ])
    model.fit(X_train, y_train)
    preds, proba = model.predict(X_test), model.predict_proba(X_test)[:, 1]
    importances = model.named_steps["clf"].feature_importances_

    metadata = {
        "model_type": "StandardScaler + GradientBoostingClassifier",
        "classes": CLASSES,
        "features": FEATURES,
        "metrics": {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "f1": round(f1_score(y_test, preds), 4),
            "roc_auc": round(roc_auc_score(y_test, proba), 4),
        },
        "feature_importances": dict(zip(FEATURES, importances.round(4).tolist())),
        # [min, max, median] per feature -> Streamlit slider bounds
        "feature_ranges": {f: [round(float(X[f].min()), 4), round(float(X[f].max()), 4),
                               round(float(X[f].median()), 4)] for f in FEATURES},
        # per-class medians -> "Typical malignant / benign" presets in the dashboard
        "class_medians": {
            CLASSES[c]: X[y == c].median().round(4).to_dict() for c in range(len(CLASSES))
        },
    }
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "cancer_model.pkl")
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Model saved | {metadata['metrics']}")


if __name__ == "__main__":
    train()
