import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from data import CLASSES, FEATURES

MODEL_DIR = Path(__file__).resolve().parents[1] / "model"


@lru_cache
def _load():
    return joblib.load(MODEL_DIR / "cancer_model.pkl"), json.loads((MODEL_DIR / "metadata.json").read_text())


def get_metadata():
    return _load()[1]


def predict_many(rows: list[dict]) -> list[dict]:
    model, _ = _load()
    X = pd.DataFrame(rows)[FEATURES]
    out = []
    for p in model.predict_proba(X):
        idx = int(p.argmax())
        out.append({
            "label": idx,
            "diagnosis": CLASSES[idx],
            "confidence": round(float(p[idx]), 4),
            "probabilities": dict(zip(CLASSES, p.round(4).tolist())),
        })
    return out
