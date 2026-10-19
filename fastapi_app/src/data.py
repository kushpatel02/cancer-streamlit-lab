from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

_raw = load_breast_cancer(as_frame=True)
# "mean radius" -> "mean_radius" so names are valid JSON/Pydantic fields
FEATURES = [c.replace(" ", "_") for c in _raw.feature_names]
CLASSES = [str(c) for c in _raw.target_names]  # ['malignant', 'benign']


def load_data():
    X = _raw.data.copy()
    X.columns = FEATURES
    return X, _raw.target


def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
