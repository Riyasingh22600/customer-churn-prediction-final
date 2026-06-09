from typing import Dict, Any
import pandas as pd
from src.models.utils import load_object

_model = None
_features = None


def load_model_and_features():
    global _model, _features
    if _model is None:
        _model = load_object("model_rf.joblib")
    if _features is None:
        _features = load_object("selected_features.joblib")
    return _model, _features


def predict_single(record: Dict[str, Any]):
    model, features = load_model_and_features()
    df = pd.DataFrame([record])
    df = df[features]  # keep only selected features
    proba = float(model.predict_proba(df)[:, 1][0])
    pred = int(model.predict(df)[0])
    return {"prediction": pred, "probability": float(proba)}