import joblib
from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parents[2] / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
def save_object(obj, name: str):
    p = MODEL_DIR / name
    joblib.dump(obj, p)
    return str(p)
def load_object(name: str):
    p = MODEL_DIR / name
    return joblib.load(p)