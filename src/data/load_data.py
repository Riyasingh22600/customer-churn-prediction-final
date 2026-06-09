import pandas as pd
from pathlib import Path
DATA_PATH = Path(__file__).resolve().parents[2] / 'data' / 'processed' / 'cleaned_customer_churn_dataset.csv'
def load_data(path: str | None = None) -> pd.DataFrame:
    p = path or DATA_PATH
    df = pd.read_csv(p)
    return df