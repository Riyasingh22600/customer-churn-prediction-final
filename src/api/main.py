from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
from src.models.inference import predict_single

app = FastAPI(title="Customer Churn Prediction API")

class Record(BaseModel):
    data: List[Dict[str, Any]]  # Accept multiple rows

@app.get("/")
async def root():
    return {"status": "ok", "message": "Customer Churn API running"}

@app.post("/predict")
async def predict(records: Record):
    """
    Accepts a list of customer feature dicts and returns predictions for all.
    """
    results = []
    for row in records.data:
        try:
            res = predict_single(row)
            # Ensure JSON-serializable types
            results.append({
                "prediction": int(res["prediction"]),
                "probability": float(res["probability"])
            })
        except Exception as e:
            results.append({"error": str(e)})

    return results
