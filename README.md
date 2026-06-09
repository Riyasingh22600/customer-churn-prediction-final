# 📊 Customer Churn Prediction — End-to-End ML Project

An **end-to-end Machine Learning project** to predict customer churn.  
The pipeline covers everything from **data preprocessing → feature selection → model training → API serving (FastAPI on Render) → interactive UI (Streamlit)**.  

Designed for **scalability and production-readiness**, following real-world .
---

## 🚀 Features

- **Data Preprocessing & Feature Engineering**  
  - Handling missing values and categorical encoding.  
  - Custom engineered features: `TotalCharges_per_Month`, tenure bucketing, etc.  

- **Feature Selection**  
  - Top 15 most important features automatically selected via model importance.  

- **Model Training & Evaluation**  
  - Algorithms: Scikit-learn RandomForest, XGBoost.  
  - Metrics: Precision, Recall, F1-score, AUC.  
  - Reproducibility with serialized `.joblib` artifacts.  

- **API & UI**  
  - REST API built with **FastAPI** (deployed on Render).  
  - Interactive **Streamlit dashboard** for:  
    - Single prediction via manual input.  
    - Batch prediction via CSV upload.  
  - Configurable API endpoint via `.streamlit/secrets.toml`.

---

## 📂 Project Structure

```
churnshield/
│── customer_churn_project-main/
│
│── 📦 Core Files
│ ├── Dockerfile
│ ├── Dockerfile.ui
│ ├── docker-compose.yml
│ ├── requirements.txt
│ └── README.md
│
│── 📊 Data
│ └── data/
│ └── processed/
│ └── cleaned_customer_churn_dataset.csv
│
│── 🤖 Models
│ └── models/
│ ├── model_rf.joblib
│ └── selected_features.joblib
│
│── 🛠️ Source Code
│ └── src/
│ ├── init.py
│ │
│ ├── api/
│ │ └── main.py # FastAPI application
│ │
│ ├── data/
│ │ ├── load_data.py
│ │ └── preprocess.py
│ │
│ ├── features/
│ │ └── featrues.py # Feature engineering
│ │
│ ├── models/
│ │ ├── inference.py
│ │ ├── train.py # Training script
│ │ └── utils.py # Helper functions
│ │
│ └── ui/
│ └── streamlit_app.py # Streamlit dashboard
│
│── 🧪 Testing
│ └── test_api_url.pdf # API testing reference

```

---

## ⚡ Quickstart

### 1️⃣ Clone repo & setup environment
```bash
git clone https://github.com/<your-username>/customer_churn_project.git
cd customer_churn_project

# Create virtual environment
python -m venv churnvenv

# Windows
churnvenv\Scripts\activate
# Linux/Mac
source churnvenv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Train the model
```bash
python -m src.models.train
```
> Trained artifacts (`.joblib`) will be stored in the `models/` directory.

### 3️⃣ Run FastAPI backend (local)
```bash
uvicorn src.api.main:app --reload --port 8000
```
- Local API: [http://localhost:8000](http://localhost:8000)  
- Local Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 🌍 Deployed API (Render)
Your API is live here:  
👉 **[https://customer-churn-project-2.onrender.com](https://customer-churn-project-2.onrender.com)**  

Docs:  
👉 **[https://customer-churn-project-2.onrender.com/docs](https://customer-churn-project-2.onrender.com/docs)**  

### 4️⃣ Run Streamlit UI
```bash
streamlit run src/ui/streamlit_app.py
```
- Dashboard: [http://localhost:8501](http://localhost:8501)

Or deploy on **Streamlit Cloud**, and configure `.streamlit/secrets.toml` with:
```toml
API_URL = "https://customer-churn-project-2.onrender.com"
```

---

## 🧪 Testing the API

You can test the deployed API directly using the included script:

**test_api.py**
```python
import requests

# Render API endpoint
API_URL = "https://customer-churn-project-2.onrender.com/predict"

# Example input with selected features
sample_input = {
    "data": {
        "TotalCharges_per_Month": 85.4,
        "Is_MonthToMonth_Encoded": 1,
        "PaymentMethod_Electronic check": 1,
        "Contract_Two year": 0,
        "SeniorCitizen": 0,
        "MonthlyCharges": 70.35,
        "tenure": 12,
        "InternetService_Fiber optic": 1,
        "OnlineSecurity_No": 1,
        "TechSupport_No": 1,
        "PaperlessBilling": 1,
        "Partner": 0,
        "Dependents": 0,
        "DeviceProtection_No": 1,
        "StreamingTV_Yes": 1
    }
}

response = requests.post(API_URL, json=sample_input)

if response.status_code == 200:
    print("✅ API Response:", response.json())
else:
    print("❌ Error:", response.status_code, response.text)
```

Run with:
```bash
python test_api.py
```

---

## 📊 Model Performance

| Metric          | Score |
|-----------------|-------|
| Accuracy        | 0.85  |
| Precision (1)   | 0.72  |
| Recall (1)      | 0.70  |
| F1-Score (1)    | 0.59  |
| AUC             | 0.776 |

---

## 🔮 Next Steps (MLOps Roadmap)

- Containerization with **Docker** (API + UI).  
- CI/CD pipeline for **automated training & deployment**.  
- Experiment tracking with **MLflow** or **Weights & Biases**.  
- Cloud deployment on **AWS / GCP / Azure**.  
- Monitoring data drift & retraining pipeline.

---

## 🧑‍💻 Author

Built by 
**SaiKiran Tumeshpeta ** 
** Riya **
** Rohini **


🔗 Connect: [LinkedIn](https://www.linkedin.com/in/saikirantumeshpeta/) | [GitHub](https://github.com/SAIKIRAN-TUMESHPETA-DATASCIENTIST)