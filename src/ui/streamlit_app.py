import streamlit as st
import requests
import pandas as pd
import sys, os
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.utils import load_object

# ---------------- Page Setup ----------------
API_URL = "https://customer-churn-project-2.onrender.com"  # Docker Compose service name
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("""
Welcome! 👋  
This tool predicts whether a **customer will churn** based on features like contract, charges, payment method, etc.

You can either:  
1. 📂 Upload a CSV file with customer data  
2. ✍️ Enter details manually  

The model returns:  
- ✅ **Prediction** → Yes / No  
- 📈 **Probability** → Confidence of the prediction
""")

# ---------------- Load Selected Features ----------------
selected_features = load_object("selected_features.joblib")

# ---------------- Feature Mapping ----------------
feature_mapping = {
    "TotalCharges_per_Month": ("💰 Monthly Charges", "Total monthly bill"),
    "Is_MonthToMonth_Encoded": ("📅 Month-to-Month Contract (Encoded)", ""),
    "Is_MonthToMonth": ("📅 Month-to-Month Contract", ""),
    "PaymentMethod_Electronic check": ("💳 Payment: Electronic Check", ""),
    "Is_FiberOptic": ("🌐 Fiber Optic Internet", ""),
    "Contract_Two year": ("📑 Two-Year Contract", ""),
    "Satisfaction_Score": ("⭐ Satisfaction Score", ""),
    "tenure-binned_Low": ("⏳ Short Tenure (Binned)", ""),
    "OnlineBackup_Yes": ("☁️ Online Backup", ""),
    "Services_Count": ("🛠️ Services Count", ""),
    "SeniorCitizen_Encoded": ("👵 Senior Citizen", ""),
    "Tenure_Low": ("⏱️ Short Tenure", ""),
    "OnlineSecurity_Yes": ("🔒 Online Security", ""),
    "Tenure_Low_Encoded": ("⏱️ Short Tenure (Encoded)", ""),
    "PaymentMethod_Credit card (automatic)": ("💳 Auto-Pay by Credit Card", "")
}

# ---------------- Helper Function ----------------
def safe_post(url, payload, retries=5):
    """Send data to FastAPI backend safely with retries"""
    for i in range(retries):
        try:
            r = requests.post(url, json=payload, timeout=10)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            st.warning(f"API request failed: {e}. Retrying ({i+1}/{retries})...")
            time.sleep(3)
    st.error("API not responding after multiple attempts.")
    return None

# ---------------- CSV Upload ----------------
uploaded = st.file_uploader("📂 Upload customer data (CSV with same columns)", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.write("👀 Preview of uploaded data:")
    st.dataframe(df.head())

    if st.button("🚀 Predict for Uploaded Data"):
        # Fill missing features with 0
        payload_data = [
            {feat: row.get(feat, 0) for feat in selected_features}
            for _, row in df.iterrows()
        ]
        payload = {"data": payload_data}
        results = safe_post(f"{API_URL}/predict", payload)

        if results:
            st.subheader("📌 Predictions")
            pred_df = pd.DataFrame(results)
            if "prediction" in pred_df.columns and "probability" in pred_df.columns:
                pred_df = pd.DataFrame({
                    "Prediction": pred_df["prediction"].map(lambda x: "Yes (Churn)" if x == 1 else "No"),
                    "Probability (%)": (pred_df["probability"] * 100).round(2)
                })
            st.dataframe(pred_df)

# ---------------- Manual Input ----------------
else:
    st.sidebar.header("✍️ Enter Customer Details")

    manual_input = {}
    for feat in selected_features:
        label, desc = feature_mapping.get(feat, (feat, ""))
        val = st.sidebar.text_input(label, help=desc)
        try:
            manual_input[feat] = float(val) if val != "" else 0
        except ValueError:
            manual_input[feat] = val

    if st.sidebar.button("🚀 Predict Manually"):
        payload = {"data": [manual_input]}  # Single row wrapped in list
        res = safe_post(f"{API_URL}/predict", payload)

        if res:
            pred = res[0].get("prediction")
            prob = res[0].get("probability")
            if pred is not None and prob is not None:
                label = "Yes (Churn)" if pred == 1 else "No"
                st.success(f"**Prediction:** {label}")
                st.info(f"**Probability of Churn:** {round(prob*100,2)} %")
            else:
                st.json(res[0])
