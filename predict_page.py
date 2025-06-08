import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open("rf_model.sav", "rb") as f:
    model = pickle.load(f)

# Load encoded training data (used to extract column names)
df_encoded = pd.read_csv("encoded_dataframe.csv")
X = df_encoded.drop("Churn", axis=1)

st.title("📊 Telco Customer Churn Predictor")
st.write("Enter customer information to predict whether they are likely to churn.")

with st.form("churn_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=500.0, step=1.0)
    TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=20000.0, step=1.0)
    tenure_group = st.selectbox("Tenure Group", ["1-12", "13-24", "25-36", "37-48", "49-60", "61-72"])

    submitted = st.form_submit_button("Predict")

if submitted:
    user_input = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
        "tenure_group": tenure_group
    }])

    user_encoded = pd.get_dummies(user_input)
    user_final = user_encoded.reindex(columns=X.columns, fill_value=0)

    prediction = model.predict(user_final)[0]
    prob = model.predict_proba(user_final)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High risk of churn! Confidence: {prob * 100:.2f}%")
    else:
        st.success(f"✅ Likely to stay. Confidence: {(1 - prob) * 100:.2f}%")
