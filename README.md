# Customer Churn Prediction 📉

## What is Customer Churn?

Customer churn refers to the phenomenon where customers stop using a company's products or services. In industries such as telecom, churn directly affects revenue and growth. Predicting churn in advance enables businesses to take proactive measures like offering discounts or improving service quality so as to retain at-risk customers.

This project uses supervised machine learning techniques to predict whether a customer is likely to churn, based on their account information, service usage, and demographics.

---

## 📊 Project Overview

In this project, I used the **Telco Customer Churn** dataset to build and evaluate machine learning models that can predict churn with good accuracy. This involves:

- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Building (Logistic Regression & Random Forest)
- Model Evaluation
- Feature Importance Interpretation

---

## 🔍 Dataset

- Source: [Kaggle - Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
- Rows: ~7,000 customers  
- Features: 21 columns including:
  - Demographics (e.g., gender, senior citizen)
  - Account info (e.g., tenure, contract type, payment method)
  - Services used (e.g., internet service, phone service)
  - Target variable: `Churn` (Yes/No)

---

## 🛠️ Tech Stack

- **Python**
- **Jupyter Notebook**
- **Pandas & NumPy** – Data manipulation
- **Matplotlib & Seaborn** – Visualization
- **Scikit-learn** – Model building
- **Missingno** – Visualizing missing data
- **SMOTEENN** – Handling class imbalance

---

## 📂 Project Structure

```bash
Customer-Churn-Prediction/
│
├── Customer-Churn-Analysis.ipynb       # Main notebook
├── encoded_dataframe.csv               # Processed dataset (one-hot encoded)
├── rf_model.sav                        # Trained Random Forest model
├── requirements.txt                    # Dependencies
├── predict_page.py                     # Streamlit interface (optional)
└── README.md                           # Project overview
