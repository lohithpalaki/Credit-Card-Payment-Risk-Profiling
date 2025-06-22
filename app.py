import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Dummy login credentials
auth_users = {"Risk_Admin": "Secure123"}

# Session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "Login"

# Sidebar Navigation
if st.session_state.logged_in:
    st.sidebar.title("Navigation")
    st.session_state.page = st.sidebar.selectbox(
        "Go to", ["Predict One", "Predict from CSV"],
        index=["Predict One", "Predict from CSV"].index(st.session_state.page)
        if st.session_state.page in ["Predict One", "Predict from CSV"] else 0
    )
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.page = "Login"
        st.rerun()
else:
    st.session_state.page = "Login"

# --------------------------
# 🔐 Login Page
# --------------------------
def login_page():
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in auth_users and auth_users[username] == password:
            st.session_state.logged_in = True
            st.session_state.page = "Predict One"
            st.success("Login successful!")
        else:
            st.error("Invalid credentials")

# --------------------------
# 🔍 Predict Single Entry
# --------------------------
def single_prediction_page():
    st.title("📌 Predict Missed Payment Risk")

    # Input fields (same features used in model training)
    Age = st.number_input("Age", 18, 100, 30)
    Gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    Account_Balance = st.number_input("Account Balance", 0.0, 1e7, 50000.0)
    Transaction_Amount = st.number_input("Transaction Amount", 0.0, 1e6, 10000.0)
    Account_Balance_After_Transaction = st.number_input("Balance After Transaction", 0.0, 1e7, 40000.0)
    Loan_Amount = st.number_input("Loan Amount", 0.0, 1e7, 200000.0)
    Interest_Rate = st.slider("Interest Rate (%)", 0.0, 25.0, 8.5)
    Loan_Term = st.number_input("Loan Term (months)", 1, 360, 60)
    Credit_Limit = st.number_input("Credit Limit", 1000.0, 1e6, 100000.0)
    Credit_Card_Balance = st.number_input("Credit Card Balance", 0.0, 1e6, 20000.0)
    Minimum_Payment_Due = st.number_input("Minimum Payment Due", 0.0, 1e6, 5000.0)

    if st.button("Predict Missed Payment"):
        # Feature Engineering
        Credit_Utilization = Credit_Card_Balance / Credit_Limit if Credit_Limit > 0 else 0.0

        # Create input DataFrame matching training structure
        input_data = pd.DataFrame([{
            'Age': Age,
            'Account_Balance': Account_Balance,
            'Transaction_Amount': Transaction_Amount,
            'Account_Balance_After_Transaction': Account_Balance_After_Transaction,
            'Loan_Amount': Loan_Amount,
            'Interest_Rate': Interest_Rate,
            'Loan_Term': Loan_Term,
            'Credit_Limit': Credit_Limit,
            'Credit_Card_Balance': Credit_Card_Balance,
            'Minimum_Payment_Due': Minimum_Payment_Due,
            'Credit_Utilization': Credit_Utilization
        }])

        # Load trained model
        model = joblib.load("lr_credit_model.pkl")
        prediction = model.predict(input_data)[0]

        label_map = {0: "❌ Likely to Miss Payment", 1: "✅ Unlikely to Miss Payment"}
        st.success(f"Prediction: {label_map[prediction]}")

# --------------------------
# 📂 Batch Prediction
# --------------------------
def batch_prediction_page():
    st.title("📂 Predict Missed Payment Risk from CSV")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("📄 Uploaded Data", df.head())

        # Required columns in uploaded CSV
        required_cols = [
            'Age', 'Account_Balance', 'Transaction_Amount', 'Account_Balance_After_Transaction',
            'Loan_Amount', 'Interest_Rate', 'Loan_Term',
            'Credit_Limit', 'Credit_Card_Balance', 'Minimum_Payment_Due'
        ]

        if all(col in df.columns for col in required_cols):
            # Feature engineering
            df['Credit_Utilization'] = df['Credit_Card_Balance'] / df['Credit_Limit'].replace(0, np.nan)
            df['Credit_Utilization'] = df['Credit_Utilization'].fillna(0)

            model = joblib.load("logistic_regression_model.pkl")
            preds = model.predict(df[required_cols + ['Credit_Utilization']])

            df['Missed_Payment_Prediction'] = ["❌ Likely to Miss" if p == 0 else "✅ Unlikely to Miss" for p in preds]
            st.write("📊 Predictions", df[['Missed_Payment_Prediction']])
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇ Download Results", csv, "missed_payment_predictions.csv", "text/csv")
        else:
            st.error("CSV missing required columns.")

# --------------------------
# App Routing
# --------------------------
if st.session_state.page == "Login":
    login_page()
elif st.session_state.page == "Predict One":
    single_prediction_page()
elif st.session_state.page == "Predict from CSV":
    batch_prediction_page()
