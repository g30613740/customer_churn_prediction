import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page title and layout
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("Customer Churn Predictor")
st.markdown("Enter customer details below to predict the probability of churn.")

# Load the saved model, scaler, and training columns
@st.cache_resource
def load_artifacts():
    model = joblib.load('churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    columns = joblib.load('training_columns.pkl')
    return model, scaler, columns

model, scaler, training_columns = load_artifacts()

# ----- Feature Engineering Functions (must match training) -----
def engineer_features(df):
    """Apply the same feature engineering as in training."""
    df_new = df.copy()
    
    # 1. Average Monthly Spend
    df_new['Avg_Monthly_Spend'] = df_new['TotalCharges'].div(df_new['tenure']).replace([float('inf'), -float('inf')], 0).fillna(0)
    
    # 2. Tenure Group (categorical)
    bins = [0, 6, 12, 24, 72]
    labels = ['New (0-6m)', 'Recent (7-12m)', 'Regular (1-2y)', 'Loyal (2y+)']
    df_new['Tenure_Group'] = pd.cut(df_new['tenure'], bins=bins, labels=labels, right=False)
    
    # 3. Service Count
    service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
                    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df_new['Service_Count'] = df_new[service_cols].apply(lambda row: (row != 'No').sum(), axis=1)
    
    # 4. Contract Tenure Ratio
    contract_mapping = {'Month-to-month': 1, 'One year': 12, 'Two year': 24}
    df_new['Contract_Tenure_Ratio'] = df_new['tenure'] / df_new['Contract'].map(contract_mapping)
    
    # Drop columns that are not needed after engineering (if any)
    # We'll keep the raw columns because they are used in encoding.
    return df_new

def preprocess_input(df_raw):
    """
    Takes raw user input (a DataFrame with raw features) and returns
    a DataFrame ready for prediction (encoded, scaled, aligned).
    """
    # 1. Fix TotalCharges (convert to numeric, fill NaN)
    df_raw['TotalCharges'] = pd.to_numeric(df_raw['TotalCharges'], errors='coerce')
    df_raw['TotalCharges'] = df_raw['TotalCharges'].fillna(0)
    
    # 2. Apply feature engineering
    df_eng = engineer_features(df_raw)
    
    # 3. One-hot encode categorical columns (including Tenure_Group)
    cat_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                'PaperlessBilling', 'PaymentMethod', 'Tenure_Group']
    
    # We need to reindex to match training columns; we'll do one-hot then reindex.
    df_encoded = pd.get_dummies(df_eng, columns=cat_cols, drop_first=True)
    
    # 4. Reindex to match training columns (fill missing with 0)
    df_aligned = df_encoded.reindex(columns=training_columns, fill_value=0)
    
    # 5. Scale numerical features (same scaler as training)
    num_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
                'Avg_Monthly_Spend', 'Service_Count', 'Contract_Tenure_Ratio']
    df_scaled = df_aligned.copy()
    df_scaled[num_cols] = scaler.transform(df_aligned[num_cols])
    
    return df_scaled

# ----- Sidebar: Input Widgets -----
st.sidebar.header("Customer Details")
with st.sidebar.form("input_form"):
    # We'll create inputs for all features.
    # For simplicity, we group them.
    
    # Demographics
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    
    # Account info
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method = st.selectbox("Payment Method", 
                                  ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=70.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=500.0)
    
    # Services
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    
    submit_button = st.form_submit_button(label="Predict Churn")

# ----- Prediction Logic -----
if submit_button:
    # Build a DataFrame from the input
    input_data = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [senior_citizen],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'PhoneService': [phone_service],
        'MultipleLines': [multiple_lines],
        'InternetService': [internet_service],
        'OnlineSecurity': [online_security],
        'OnlineBackup': [online_backup],
        'DeviceProtection': [device_protection],
        'TechSupport': [tech_support],
        'StreamingTV': [streaming_tv],
        'StreamingMovies': [streaming_movies],
        'Contract': [contract],
        'PaperlessBilling': [paperless_billing],
        'PaymentMethod': [payment_method],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    })
    
    # Preprocess the input
    preprocessed = preprocess_input(input_data)
    
    # Predict probability
    prob = model.predict_proba(preprocessed)[0, 1]  # probability of churn
    prediction = "Churn" if prob >= 0.5 else "No Churn"
    
    # Display results
    st.subheader("Prediction Result")
    col1, col2 = st.columns(2)
    col1.metric("Prediction", prediction)
    col2.metric("Churn Probability", f"{prob*100:.1f}%")
    
    # Optional: Gauge or progress bar
    st.progress(prob)
    
    # Interpretation
    if prob >= 0.5:
        st.warning("This customer is at high risk of churn. Consider retention actions.")
    else:
        st.success("This customer is likely to stay. Keep up the good service!")
else:
    st.info("Fill in the details and click 'Predict Churn' to see the result.")