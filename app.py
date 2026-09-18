import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Set page config
st.set_page_config(page_title="Insurance Predictor", page_icon="🏥", layout="centered")

# Load trained linear regression model
@st.cache_resource
def load_model():
    return joblib.load('Insurance_model_ensemble_rf.pkl')

model = load_model()

st.title("🏥 Medical Insurance Cost Predictor")
st.write("Enter individual health and demographic details below to estimate annual insurance charges.")

# Form Layout
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0, step=1)
        
    with col2:
        gender = st.selectbox("Gender", options=["Male", "Female"])
        smoker = st.selectbox("Smoker?", options=["No", "Yes"])
        region = st.selectbox("Region", options=["Southeast", "Other (Northwest, Northeast, Southwest)"])

    submitted = st.form_submit_button("Calculate Estimated Cost")

if submitted:
    # 1. Map UI inputs to match your model's exact features
    is_smoker = 1 if smoker == "Yes" else 0
    is_female = 1 if gender == "Female" else 0
    region_southeast = 1 if region == "Southeast" else 0
    bmi_category_obese = 1 if bmi >= 30.0 else 0

    # 2. Structure as DataFrame with exact feature names
    input_data = pd.DataFrame([{
        'age': age,
        'bmi': bmi,
        'children': children,
        'is_smoker': is_smoker,
        'region_southeast': region_southeast,
        'is_female': is_female,
        'bmi_category_obese': bmi_category_obese
    }])

    # 3. Generate prediction
    prediction = model.predict(input_data)[0]
    
    # Floor at 0 in case linear regression outputs negative values
    final_cost = max(0.0, float(prediction))
    
    st.success(f"### Estimated Insurance Charges: **Rs. {final_cost:,.2f}**")