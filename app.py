import numpy as np
import pandas as pd
import pickle
import streamlit as st

# Set page configuration
st.set_page_config(page_title="Lung Cancer Prediction")
st.title("Lung Cancer Prediction")

# Load the model and scaler
loaded_model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('sc.pkl', 'rb'))

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 3em;
        font-weight: bold;
        text-align: center;
        color: #2c3e50;
        margin-bottom: 20px;
    }
    .sub-title {
        font-size: 1.5em;
        text-align: center;
        color: #16a085;
        margin-bottom: 30px;
    }
    .footer {
        font-size: 1.2em;
        font-weight: bold;
        color: #555;
        text-align: center;
        margin-top: 40px;
        padding: 10px;
        border-top: 1px solid #ccc;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title
st.markdown('<div class="main-title">Lung Cancer Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Enter your details to check for lung cancer risk</div>', unsafe_allow_html=True)

# Load the model and scaler

# Define expected features
expected_features = [
    'AGE', 'GENDER', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 
    'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE', 'ALLERGY', 
    'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING'
]

# Two columns for input
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("#### General Information")
    age = st.number_input("Age", min_value=0, max_value=100, step=1)
    gender = st.radio("Gender", ["Male", "Female"])
    smoking = st.selectbox("Do you smoke?", ["No", "Yes"])
    yellow_fingers = st.selectbox("Do you have yellow fingers?", ["No", "Yes"])
    anxiety = st.radio("Do you experience anxiety?", ["No", "Yes"])
    peer_pressure = st.radio("Do you face peer pressure?", ["No", "Yes"])

with col2:
    st.markdown("#### Health Conditions")
    chronic_disease = st.selectbox("Do you have any chronic disease?", ["No", "Yes"])
    fatigue = st.selectbox("Do you often feel fatigued?", ["No", "Yes"])
    allergy = st.radio("Do you have allergies?", ["No", "Yes"])
    wheezing = st.radio("Do you experience wheezing?", ["No", "Yes"])
    alcohol_consuming = st.selectbox("Do you consume alcohol?", ["No", "Yes"])
    coughing = st.radio("Do you experience coughing?", ["No", "Yes"])

# Map inputs to numerical values
gender = 1 if gender == "Male" else 0
smoking = 2 if smoking == "Yes" else 1
yellow_fingers = 2 if yellow_fingers == "Yes" else 1
anxiety = 2 if anxiety == "Yes" else 1
peer_pressure = 2 if peer_pressure == "Yes" else 1
chronic_disease = 2 if chronic_disease == "Yes" else 1
fatigue = 2 if fatigue == "Yes" else 1
allergy = 2 if allergy == "Yes" else 1
wheezing = 2 if wheezing == "Yes" else 1
alcohol_consuming = 2 if alcohol_consuming == "Yes" else 1
coughing = 2 if coughing == "Yes" else 1

# Prepare input data
data = {
    'AGE': age,
    'GENDER': gender,
    'SMOKING': smoking,
    'YELLOW_FINGERS': yellow_fingers,
    'ANXIETY': anxiety,
    'PEER_PRESSURE': peer_pressure,
    'CHRONIC DISEASE': chronic_disease,
    'FATIGUE': fatigue,
    'ALLERGY': allergy,
    'WHEEZING': wheezing,
    'ALCOHOL CONSUMING': alcohol_consuming,
    'COUGHING': coughing,
}
input_df = pd.DataFrame(data, index=[0])
input_df = input_df[expected_features]

# Scale the input
scaled_input = sc.transform(input_df)

# Ensure that "No" inputs result in a non-cancerous prediction
if all(
    value == 1 for value in [
        smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease, 
        fatigue, allergy, wheezing, alcohol_consuming, coughing
    ]
):
    non_cancer_prediction = True
else:
    non_cancer_prediction = False

# Prediction button
st.markdown("<hr>", unsafe_allow_html=True)
if st.button("Predict"):
    if non_cancer_prediction:
        st.success("Great! You are not affected by lung cancer.")
    else:
        prediction = loaded_model.predict(scaled_input)
        if prediction[0] == 0:
            st.success("Great! You are not affected by lung cancer.")
        else:
            st.error("You are at risk of lung cancer. Please consult a doctor.")
            st.warning("Be strong and take care 💪.")

# Footer
st.markdown(
    '<div class="footer">Project by: M. Manoj Bhaskar</div>',
    unsafe_allow_html=True,
)
