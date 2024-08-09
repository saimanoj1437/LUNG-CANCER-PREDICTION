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

# Define the exact feature names the model expects (12 features)
expected_features = ['AGE', 'GENDER', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 
                     'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE', 'ALLERGY', 
                     'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING']

# Get user input
def user_input_features():
    st.markdown("#### Gender: 0 = Female, 1 = Male")
    gender = st.selectbox("Gender", options=["M", "F"])

    st.markdown("#### Age")
    age = st.number_input("Age", min_value=0, max_value=100, step=1)

    st.markdown("#### Smoking: 1 = No, 2 = Yes")
    smoking = st.number_input("Smoking", min_value=1, max_value=2, step=1)

    st.markdown("#### Yellow Fingers: 1 = No, 2 = Yes")
    yellow_fingers = st.number_input("Yellow Fingers", min_value=1, max_value=2, step=1)

    st.markdown("#### Anxiety: 1 = No, 2 = Yes")
    anxiety = st.number_input("Anxiety", min_value=1, max_value=2, step=1)

    st.markdown("#### Peer Pressure: 1 = No, 2 = Yes")
    peer_pressure = st.number_input("Peer Pressure", min_value=1, max_value=2, step=1)

    st.markdown("#### Chronic Disease: 1 = No, 2 = Yes")
    chronic_disease = st.number_input("Chronic Disease", min_value=1, max_value=2, step=1)

    st.markdown("#### Fatigue: 1 = No, 2 = Yes")
    fatigue = st.number_input("Fatigue", min_value=1, max_value=2, step=1)

    st.markdown("#### Allergy: 1 = No, 2 = Yes")
    allergy = st.number_input("Allergy", min_value=1, max_value=2, step=1)

    st.markdown("#### Wheezing: 1 = No, 2 = Yes")
    wheezing = st.number_input("Wheezing", min_value=1, max_value=2, step=1)

    st.markdown("#### Alcohol Consuming: 1 = No, 2 = Yes")
    alcohol_consuming = st.number_input("Alcohol Consuming", min_value=1, max_value=2, step=1)

    st.markdown("#### Coughing: 1 = No, 2 = Yes")
    coughing = st.number_input("Coughing", min_value=1, max_value=2, step=1)

    # Map gender to a numerical value
    gender = 1 if gender == "M" else 0

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
        'COUGHING': coughing
    }

    input_df = pd.DataFrame(data, index=[0])

    # Reorder columns to match the order expected by the model
    input_df = input_df[expected_features]

    return input_df

# Get user input
input_df = user_input_features()

# Scale the input
scaled_input = sc.transform(input_df)

# Create a button to trigger prediction
if st.button('Predict'):
    prediction = loaded_model.predict(scaled_input)

    if prediction[0] == 0:
        st.write("YOU ARE NOT AFFECTED BY LUNG CANCER")
    else:
        st.write("YOU ARE AFFECTED BY LUNG CANCER")
        st.write("Consult the Doctor! Be Strong 💪")

st.sidebar.markdown(
    "Dataset Features\n\n"
    "- **AGE**: Age of the patient.\n"
    "- **GENDER**: Gender of the patient.\n"
    "- **SMOKING**: Smoking habit of the patient.\n"
    "- **YELLOW_FINGERS**: Presence of yellow fingers.\n"
    "- **ANXIETY**: Anxiety levels.\n"
    "- **PEER_PRESSURE**: Peer pressure experienced.\n"
    "- **CHRONIC DISEASE**: Presence of chronic disease.\n"
    "- **FATIGUE**: Fatigue experienced.\n"
    "- **ALLERGY**: Presence of allergies.\n"
    "- **WHEEZING**: Wheezing symptoms.\n"
    "- **ALCOHOL CONSUMING**: Alcohol consumption habit.\n"
    "- **COUGHING**: Coughing symptoms."
)

st.markdown("PROJECT BY : M.MANOJ BHASKAR")
