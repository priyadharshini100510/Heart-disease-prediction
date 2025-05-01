import streamlit as st
import pickle
import pandas as pd

# Load the trained model
with open('c:/Users/sabar/Desktop/Disease predicition/Heart-disease-prediction/ensemble_model.pkl', 'rb') as model_file:
    ensemble_model = pickle.load(model_file)

# Streamlit app
st.title("Heart Disease Prediction")

# Input fields for prediction
age = st.number_input("Age", min_value=0)
cholesterol = st.number_input("Cholesterol", min_value=0)
bp = st.number_input("Blood Pressure", min_value=0)
sex_female = st.radio("Sex", options=["Female", "Male"]) == "Female"
diabetes_yes = st.radio("Diabetes", options=["Yes", "No"]) == "Yes"

# Create input DataFrame for prediction
input_data = pd.DataFrame({
    "age": [age],
    "cholesterol": [cholesterol],
    "bp": [bp],
    "sex_Female": [int(sex_female)],
    "sex_Male": [int(not sex_female)],
    "diabetes_No": [int(not diabetes_yes)],
    "diabetes_Yes": [int(diabetes_yes)],
    "age_cholesterol": [age * cholesterol],
    "bp_cholesterol": [bp * cholesterol]
})

# Predict button
if st.button("Predict Heart Disease Risk"):
    prediction = ensemble_model.predict(input_data)
    prediction_prob = ensemble_model.predict_proba(input_data)[0][1] * 100
    st.write(f"Prediction: {'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'}")
    st.write(f"Heart Disease Risk Rate: {prediction_prob:.2f}%")