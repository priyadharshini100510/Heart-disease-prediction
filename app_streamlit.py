import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go

# Load the trained model
with open('ensemble_model.pkl', 'rb') as model_file:
    ensemble_model = pickle.load(model_file)

# Streamlit app
st.title("Heart Disease Prediction")

# Tabs for different sections
tab1, tab2, tab3 = st.tabs(["CSV Guide", "CSV Prediction", "Single Prediction"])

with tab1:
    # CSV Format Guide
    st.markdown("""
    ### CSV Format Guide
    The CSV file should have the following columns:
    - `age`
    - `cholesterol`
    - `bp`
    - `sex`
    - `diabetes`
    - `heart_disease`

    #### Cholesterol Levels
    - **Normal**: Less than 200 mg/dL
    - **Borderline High**: 200-239 mg/dL
    - **High**: 240 mg/dL and above

    #### Blood Pressure Levels
    - **Normal**: Less than 120/80 mmHg
    - **Elevated**: 120-129/<80 mmHg
    - **High (Hypertension Stage 1)**: 130-139/80-89 mmHg
    - **High (Hypertension Stage 2)**: 140 and above/90 and above mmHg
    - **Hypertensive Crisis**: Higher than 180/120 mmHg
    """)
    
    # CSV Template Download
    csv_template = pd.DataFrame({
        "age": [0],
        "cholesterol": [0],
        "bp": [0],
        "sex": ["Male"],
        "diabetes": ["No"],
        "heart_disease": [0]
    })
    st.download_button("Download CSV Template", csv_template.to_csv(index=False), "template.csv", "text/csv")

with tab2:
    # CSV File Upload
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        # Preprocess data: Encode categorical columns
        data['sex_Female'] = (data['sex'] == 'Female').astype(int)
        data['sex_Male'] = (data['sex'] == 'Male').astype(int)
        data['diabetes_No'] = (data['diabetes'] == 'No').astype(int)
        data['diabetes_Yes'] = (data['diabetes'] == 'Yes').astype(int)
        
        # Feature engineering: create interaction features
        data['age_cholesterol'] = data['age'] * data['cholesterol']
        data['bp_cholesterol'] = data['bp'] * data['cholesterol']
        
        # Drop unnecessary columns
        data = data.drop(columns=['sex', 'diabetes', 'heart_disease'])
        
        st.write("Uploaded CSV Data:")
        st.write(data)
        
        # Predict using the ensemble model
        predictions = ensemble_model.predict(data)
        prediction_probs = ensemble_model.predict_proba(data)[:, 1] * 100
        data['Prediction'] = ['Heart Disease' if pred == 1 else 'No Heart Disease' for pred in predictions]
        data['Risk Rate (%)'] = prediction_probs

        # Define risk categories
        data['Risk Category'] = pd.cut(data['Risk Rate (%)'], bins=[0, 30, 70, 100], labels=['Low', 'Moderate', 'High'])

        st.write("Predictions:")
        st.write(data)

        # Visualize risk categories with a histogram
        st.subheader("Risk Category Distribution")
        fig = go.Figure(data=[go.Histogram(x=data['Risk Category'])])
        fig.update_layout(title='Risk Category Distribution', xaxis_title='Risk Category', yaxis_title='Count')
        st.plotly_chart(fig)

        # Separate high-risk, moderate-risk, and low-risk patients
        high_risk = data[data['Risk Category'] == 'High']
        moderate_risk = data[data['Risk Category'] == 'Moderate']
        low_risk = data[data['Risk Category'] == 'Low']

        # Download buttons for high-risk, moderate-risk, and low-risk patients
        st.download_button("Download High Risk Patients", high_risk.to_csv(index=False), "high_risk.csv", "text/csv")
        st.download_button("Download Moderate Risk Patients", moderate_risk.to_csv(index=False), "moderate_risk.csv", "text/csv")
        st.download_button("Download Low Risk Patients", low_risk.to_csv(index=False), "low_risk.csv", "text/csv")

with tab3:
    st.header("Single User Prediction")

    # Input fields for single prediction
    st.subheader("Enter Patient Details")
    age = st.number_input("Age", min_value=0)
    cholesterol = st.number_input("Cholesterol", min_value=0)
    bp = st.number_input("Blood Pressure", min_value=0)
    sex_female = st.radio("Sex", options=["Female", "Male"]) == "Female"
    diabetes_yes = st.radio("Diabetes", options=["Yes", "No"]) == "Yes"

    # Create input DataFrame for single prediction
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

    # Predict button for single prediction
    if st.button("Predict Heart Disease Risk"):
        st.subheader("Prediction Results")
        prediction = ensemble_model.predict(input_data)
        prediction_prob = ensemble_model.predict_proba(input_data)[0][1] * 100  # Probability of heart disease
        st.write(f"Prediction: {'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'}")
        st.write(f"Heart Disease Risk Rate: {prediction_prob:.2f}%")

        # Speedometer effect using Plotly
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction_prob,
            title={'text': "Risk Rate"},
            gauge={'axis': {'range': [0, 100]}}
        ))
        st.plotly_chart(fig)