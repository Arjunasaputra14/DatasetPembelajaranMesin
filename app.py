
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Define the path to the joblib files
model_path = 'best_model.joblib'
scaler_path = 'scaler.joblib'

# Load the saved model and scaler
try:
    best_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except FileNotFoundError:
    st.error(f"Error: Model or scaler file not found. Please ensure '{model_path}' and '{scaler_path}' are in the same directory as the app.")
    st.stop()

# Set up the Streamlit application
st.title("Stroke Prediction App")
st.write("Enter the patient's details to predict the likelihood of stroke.")

# Create input fields for each feature
age = st.number_input("Age", min_value=0.08, max_value=120.0, value=45.0)
hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
ever_married = st.selectbox("Ever Married", ['Yes', 'No'])
work_type = st.selectbox("Work Type", ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
Residence_type = st.selectbox("Residence Type", ['Urban', 'Rural'])
avg_glucose_level = st.number_input("Average Glucose Level", min_value=50.0, max_value=300.0, value=100.0)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
smoking_status = st.selectbox("Smoking Status", ['never smoked', 'Unknown', 'formerly smoked', 'smokes'])

# Map categorical inputs to the format used during training (one-hot encoding)
gender_Male = st.selectbox("Gender", ['Female', 'Male']) == 'Male' # Assuming 'Female' is the reference category

# Create a dictionary from the input values
input_data = {
    'age': age,
    'hypertension': hypertension,
    'heart_disease': heart_disease,
    'bmi': bmi,
    'avg_glucose_level_log': np.log1p(avg_glucose_level), # Apply log transform
    'gender_Male': gender_Male,
    'ever_married_Yes': ever_married == 'Yes',
    'work_type_Never_worked': work_type == 'Never_worked',
    'work_type_Private': work_type == 'Private',
    'work_type_Self-employed': work_type == 'Self-employed',
    'work_type_children': work_type == 'children',
    'Residence_type_Urban': Residence_type == 'Urban',
    'smoking_status_formerly smoked': smoking_status == 'formerly smoked',
    'smoking_status_never smoked': smoking_status == 'never smoked',
    'smoking_status_smokes': smoking_status == 'smokes'
}

# Create a DataFrame from the input data
input_df = pd.DataFrame([input_data])

# Ensure the order of columns in the input DataFrame matches the training data
# You might need to load the training data column order or store it separately
# For this example, we'll assume a specific order based on the notebook's preprocessing
expected_columns = ['age', 'hypertension', 'heart_disease', 'bmi', 'avg_glucose_level_log',
                    'gender_Male', 'ever_married_Yes', 'work_type_Never_worked', 'work_type_Private',
                    'work_type_Self-employed', 'work_type_children', 'Residence_type_Urban',
                    'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes']

# Reindex the input_df to match the expected column order, adding missing columns with 0
input_df = input_df.reindex(columns=expected_columns, fill_value=0)


# Create a predict button
if st.button("Predict"):
    # Apply the scaler to the input data
    input_scaled = scaler.transform(input_df)

    # Make a prediction
    prediction = best_model.predict(input_scaled)

    # Display the prediction result
    if prediction[0] == 1:
        st.error("Prediction: High likelihood of stroke")
    else:
        st.success("Prediction: Low likelihood of stroke")
