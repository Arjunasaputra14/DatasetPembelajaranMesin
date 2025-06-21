import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained XGBoost model
try:
    model = joblib.load('model_stroke.pkl')
except FileNotFoundError:
    st.error("Model file 'model_stroke.pkl' not found. Please make sure the model is saved in the same directory.")
    st.stop()

# Define the columns used during training (excluding 'stroke' and 'id')
# This list should match the columns of X_train used during training
# Make sure to include all possible dummy variables
trained_columns = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
                  'gender_Male', 'gender_Other', 'ever_married_Yes',
                  'work_type_Never_worked', 'work_type_Private', 'work_type_Self-employed',
                  'work_type_children', 'Residence_type_Urban',
                  'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes']

# Streamlit App Title and Description
st.title("Stroke Disease Prediction")
st.write("Enter the patient's information to predict the likelihood of stroke.")

# Create input fields for features
st.header("Patient Information")

age = st.number_input("Age", min_value=0, max_value=120, value=30)
hypertension = st.selectbox("Hypertension", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
heart_disease = st.selectbox("Heart Disease", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
ever_married = st.selectbox("Ever Married", options=['Yes', 'No'])
work_type = st.selectbox("Work Type", options=['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
residence_type = st.selectbox("Residence Type", options=['Urban', 'Rural'])
avg_glucose_level = st.number_input("Average Glucose Level", min_value=50.0, max_value=300.0, value=100.0)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
smoking_status = st.selectbox("Smoking Status", options=['never smoked', 'formerly smoked', 'smokes', 'Unknown'])

# Create a dictionary with input values
input_data = {
    'age': age,
    'hypertension': hypertension,
    'heart_disease': heart_disease,
    'avg_glucose_level': avg_glucose_level,
    'bmi': bmi,
    'gender': 'Male', # Default gender, will be handled by one-hot encoding
    'ever_married': ever_married,
    'work_type': work_type,
    'Residence_type': residence_type,
    'smoking_status': smoking_status
}

# Add a selectbox for gender after the initial input fields
gender = st.selectbox("Gender", options=['Male', 'Female', 'Other'])
input_data['gender'] = gender


# Preprocess the input data
def preprocess_input(data, trained_cols):
    df_input = pd.DataFrame([data])

    # Handle categorical features using one-hot encoding
    categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    df_input = pd.get_dummies(df_input, columns=categorical_cols, drop_first=True)

    # Ensure all columns from training are present, add missing ones with 0
    for col in trained_cols:
        if col not in df_input.columns:
            df_input[col] = 0

    # Ensure the order of columns is the same as during training and drop extra columns
    df_input = df_input[trained_cols]

    return df_input

# Prediction button
if st.button("Predict Stroke Risk"):
    processed_input = preprocess_input(input_data, trained_columns)

    # Make prediction
    prediction = model.predict(processed_input)
    prediction_proba = model.predict_proba(processed_input)[:, 1] # Probability of stroke

    st.header("Prediction Result")
    if prediction[0] == 1:
        st.error(f"Based on the provided information, the model predicts a HIGH risk of stroke.")
        st.write(f"Probability of stroke: **{prediction_proba[0]:.2f}**")
    else:
        st.success(f"Based on the provided information, the model predicts a LOW risk of stroke.")
        st.write(f"Probability of stroke: **{prediction_proba[0]:.2f}**")

    st.write("---")
    st.write("Disclaimer: This is a prediction based on a machine learning model and should not be considered as medical advice. Consult a healthcare professional for any health concerns.")

st.sidebar.header("About the Model")
st.sidebar.write("This application uses an XGBoost model trained on a dataset of patient health information to predict the likelihood of stroke.")
st.sidebar.write("The model was trained to handle class imbalance using `scale_pos_weight`.")
st.sidebar.write("The features used for prediction are:")
st.sidebar.write("- Age")
st.sidebar.write("- Hypertension")
st.sidebar.write("- Heart Disease")
st.sidebar.write("- Ever Married")
st.sidebar.write("- Work Type")
st.sidebar.write("- Residence Type")
st.sidebar.write("- Average Glucose Level")
st.sidebar.write("- BMI")
st.sidebar.write("- Smoking Status")
st.sidebar.write("- Gender")
