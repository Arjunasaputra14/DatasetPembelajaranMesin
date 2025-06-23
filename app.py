import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.title("Stroke Prediction Web Application")

# Load the saved model and scaler
try:
    best_model = joblib.load('kneighbors_model.pkl')
    scaler = joblib.load('scaler.pkl')
    st.write("Model and scaler loaded successfully.")
except FileNotFoundError:
    st.error("Error loading model or scaler. Make sure 'kneighbors_model.pkl' and 'scaler.pkl' are in the same directory.")
    st.stop()


st.header("Enter Patient Information:")

# Create input fields for each feature
age = st.number_input("Age", min_value=0.08, max_value=100.0, value=40.0, step=0.1)
gender = st.selectbox("Gender", ['Female', 'Male'])
hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
ever_married = st.selectbox("Ever Married", ['No', 'Yes'])
work_type = st.selectbox("Work Type", ['children', 'Govt_job', 'Never_worked', 'Private', 'Self-employed'])
Residence_type = st.selectbox("Residence Type", ['Rural', 'Urban'])
avg_glucose_level = st.number_input("Average Glucose Level", min_value=50.0, max_value=300.0, value=100.0, step=0.1)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
smoking_status = st.selectbox("Smoking Status", ['Unknown', 'formerly smoked', 'never smoked', 'smokes'])

if st.button("Predict Stroke"):
    # Create a dictionary from user inputs
    user_input = {
        'age': age,
        'gender': gender,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': ever_married,
        'work_type': work_type,
        'Residence_type': Residence_type,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': smoking_status
    }

    # Convert dictionary to DataFrame
    input_df = pd.DataFrame([user_input])

    # Apply the same one-hot encoding as during training
    # We need to make sure all categories from the training data are present
    # to avoid shape mismatch issues during prediction.
    # Create dummy variables for categorical columns
    categorical_cols_train = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

    # Apply one-hot encoding to the input DataFrame
    input_encoded = pd.get_dummies(input_df, columns=categorical_cols_train, drop_first=True)

    # Ensure all training columns are present and in the same order
    # Get the list of columns from the training data (excluding the target variable 'stroke')
    # Access X_resampled from the notebook's global scope
    train_cols = X_resampled.columns

    # Add missing columns to the input DataFrame and set their values to 0
    for col in train_cols:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    # Reindex the input DataFrame to match the order of training columns
    input_processed = input_encoded[train_cols]

    # Apply the same log transformation to avg_glucose_level if it was applied during training
    if 'avg_glucose_level_log' in train_cols and 'avg_glucose_level' in user_input:
        # Check if the original 'avg_glucose_level' column exists in the processed input
        if 'avg_glucose_level' in input_processed.columns:
            # Log transform and create the log-transformed column
            input_processed['avg_glucose_level_log'] = np.log1p(input_processed['avg_glucose_level'])
            # Drop the original 'avg_glucose_level' column
            input_processed = input_processed.drop('avg_glucose_level', axis=1)
            # Reindex again to ensure 'avg_glucose_level_log' is in the correct position
            input_processed = input_processed.reindex(columns=train_cols, fill_value=0)


    # Scale the numerical features
    # Identify numerical columns from the training data that are not dummy variables
    # Access X_resampled from the notebook's global scope
    numerical_cols_train = X_resampled.select_dtypes(include=np.number).columns.tolist()

    # Remove dummy variables from the list of numerical columns to be scaled
    dummy_cols = [col for col in input_processed.columns if '_' in col]
    numerical_cols_to_scale = [col for col in numerical_cols_train if col not in dummy_cols and col in input_processed.columns]

    # Scale only the identified numerical columns
    input_processed[numerical_cols_to_scale] = scaler.transform(input_processed[numerical_cols_to_scale])

    # Make prediction
    prediction = best_model.predict(input_processed)
    prediction_proba = best_model.predict_proba(input_processed)[:, 1] # Probability of stroke

    st.write("Processed Input:")
    st.write(input_processed)

    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.error(f"Based on the input data, the model predicts a HIGH risk of stroke.")
    else:
        st.success(f"Based on the input data, the model predicts a LOW risk of stroke.")

    st.write(f"Probability of Stroke: {prediction_proba[0]:.2f}")
