import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.title("Stroke Prediction Web Application")

# Load the saved model and scaler
try:
    best_model = joblib.load('kneighbors_model.pkl')
    scaler = joblib.load('scaler.pkl')
    st.success("Model and scaler loaded successfully.")
except FileNotFoundError:
    st.error("Error loading model or scaler. Make sure 'kneighbors_model.pkl' and 'scaler.pkl' are in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred while loading model or scaler: {e}")
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
    try:
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
        categorical_cols_train = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

        # Apply one-hot encoding
        input_encoded = pd.get_dummies(input_df, columns=categorical_cols_train, drop_first=True)

        # Manually apply log transformation to avg_glucose_level
        # We do this BEFORE aligning columns to ensure 'avg_glucose_level_log' is created correctly
        input_encoded['avg_glucose_level_log'] = np.log1p(input_encoded['avg_glucose_level'])
        # Drop the original avg_glucose_level column
        input_encoded = input_encoded.drop('avg_glucose_level', axis=1)


        # Define the exact column names and their order as in the training data (X_resampled)
        # This list must be exactly the same as the columns in X_resampled after all preprocessing in the notebook
        train_cols = [
            'age', 'hypertension', 'heart_disease', 'bmi', 'avg_glucose_level_log',
            'gender_Male',
            'ever_married_Yes',
            'work_type_Never_worked', 'work_type_Private', 'work_type_Self-employed', 'work_type_children',
            'Residence_type_Urban',
            'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes'
        ]

        # Add missing columns to the input DataFrame and set their values to 0
        for col in train_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0

        # Reindex the input DataFrame to match the order of training columns
        input_processed = input_encoded[train_cols]

        st.write("Columns in processed input before scaling/prediction:") # Debugging line
        st.write(input_processed.columns.tolist()) # Debugging line
        st.write("Expected columns from training:") # Debugging line
        st.write(train_cols) # Debugging line


        # Scale the numerical features - ensure only numerical columns are scaled
        # Identify numerical columns explicitly
        numerical_cols_to_scale = ['age', 'bmi', 'avg_glucose_level_log']

        # Ensure that the numerical columns to scale actually exist in the processed input
        cols_to_scale_present = [col for col in numerical_cols_to_scale if col in input_processed.columns]

        if cols_to_scale_present:
             st.write(f"Scaling columns: {cols_to_scale_present}") # Debugging line
             input_processed[cols_to_scale_present] = scaler.transform(input_processed[cols_to_scale_present])
             st.write("Scaling applied.") # Debugging line
        else:
             st.warning("None of the expected numerical columns for scaling were found in the processed input.")


        # Make prediction
        st.write("Attempting prediction...") # Debugging line
        prediction = best_model.predict(input_processed)
        prediction_proba = best_model.predict_proba(input_processed)[:, 1] # Probability of stroke
        st.write("Prediction successful.") # Debugging line


        st.subheader("Prediction Result:")
        if prediction[0] == 1:
            st.error(f"Based on the input data, the model predicts a HIGH risk of stroke.")
        else:
            st.success(f"Based on the input data, the model predicts a LOW risk of stroke.")

        st.write(f"Probability of Stroke: {prediction_proba[0]:.2f}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.write("Please check the input values and try again.")
