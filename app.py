
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load('best_rf_model.pkl')

# Load the scaler used for numeric features
# We need to recreate and fit the scaler on a dummy dataset with the same structure
# as the training data to ensure consistent scaling in the Streamlit app.
# In a real application, you would save and load the fitted scaler as well.
# For demonstration purposes, we will fit it on a sample of the original data (before scaling)
# assuming the structure of the data remains consistent.

# It's better to save the fitted scaler in the notebook and load it here.
# Let's assume you have saved the scaler as 'scaler.pkl' in your notebook.
# If not, you would need to re-fit the scaler or manually define the mean and std deviation
# for each numeric feature based on your training data.

# For this example, let's create a dummy scaler for demonstration.
# Replace this with loading your saved scaler if available.
# If you saved your scaler using joblib.dump(scaler, 'scaler.pkl'), uncomment the line below:
# scaler = joblib.load('scaler.pkl')

# --- If you did NOT save the scaler, you can manually define it based on your training data ---
# IMPORTANT: Replace these values with the actual mean and std deviation from your training data
# You can get these from the fitted scaler object in your notebook after scaling the training data.
# For example, after running the scaler.fit_transform(X_train[numeric_features]) cell,
# you can access scaler.mean_ and scaler.scale_

# Dummy values (replace with actual mean and std from your training data)
# To get actual values, add print(scaler.mean_) and print(scaler.scale_) after scaling in your notebook
# and use those values here.
numeric_features_means = [0.0, 43.476, 106.147, 28.89] # Replace with actual means
numeric_features_stds = [0.0, 1.99, 45.28, 7.7] # Replace with actual standard deviations

class DummyScaler:
    def __init__(self, mean, scale):
        self.mean_ = mean
        self.scale_ = scale

    def transform(self, X):
        return (X - self.mean_) / self.scale_

# Instantiate the dummy scaler (replace with loading your saved scaler)
scaler = DummyScaler(numeric_features_means, numeric_features_stds)

numeric_features = ['id', 'age', 'avg_glucose_level', 'bmi']
categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']


# Define the order of columns expected by the model after preprocessing
# This is crucial for making predictions on new data
# Get the columns from the X_train DataFrame after preprocessing in your notebook
# For example, after running the one-hot encoding and scaling cells, print X_train.columns
# and use that list here.
# Example columns (replace with actual columns from your preprocessed X_train)
expected_columns = ['id', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
                   'gender_Male', 'gender_Other', 'ever_married_Yes',
                   'work_type_Never_worked', 'work_type_Private', 'work_type_Self-employed', 'work_type_children',
                   'Residence_type_Urban', 'smoking_status_formerly smoked',
                   'smoking_status_never smoked', 'smoking_status_smokes']


# Streamlit App
st.title("Aplikasi Deteksi Risiko Stroke")
st.write("Masukkan informasi pasien di bawah ini untuk memprediksi risiko stroke.")

# Input fields for user data
st.sidebar.header("Informasi Pasien")

gender = st.sidebar.selectbox("Jenis Kelamin", ['Female', 'Male', 'Other'])
age = st.sidebar.slider("Usia", 0, 120, 30)
hypertension = st.sidebar.selectbox("Hipertensi", ['No', 'Yes'])
heart_disease = st.sidebar.selectbox("Penyakit Jantung", ['No', 'Yes'])
ever_married = st.sidebar.selectbox("Status Pernikahan", ['No', 'Yes'])
work_type = st.sidebar.selectbox("Tipe Pekerjaan", ['children', 'Govt_job', 'Never_worked', 'Private', 'Self-employed'])
Residence_type = st.sidebar.selectbox("Tipe Tempat Tinggal", ['Rural', 'Urban'])
avg_glucose_level = st.sidebar.number_input("Tingkat Glukosa Rata-rata", value=100.0)
bmi = st.sidebar.number_input("BMI", value=25.0)
smoking_status = st.sidebar.selectbox("Status Merokok", ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])

# Convert categorical inputs to numerical/encoded format
input_data = {
    'gender': gender,
    'age': age,
    'hypertension': 1 if hypertension == 'Yes' else 0,
    'heart_disease': 1 if heart_disease == 'Yes' else 0,
    'ever_married': ever_married,
    'work_type': work_type,
    'Residence_type': Residence_type,
    'avg_glucose_level': avg_glucose_level,
    'bmi': bmi,
    'smoking_status': smoking_status
}

# Create a DataFrame from input data
input_df = pd.DataFrame([input_data])

# Apply one-hot encoding to the input data
input_df = pd.get_dummies(input_df, columns=categorical_features, drop_first=True)

# Ensure all expected columns are present and in the correct order
for col in expected_columns:
    if col not in input_df.columns:
        input_df[col] = 0 # Add missing columns with a value of 0

# Reorder columns to match the training data
input_df = input_df[expected_columns]

# Scale numeric features using the loaded scaler
# Note: The 'id' column was included in numeric features during training.
# In a real application, you might exclude 'id' or handle it differently.
# For consistency with the training data, we include it here for scaling.
input_df[numeric_features] = scaler.transform(input_df[numeric_features])


# Make prediction
# Baris ke-121
if st.sidebar.button("Prediksi Risiko Stroke"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)[:, 1]  # Probabilitas stroke

    st.subheader("Hasil Prediksi")
    if prediction[0] == 1:
        st.error(f"Pasien memiliki risiko tinggi terkena stroke. (Probabilitas: {prediction_proba[0]:.2f})")
    else:
        st.success(f"Pasien memiliki risiko rendah terkena stroke. (Probabilitas: {prediction_proba[0]:.2f})")

    # Perbaikan: indentasi disesuaikan agar berada di dalam blok if-button
    st.write("Catatan: Prediksi ini didasarkan pada model machine learning dan tidak menggantikan diagnosis dari tenaga medis.")



