import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Coba load model
model_path = 'best_rf_model.pkl'
if not os.path.exists(model_path):
    st.error(f"File model '{model_path}' tidak ditemukan. Pastikan file ini sudah di-upload.")
    st.stop()

model = joblib.load(model_path)

# Coba load scaler
scaler_path = 'scaler.pkl'
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
else:
    # DummyScaler jika scaler.pkl tidak ada
    numeric_features_means = [0.0, 43.476, 106.147, 28.89]  # Ganti dengan mean asli
    numeric_features_stds = [0.0, 1.99, 45.28, 7.7]         # Ganti dengan std asli

    class DummyScaler:
    def _init_(self, mean, scale):  # <- dua garis bawah, dan diberi indentasi
        self.mean_ = mean
        self.scale_ = scale

    def transform(self, X):
        return (X - self.mean_) / self.scale_

    scaler = DummyScaler(numeric_features_means, numeric_features_stds)

# Fitur numerik dan kategorikal
numeric_features = ['id', 'age', 'avg_glucose_level', 'bmi']
categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

# Kolom akhir setelah preprocessing (disesuaikan dengan X_train saat training)
expected_columns = [
    'id', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
    'gender_Male', 'gender_Other', 'ever_married_Yes',
    'work_type_Never_worked', 'work_type_Private', 'work_type_Self-employed', 'work_type_children',
    'Residence_type_Urban',
    'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes'
]

# Tampilan aplikasi
st.title("🧠 Aplikasi Deteksi Risiko Stroke")
st.write("Masukkan informasi pasien untuk memprediksi risiko stroke menggunakan model Machine Learning.")

# Input dari pengguna
st.sidebar.header("📝 Input Data Pasien")
gender = st.sidebar.selectbox("Jenis Kelamin", ['Female', 'Male', 'Other'])
age = st.sidebar.slider("Usia", 0, 120, 30)
hypertension = st.sidebar.selectbox("Hipertensi", ['No', 'Yes'])
heart_disease = st.sidebar.selectbox("Penyakit Jantung", ['No', 'Yes'])
ever_married = st.sidebar.selectbox("Status Pernikahan", ['No', 'Yes'])
work_type = st.sidebar.selectbox("Tipe Pekerjaan", ['children', 'Govt_job', 'Never_worked', 'Private', 'Self-employed'])
Residence_type = st.sidebar.selectbox("Tempat Tinggal", ['Rural', 'Urban'])
avg_glucose_level = st.sidebar.number_input("Tingkat Glukosa Rata-rata", value=100.0)
bmi = st.sidebar.number_input("BMI", value=25.0)
smoking_status = st.sidebar.selectbox("Status Merokok", ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])

# Siapkan data
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

input_df = pd.DataFrame([input_data])

# One-hot encoding
input_df = pd.get_dummies(input_df, columns=categorical_features, drop_first=True)

# Tambahkan kolom yang tidak ada dengan nilai 0
for col in expected_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Susun kolom sesuai urutan training
input_df = input_df[expected_columns]

# Skala fitur numerik
input_df[numeric_features] = scaler.transform(input_df[numeric_features])

# Prediksi saat tombol diklik
if st.sidebar.button("🔍 Prediksi Risiko Stroke"):
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[:, 1]

    st.subheader("📊 Hasil Prediksi:")
    if prediction[0] == 1:
        st.error(f"⚠ Pasien memiliki risiko tinggi terkena stroke.\nProbabilitas: {probability[0]:.2f}")
    else:
        st.success(f"✅ Pasien memiliki risiko rendah terkena stroke.\nProbabilitas: {probability[0]:.2f}")

    st.info("🔔 Catatan: Hasil ini berbasis model Machine Learning dan bukan diagnosis medis resmi.")
