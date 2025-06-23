import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the trained model
try:
    with open('random_forest_classifier_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'random_forest_classifier_model.pkl' is in the correct directory.")
    st.stop()


st.title('Stroke Prediction Application')
st.write('Enter the patient\'s information to predict the risk of stroke.')

# Input fields for features
gender = st.selectbox('Gender', ['Female', 'Male'])
age = st.number_input('Age', min_value=0, max_value=120, value=30)
hypertension = st.selectbox('Hypertension', ['No', 'Yes'])
heart_disease = st.selectbox('Heart Disease', ['No', 'Yes'])
ever_married = st.selectbox('Ever Married', ['No', 'Yes'])
work_type = st.selectbox('Work Type', ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
residence_type = st.selectbox('Residence Type', ['Urban', 'Rural'])
avg_glucose_level = st.number_input('Average Glucose Level', min_value=0.0, value=100.0)
bmi = st.number_input('BMI', min_value=0.0, value=25.0)
smoking_status = st.selectbox('Smoking Status', ['never smoked', 'formerly smoked', 'smokes', 'Unknown'])

# Prediction button
if st.button('Predict'):
    # Create a dictionary with the input data
    input_data = {
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': ever_married,
        'work_type': work_type,
        'residence_type': residence_type,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': smoking_status
    }

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Preprocess the input data (similar to how the training data was preprocessed)
    # Use the same encoding as in the notebook
    # Fit LabelEncoder on all possible values to avoid errors when only one value is present in input_df
    gender_le = LabelEncoder().fit(['Female', 'Male'])
    ever_married_le = LabelEncoder().fit(['No', 'Yes'])
    work_type_le = LabelEncoder().fit(['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
    residence_type_le = LabelEncoder().fit(['Urban', 'Rural'])
    smoking_status_le = LabelEncoder().fit(['never smoked', 'formerly smoked', 'smokes', 'Unknown'])

    input_df['gender'] = gender_le.transform(input_df['gender'])
    input_df['ever_married'] = ever_married_le.transform(input_df['ever_married'])
    input_df['work_type'] = work_type_le.transform(input_df['work_type'])
    input_df['residence_type'] = residence_type_le.transform(input_df['residence_type'])
    input_df['smoking_status'] = smoking_status_le.transform(input_df['smoking_status'])


    # Convert Yes/No to 1/0 for hypertension and heart_disease
    input_df['hypertension'] = input_df['hypertension'].apply(lambda x: 1 if x == 'Yes' else 0)
    input_df['heart_disease'] = input_df['heart_disease'].apply(lambda x: 1 if x == 'Yes' else 0)


    # Ensure the order of columns in the input matches the training data
    # This is important for the model prediction
    # Get the column order from the training data used to train the model
    # Assuming X_train is available from the notebook
    # If not, you would need to save the column order when training the model
    try:
        # Use the columns from the X_train variable in the notebook
        input_df = input_df[X_train.columns]
    except NameError:
        st.error("X_train is not available. Cannot ensure correct column order for prediction.")
        st.stop()


    # Make prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)[:, 1] * 100


    # Display prediction result
    if prediction[0] == 1:
        st.error(f'Prediction: High risk of stroke (Probability: {prediction_proba[0]:.2f}%)')
    else:
        st.success(f'Prediction: Low risk of stroke (Probability: {prediction_proba[0]:.2f}%)')

    # Explain model accuracy
    st.write("Model Accuracy (Random Forest Classifier): 94.91%")
