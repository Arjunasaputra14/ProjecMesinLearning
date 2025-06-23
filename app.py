
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Set the title of the app
st.title('Stroke Prediction App')

# Add input fields for each feature
st.header('Enter Patient Data:')

# Define input fields based on the features used in the model
# Numerical features: 'age', 'avg_glucose_level', 'bmi', 'hypertension', 'heart_disease'
age = st.number_input('Age', min_value=0.0, max_value=120.0, value=30.0, step=0.1)
avg_glucose_level = st.number_input('Average Glucose Level', min_value=0.0, value=100.0, step=0.1)
bmi = st.number_input('BMI', min_value=0.0, value=25.0, step=0.1)
hypertension = st.selectbox('Hypertension', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
heart_disease = st.selectbox('Heart Disease', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

# Categorical features: 'gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'
gender = st.selectbox('Gender', options=['Female', 'Male', 'Other'])
ever_married = st.selectbox('Ever Married', options=['Yes', 'No'])
work_type = st.selectbox('Work Type', options=['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
Residence_type = st.selectbox('Residence Type', options=['Urban', 'Rural'])
smoking_status = st.selectbox('Smoking Status', options=['formerly smoked', 'never smoked', 'smokes', 'Unknown'])

# Create a button to trigger prediction
if st.button('Predict Stroke'):
    # Create a dictionary from user input
    user_data = {
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': ever_married,
        'work_type': work_type,
        'Residence_type': Residence_type,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': smoking_status
    }

    # Convert user data to DataFrame
    user_df = pd.DataFrame([user_data])

    # Load the trained model and preprocessor
    try:
        with open('best_xgb_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)

        # Preprocess the user input
        user_processed = preprocessor.transform(user_df)

        # Make prediction
        prediction = model.predict(user_processed)
        prediction_proba = model.predict_proba(user_processed)[:, 1] # Probability of stroke

        # Display the prediction result
        st.header('Prediction Result:')
        if prediction[0] == 1:
            st.error(f'Based on the input data, there is a high risk of stroke. (Probability: {prediction_proba[0]:.4f})')
        else:
            st.success(f'Based on the input data, the risk of stroke is low. (Probability: {prediction_proba[0]:.4f})')

    except FileNotFoundError:
        st.error("Error: Model or preprocessor file not found. Please ensure 'best_xgb_model.pkl' and 'preprocessor.pkl' are in the same directory.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# Placeholder for displaying the prediction result
# This part is now handled within the button click
