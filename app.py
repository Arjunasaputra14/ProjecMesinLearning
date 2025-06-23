import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder # Need these classes for loading preprocessor
from sklearn.impute import SimpleImputer # Need this class for loading preprocessor
from sklearn.pipeline import Pipeline # Need this class for loading preprocessor
from sklearn.compose import ColumnTransformer # Need this class for loading preprocessor
from sklearn.ensemble import RandomForestClassifier # Need this class for loading the model

# Load the trained model and preprocessor
try:
    # Make sure the class definitions used in the preprocessor pipeline are available
    # This includes SimpleImputer, StandardScaler, OneHotEncoder, Pipeline, ColumnTransformer
    # Also make sure the model class (e.g., LogisticRegression or RandomForestClassifier) is imported

    best_model = joblib.load('best_model.pkl') # Assuming 'best_model.pkl' is the saved tuned RF model
    preprocessor = joblib.load('preprocessor.pkl')
    st.success("Model and preprocessor loaded successfully!")
except Exception as e:
    st.error(f"Error loading model or preprocessor: {e}")
    st.stop() # Stop execution if loading fails

# Define the list of features that the model was trained on
# This list is crucial for ensuring the input data for prediction is in the correct format and order
# You can get this list from the columns of X_train after preprocessing and feature selection (if any)
# Based on the notebook, the final X_train_res had 21 columns.
# Let's use the columns from df_processed_transformed as the expected input feature names for the model.
# We need to make sure the saved 'best_model.pkl' was trained on data with these columns in this order.
# Assuming 'best_model.pkl' is the tuned RF model trained on X_train_res
# The columns of X_train_res are the same as df_processed_transformed.columns
model_feature_names = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
                       'gender_Female', 'gender_Male', 'gender_Other', 'ever_married_No',
                       'ever_married_Yes', 'work_type_Govt_job', 'work_type_Never_worked',
                       'work_type_Private', 'work_type_Self-employed', 'work_type_children',
                       'Residence_type_Rural', 'Residence_type_Urban', 'smoking_status_Unknown',
                       'smoking_status_formerly smoked', 'smoking_status_never smoked',
                       'smoking_status_smokes']


st.title("Aplikasi Deteksi Penyakit Stroke")

st.write("Aplikasi ini menggunakan model Machine Learning untuk memprediksi risiko stroke berdasarkan input pengguna.")

# Create input fields for the user
st.header("Input Data Pasien:")

# Collect user input for each original feature
gender = st.selectbox("Jenis Kelamin", ['Female', 'Male', 'Other'])
age = st.number_input("Usia", min_value=0.0, max_value=120.0, value=30.0)
hypertension = st.selectbox("Hipertensi", [0, 1], format_func=lambda x: 'Ya' if x == 1 else 'Tidak')
heart_disease = st.selectbox("Penyakit Jantung", [0, 1], format_func=lambda x: 'Ya' if x == 1 else 'Tidak')
ever_married = st.selectbox("Pernah Menikah", ['Yes', 'No'])
work_type = st.selectbox("Tipe Pekerjaan", ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
Residence_type = st.selectbox("Tipe Tempat Tinggal", ['Urban', 'Rural'])
avg_glucose_level = st.number_input("Rata-rata Tingkat Glukosa", min_value=0.0, value=100.0)
bmi = st.number_input("BMI", min_value=0.0, value=25.0)
smoking_status = st.selectbox("Status Merokok", ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])

# Create a dictionary from user input
user_input = {
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

# Convert user input to a pandas DataFrame
user_input_df = pd.DataFrame([user_input])

# Reorder columns to match the order used during preprocessing (all original columns except id and stroke)
# This is important because the preprocessor expects the columns in a specific order
original_columns_order = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']
user_input_df = user_input_df[original_columns_order]


# Add a button to make a prediction
if st.button("Prediksi Risiko Stroke"):
    try:
        # Apply the preprocessor to the user input
        # The preprocessor will handle missing values (if any in input, although st.number_input handles this),
        # one-hot encode categorical features, and scale numerical features.
        user_input_processed = preprocessor.transform(user_input_df)

        # Convert the processed input back to a DataFrame with the correct column names
        # Need to use the column names that the preprocessor outputs
        # Based on the preprocessor definition and the df_processed_transformed created earlier
        # the order should be numerical_features + one-hot encoded categorical features + remainder features
        # Let's use the columns from df_processed_transformed as the reference
        processed_column_names = df_processed_transformed.columns.tolist()
        user_input_processed_df = pd.DataFrame(user_input_processed, columns=processed_column_names)


        # Ensure the columns match the order expected by the model
        # Although the preprocessor should output in a consistent order,
        # explicitly reindexing can prevent issues if feature selection was applied later.
        # However, in this case, the 'best_model.pkl' was likely trained on the full
        # preprocessed features (X_train_res), so we should use those column names.
        # Let's assume 'best_model.pkl' was trained on data with columns exactly matching
        # the columns in df_processed_transformed.

        # Make prediction using the loaded model
        prediction = best_model.predict(user_input_processed_df)
        prediction_proba = best_model.predict_proba(user_input_processed_df)[:, 1] # Probability of stroke (class 1)

        st.subheader("Hasil Prediksi:")
        if prediction[0] == 1:
            st.write("Hasil: Risiko Stroke Tinggi")
            st.warning(f"Probabilitas Stroke: {prediction_proba[0]:.4f}")
        else:
            st.write("Hasil: Risiko Stroke Rendah")
            st.success(f"Probabilitas Stroke: {prediction_proba[0]:.4f}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.error(f"Error details: {e}") # Print error details for debugging
