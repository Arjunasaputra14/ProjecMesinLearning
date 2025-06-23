import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Import necessary classes for loading the preprocessor and model
# These classes must be available in the environment where the app is run
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier # Import the actual model class used

# Define the expected column names after preprocessing
# This list should match the columns in X_train_res or df_processed_transformed
# This is crucial for creating a DataFrame from the processed user input
# Assuming df_processed_transformed is available from the last execution state
# Use the columns from the transformed training data
try:
    # Attempt to load a dummy version or assume the structure based on the notebook state
    # A more robust way in a real app is to save these column names during training
    # For this case, let's hardcode based on the notebook's df_processed_transformed columns
    processed_feature_names = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
                               'gender_Female', 'gender_Male', 'gender_Other', 'ever_married_No',
                               'ever_married_Yes', 'work_type_Govt_job', 'work_type_Never_worked',
                               'work_type_Private', 'work_type_Self-employed', 'work_type_children',
                               'Residence_type_Rural', 'Residence_type_Urban', 'smoking_status_Unknown',
                               'smoking_status_formerly smoked', 'smoking_status_never smoked',
                               'smoking_status_smokes']
except Exception as e:
    st.error(f"Could not determine processed feature names: {e}")
    st.stop() # Stop if feature names cannot be determined


# Load the trained model and preprocessor
try:
    # Ensure the filename matches the saved file name
    best_model = joblib.load('best_model.pkl') # Assuming this saved the tuned RF model
    preprocessor = joblib.load('preprocessor.pkl')
    st.success("Model and preprocessor loaded successfully!")
except FileNotFoundError:
    st.error("Error: Model or preprocessor file not found. Make sure 'best_model.pkl' and 'preprocessor.pkl' are in the same directory.")
    st.stop() # Stop execution if files are not found
except Exception as e:
    st.error(f"Error loading model or preprocessor: {e}")
    st.stop() # Stop execution if other loading errors occur


st.title("Aplikasi Deteksi Penyakit Stroke")

st.write("Aplikasi ini menggunakan model Machine Learning untuk memprediksi risiko stroke berdasarkan input pengguna.")

# Create input fields for the user, matching the original dataset columns
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
bmi = st.number_input("BMI", min_value=0.0, value=25.0) # Note: BMI had missing values, imputing strategy was median
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

# Ensure the order of columns in the input DataFrame matches the order the preprocessor expects
# This order should be the same as the original columns (excluding 'id' and 'stroke')
original_columns_order = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']
user_input_df = user_input_df[original_columns_order]


# Add a button to make a prediction
if st.button("Prediksi Risiko Stroke"):
    try:
        # Apply the preprocessor to the user input DataFrame
        # The preprocessor handles imputation, one-hot encoding, and scaling
        user_input_processed = preprocessor.transform(user_input_df)

        # Convert the processed input back to a DataFrame with the correct column names
        # Use the predefined processed_feature_names
        user_input_processed_df = pd.DataFrame(user_input_processed, columns=processed_feature_names)

        # Ensure the columns in the processed user input match the columns the model was trained on
        # Although handled by using processed_feature_names, this is a safeguard.
        # If the model was trained on a subset of features after preprocessing,
        # this is where you would select those specific features.
        # Assuming best_model was trained on all processed_feature_names.


        # Make prediction using the loaded model
        prediction = best_model.predict(user_input_processed_df)
        prediction_proba = best_model.predict_proba(user_input_processed_df)[:, 1] # Probability of stroke (class 1)

        st.subheader("Hasil Prediksi:")
        if prediction[0] == 1:
            st.write("Hasil: **Risiko Stroke Tinggi**")
            st.warning(f"Probabilitas Stroke: {prediction_proba[0]:.4f}")
        else:
            st.write("Hasil: **Risiko Stroke Rendah**")
            st.success(f"Probabilitas Stroke: {prediction_proba[0]:.4f}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        # Optionally, print traceback for debugging
        # import traceback
        # st.error(traceback.format_exc())
