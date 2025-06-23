import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model and preprocessor
try:
    best_model = joblib.load('best_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    st.success("Model and preprocessor loaded successfully!")
except Exception as e:
    st.error(f"Error loading model or preprocessor: {e}")

# Define the list of selected features used during training
# This list should match the 'selected_features' list from the feature selection step
# Make sure the order of features here is the same as in the training data X
selected_features = ['age', 'avg_glucose_level', 'bmi', 'hypertension', 'ever_married_Yes', 'heart_disease', 'ever_married_No', 'work_type_Self-employed', 'smoking_status_never smoked', 'work_type_Private', 'gender_Female', 'Residence_type_Urban', 'smoking_status_formerly smoked', 'Residence_type_Rural', 'gender_Male']


st.title("Aplikasi Deteksi Penyakit Stroke")

st.write("Aplikasi ini menggunakan model Machine Learning untuk memprediksi risiko stroke berdasarkan input pengguna.")

# Create input fields for the user
st.header("Input Data Pasien:")

gender = st.selectbox("Jenis Kelamin", ['Female', 'Male', 'Other'])
age = st.number_input("Usia", min_value=0, max_value=120, value=30)
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
        user_input_processed = preprocessor.transform(user_input_df)

        # Convert the processed input back to a DataFrame with the correct column names
        # Need to get the feature names after preprocessing
        # The ColumnTransformer 'remainder='passthrough'' keeps the 'hypertension' and 'heart_disease'
        # The one-hot encoder creates new columns for categorical features
        # We need to get the order of columns that the preprocessor outputs
        preprocessor_output_features = []
        # Add numerical feature names (from numerical_features)
        preprocessor_output_features.extend(['age', 'avg_glucose_level', 'bmi']) # Ensure these match the order in numerical_features

        # Add one-hot encoded feature names (from onehot_feature_names)
        # We need to get the exact output order from the one-hot encoder part of the preprocessor
        # A robust way is to fit a dummy dataset and get the column names
        # However, since we know the structure, we can reconstruct it.
        # This part can be tricky if the preprocessor changes the order.
        # A safer approach might involve saving the feature names list generated after preprocessing

        # For now, let's try to reconstruct based on the original preprocessor code
        # This assumes the order of categorical features and their one-hot encoding
        # remains consistent.
        # Let's get the feature names from the onehot encoder directly if possible
        # Accessing the onehot encoder from the fitted preprocessor
        onehot_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        categorical_output_features = onehot_encoder.get_feature_names_out(categorical_features)
        preprocessor_output_features.extend(categorical_output_features)

        # Add the 'passthrough' columns (hypertension, heart_disease) - they were originally at the end
        # Need to make sure they are added in the correct order
        # Assuming they were added in the order they appeared after the categorical features
        # in the original df_processed DataFrame before dropping
        # original_columns_after_cat = ['hypertension', 'heart_disease'] # This is incorrect, remainder=passthrough keeps original columns not dropped
        # The remainder columns kept were 'hypertension' and 'heart_disease' and their order is preserved
        # Let's check the original df_processed columns order after numerical and categorical
        # This reconstruction is error-prone. A better way is to save the final feature names.
        # Let's revisit the preprocessing step and add a line to save the column names after transformation.

        # For now, let's assume the order from the original processing is:
        # numerical_features + onehot_encoded_features + hypertension + heart_disease
        # The ColumnTransformer with remainder='passthrough' puts the remainder columns at the end of the transformed array.
        # So the order is numerical_transformed + categorical_transformed + remainder_columns

        # Let's get the remainder feature names from the preprocessor config
        # This is also not directly exposed easily.

        # Let's assume the order is: numerical_features + onehot_feature_names + hypertension + heart_disease
        # This worked when creating df_processed_transformed initially.
        all_feature_names_after_preprocessing = list(numerical_features) + list(categorical_output_features) + ['hypertension', 'heart_disease']


        user_input_processed_df = pd.DataFrame(user_input_processed, columns=all_feature_names_after_preprocessing)

        # Now, select only the features that were used for training the model (selected_features)
        user_input_final = user_input_processed_df[selected_features]


        # Make prediction
        prediction = best_model.predict(user_input_final)
        prediction_proba = best_model.predict_proba(user_input_final)[:, 1] # Probability of stroke (class 1)

        st.subheader("Hasil Prediksi:")
        if prediction[0] == 1:
            st.write("Hasil: Risiko Stroke Tinggi")
            st.warning(f"Probabilitas Stroke: {prediction_proba[0]:.4f}")
        else:
            st.write("Hasil: Risiko Stroke Rendah")
            st.success(f"Probabilitas Stroke: {prediction_proba[0]:.4f}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
