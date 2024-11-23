import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the trained model
model_file = "calorie_model.pkl"  # Ensure this matches your saved model file name
with open(model_file, "rb") as file:
    model = pickle.load(file)

# Pre-trained PCA components for consistent transformation
pca_physical = PCA(n_components=1)
pca_exertion = PCA(n_components=1)

# Feature scaling setup
scaler = StandardScaler()

# Title and introduction
st.title("Calorie Burn Prediction")
st.write(
    """
    This app predicts the number of calories burned based on physical and physiological metrics.
    Fill in the details below, and click **Predict** to get the result.
    """
)

# Input form
st.sidebar.header("Input Features")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.number_input("Age", min_value=1, max_value=80, value=25)
height = st.sidebar.number_input("Height (cm)", min_value=50, max_value=250, value=170)
weight = st.sidebar.number_input("Weight (kg)", min_value=10, max_value=200, value=70)
duration = st.sidebar.number_input("Duration (mins)", min_value=1, max_value=500, value=30)
heart_rate = st.sidebar.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=80)
body_temp = st.sidebar.number_input("Body Temperature (°C)", min_value=30.0, max_value=45.0, value=37.0)

# Feature engineering and preprocessing
def preprocess_input():
    # Gender encoding
    gender_encoded = 0 if gender == "Male" else 1
    
    # BMI calculation
    bmi = weight / (height / 100) ** 2
    
    # Age group assignment
    age_group = "Young" if age <= 30 else "Middle-aged" if age <= 60 else "Old"
    age_group_encoded = [1, 0] if age_group == "Middle-aged" else [0, 1] if age_group == "Old" else [0, 0]
    
    # Create a DataFrame for scaling
    input_data = pd.DataFrame(
        [[age, height, weight, duration, heart_rate, body_temp, bmi]],
        columns=["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp", "BMI"]
    )
    
    # Apply scaling
    columns_to_scale = ["Height", "Weight", "Duration", "Heart_Rate", "Body_Temp", "Age", "BMI"]
    input_data[columns_to_scale] = scaler.fit_transform(input_data[columns_to_scale])
    
    # PCA for correlated groups
    physical_factor = pca_physical.fit_transform(input_data[["Height", "Weight"]])[0, 0]
    exertion_factor = pca_exertion.fit_transform(input_data[["Duration", "Heart_Rate", "Body_Temp"]])[0, 0]
    
    # Final input array
    final_input = np.hstack([input_data.drop(columns=["Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]).values[0],
                             physical_factor, exertion_factor, gender_encoded, age_group_encoded])
    
    return final_input.reshape(1, -1)

# Predict button
if st.button("Predict"):
    input_features = preprocess_input()
    prediction = model.predict(input_features)[0] ** 2  # Undo sqrt transformation
    st.success(f"The predicted calories burned: **{prediction:.2f}** kcal")
