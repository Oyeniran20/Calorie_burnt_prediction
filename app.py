# Import libraries
import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# App Title
st.title("Calorie Burn Prediction App")

# Sidebar for Inputs
st.sidebar.header("Enter Personal and Activity Details:")

# Input Features
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.number_input("Age (years)", min_value=1, max_value=100, value=25)
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0)
duration = st.sidebar.number_input("Duration (minutes)", min_value=1, max_value=300, value=30)
heart_rate = st.sidebar.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=120)
body_temp = st.sidebar.number_input("Body Temperature (°C)", min_value=30.0, max_value=45.0, value=36.5)

# Encode Gender
gender_code = 0 if gender == "Male" else 1

# Predict Button
if st.sidebar.button("Predict"):
    # Prepare input data
    input_data = np.array([[gender_code, age, bmi, duration, heart_rate, body_temp]])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Transform prediction back from square root to actual calories
    calories_burned = prediction[0] ** 2
    
    # Display result
    st.success(f"Estimated Calories Burned: {calories_burned:.2f} kcal")


# use `streamlit run app.py` to execute the code on the prompt
