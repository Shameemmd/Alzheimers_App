import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("alzheimers_small_model.pkl")

st.set_page_config(page_title="Alzheimer Prediction", layout="centered")

st.title("🧠 Alzheimer's Disease Prediction System")

st.markdown("Enter patient medical details below:")

# Input fields
feature1 = st.number_input("Functional Assessment")
feature2 = st.number_input("ADL Score")
feature3 = st.number_input("MMSE Score")
feature4 = st.number_input("Memory Complaints (0/1)", min_value=0, max_value=1)
feature5 = st.number_input("Behavioral Problems (0/1)", min_value=0, max_value=1)
feature6 = st.number_input("Physical Activity Level")
feature7 = st.number_input("Cholesterol HDL")
feature8 = st.number_input("Sleep Quality")
feature9 = st.number_input("Triglycerides")
feature10 = st.number_input("Total Cholesterol")

if st.button("🔍 Predict"):
    
    input_data = np.array([[feature1, feature2, feature3, feature4,
                            feature5, feature6, feature7,
                            feature8, feature9, feature10]])

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)
    confidence = round(max(probability[0]) * 100, 2)

    result = "Alzheimer's Disease" if prediction[0] == 1 else "Normal"

    st.subheader(f"Prediction: {result}")
    st.success(f"Confidence: {confidence}%")