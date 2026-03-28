import streamlit as st
import joblib
import numpy as np
from datetime import datetime
import io

from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

# Page config
st.set_page_config(page_title="Multimodal Alzheimer System", page_icon="🧠", layout="wide")

# Load NEW multimodal model
model = joblib.load("alzheimers_multimodal_model.pkl")

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧠 Navigation")
menu = st.sidebar.radio("Go to", ["Prediction System", "About"])

if menu == "About":
    st.title("📘 About Project")
    st.write("""
    This project implements Multimodal Data Fusion for Alzheimer's Disease Prediction.
    It combines brain imaging, genetic, clinical, and cognitive data for improved accuracy.
    """)
    st.stop()

# ---------------- MAIN ----------------
st.title("🏥 Multimodal Brain Data Fusion for Alzheimer’s Prediction")

# ---------------- PATIENT INFO ----------------
st.subheader("👤 Patient Information")

col1, col2 = st.columns(2)

with col1:
    patient_name = st.text_input("Patient Name")
    patient_id = st.text_input("Patient ID")

with col2:
    age = st.number_input("Age", 0, 120)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])

date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ---------------- MRI FEATURES ----------------
st.subheader("🧠 Brain Imaging Features (Simulated MRI)")

col3, col4 = st.columns(2)

with col3:
    brain_volume = st.number_input("Brain Volume", min_value=0.0)
    hippocampus_size = st.number_input("Hippocampus Size", min_value=0.0)

with col4:
    apoe = st.selectbox("APOE Gene Risk (0=Low,1=Medium,2=High)", [0,1,2])

# ---------------- CLINICAL ----------------
st.subheader("🏥 Clinical Data")

adl = st.number_input("ADL Score")
functional = st.number_input("Functional Assessment")
cholesterol = st.number_input("Total Cholesterol")

# ---------------- COGNITIVE ----------------
st.subheader("🧠 Cognitive Data")

mmse = st.number_input("MMSE Score")
memory = st.number_input("Memory Complaints (0/1)", 0, 1)

# ---------------- PREDICT ----------------
if st.button("🔍 Predict", use_container_width=True):

    input_data = np.array([[
        brain_volume,
        hippocampus_size,
        apoe,
        age,
        adl,
        functional,
        cholesterol,
        mmse,
        memory
    ]])

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    confidence = round(max(probability[0]) * 100, 2)

    # 🔁 3-class output
    if prediction[0] == 0:
        result = "Normal"
    elif prediction[0] == 1:
        result = "Mild Cognitive Impairment (MCI)"
    else:
        result = "Alzheimer's Disease"

    # Risk logic
    if result == "Alzheimer's Disease" and confidence > 80:
        risk = "High Risk"
    elif confidence > 60:
        risk = "Moderate Risk"
    else:
        risk = "Low Risk"

    st.markdown("---")

    # RESULT CARDS
    col5, col6, col7 = st.columns(3)

    col5.metric("🧠 Prediction", result)
    col6.metric("📊 Confidence", f"{confidence}%")
    col7.metric("⚠ Risk Level", risk)

    # ---------------- TABLE ----------------
    st.subheader("📋 Patient Report")

    table_data = [
        ["Field", "Value"],
        ["Patient Name", patient_name],
        ["Patient ID", patient_id],
        ["Age", age],
        ["Gender", gender],
        ["Date & Time", date_time],
        ["Brain Volume", brain_volume],
        ["Hippocampus Size", hippocampus_size],
        ["APOE", apoe],
        ["ADL Score", adl],
        ["Functional Assessment", functional],
        ["Cholesterol", cholesterol],
        ["MMSE Score", mmse],
        ["Memory Complaints", memory],
        ["Prediction", result],
        ["Confidence", f"{confidence}%"],
        ["Risk Level", risk]
    ]

    st.table(table_data)

    # ---------------- PDF ----------------
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    elements = []

    styles = getSampleStyleSheet()

    # Logo (optional)
    try:
        logo = Image("logo.png", width=60, height=60)
        elements.append(logo)
    except:
        pass

    elements.append(Paragraph("Multimodal Alzheimer Prediction Report", styles['Title']))
    elements.append(Spacer(1, 10))

    table = Table(table_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))

    elements.append(table)
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("Doctor Signature: ____________________", styles['Normal']))

    doc.build(elements)

    buffer.seek(0)

    st.download_button(
        "📥 Download Report",
        buffer,
        file_name="alzheimers_multimodal_report.pdf",
        mime="application/pdf"
    )