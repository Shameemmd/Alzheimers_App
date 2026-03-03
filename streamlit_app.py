import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

# Page config
st.set_page_config(page_title="Alzheimer Prediction System", page_icon="🧠")

# Load model
model = joblib.load("alzheimers_small_model.pkl")

# ---- Sidebar ----
st.sidebar.title("🧠 Model Information")
st.sidebar.write("Algorithm: Random Forest Classifier")
st.sidebar.write("Estimated Accuracy: 87%")
st.sidebar.write("Developed as MCA ML Project")

# ---- Title ----
st.title("🧠 Alzheimer's Disease Prediction System")

st.markdown("### Enter Patient Medical Details:")

# Inputs
feature1 = st.number_input("Functional Assessment", min_value=0.0)
feature2 = st.number_input("ADL Score", min_value=0.0)
feature3 = st.number_input("MMSE Score", min_value=0.0)
feature4 = st.number_input("Memory Complaints (0/1)", min_value=0, max_value=1)
feature5 = st.number_input("Behavioral Problems (0/1)", min_value=0, max_value=1)
feature6 = st.number_input("Physical Activity Level", min_value=0.0)
feature7 = st.number_input("Cholesterol HDL", min_value=0.0)
feature8 = st.number_input("Sleep Quality", min_value=0.0)
feature9 = st.number_input("Triglycerides", min_value=0.0)
feature10 = st.number_input("Total Cholesterol", min_value=0.0)

if st.button("🔍 Predict"):

    input_data = np.array([[feature1, feature2, feature3, feature4,
                            feature5, feature6, feature7,
                            feature8, feature9, feature10]])

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    confidence = round(max(probability[0]) * 100, 2)
    result = "Alzheimer's Disease" if prediction[0] == 1 else "Normal"

    st.markdown("---")

    # Display Result
    if result == "Alzheimer's Disease":
        st.error(f"⚠ Prediction: {result}")
    else:
        st.success(f"✅ Prediction: {result}")

    st.info(f"Confidence Level: {confidence}%")

    # ---- Graph Visualization ----
    st.subheader("📊 Prediction Confidence Chart")

    labels = ["Normal", "Alzheimer's"]
    values = probability[0] * 100

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_ylabel("Probability (%)")
    ax.set_ylim([0, 100])

    st.pyplot(fig)

    # ---- PDF Report Generation ----
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    pdf.drawString(100, 750, "Alzheimer's Disease Prediction Report")
    pdf.drawString(100, 730, f"Prediction Result: {result}")
    pdf.drawString(100, 710, f"Confidence Level: {confidence}%")
    pdf.save()

    buffer.seek(0)

    st.download_button(
        label="📥 Download PDF Report",
        data=buffer,
        file_name="alzheimers_prediction_report.pdf",
        mime="application/pdf"
    )

# Footer
st.markdown("---")
st.caption("© 2026 Alzheimer's ML Prediction App | Developed by Shameem Mohammad")