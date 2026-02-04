"""Streamlit application for diabetes risk prediction."""

import streamlit as st

from src.predict import predict_risk

st.set_page_config(page_title="Diabetes Risk Analysis", page_icon="🩺")

st.title("Diabetes Risk Analysis")
st.write(
    "Enter your lifestyle and health metrics to estimate diabetes risk. "
    "This tool is for educational use only and not a medical diagnosis."
)

with st.form("risk_form"):
    age = st.number_input("Age", min_value=1, max_value=120, value=35)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    physical_activity = st.selectbox(
        "Physical Activity Level", ["Low", "Moderate", "High"]
    )
    diet = st.selectbox("Diet Habit", ["Balanced", "Low Carb", "High Sugar", "High Fat"])
    family_history = st.selectbox("Family History / Genetic Risk", ["No", "Yes"])
    blood_pressure = st.number_input(
        "Blood Pressure", min_value=60, max_value=200, value=120
    )
    glucose = st.number_input("Glucose Level", min_value=60, max_value=300, value=100)

    submitted = st.form_submit_button("Predict Risk")

if submitted:
    input_payload = {
        "age": age,
        "bmi": bmi,
        "physical_activity": physical_activity,
        "diet": diet,
        "family_history": family_history,
        "blood_pressure": blood_pressure,
        "glucose": glucose,
    }

    result = predict_risk(input_payload)
    probability_pct = result["probability"] * 100

    st.subheader("Prediction Result")
    st.metric("Prediction", result["prediction"])
    st.metric("Probability", f"{probability_pct:.1f}%")

    if result["prediction"] == "Diabetic":
        recommendation = (
            "Consider scheduling a check-up with a healthcare professional, "
            "maintain a balanced diet, and increase physical activity."
        )
    else:
        recommendation = (
            "Great job! Keep maintaining a healthy lifestyle with regular exercise "
            "and balanced nutrition."
        )

    st.info(recommendation)
