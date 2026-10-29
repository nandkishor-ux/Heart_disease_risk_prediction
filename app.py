import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="🫀",
    layout="centered"
)

# Load model and scaler
@st.cache_resource
def load_artifacts():
    model = joblib.load("heart_disease_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# Header
st.title("🫀 Heart Disease Risk Predictor")
st.markdown("Enter patient clinical parameters below and click **Predict Risk**.")

# Input form
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50, step=1)
        trestbps = st.number_input("Resting BP (mmHg)", min_value=50, max_value=250, value=120, step=1)
        restecg = st.selectbox(
            "Resting ECG",
            options=[0, 1, 2],
            format_func=lambda x: {0: "Normal", 1: "ST-T Abnormality", 2: "LV Hypertrophy"}[x]
        )
        oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        thal = st.selectbox(
            "Thalassemia",
            options=[0, 1, 2, 3],
            format_func=lambda x: {0: "Unknown", 1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}[x],
            index=1
        )

    with col2:
        sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
        chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200, step=1)
        thalach = st.number_input("Max Heart Rate", min_value=60, max_value=250, value=150, step=1)
        slope = st.selectbox(
            "Slope of ST Segment",
            options=[0, 1, 2],
            format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x],
            index=1
        )

    with col3:
        cp = st.selectbox(
            "Chest Pain Type",
            options=[0, 1, 2, 3],
            format_func=lambda x: {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-anginal Pain", 3: "Asymptomatic"}[x]
        )
        fbs = st.selectbox("Fasting Blood Sugar > 120?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        ca = st.selectbox("Major Vessels Colored (ca)", options=[0, 1, 2, 3], index=0)

    submitted = st.form_submit_button("🔍 Predict Risk", use_container_width=True)

if submitted:
    feature_cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
    
    input_data = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }

    df = pd.DataFrame([input_data])[feature_cols]
    scaled_data = scaler.transform(df)

    prediction = model.predict(scaled_data)[0]
    probability = float(model.predict_proba(scaled_data)[0][1])

    st.divider()

    if probability < 0.30:
        risk_level = "LOW RISK"
        alert_box = st.success
    elif probability < 0.60:
        risk_level = "MODERATE RISK"
        alert_box = st.warning
    else:
        risk_level = "HIGH RISK"
        alert_box = st.error

    message = "Heart disease risk detected." if prediction == 1 else "No immediate heart disease detected."
    alert_box(f"### {message}")

    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted Class", "Disease" if prediction == 1 else "No Disease")
    m2.metric("Probability", f"{probability * 100:.1f}%")
    m3.metric("Risk Assessment", risk_level)

    st.caption("⚠️ For educational and informational purposes only. Not a substitute for professional medical advice.")