import streamlit as st
import pandas as pd
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .main-title {
        color: white;
        font-size: 3rem;
        font-weight: bold;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    .main-subtitle {
        color: white;
        font-size: 1.2rem;
        margin-top: 0.5rem;
        opacity: 0.9;
    }

    .section-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1.5rem 0 1rem 0;
        font-size: 1.3rem;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .info-box {
        background-color: #f8f9fa;
        border-left: 4px solid #4ECDC4;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }

    .prediction-container {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-top: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .feature-description {
        font-size: 0.9rem;
        color: #666;
        font-style: italic;
        margin-top: 0.2rem;
    }

    .stButton > button {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-size: 1.1rem;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1 class="main-title">❤️ Heart Disease Prediction</h1>
    <p class="main-subtitle">Advanced AI-powered cardiac risk assessment tool</p>
</div>
""", unsafe_allow_html=True)

# File path checking (keeping original logic)
model_path = '../models/final_model.pkl'
preprocessor_path = '../models/preprocessor.pkl'

if not os.path.exists(model_path):
    st.error(f"خطأ: ملف النموذج غير موجود في المسار: {model_path}")
    st.stop()

if not os.path.exists(preprocessor_path):
    st.error(f"خطأ: ملف المعالج غير موجود في المسار: {preprocessor_path}")
    st.stop()

# Load model and preprocessor (keeping original logic)
try:
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
except Exception as e:
    st.error(f"خطأ أثناء تحميل الملفات: {str(e)}")
    st.stop()

# Feature names (keeping original)
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca',
                 'thal']

# Information box
st.markdown("""
<div class="info-box">
    <strong>📋 Instructions:</strong> Please fill in all the patient information below. 
    All fields are required for accurate prediction. The AI model will analyze the data 
    and provide a risk assessment for heart disease.
</div>
""", unsafe_allow_html=True)

# Initialize input data dictionary
input_data = {}

# Create columns for better layout
col1, col2 = st.columns(2)

with col1:
    # Patient Demographics Section
    st.markdown('<div class="section-header">👤 Patient Demographics</div>', unsafe_allow_html=True)

    input_data['age'] = st.number_input(
        "Age (years)",
        value=50.0,
        step=1.0,
        min_value=0.0,
        max_value=120.0,
        help="Patient's age in years"
    )
    st.markdown('<p class="feature-description">Enter the patient\'s current age</p>', unsafe_allow_html=True)

    input_data['sex'] = st.selectbox(
        "Sex",
        options=[0, 1],
        format_func=lambda x: "Female" if x == 0 else "Male",
        help="Patient's biological sex"
    )
    st.markdown('<p class="feature-description">0 = Female, 1 = Male</p>', unsafe_allow_html=True)

    # Chest Pain and Symptoms Section
    st.markdown('<div class="section-header">💔 Chest Pain & Symptoms</div>', unsafe_allow_html=True)

    input_data['cp'] = st.selectbox(
        "Chest Pain Type",
        options=[0, 1, 2, 3],
        format_func=lambda x: {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-Anginal Pain", 3: "Asymptomatic"}[x],
        help="Type of chest pain experienced"
    )
    st.markdown(
        '<p class="feature-description">0 = Typical Angina, 1 = Atypical Angina, 2 = Non-Anginal Pain, 3 = Asymptomatic</p>',
        unsafe_allow_html=True)

    input_data['trestbps'] = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        value=120.0,
        step=1.0,
        min_value=0.0,
        help="Resting blood pressure in mm Hg"
    )
    st.markdown('<p class="feature-description">Normal range: 90-140 mm Hg</p>', unsafe_allow_html=True)

    input_data['chol'] = st.number_input(
        "Serum Cholesterol (mg/dl)",
        value=200.0,
        step=1.0,
        min_value=0.0,
        help="Serum cholesterol level in mg/dl"
    )
    st.markdown('<p class="feature-description">Normal range: < 200 mg/dl</p>', unsafe_allow_html=True)

    input_data['fbs'] = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
        help="Whether fasting blood sugar is greater than 120 mg/dl"
    )
    st.markdown('<p class="feature-description">0 = No (≤ 120 mg/dl), 1 = Yes (> 120 mg/dl)</p>',
                unsafe_allow_html=True)

with col2:
    # Heart Rate and Exercise Section
    st.markdown('<div class="section-header">💓 Heart Rate & Exercise</div>', unsafe_allow_html=True)

    input_data['restecg'] = st.selectbox(
        "Resting ECG Results",
        options=[0, 1, 2],
        format_func=lambda x: {0: "Normal", 1: "ST-T Wave Abnormality", 2: "Left Ventricular Hypertrophy"}[x],
        help="Resting electrocardiographic results"
    )
    st.markdown(
        '<p class="feature-description">0 = Normal, 1 = ST-T Wave Abnormality, 2 = Left Ventricular Hypertrophy</p>',
        unsafe_allow_html=True)

    input_data['thalach'] = st.number_input(
        "Maximum Heart Rate Achieved",
        value=150.0,
        step=1.0,
        min_value=0.0,
        help="Maximum heart rate achieved during exercise"
    )
    st.markdown('<p class="feature-description">Typical range: 60-220 bpm</p>', unsafe_allow_html=True)

    input_data['exang'] = st.selectbox(
        "Exercise Induced Angina",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
        help="Whether exercise induces angina"
    )
    st.markdown('<p class="feature-description">0 = No, 1 = Yes</p>', unsafe_allow_html=True)

    # Additional Cardiac Indicators Section
    st.markdown('<div class="section-header">📊 Additional Cardiac Indicators</div>', unsafe_allow_html=True)

    input_data['oldpeak'] = st.number_input(
        "ST Depression (Oldpeak)",
        value=0.0,
        step=0.1,
        min_value=0.0,
        help="ST depression induced by exercise relative to rest"
    )
    st.markdown('<p class="feature-description">ST depression value (usually 0-6)</p>', unsafe_allow_html=True)

    input_data['slope'] = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        options=[0, 1, 2],
        format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x],
        help="Slope of the peak exercise ST segment"
    )
    st.markdown('<p class="feature-description">0 = Upsloping, 1 = Flat, 2 = Downsloping</p>', unsafe_allow_html=True)

    input_data['ca'] = st.number_input(
        "Number of Major Vessels (0-3)",
        value=0.0,
        step=1.0,
        min_value=0.0,
        max_value=3.0,
        help="Number of major vessels colored by fluoroscopy"
    )
    st.markdown('<p class="feature-description">Number of major vessels (0-3) colored by fluoroscopy</p>',
                unsafe_allow_html=True)

    input_data['thal'] = st.selectbox(
        "Thalassemia",
        options=[0, 1, 2, 3],
        format_func=lambda x: {0: "Normal", 1: "Fixed Defect", 2: "Reversible Defect", 3: "Not Described"}[x],
        help="Thalassemia test result"
    )
    st.markdown(
        '<p class="feature-description">0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect, 3 = Not Described</p>',
        unsafe_allow_html=True)

# Convert input to DataFrame (keeping original logic)
input_df = pd.DataFrame([input_data], columns=feature_names)

# Prediction section
st.markdown('<div class="section-header">🔮 Prediction Results</div>', unsafe_allow_html=True)

# Center the prediction button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🔍 Analyze Heart Disease Risk", use_container_width=True)

if predict_button:
    try:
        # Process input and make prediction (keeping original logic)
        input_processed = preprocessor.transform(input_df)
        prediction = model.predict(input_processed)[0]
        probability = model.predict_proba(input_processed)[0][1]

        # Display results with enhanced styling
        if prediction == 1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #ff6b6b, #ee5a52); color: white; padding: 2rem; border-radius: 10px; text-align: center; margin: 1rem 0; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                <h2 style="margin: 0; font-size: 1.8rem;">⚠️ High Risk Alert</h2>
                <p style="font-size: 1.2rem; margin: 1rem 0;">The patient shows indicators of potential heart disease</p>
                <p style="font-size: 2rem; font-weight: bold; margin: 0;">Risk Probability: {probability:.1%}</p>
                <p style="font-size: 1rem; margin-top: 1rem; opacity: 0.9;">Please consult with a cardiologist for further evaluation</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #4ecdc4, #44a08d); color: white; padding: 2rem; border-radius: 10px; text-align: center; margin: 1rem 0; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                <h2 style="margin: 0; font-size: 1.8rem;">✅ Low Risk</h2>
                <p style="font-size: 1.2rem; margin: 1rem 0;">The patient shows low risk indicators for heart disease</p>
                <p style="font-size: 2rem; font-weight: bold; margin: 0;">Risk Probability: {probability:.1%}</p>
                <p style="font-size: 1rem; margin-top: 1rem; opacity: 0.9;">Continue maintaining a healthy lifestyle</p>
            </div>
            """, unsafe_allow_html=True)

        # Additional information
        st.markdown("""
        <div class="info-box">
            <strong>📌 Important Note:</strong> This prediction is based on AI analysis and should not replace professional medical advice. 
            Always consult with healthcare professionals for proper diagnosis and treatment.
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"خطأ أثناء التوقع: {str(e)}")

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #666; border-top: 1px solid #eee; margin-top: 3rem;">
    <p>❤️ Heart Disease Prediction System | Powered by Advanced Machine Learning</p>
</div>
""", unsafe_allow_html=True)