import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import hashlib
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Set Background Image
import streamlit as st

# Apply Custom CSS
page_bg_img = f"""
<style>
/* Background Styling */
[data-testid="stAppViewContainer"] {{
    background: linear-gradient(to bottom, rgba(255,255,255,0.7), rgba(0,0,0,0.9)), 
                url("https://www.guideir.com/Public/Uploads/uploadfile/images/20220609/16MedicalDiagnosis2.jpg"); 
    background-size: cover;
    background-position: top;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

/* Fix scrolling */
html, body, [data-testid="stAppViewContainer"] {{
    height: 100vh !important;
    overflow-y: auto !important;
}}

[data-testid="stVerticalBlock"] {{
    max-height: 90vh !important;
    overflow-y: auto !important;
}}

/* Improve label contrast */
label {{
    color: #222222 !important;  /* Dark gray for better visibility */
    font-weight: bold !important;
}}

/* Darken form elements (inputs, selects, sliders) */
input, select, textarea {{
    color: black !important;
    font-weight: bold !important;
}}

/* Make all buttons Neon Green */
button, .stButton button {{
    background-color: #90EE90 !important;  /* Light Green */
    color: black !important;
    font-weight: bold !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    transition: 0.3s;
}}

button:hover, .stButton button:hover {{
    background-color: #77DD77 !important; /* Slightly darker pastel green */
}}

/* Make "No Disease Detected" message Dark Blue */
.st-alert {{
    background-color: #002D62 !important; /* Dark blue */
    color: white !important;
    font-weight: bold !important;
    border-radius: 8px;
    padding: 10px;
}}

/* Make "File uploaded successfully" & uploaded file name (cancer.csv) even darker */
.st-emotion-cache-1itfubd p, .stFileUploader p {{
    color: #111111 !important;  /* Even Darker Black */
    font-weight: bold !important;
}}

</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# Load Model and Scaler
@st.cache_resource
def load_model(disease):
    model_path = f"{disease}_model.pkl"
    model= None
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            model = joblib.load(file)
    else:
        st.error(f"Error: {model_path} not found!")
    
    return model

# Sidebar Disease Selection
diseases = ["Blood cancer", "Diabetes", "Cardiovascular disease", "Hypothyroid", "Kidney disease", "Liver disease", "Parkinson disease", "Tuberculosis"]
selected_disease = st.sidebar.selectbox("Choose a Disease", diseases)
st.title(f"\U0001FA7A AI-Powered {selected_disease} Prediction")

# Load Model
model= load_model(selected_disease.lower())

# Sidebar Inputs
st.sidebar.title("Enter Patient Data")

if selected_disease == "Blood cancer":
    age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=45)
    white_blood_cells = st.sidebar.number_input("White Blood Cells (x10^9/L)", min_value=1, max_value=500, value=10)
    rbc_count = st.sidebar.number_input("RBC Count (million cells/µL)", min_value=2.0, max_value=7.0, value=4.5)
    hemoglobin = st.sidebar.number_input("Hemoglobin (g/dL)", min_value=3.0, max_value=20.0, value=13.5)
    platelet_count = st.sidebar.number_input("Platelet Count (x10^9/L)", min_value=10, max_value=500, value=250)
    lymphocyte_percentage = st.sidebar.number_input("Lymphocyte %", min_value=1.0, max_value=100.0, value=30.0)
    blast_cells = st.sidebar.number_input("Blast Cells %", min_value=0.0, max_value=100.0, value=2.0)
    bone_marrow = st.sidebar.number_input("Bone Marrow Value", min_value=30.0, max_value=100.0, value=50.0)
    

    input_features = np.array([
        age, white_blood_cells,rbc_count, hemoglobin, platelet_count,
        lymphocyte_percentage, blast_cells, bone_marrow]).reshape(1, -1)

elif selected_disease == "Diabetes":
    age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=45)
    glucose = st.sidebar.number_input("Glucose Level", min_value=50, max_value=300, value=100)
    blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=60, max_value=200, value=120)
    insulin = st.sidebar.number_input("Insulin Level", min_value=0, max_value=1000, value=80)
    bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    family_history = st.sidebar.selectbox("Family History of Diabetes", ["No", "Yes"])
    physical_activity = st.sidebar.selectbox("Physical Activity Level", ["Low", "Medium", "High"])
    
    # Convert categorical to numerical
    family_history_map = {"No": 0, "Yes": 1}
    physical_activity_map = {"Low": 0, "Medium": 1, "High": 2}
    
    input_features = np.array([
        age, glucose, blood_pressure, insulin, bmi,
        family_history_map[family_history], physical_activity_map[physical_activity]
    ]).reshape(1, -1)

if selected_disease == "Cardiovascular disease":
    age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=50)
    cholesterol = st.sidebar.number_input("Cholesterol (mg/dL)", min_value=100, max_value=500, value=200)
    resting_bp = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
    max_heart_rate = st.sidebar.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
    fasting_blood_sugar = st.sidebar.number_input("Fasting Blood Sugar (mg/dL)", min_value=50, max_value=300, value=100)
    st_depression = st.sidebar.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0)
    exercise_angina = st.sidebar.selectbox("Exercise-Induced Angina", ["No", "Yes"])
    blood_pressure = st.sidebar.number_input("Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
    
    # Convert categorical to numerical
    exercise_angina_map = {"No": 0, "Yes": 1}
    input_features = np.array([
        age, cholesterol, resting_bp, max_heart_rate,
        fasting_blood_sugar, st_depression,
        exercise_angina_map[exercise_angina], blood_pressure
    ]).reshape(1, -1)
    

elif  selected_disease == "Hypothyroid":
    age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=50)
    tsh = st.sidebar.number_input("Thyroid Stimulating Hormone (TSH) (mIU/L)", min_value=0.1, max_value=10.0, value=2.0)
    t4 = st.sidebar.number_input("Free Thyroxine (T4) (ng/dL)", min_value=0.1, max_value=3.0, value=1.2)
    t3 = st.sidebar.number_input("Triiodothyronine (T3) (pg/mL)", min_value=0.5, max_value=5.0, value=2.5)
    weight_gain = st.sidebar.selectbox("Weight Gain", ["No", "Yes"])
    fatigue = st.sidebar.selectbox("Fatigue", ["No", "Yes"])
    cold_intolerance = st.sidebar.selectbox("Cold Intolerance", ["No", "Yes"])
    hair_thinning = st.sidebar.selectbox("Hair Thinning", ["No", "Yes"])
    
    # Convert categorical to numerical
    binary_map = {"No": 0, "Yes": 1}
    input_features = np.array([
        age, tsh, t4, t3, 
        binary_map[weight_gain], binary_map[fatigue], 
        binary_map[cold_intolerance], binary_map[hair_thinning]
    ]).reshape(1, -1)


elif selected_disease == "Kidney disease":
    age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=45)
    blood_urea = st.sidebar.number_input("Blood Urea Nitrogen (mg/dL)", min_value=0.0, max_value=100.0, value=20.0)
    serum_creatinine = st.sidebar.number_input("Serum Creatinine (mg/dL)", min_value=0.1, max_value=10.0, value=1.2)
    gfr = st.sidebar.number_input("Glomerular Filtration Rate (mL/min/1.73 m²)", min_value=0.0, max_value=150.0, value=90.0)
    sodium = st.sidebar.number_input("Sodium (mEq/L)", min_value=100, max_value=160, value=140)
    potassium = st.sidebar.number_input("Potassium (mEq/L)", min_value=2.0, max_value=8.0, value=4.5)
    hemoglobin = st.sidebar.number_input("Hemoglobin (g/dL)", min_value=5.0, max_value=20.0, value=13.0)
    protein_in_urine = st.sidebar.selectbox("Protein in Urine", ["No", "Yes"])
    fatigue = st.sidebar.selectbox("Fatigue", ["No", "Yes"])

    # Convert categorical to numerical
    protein_in_urine_map = {"No": 0, "Yes": 1}
    fatigue_map = {"No": 0, "Yes": 1}

    # Creating input features array
    input_features = np.array([
        age, blood_urea, serum_creatinine, gfr, sodium, potassium, 
        hemoglobin, protein_in_urine_map[protein_in_urine], fatigue_map[fatigue]
    ]).reshape(1, -1)
    
elif selected_disease == "Liver disease":
    age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=45)
    bilirubin = st.sidebar.number_input("Bilirubin Level (mg/dL)", min_value=0.1, max_value=10.0, value=1.0)
    alkaline_phosphatase = st.sidebar.number_input("Alkaline Phosphatase (U/L)", min_value=50, max_value=400, value=100)
    aspartate_aminotransferase = st.sidebar.number_input("Aspartate Aminotransferase (U/L)", min_value=10, max_value=500, value=30)
    alanine_aminotransferase = st.sidebar.number_input("Alanine Aminotransferase (U/L)", min_value=10, max_value=500, value=30)
    albumin = st.sidebar.number_input("Albumin (g/dL)", min_value=2.0, max_value=6.0, value=4.0)
    prothrombin_time = st.sidebar.number_input("Prothrombin Time (seconds)", min_value=8.0, max_value=25.0, value=12.0)
    jaundice = st.sidebar.selectbox("Jaundice", ["No", "Yes"])
    fatigue = st.sidebar.selectbox("Fatigue", ["No", "Yes"])

    # Convert categorical to numerical
    jaundice_map = {"No": 0, "Yes": 1}
    fatigue_map = {"No": 0, "Yes": 1}

    # Creating input features array
    input_features = np.array([
        age, bilirubin, alkaline_phosphatase, aspartate_aminotransferase,
        alanine_aminotransferase, albumin, prothrombin_time, 
        jaundice_map[jaundice], fatigue_map[fatigue]
    ]).reshape(1, -1)

elif selected_disease == "Parkinson disease":
    age = st.sidebar.number_input("Age", min_value=30, max_value=100, value=60)
    tremor_intensity = st.sidebar.slider("Tremor Intensity (0-10)", min_value=0.0, max_value=10.0, value=2.0)
    muscle_rigidity = st.sidebar.slider("Muscle Rigidity (0-10)", min_value=0.0, max_value=10.0, value=2.0)
    bradykinesia = st.sidebar.slider("Bradykinesia (0-10)", min_value=0.0, max_value=10.0, value=2.0)
    postural_instability = st.sidebar.slider("Postural Instability (0-10)", min_value=0.0, max_value=10.0, value=2.0)
    jitter = st.sidebar.number_input("Jitter (%)", min_value=0.0, max_value=5.0, value=0.5)
    speech_impairment = st.sidebar.slider("Speech Impairment (0-10)", min_value=0.0, max_value=10.0, value=2.0)
    shuffling_gait = st.sidebar.selectbox("Shuffling Gait", ["No", "Yes"])
    depression = st.sidebar.selectbox("Depression", ["No", "Yes"])

    # Convert categorical to numerical
    shuffling_map = {"No": 0, "Yes": 1}
    depression_map = {"No": 0, "Yes": 1}

    # Creating input features array
    input_features = np.array([
        age, tremor_intensity, muscle_rigidity, bradykinesia, postural_instability, 
        jitter, speech_impairment, shuffling_map[shuffling_gait], depression_map[depression]
    ]).reshape(1, -1)

elif selected_disease == "Tuberculosis":
    age = st.sidebar.number_input("Age", min_value=10, max_value=100, value=30)
    fever_duration = st.sidebar.slider("Fever Duration (days)", min_value=0.0, max_value=50.0, value=5.0)
    cough_duration = st.sidebar.slider("Cough Duration (days)", min_value=0.0, max_value=50.0, value=10.0)
    weight_loss = st.sidebar.slider("Weight Loss (kg)", min_value=0.0, max_value=15.0, value=3.0)
    chest_pain = st.sidebar.selectbox("Chest Pain", ["No", "Yes"])
    night_sweats = st.sidebar.selectbox("Night Sweats", ["No", "Yes"])
    fatigue = st.sidebar.selectbox("Fatigue", ["No", "Yes"])
    sputum_production = st.sidebar.selectbox("Sputum Production", ["No", "Yes"])

    # Convert categorical to numerical
    binary_map = {"No": 0, "Yes": 1}

    # Creating input features array
    input_features = np.array([
        age, fever_duration, cough_duration, weight_loss,
        binary_map[chest_pain], binary_map[night_sweats], 
        binary_map[fatigue], binary_map[sputum_production]
    ]).reshape(1, -1)

# Prediction Button
if st.sidebar.button("Predict"):
    if model:
        prediction = model.predict(input_features)
        result = "\U0001F6D1 Disease Detected!" if prediction[0] == 1 else "\U00002705 No Disease Detected"
        st.sidebar.success(f"Prediction: {result}")
    else:
        st.sidebar.error("Model not found! Train and save the model first.")
    
# Simulating ML predictions with random probabilities (Replace this with actual model predictions)
def predict_disease():
    diseases = ["Blood Cancer", "Diabetes", "Hypothyroid", "Cardiovascular Disease", 
                "Kidney Disease", "Liver Disease", "Parkinson’s Disease", "Tuberculosis"]
    return {disease: np.random.uniform(0, 1) for disease in diseases}

disease_symptoms = {
    "Blood Cancer": ["Fatigue", "Frequent infections", "Unexplained weight loss", "Easy bruising", "Swollen lymph nodes"],
    "Diabetes": ["Frequent urination", "Excessive thirst", "Unexplained weight loss", "Fatigue", "Blurred vision"],
    "Hypothyroid": ["Fatigue", "Weight gain", "Cold intolerance", "Depression", "Slow heart rate"],
    "Cardiovascular Disease": ["Chest pain", "Shortness of breath", "Fatigue", "Swelling in legs", "Dizziness"],
    "Kidney Disease": ["Swelling in feet", "Fatigue", "High blood pressure", "Nausea", "Frequent urination"],
    "Liver Disease": ["Jaundice", "Abdominal pain", "Loss of appetite", "Dark urine", "Swelling in legs"],
    "Parkinson’s Disease": ["Tremors", "Slow movement", "Stiffness", "Loss of balance", "Speech changes"],
    "Tuberculosis": ["Chronic cough", "Night sweats", "Fever", "Unexplained weight loss", "Chest pain"]
}

def assess_risk(selected_symptoms, disease_symptoms):
    risk_score = sum([1 for symptoms in disease_symptoms.values() if any(symptom in symptoms for symptom in selected_symptoms)])
    # Display risk level progress with a message
    st.subheader("\U0001F4CA Health Risk Assessment Progress")
    # Progress bar with dynamic text
    progress = risk_score / len(disease_symptoms) 
    st.progress(progress)
    progress_percentage = (risk_score / len(disease_symptoms)) * 100
    st.write(f"**Risk Score Progress: {progress_percentage:.2f}%**")
    if risk_score >= 5:
        return risk_score, "High Risk"
    elif risk_score >= 3:
        return risk_score, "Medium Risk"
    else:
        return risk_score, "Low Risk"
    

def generate_report(name, risk_score, risk_level, selected_symptoms, disease_predictions):
    report_html = f"""
    <div style="background-color: white; padding: 20px; border-radius: 10px;">
        <h2 style="text-align: center;">&#x1F4DD; Diagnosis Report</h2>
        <p><strong>&#x1F9D1;&#x200D;&#x2695;&#xFE0F; Patient Name:</strong> {name}</p>
        <p><strong>&#x1F4CA; Risk Score:</strong> {risk_score} / {len(disease_symptoms)}</p>
        <p><strong>&#x26A0;&#xFE0F; Risk Level:</strong> {risk_level}</p>
        <h3>&#x1FA7A; Selected Symptoms:</h3>
        <ul>
            {''.join(f'<li>{symptom}</li>' for symptom in selected_symptoms)}
        </ul>
        <h3>&#x1F52C; Disease Prediction Probabilities:</h3>
        <ul>
            {''.join(f'<li><strong>{disease}:</strong> {prob:.2%}</li>' for disease, prob in disease_predictions.items())}
        </ul>
        <h3>&#x1F3E5; Health Recommendations:</h3>
        <p style="color: {'red' if risk_level == 'High Risk' else 'orange' if risk_level == 'Medium Risk' else 'green'};">
            { '&#x1F6A8; Immediate medical attention recommended!' if risk_level == 'High Risk' else 
               '&#x26A0;&#xFE0F; Monitor your symptoms and seek medical advice.' if risk_level == 'Medium Risk' else 
               '&#x2705; Low risk, but stay alert!' }
        </p>
    </div>
    """
    return report_html

st.title("&#x1FA7A; Disease Risk Assessment & Prediction")

name = st.text_input("Enter your name:")
selected_symptoms = st.multiselect("Select your symptoms:", sum(disease_symptoms.values(), []))

if st.button("Predict & Generate Report"):
    if name and selected_symptoms:
        risk_score, risk_level = assess_risk(selected_symptoms, disease_symptoms)
        disease_predictions = predict_disease()
        report_html = generate_report(name, risk_score, risk_level, selected_symptoms, disease_predictions)
        st.markdown(report_html, unsafe_allow_html=True)
    else:
        st.warning("&#x26A0;&#xFE0F; Please enter your name and select at least one symptom.")       

# Longitudinal Tracking - Patient Progression
uploaded_progress = st.file_uploader("Upload CSV for Patient Tracking", type=["csv"], key="patient_tracking_upload")
if uploaded_progress:
    df_progress = pd.read_csv(uploaded_progress)
    st.success("File uploaded successfully!")
    
    st.write("Columns in uploaded file:", df_progress.columns)
    df_progress.columns = df_progress.columns.str.strip().str.lower()

    # Generate patient_id if not present
    if "patient_id" not in df_progress.columns:
        df_progress["patient_id"] = df_progress.apply(lambda row: hashlib.md5(str(row.values).encode()).hexdigest(), axis=1)
        st.warning("No 'Patient_ID' column found. Generated unique IDs based on row data.")

    patient_ids = df_progress["patient_id"].unique()
    selected_patient = st.selectbox("Select Patient ID", patient_ids)

    if st.button("Show Progress"):
        # Filter data for the selected patient
        patient_data = df_progress[df_progress["patient_id"] == selected_patient].copy()

        # Check if there are multiple entries to create a sequential time index
        if len(patient_data) > 1:
            patient_data["time_index"] = range(1, len(patient_data) + 1)
            index_column = "time_index"
        else:
            index_column = None  # No need for an index if only one row is available

        # Select only numerical columns (excluding patient_id)
        numeric_columns = patient_data.select_dtypes(include=["number"]).columns.tolist()
        numeric_columns = [col for col in numeric_columns if col != "patient_id"]

        if numeric_columns:
            if index_column:
                st.line_chart(patient_data.set_index(index_column)[numeric_columns])
            else:
                st.line_chart(patient_data[numeric_columns])
        else:
            st.error("No valid numerical columns found for visualization.")

# Demographic-Specific Analytics
uploaded_demo = st.file_uploader("Upload CSV for Demographic Analysis", type=["csv"])
if uploaded_demo:
    df_demo = pd.read_csv(uploaded_demo)
    st.write("### ## \U0001F4CA Demographic-Based Prediction Analysis")
    age_bins = [0, 20, 40, 60, 80, 100]
    df_demo["Age Group"] = pd.cut(df_demo["Age"], bins=age_bins, labels=["0-20", "21-40", "41-60", "61-80", "81-100"])
    plt.figure(figsize=(8, 5))
    sns.countplot(x="Age Group", hue="Diagnosis", data=df_demo, palette="coolwarm")
    st.pyplot(plt)     

# Dataset Upload & Visualization
st.write("## \U0001F4CA Data Visualization")
uploaded_file = st.file_uploader("Upload a CSV dataset for analysis", type=["csv"])

if uploaded_file is not None:  # Ensure the file is uploaded before processing
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    st.write("### Dataset Overview")
    st.write(df.head())

    # Convert categorical columns to numeric (if applicable)
    df_numeric = df.select_dtypes(include=["number"])  # Keep only numeric columns

    if df_numeric.shape[1] > 1:  # Ensure there are at least two numeric columns
        st.write("### Feature Correlations")
        plt.figure(figsize=(10, 6))
        sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        st.pyplot(plt)
    else:
        st.warning("Not enough numeric columns for correlation analysis.")


# Lifestyle & Diet Recommendations
health_tips = {
    "Blood Cancer": {
        "Diet": [
            "\U0001F34E Consume iron-rich foods.",
            "\U0001F969 Avoid processed meats.",
            "\U0001F4A7 Stay hydrated."
        ],
        "Lifestyle": [
            "\U0001F468\u200D Regular check-ups are essential." 
        ],
        "Exercise": [
            "\U0001F3CB\uFE0F Engage in light physical activity."
        ]
    },
    "Diabetes": {
        "Diet": [
            "\U0001F36C Avoid sugary foods.",
            "\U0001F33F Consume whole grains and high-fiber foods."
        ],
        "Lifestyle": [
            "\U0001F4AA Maintain a healthy weight."
        ],
        "Exercise": [
            "\U0001F3C3 Engage in a regular exercise routine."
        ]
    },
    "Cardiovascular Disease": {
        "Diet": [
            "\U0001F9C0 Reduce salt intake.",
            "\U0001F41F Eat heart-healthy foods like nuts & fish."
        ],
        "Lifestyle": [
            "\U0001F6AC Avoid smoking and alcohol."
        ],
        "Exercise": [
            "\U0001F3CB\uFE0F Engage in regular physical activity."
        ]
    },
    "Hypothyroid": {
        "Diet": [
            "\U0001F969 Eat iodine-rich foods like seaweed.",
            "\U0001F344 Avoid goitrogens (cabbage, soy)."
        ],
        "Lifestyle": [
            "\U000026A1 Ensure sufficient selenium intake."
        ],
        "Exercise": [
            "\U0001F3C3 Engage in moderate physical activity."
        ]
    },
    "Kidney Disease": {
        "Diet": [
            "\U0001F950 Monitor protein intake.",
            "\U0001F9C2 Reduce sodium and avoid processed foods."
        ],
        "Lifestyle": [
            "\U0001F4A7 Stay hydrated with clean water."
        ],
        "Exercise": [
            "\U0001F3C3 Engage in gentle physical movement."
        ]
    },
    "Liver Disease": {
        "Diet": [
            "\U0001F37A Limit alcohol consumption.",
            "\U0001F355 Avoid fatty foods.",
            "\U0001F34C Increase antioxidant intake (fruits & veggies)."
        ],
        "Lifestyle": [
            "\U0001F4A7 Stay hydrated."
        ],
        "Exercise": [
            "\U0001F3CB\uFE0F Engage in light physical activity."
        ]
    },
    "Parkinson’s Disease": {
        "Diet": [
            "\U0001F34E Eat antioxidant-rich foods (berries, greens).",
            "\U0001F95B Avoid excess dairy."
        ],
        "Lifestyle": [
            "\U0001F3CB\uFE0F Maintain an active lifestyle."
        ],
        "Exercise": [
            "\U0001F3C3 Practice balance & coordination exercises."
        ]
    },
    "Tuberculosis": {
        "Diet": [
            "\U0001F969 Increase protein intake.",
            "\U0001F31E Consume vitamin D-rich foods."
        ],
        "Lifestyle": [
            "\U0001F6AC Avoid smoking and alcohol consumption."
        ],
        "Exercise": [
            "\U0001F3C3 Engage in breathing exercises."
        ]
    }
}

st.subheader("\U0001F50D Get Lifestyle & Diet Recommendations")   

# Input field to manually enter a disease for recommendations
predicted_disease = st.selectbox("Select a disease:", [""] + list(health_tips.keys()))

# Button to show recommendations
if st.button("Show Recommendations"):
    if predicted_disease:
        tips = health_tips[predicted_disease]
        st.markdown(
            f"""
            <div style="background-color: white; padding: 20px; border-radius: 10px;">
                <h3>\U0001F4A1 Lifestyle & Diet Recommendations for {predicted_disease}</h3>
                <h4>\U0001F374 Diet:</h4>
                <ul style="color: black;">{''.join(f'<li>{tip}</li>' for tip in tips['Diet'])}</ul>
                <h4>\U0001F3E5 Lifestyle:</h4>
                <ul style="color: black;">{''.join(f'<li>{tip}</li>' for tip in tips['Lifestyle'])}</ul>
                <h4>\U0001F3CB\uFE0F Exercise:</h4>
                <ul style="color: black;">{''.join(f'<li>{tip}</li>' for tip in tips['Exercise'])}</ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("\U000026A0 Please select a disease to get recommendations.")
