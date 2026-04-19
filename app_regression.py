import streamlit as st
import pandas as pd
import joblib
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

model = joblib.load("best_model_regression.pkl")

st.set_page_config(page_title="Salary Predictor", layout="wide")

st.title("💰 Student Salary Prediction")

# ================= LAYOUT =================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Academic")

    branch = st.selectbox("Branch", ["CE", "CSE", "IT", "ECE"])
    cgpa = st.slider("CGPA", 0.0, 10.0, 7.0)
    twelfth = st.slider("12th Percentage", 0.0, 100.0, 60.0)
    backlogs = st.number_input("Backlogs", 0, 10, 0)

    st.subheader("💻 Skills")

    coding = st.slider("Coding Skill", 0, 10, 5)
    aptitude = st.slider("Aptitude Skill", 0, 10, 5)
    communication = st.slider("Communication Skill", 0, 10, 5)

with col2:
    st.subheader("📁 Experience")

    projects = st.number_input("Projects", 0, 10, 1)
    internships = st.number_input("Internships", 0, 10, 0)
    hackathons = st.number_input("Hackathons", 0, 10, 0)
    certs = st.number_input("Certifications", 0, 10, 0)

    st.subheader("📈 Study & Activity")

    study_hours = st.slider("Study Hours per Day", 0.0, 12.0, 4.0)
    attendance = st.slider("Attendance (%)", 0.0, 100.0, 80.0)
    internet = st.selectbox("Internet Access", ["Yes", "No"])



input_df = pd.DataFrame([{
    "branch": branch,
    "cgpa": cgpa,
    "twelfth_percentage": twelfth,
    "tenth_percentage": 70.0, 
    "backlogs": backlogs,
    "projects_completed": projects,
    "internships_completed": internships,
    "coding_skill_rating": coding,
    "aptitude_skill_rating": aptitude,
    "communication_skill_rating": communication,
    "hackathons_participated": hackathons,
    "certifications_count": certs,
    "internet_access": internet,
    "study_hours_per_day": study_hours,
    "attendance_percentage": attendance,
    "stress_level": 5  # default
}])

# ================= PREDICT =================
st.divider()

if st.button("💰 Predict Salary", use_container_width=True):
    try:
        pred = model.predict(input_df)[0]

            if pred < 2:
                st.warning("Salary is very low")
            else:
                st.success(f"💸 Estimated Salary: {pred:.2f} LPA")

    except Exception as e:
        st.error(f"Error: {e}")
