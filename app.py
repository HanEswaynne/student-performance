import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Student Grade Predictor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Minimal, clean styling ---
st.markdown("""
<style>
    .block-container {
        max-width: 800px;
        padding-top: 2rem;
    }

    h1 {
        color: #1f2937;
    }

    .subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin-top: -0.5rem;
        margin-bottom: 1.5rem;
    }

    .grade-box {
        text-align: center;
        padding: 1.5rem;
        border-radius: 12px;
        background: #f0fdfa;
        border: 1px solid #99d8d3;
        margin: 1rem 0;
    }

    .grade-box .grade {
        font-size: 3rem;
        font-weight: 800;
        color: #0f766e;
        margin: 0;
    }

    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem;
    }

    .footer-note {
        color: #9ca3af;
        font-size: 0.8rem;
        text-align: center;
        margin-top: 2rem;
    }

    .profile-note {
        color: #6b7280;
        font-size: 0.85rem;
        font-style: italic;
        margin-bottom: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load("student_performance_preprocessor.joblib")
    model = joblib.load("student_performance_best_model.joblib")
    return preprocessor, model


try:
    preprocessor, model = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files were not found. Upload "
        "`student_performance_preprocessor.joblib` and "
        "`student_performance_best_model.joblib` to the same folder as `app.py`."
    )
    st.stop()


# --- Header ---
st.title("🎓 Student Grade Predictor")
st.markdown(
    '<p class="subtitle">Estimate a student\'s likely final grade from study habits, '
    'attendance, and subject scores.</p>',
    unsafe_allow_html=True
)
st.info("Fill in the student profile, then click **Predict Final Grade**. "
        "For educational analysis, not final academic decisions.")

# --- Form ---
with st.form("student_prediction_form"):

    st.subheader("👤 Student Profile")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=100, value=17, step=1)
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        school_type = st.selectbox("School Type", ["Public", "Private"])

    with col2:
        parent_education = st.selectbox(
            "Parent Education Level",
            ["High School", "Diploma", "Bachelor", "Master", "PhD"]
        )
        internet_access = st.selectbox("Internet Access", ["Yes", "No"])
        extra_activities = st.selectbox("Extra Activities", ["Yes", "No"])

    st.divider()
    st.subheader("📚 Study & Academic Records")
    col3, col4 = st.columns(2)

    with col3:
        study_hours = st.number_input(
            "Daily Study Hours", min_value=0.0, max_value=24.0, value=3.0, step=0.5
        )
        study_method = st.selectbox(
            "Study Method",
            ["Self-study", "Group Study", "Online Learning", "Tutoring"]
        )
        attendance_percentage = st.slider("Attendance Percentage", 0, 100, 85)

    with col4:
        math_score = st.number_input("Math Score", 0.0, 100.0, 70.0, step=1.0)
        science_score = st.number_input("Science Score", 0.0, 100.0, 70.0, step=1.0)
        english_score = st.number_input("English Score", 0.0, 100.0, 70.0, step=1.0)

    st.write("")
    submitted = st.form_submit_button("✨ Predict Final Grade")


# --- Prediction ---
if submitted:

    new_student = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "school_type": school_type,
        "parent_education": parent_education,
        "study_hours": study_hours,
        "study_method": study_method,
        "internet_access": internet_access,
        "attendance_percentage": attendance_percentage,
        "extra_activities": extra_activities,
        "math_score": math_score,
        "science_score": science_score,
        "english_score": english_score
    }])

    processed_student = preprocessor.transform(new_student)
    prediction = model.predict(processed_student)[0]
    grade = str(prediction).strip().upper()
    average_subject_score = (math_score + science_score + english_score) / 3

    st.markdown(f"""
    <div class="grade-box">
        <p style="margin:0; color:#374151;">Predicted Final Grade</p>
        <p class="grade">{grade}</p>
    </div>
    """, unsafe_allow_html=True)

    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(processed_student)[0]
            class_labels = model.classes_
            pred_idx = list(class_labels).index(prediction)
            confidence = proba[pred_idx]
            st.markdown("**Model Confidence**")
            st.progress(float(confidence))
            st.caption(f"{confidence * 100:.1f}%")
        except (AttributeError, IndexError, ValueError, TypeError):
            pass

    if grade in ["A", "B"]:
        st.success("Strong predicted outcome — patterns associated with higher performance.")
    elif grade == "C":
        st.warning("Moderate predicted outcome — consistent study support may help.")
    else:
        st.error("May benefit from additional academic support or attendance monitoring.")

    st.subheader("Academic Summary")
    sum1, sum2, sum3, sum4 = st.columns(4)
    sum1.metric("Math Score", f"{math_score:.1f}")
    sum2.metric("Science Score", f"{science_score:.1f}")
    sum3.metric("English Score", f"{english_score:.1f}")
    sum4.metric("Average Subject Score", f"{average_subject_score:.1f}")

    st.subheader("Learning Profile")
    st.markdown(
        '<p class="profile-note">These are input-based indicators, not direct '
        "explanations of the model prediction.</p>",
        unsafe_allow_html=True
    )

    strengths = []
    areas_for_support = []

    if average_subject_score >= 75:
        strengths.append("Strong overall subject performance")
    if average_subject_score < 60:
        areas_for_support.append("Subject-score improvement may be needed")
    if attendance_percentage >= 90:
        strengths.append("Strong attendance record")
    if attendance_percentage < 75:
        areas_for_support.append("Attendance may need improvement")
    if study_hours >= 2:
        strengths.append("Consistent daily study time")
    if study_hours < 2:
        areas_for_support.append("More regular study time may be beneficial")
    if internet_access == "Yes":
        strengths.append("Access to online learning resources")

    profile_left, profile_right = st.columns(2)

    with profile_left:
        st.markdown("**Strengths**")
        if strengths:
            for item in strengths:
                st.markdown(f"- {item}")
        else:
            st.markdown("_No specific strengths identified from inputs._")

    with profile_right:
        st.markdown("**Areas for Support**")
        if areas_for_support:
            for item in areas_for_support:
                st.markdown(f"- {item}")
        else:
            st.markdown("_No specific support areas identified from inputs._")

    report_df = pd.DataFrame([{
        "predicted_final_grade": grade,
        "age": age,
        "gender": gender,
        "school_type": school_type,
        "parent_education": parent_education,
        "study_hours": study_hours,
        "study_method": study_method,
        "internet_access": internet_access,
        "attendance_percentage": attendance_percentage,
        "extra_activities": extra_activities,
        "math_score": math_score,
        "science_score": science_score,
        "english_score": english_score,
        "average_subject_score": average_subject_score
    }])

    st.download_button(
        label="⬇️ Download Prediction Report",
        data=report_df.to_csv(index=False),
        file_name="student_grade_prediction.csv",
        mime="text/csv"
    )

with st.expander("ℹ️ About this prediction system"):
    st.markdown("""
This system predicts a student's **final grade** using **12 selected features** from the
student profile and academic records entered in the form.

**Features used:** age, gender, school type, parent education level, daily study hours,
study method, internet access, attendance percentage, extra activities, and scores in
math, science, and English.

**Excluded from the model:** student identifiers, travel time, and overall score.
Overall score is intentionally excluded to **reduce target leakage**, since it is closely
related to the final grade being predicted.

**Model selection:** Logistic Regression, K-Nearest Neighbors (KNN), Support Vector
Machine (SVM), and Artificial Neural Network (ANN) were trained and compared. The best
performing model was selected using **weighted F1-score**.
    """)

st.markdown(
    '<p class="footer-note">Disclaimer: This prediction is generated from patterns in the '
    "training dataset. It is intended for educational analysis and student-support "
    "purposes only, not as a final academic decision.</p>",
    unsafe_allow_html=True
)
