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

    st.markdown(f"""
    <div class="grade-box">
        <p style="margin:0; color:#374151;">Predicted Final Grade</p>
        <p class="grade">{grade}</p>
    </div>
    """, unsafe_allow_html=True)

    if grade in ["A", "B"]:
        st.success("Strong predicted outcome — patterns associated with higher performance.")
    elif grade == "C":
        st.warning("Moderate predicted outcome — consistent study support may help.")
    else:
        st.error("May benefit from additional academic support or attendance monitoring.")

    st.subheader("Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Average Score", f"{(math_score + science_score + english_score) / 3:.1f}")
    c2.metric("Attendance", f"{attendance_percentage}%")
    c3.metric("Daily Study Time", f"{study_hours:.1f} hrs")

st.markdown(
    '<p class="footer-note">Disclaimer: Predictions are based on training data patterns '
    'and are for educational support purposes only.</p>',
    unsafe_allow_html=True
)
