import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Student Grade Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stApp {
        background: #f6f8fc;
    }

    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        background: linear-gradient(135deg, #12355b 0%, #1f6f8b 55%, #35a29f 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 22px rgba(18, 53, 91, 0.20);
    }

    .hero h1 {
        color: white;
        margin: 0;
        font-size: 2.2rem;
    }

    .hero p {
        color: #e5f6f6;
        font-size: 1.05rem;
        margin: 0.6rem 0 0 0;
    }

    .section-title {
        color: #12355b;
        font-size: 1.2rem;
        font-weight: 700;
        margin-top: 0.4rem;
        margin-bottom: 0.6rem;
    }

    .info-card {
        background: white;
        padding: 1rem 1.2rem;
        border-radius: 12px;
        border-left: 5px solid #35a29f;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.06);
        margin-bottom: 1.2rem;
    }

    .grade-result {
        background: linear-gradient(135deg, #e8f7f5, #ffffff);
        border: 1px solid #99d8d3;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 5px 14px rgba(31, 111, 139, 0.12);
        margin-top: 1rem;
    }

    .grade-result h2 {
        color: #12355b;
        margin-bottom: 0.5rem;
    }

    .grade-value {
        color: #16847e;
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
    }

    .stButton > button {
        width: 100%;
        border: none;
        border-radius: 10px;
        background: linear-gradient(135deg, #12355b, #1f6f8b);
        color: white;
        font-size: 1.05rem;
        font-weight: 700;
        padding: 0.7rem 1rem;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #0d2947, #15546a);
        color: white;
    }

    div[data-testid="stNumberInput"] input,
    div[data-testid="stSelectbox"] div {
        border-radius: 8px;
    }

    .footer-note {
        color: #6c757d;
        font-size: 0.82rem;
        text-align: center;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load(
        "student_performance_preprocessor.joblib"
    )
    model = joblib.load(
        "student_performance_best_model.joblib"
    )
    return preprocessor, model


try:
    preprocessor, model = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files were not found. Upload "
        "`student_performance_preprocessor.joblib` and "
        "`student_performance_best_model.joblib` to the same GitHub folder as `app.py`."
    )
    st.stop()


st.markdown("""
<div class="hero">
    <h1>🎓 Student Grade Predictor</h1>
    <p>
        Estimate a student's likely final grade using study habits,
        attendance, school context, and subject scores.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-card">
    Complete the student profile below, then select <b>Predict Final Grade</b>.
    This tool is designed to support educational analysis—not to make final academic decisions.
</div>
""", unsafe_allow_html=True)


with st.form("student_prediction_form"):

    left_column, right_column = st.columns(2)

    with left_column:
        st.markdown('<p class="section-title">👤 Student Profile</p>', unsafe_allow_html=True)

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=100,
            value=17,
            step=1
        )

        gender = st.selectbox(
            "Gender",
            ["Female", "Male", "Other"]
        )

        school_type = st.selectbox(
            "School Type",
            ["Public", "Private"]
        )

        parent_education = st.selectbox(
            "Parent Education Level",
            ["High School", "Diploma", "Bachelor", "Master", "PhD"]
        )

        internet_access = st.selectbox(
            "Internet Access",
            ["Yes", "No"]
        )

        extra_activities = st.selectbox(
            "Participates in Extra Activities",
            ["Yes", "No"]
        )

    with right_column:
        st.markdown('<p class="section-title">📚 Study and Academic Records</p>', unsafe_allow_html=True)

        study_hours = st.number_input(
            "Daily Study Hours",
            min_value=0.0,
            max_value=24.0,
            value=3.0,
            step=0.5
        )

        study_method = st.selectbox(
            "Study Method",
            ["Self-study", "Group Study", "Online Learning", "Tutoring"]
        )

        attendance_percentage = st.slider(
            "Attendance Percentage",
            min_value=0,
            max_value=100,
            value=85
        )

        math_score = st.number_input(
            "Math Score",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=1.0
        )

        science_score = st.number_input(
            "Science Score",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=1.0
        )

        english_score = st.number_input(
            "English Score",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=1.0
        )

    st.write("")
    submitted = st.form_submit_button("✨ Predict Final Grade")


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
    <div class="grade-result">
        <h2>Predicted Final Grade</h2>
        <p class="grade-value">{grade}</p>
    </div>
    """, unsafe_allow_html=True)

    if grade in ["A", "B"]:
        st.success(
            "Strong predicted academic outcome. The student profile shows "
            "patterns associated with higher performance in the training data."
        )
    elif grade == "C":
        st.warning(
            "Moderate predicted academic outcome. Continued monitoring and "
            "consistent study support may be beneficial."
        )
    else:
        st.error(
            "The student may benefit from additional academic support, "
            "attendance monitoring, or targeted subject assistance."
        )

    st.markdown("### Student Summary")

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    summary_col1.metric(
        "Average Subject Score",
        f"{(math_score + science_score + english_score) / 3:.1f}"
    )

    summary_col2.metric(
        "Attendance",
        f"{attendance_percentage}%"
    )

    summary_col3.metric(
        "Daily Study Time",
        f"{study_hours:.1f} hrs"
    )

st.markdown("""
<p class="footer-note">
Disclaimer: This prediction is generated from patterns in the training dataset.
It is for educational analysis and student-support purposes only, not a final academic decision.
</p>
""", unsafe_allow_html=True)
