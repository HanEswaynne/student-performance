import streamlit as st
import pandas as pd
import joblib

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Grade Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background: #f4f7fb;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Remove unnecessary top spacing */
    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* ---------- HERO ---------- */

    .hero {
        background: linear-gradient(
            135deg,
            #102a43 0%,
            #155e75 50%,
            #159a9c 100%
        );

        padding: 2.8rem 3rem;
        border-radius: 24px;
        color: white;
        margin-bottom: 1.5rem;

        box-shadow:
            0 12px 30px rgba(16, 42, 67, 0.18);

        position: relative;
        overflow: hidden;
    }

    .hero::after {
        content: "🎓";
        position: absolute;
        right: 50px;
        top: 25px;
        font-size: 7rem;
        opacity: 0.12;
    }

    .hero h1 {
        color: white;
        font-size: 2.7rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -1px;
    }

    .hero p {
        color: #d9f5f4;
        font-size: 1.08rem;
        margin-top: 0.7rem;
        max-width: 700px;
        line-height: 1.6;
    }

    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.25);
        padding: 0.35rem 0.8rem;
        border-radius: 50px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
        backdrop-filter: blur(5px);
    }

    /* ---------- INFO CARD ---------- */

    .info-card {
        background: white;
        padding: 1.1rem 1.3rem;
        border-radius: 14px;

        border: 1px solid #e3eaf2;
        border-left: 5px solid #159a9c;

        box-shadow:
            0 4px 14px rgba(16, 42, 67, 0.05);

        margin-bottom: 1.5rem;

        color: #52606d;
        line-height: 1.6;
    }

    .info-card b {
        color: #102a43;
    }

    /* ---------- SECTION HEADERS ---------- */

    .section-title {
        color: #102a43;
        font-size: 1.18rem;
        font-weight: 750;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e8eef5;
    }

    .section-subtitle {
        color: #7b8794;
        font-size: 0.88rem;
        margin-top: -0.5rem;
        margin-bottom: 1rem;
    }

    /* ---------- FORM CARD ---------- */

    div[data-testid="stForm"] {
        background: white;
        border: 1px solid #e3eaf2;
        border-radius: 20px;
        padding: 1.5rem 1.8rem 1.8rem 1.8rem;

        box-shadow:
            0 8px 25px rgba(16, 42, 67, 0.06);
    }

    /* ---------- INPUTS ---------- */

    div[data-baseweb="select"] > div {
        border-radius: 10px;
        border-color: #d9e2ec;
    }

    div[data-testid="stNumberInput"] input {
        border-radius: 10px;
        border-color: #d9e2ec;
    }

    div[data-testid="stSlider"] {
        padding-top: 0.3rem;
    }

    label {
        color: #334e68 !important;
        font-weight: 600 !important;
    }

    /* ---------- PREDICT BUTTON ---------- */

    .stButton > button,
    button[kind="primaryFormSubmit"] {

        width: 100%;

        border: none;
        border-radius: 12px;

        background: linear-gradient(
            135deg,
            #102a43,
            #155e75
        );

        color: white;

        font-size: 1.05rem;
        font-weight: 700;

        padding: 0.75rem 1rem;

        box-shadow:
            0 6px 15px rgba(16, 42, 67, 0.18);

        transition: all 0.2s ease;
    }

    .stButton > button:hover,
    button[kind="primaryFormSubmit"]:hover {

        background: linear-gradient(
            135deg,
            #0b2033,
            #104c60
        );

        color: white;

        transform: translateY(-1px);

        box-shadow:
            0 8px 18px rgba(16, 42, 67, 0.25);
    }

    /* ---------- RESULT ---------- */

    .result-card {

        background: linear-gradient(
            135deg,
            #e8f8f6,
            #ffffff
        );

        border: 1px solid #b5e3df;

        border-radius: 22px;

        padding: 2rem;

        text-align: center;

        box-shadow:
            0 10px 28px rgba(21, 154, 156, 0.10);

        margin-top: 1.8rem;
    }

    .result-label {
        color: #52606d;
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }

    .grade-value {
        color: #159a9c;
        font-size: 4.5rem;
        line-height: 1;
        font-weight: 900;
        margin: 0.4rem 0;
    }

    .result-description {
        color: #627d98;
        font-size: 0.95rem;
    }

    /* ---------- METRIC CARDS ---------- */

    .metric-card {

        background: white;

        border: 1px solid #e3eaf2;

        border-radius: 15px;

        padding: 1.2rem;

        text-align: center;

        box-shadow:
            0 5px 16px rgba(16, 42, 67, 0.05);
    }

    .metric-icon {
        font-size: 1.5rem;
        margin-bottom: 0.3rem;
    }

    .metric-title {
        color: #7b8794;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .metric-value {
        color: #102a43;
        font-size: 1.45rem;
        font-weight: 800;
        margin-top: 0.25rem;
    }

    /* ---------- HOW IT WORKS ---------- */

    .how-card {

        background: white;

        border: 1px solid #e3eaf2;

        border-radius: 15px;

        padding: 1.2rem;

        height: 100%;

        box-shadow:
            0 5px 15px rgba(16, 42, 67, 0.04);
    }

    .how-number {

        width: 34px;
        height: 34px;

        border-radius: 50%;

        background: #e8f8f6;

        color: #159a9c;

        display: flex;
        align-items: center;
        justify-content: center;

        font-weight: 800;

        margin-bottom: 0.7rem;
    }

    .how-title {
        color: #102a43;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }

    .how-text {
        color: #7b8794;
        font-size: 0.88rem;
        line-height: 1.5;
    }

    /* ---------- FOOTER ---------- */

    .footer {

        margin-top: 3rem;
        padding-top: 1.2rem;

        border-top: 1px solid #d9e2ec;

        text-align: center;

        color: #9aa5b1;

        font-size: 0.78rem;
        line-height: 1.5;
    }

    /* ---------- MOBILE ---------- */

    @media (max-width: 768px) {

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .hero {
            padding: 2rem 1.5rem;
        }

        .hero h1 {
            font-size: 2rem;
        }

        .hero::after {
            display: none;
        }

        div[data-testid="stForm"] {
            padding: 1.2rem;
        }

        .grade-value {
            font-size: 3.5rem;
        }
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

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
        "Model files were not found. Please upload "
        "`student_performance_preprocessor.joblib` and "
        "`student_performance_best_model.joblib` "
        "to the same GitHub folder as `app.py`."
    )

    st.stop()


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">

    <div class="hero-badge">
        🤖 Machine Learning • Educational Analytics
    </div>

    <h1>Student Grade Predictor</h1>

    <p>
        Predict a student's likely final grade using academic records,
        study habits, attendance, and personal learning factors.
    </p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# INFO
# ============================================================

st.markdown("""
<div class="info-card">
    <b>💡 How to use this predictor</b><br>
    Enter the student's information below and click
    <b>Predict Final Grade</b>. The machine learning model will
    analyse the provided information and estimate the student's
    likely final grade.
</div>
""", unsafe_allow_html=True)


# ============================================================
# INPUT FORM
# ============================================================

with st.form("student_prediction_form"):

    left_column, right_column = st.columns(2, gap="large")

    # --------------------------------------------------------
    # LEFT COLUMN
    # --------------------------------------------------------

    with left_column:

        st.markdown(
            '<p class="section-title">👤 Student Profile</p>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<p class="section-subtitle">'
            'Basic student and school information'
            '</p>',
            unsafe_allow_html=True
        )

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
            [
                "High School",
                "Diploma",
                "Bachelor",
                "Master",
                "PhD"
            ]
        )

        internet_access = st.selectbox(
            "Internet Access",
            ["Yes", "No"]
        )

        extra_activities = st.selectbox(
            "Participates in Extra Activities",
            ["Yes", "No"]
        )

    # --------------------------------------------------------
    # RIGHT COLUMN
    # --------------------------------------------------------

    with right_column:

        st.markdown(
            '<p class="section-title">📚 Academic Information</p>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<p class="section-subtitle">'
            'Study habits, attendance, and subject performance'
            '</p>',
            unsafe_allow_html=True
        )

        study_hours = st.number_input(
            "Daily Study Hours",
            min_value=0.0,
            max_value=24.0,
            value=3.0,
            step=0.5
        )

        study_method = st.selectbox(
            "Study Method",
            [
                "Self-study",
                "Group Study",
                "Online Learning",
                "Tutoring"
            ]
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

    submitted = st.form_submit_button(
        "✨ Predict Final Grade",
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

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

    # Process input
    processed_student = preprocessor.transform(
        new_student
    )

    # Prediction
    prediction = model.predict(
        processed_student
    )[0]

    grade = str(prediction).strip().upper()

    average_score = (
        math_score +
        science_score +
        english_score
    ) / 3

    # ========================================================
    # RESULT
    # ========================================================

    if grade in ["A", "B"]:

        result_text = (
            "Strong predicted academic outcome based on "
            "patterns learned from the training data."
        )

    elif grade == "C":

        result_text = (
            "Moderate predicted academic outcome. "
            "Consistent study and attendance may help improve performance."
        )

    else:

        result_text = (
            "Additional academic support or targeted subject "
            "assistance may be beneficial."
        )

    st.markdown(f"""
    <div class="result-card">

        <div class="result-label">
            🎯 MACHINE LEARNING PREDICTION
        </div>

        <div class="grade-value">
            {grade}
        </div>

        <div class="result-description">
            {result_text}
        </div>

    </div>
    """, unsafe_allow_html=True)

    # ========================================================
    # SUMMARY
    # ========================================================

    st.markdown(
        '<p class="section-title" style="margin-top: 2rem;">📊 Student Summary</p>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:

        st.markdown(f"""
        <div class="metric-card">

            <div class="metric-icon">📈</div>

            <div class="metric-title">
                Average Subject Score
            </div>

            <div class="metric-value">
                {average_score:.1f}
            </div>

        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown(f"""
        <div class="metric-card">

            <div class="metric-icon">📅</div>

            <div class="metric-title">
                Attendance
            </div>

            <div class="metric-value">
                {attendance_percentage}%
            </div>

        </div>
        """, unsafe_allow_html=True)

    with col3:

        st.markdown(f"""
        <div class="metric-card">

            <div class="metric-icon">⏱️</div>

            <div class="metric-title">
                Daily Study Time
            </div>

            <div class="metric-value">
                {study_hours:.1f} hrs
            </div>

        </div>
        """, unsafe_allow_html=True)


# ============================================================
# HOW IT WORKS
# ============================================================

st.markdown(
    '<p class="section-title" style="margin-top: 2.5rem;">⚙️ How It Works</p>',
    unsafe_allow_html=True
)

how1, how2, how3 = st.columns(3, gap="medium")

with how1:

    st.markdown("""
    <div class="how-card">

        <div class="how-number">1</div>

        <div class="how-title">
            Enter Student Data
        </div>

        <div class="how-text">
            Provide information such as attendance,
            study hours, subject scores, and learning habits.
        </div>

    </div>
    """, unsafe_allow_html=True)


with how2:

    st.markdown("""
    <div class="how-card">

        <div class="how-number">2</div>

        <div class="how-title">
            Process the Data
        </div>

        <div class="how-text">
            The saved preprocessing pipeline transforms
            the student's information into the format
            required by the trained machine learning model.
        </div>

    </div>
    """, unsafe_allow_html=True)


with how3:

    st.markdown("""
    <div class="how-card">

        <div class="how-number">3</div>

        <div class="how-title">
            Generate Prediction
        </div>

        <div class="how-text">
            The trained model analyses the processed data
            and predicts the student's likely final grade.
        </div>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

    🎓 <b>Student Grade Predictor</b><br>

    Machine Learning • Educational Analytics<br><br>

    This prediction is generated from patterns in the training dataset.
    It is intended for educational analysis and student-support purposes
    only and should not be treated as a final academic decision.

</div>
""", unsafe_allow_html=True)
