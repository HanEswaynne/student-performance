import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Student Grade Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Styling ---
st.markdown("""
<style>
    .block-container {
        max-width: 960px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .hero {
        background: linear-gradient(135deg, #0f766e 0%, #115e59 55%, #134e4a 100%);
        color: #ffffff;
        padding: 2rem 2.25rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(15, 118, 110, 0.18);
    }

    .hero h1 {
        color: #ffffff !important;
        font-size: 2rem;
        margin-bottom: 0.35rem;
    }

    .hero p {
        color: #ccfbf1;
        font-size: 1.02rem;
        margin: 0;
        line-height: 1.6;
    }

    .section-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1.25rem 1.35rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
    }

    .section-card h3 {
        margin-top: 0;
        margin-bottom: 0.75rem;
        color: #111827;
        font-size: 1.05rem;
    }

    .info-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 0.75rem;
        margin-top: 0.75rem;
    }

    .info-pill {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.85rem 0.9rem;
        min-height: 118px;
    }

    .info-pill .icon {
        font-size: 1.25rem;
        margin-bottom: 0.35rem;
    }

    .info-pill .title {
        font-weight: 700;
        color: #0f766e;
        font-size: 0.88rem;
        margin-bottom: 0.35rem;
    }

    .info-pill .desc {
        color: #64748b;
        font-size: 0.78rem;
        line-height: 1.45;
        margin: 0;
    }

    .how-it-works {
        background: #f0fdfa;
        border: 1px solid #99f6e4;
        border-radius: 14px;
        padding: 1.1rem 1.25rem;
        margin: 1rem 0 1.5rem 0;
    }

    .how-it-works p {
        color: #334155;
        margin: 0;
        line-height: 1.65;
        font-size: 0.95rem;
    }

    .grade-box {
        text-align: center;
        padding: 1.75rem 1.5rem;
        border-radius: 16px;
        background: linear-gradient(180deg, #f0fdfa 0%, #ecfeff 100%);
        border: 1px solid #99d8d3;
        margin: 1rem 0 0.75rem 0;
        box-shadow: 0 8px 24px rgba(15, 118, 110, 0.08);
    }

    .grade-box .label {
        margin: 0;
        color: #475569;
        font-size: 0.95rem;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }

    .grade-box .grade {
        font-size: 3.5rem;
        font-weight: 800;
        color: #0f766e;
        margin: 0.15rem 0 0 0;
        line-height: 1;
    }

    .confidence-box {
        background: #ffffff;
        border: 1px solid #dbeafe;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        margin-bottom: 1rem;
    }

    .confidence-box .label {
        color: #1e3a8a;
        font-weight: 600;
        font-size: 0.92rem;
        margin-bottom: 0.35rem;
    }

    .confidence-box .value {
        color: #1d4ed8;
        font-weight: 700;
        font-size: 1.1rem;
        margin-top: 0.35rem;
    }

    .result-section-title {
        color: #0f172a;
        font-size: 1.15rem;
        font-weight: 700;
        margin: 1.25rem 0 0.75rem 0;
        padding-bottom: 0.35rem;
        border-bottom: 2px solid #e2e8f0;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 0.85rem 1rem;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
    }

    div[data-testid="stMetric"] label {
        color: #64748b !important;
        font-size: 0.82rem !important;
    }

    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #0f766e !important;
        font-weight: 700 !important;
    }

    .profile-card {
        border-radius: 14px;
        padding: 1rem 1.1rem;
        min-height: 180px;
        border: 1px solid transparent;
    }

    .profile-card.strengths {
        background: #ecfdf5;
        border-color: #a7f3d0;
    }

    .profile-card.support {
        background: #fff7ed;
        border-color: #fed7aa;
    }

    .profile-card h4 {
        margin: 0 0 0.65rem 0;
        font-size: 1rem;
    }

    .profile-card.strengths h4 { color: #047857; }
    .profile-card.support h4 { color: #c2410c; }

    .profile-card ul {
        margin: 0;
        padding-left: 1.1rem;
        color: #334155;
        line-height: 1.65;
        font-size: 0.92rem;
    }

    .profile-note {
        color: #64748b;
        font-size: 0.86rem;
        font-style: italic;
        margin-bottom: 0.85rem;
    }

    .form-section-label {
        color: #0f766e;
        font-weight: 700;
        font-size: 0.95rem;
        margin: 0.25rem 0 0.75rem 0;
    }

    .stButton > button {
        width: 100%;
        border-radius: 10px;
        font-weight: 700;
        padding: 0.7rem;
        background: linear-gradient(135deg, #0f766e, #115e59);
        color: white;
        border: none;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #115e59, #134e4a);
        color: white;
        border: none;
    }

    .footer-note {
        color: #94a3b8;
        font-size: 0.82rem;
        text-align: center;
        margin-top: 2rem;
        line-height: 1.6;
    }

    @media (max-width: 900px) {
        .info-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }

    @media (max-width: 600px) {
        .info-grid {
            grid-template-columns: 1fr;
        }
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
st.markdown("""
<div class="hero">
    <h1>🎓 Student Grade Predictor</h1>
    <p>
        Analyze a student's profile and estimate their likely final grade using patterns
        learned from historical student records — combining demographics, learning habits,
        school engagement, and subject performance.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="section-card">
    <h3>📊 About the data behind this prediction</h3>
    <p style="color:#475569; line-height:1.65; margin-top:0;">
        This tool is built from individually structured student records, where each row
        represents one student with their demographic profile, educational background,
        learning habits, and academic performance. The dataset blends behavioral,
        environmental, and academic factors — making it useful for educational analysis
        and student-support planning.
    </p>
    <div class="info-grid">
        <div class="info-pill">
            <div class="icon">👤</div>
            <div class="title">Demographics</div>
            <p class="desc">Age, gender, and school type</p>
        </div>
        <div class="info-pill">
            <div class="icon">🏠</div>
            <div class="title">Family Background</div>
            <p class="desc">Parent education level</p>
        </div>
        <div class="info-pill">
            <div class="icon">📖</div>
            <div class="title">Study Habits</div>
            <p class="desc">Daily study hours, study method, internet access</p>
        </div>
        <div class="info-pill">
            <div class="icon">🏫</div>
            <div class="title">School Engagement</div>
            <p class="desc">Attendance, travel time, extra activities</p>
        </div>
        <div class="info-pill">
            <div class="icon">📝</div>
            <div class="title">Academic Records</div>
            <p class="desc">Marks in Math, Science, and English</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="how-it-works">
    <p>
        <strong>What does this prediction do?</strong> Enter a student's details below and
        the trained machine learning model will estimate their <strong>final letter grade</strong>
        (A–F) based on 12 selected inputs. It does <em>not</em> replace official grading —
        it highlights patterns from past students so teachers and advisors can spot students
        who may benefit from extra support early.
    </p>
</div>
""", unsafe_allow_html=True)

# --- Form ---
with st.form("student_prediction_form"):

    st.markdown('<p class="form-section-label">👤 Student Profile</p>', unsafe_allow_html=True)
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
    st.markdown('<p class="form-section-label">📚 Study & Academic Records</p>', unsafe_allow_html=True)
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

    st.markdown("---")
    st.markdown('<p class="result-section-title">🎯 Prediction Result</p>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="grade-box">
        <p class="label">Predicted Final Grade</p>
        <p class="grade">{grade}</p>
    </div>
    """, unsafe_allow_html=True)

    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(processed_student)[0]
            class_labels = model.classes_
            pred_idx = list(class_labels).index(prediction)
            confidence = proba[pred_idx]
            st.markdown(f"""
            <div class="confidence-box">
                <div class="label">Model Confidence</div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(float(confidence))
            st.markdown(
                f'<p class="value" style="color:#1d4ed8; font-weight:700; margin-top:-0.5rem;">'
                f"{confidence * 100:.1f}%</p>",
                unsafe_allow_html=True
            )
        except (AttributeError, IndexError, ValueError, TypeError):
            pass

    if grade in ["A", "B"]:
        st.success("Strong predicted outcome — patterns associated with higher performance.")
    elif grade == "C":
        st.warning("Moderate predicted outcome — consistent study support may help.")
    else:
        st.error("May benefit from additional academic support or attendance monitoring.")

    st.markdown('<p class="result-section-title">📈 Academic Summary</p>', unsafe_allow_html=True)
    sum1, sum2, sum3, sum4 = st.columns(4)
    sum1.metric("Math Score", f"{math_score:.1f}")
    sum2.metric("Science Score", f"{science_score:.1f}")
    sum3.metric("English Score", f"{english_score:.1f}")
    sum4.metric("Average Subject Score", f"{average_subject_score:.1f}")

    st.markdown('<p class="result-section-title">🧭 Learning Profile</p>', unsafe_allow_html=True)
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

    strengths_html = "".join(f"<li>{item}</li>" for item in strengths) or (
        "<li><em>No specific strengths identified from inputs.</em></li>"
    )
    support_html = "".join(f"<li>{item}</li>" for item in areas_for_support) or (
        "<li><em>No specific support areas identified from inputs.</em></li>"
    )

    with profile_left:
        st.markdown(f"""
        <div class="profile-card strengths">
            <h4>✅ Strengths</h4>
            <ul>{strengths_html}</ul>
        </div>
        """, unsafe_allow_html=True)

    with profile_right:
        st.markdown(f"""
        <div class="profile-card support">
            <h4>🛟 Areas for Support</h4>
            <ul>{support_html}</ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
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

Each training record represents one student with demographic, behavioral, environmental,
and academic information — similar to the categories shown above.

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
