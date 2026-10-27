
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# LENDING RISK AI — BURGUNDY & GOLD
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
LOGISTIC_PATH = MODEL_DIR / "logistic_regression.joblib"
FOREST_PATH = MODEL_DIR / "random_forest.joblib"

BURGUNDY = "#4A102A"
BURGUNDY_LIGHT = "#8C2448"
GOLD = "#D6AE61"
IVORY = "#FBF6EF"
WHITE = "#FFFFFF"
TEXT = "#3B1830"
MUTED = "#735E68"
BORDER = "#E7D8C7"
PALE_GOLD = "#F4E9D5"

# Educational demonstration rules only.
# These are not official bank lending standards.
RISK_THRESHOLD = 0.50
LOAN_TO_INCOME_REVIEW_LIMIT = 0.50
MODEL_SCORE_DIFFERENCE_LIMIT = 0.20


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Lending Risk AI | Loan Credit Analysis",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# GLOBAL THEME
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --burgundy: #4A102A;
        --burgundy-light: #8C2448;
        --gold: #D6AE61;
        --ivory: #FBF6EF;
        --text: #3B1830;
        --muted: #735E68;
        --border: #E7D8C7;
    }

    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stMain"], [data-testid="stMainBlockContainer"] {
        background-color: #FBF6EF !important;
        color: #3B1830 !important;
    }

    [data-testid="stHeader"] {
        background-color: #FBF6EF !important;
    }

    .block-container {
        max-width: 1280px !important;
        padding-top: 1.8rem !important;
        padding-bottom: 3rem !important;
    }

    /* Main page headings and text */
    [data-testid="stMain"] h1,
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3,
    [data-testid="stMain"] h4,
    [data-testid="stMain"] h5,
    [data-testid="stMain"] h6 {
        color: #4A102A !important;
    }

    [data-testid="stMain"] p,
    [data-testid="stMain"] label,
    [data-testid="stMain"] [data-testid="stWidgetLabel"] {
        color: #3B1830 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div:first-child {
        background-color: #4A102A !important;
        border-right: 2px solid #D6AE61 !important;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #FBF6EF !important;
    }

    [data-testid="stSidebar"] h3 {
        font-size: 1.45rem !important;
        font-weight: 750 !important;
        line-height: 1.3 !important;
        margin-top: 0.4rem !important;
    }

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #E6D6DB !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: #D6AE61 !important;
    }

    .sidebar-eyebrow {
        color: #D6AE61 !important;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.14em;
        text-transform: uppercase;
    }

    /* Hero banner */
    .hero {
        background: linear-gradient(
            115deg, #4A102A 0%, #8C2448 100%
        ) !important;
        color: #FBF6EF !important;
        padding: 2.2rem 2.25rem;
        border: 1px solid #D6AE61 !important;
        border-bottom: 4px solid #D6AE61 !important;
        border-radius: 18px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 28px rgba(74, 16, 42, 0.12);
    }

    .hero h1 {
        color: #FBF6EF !important;
        font-size: 2.35rem !important;
        font-weight: 750 !important;
        margin: 0 0 0.5rem 0;
    }

    .hero p {
        color: #FBF6EF !important;
        line-height: 1.6;
        margin: 0.25rem 0;
    }

    .eyebrow {
        color: #D6AE61 !important;
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }

    .section-kicker {
        color: #8C2448 !important;
        font-size: 0.76rem;
        font-weight: 750;
        letter-spacing: 0.11em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }

    .section-title {
        color: #4A102A !important;
        font-size: 1.4rem;
        font-weight: 750;
        margin-bottom: 0.5rem;
    }

    /* Form and input controls */
    [data-testid="stForm"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E7D8C7 !important;
        border-radius: 16px !important;
        padding: 1.3rem !important;
        box-shadow: 0 5px 16px rgba(74, 16, 42, 0.05);
    }

    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stDateInput"] input {
        background-color: #FBF6EF !important;
        color: #3B1830 !important;
        border: 1px solid #D6AE61 !important;
        border-radius: 8px !important;
        -webkit-text-fill-color: #3B1830 !important;
    }

    [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stMultiSelect"] [data-baseweb="select"] > div {
        background-color: #FBF6EF !important;
        color: #3B1830 !important;
        border-color: #D6AE61 !important;
        border-radius: 8px !important;
    }

    [data-baseweb="popover"], [data-baseweb="menu"] {
        background-color: #FFFFFF !important;
    }

    [data-baseweb="menu"] *, [data-baseweb="select"] * {
        color: #3B1830 !important;
    }

    [data-testid="stNumberInput"] button {
        background-color: #F4E9D5 !important;
        color: #4A102A !important;
        border-color: #D6AE61 !important;
    }

    /* Primary buttons and their labels */
    [data-testid="stFormSubmitButton"] button,
    [data-testid="stFormSubmitButton"] button *,
    div.stButton > button[kind="primary"],
    div.stButton > button[kind="primary"] * {
        background-color: #4A102A !important;
        color: #FBF6EF !important;
        -webkit-text-fill-color: #FBF6EF !important;
        border-color: #D6AE61 !important;
        opacity: 1 !important;
    }

    [data-testid="stFormSubmitButton"] button,
    div.stButton > button {
        border: 1px solid #D6AE61 !important;
        border-radius: 9px !important;
        min-height: 2.9rem;
        font-weight: 700 !important;
    }

    [data-testid="stFormSubmitButton"] button:hover,
    div.stButton > button[kind="primary"]:hover {
        background-color: #8C2448 !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E7D8C7 !important;
        border-top: 3px solid #D6AE61 !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        box-shadow: 0 4px 12px rgba(74, 16, 42, 0.04);
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p {
        color: #735E68 !important;
    }

    [data-testid="stMetricValue"] {
        color: #4A102A !important;
        font-weight: 750 !important;
    }

    [data-testid="stProgress"] > div > div {
        background-color: #8C2448 !important;
    }

    /* Screening result */
    .result-panel {
        background-color: #FFFFFF !important;
        border: 1px solid #D6AE61 !important;
        border-left: 6px solid #8C2448 !important;
        border-radius: 14px !important;
        padding: 1.45rem 1.6rem;
        margin: 0.8rem 0 1.2rem 0;
        box-shadow: 0 6px 16px rgba(74, 16, 42, 0.05);
    }

    .result-label {
        color: #735E68 !important;
        font-size: 0.78rem;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    .result-value {
        color: #4A102A !important;
        font-size: 1.55rem;
        font-weight: 800;
        line-height: 1.4;
        margin-top: 0.4rem;
    }

    [data-baseweb="tab"][aria-selected="true"] {
        color: #4A102A !important;
        border-bottom-color: #D6AE61 !important;
    }

    [data-testid="stAlert"], [data-testid="stExpander"] {
        border-color: #E7D8C7 !important;
        border-radius: 10px !important;
    }

    [data-testid="stExpander"] {
        background-color: #FFFFFF !important;
    }

    hr {
        border-color: #E7D8C7 !important;
    }

    .footer {
        color: #735E68 !important;
        text-align: center;
        font-size: 0.82rem;
        padding: 1rem 0;
        border-top: 1px solid #E7D8C7;
    }

    @media (max-width: 700px) {
        .hero { padding: 1.5rem; }
        .hero h1 { font-size: 1.8rem !important; }
        .result-value { font-size: 1.25rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD TRAINED MODELS
# ============================================================

@st.cache_resource
def load_models():
    """Load the trained Scikit-learn model pipelines."""

    if not LOGISTIC_PATH.exists():
        raise FileNotFoundError(
            f"Logistic Regression model not found: {LOGISTIC_PATH}"
        )

    if not FOREST_PATH.exists():
        raise FileNotFoundError(
            f"Random Forest model not found: {FOREST_PATH}"
        )

    return (
        joblib.load(LOGISTIC_PATH),
        joblib.load(FOREST_PATH),
    )


# ============================================================
# HERO BANNER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">
            FINANCIAL ANALYTICS | MACHINE LEARNING
        </div>
        <h1>Lending Risk AI</h1>
        <p>Loan Credit Analysis and Default Risk Assessment</p>
        <p>
            Review applicant information, compare model scores,
            and inspect a preliminary screening result.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# Uses Streamlit text components so the HTML source is not shown.
# ============================================================

with st.sidebar:
    st.markdown(
        '<div class="sidebar-eyebrow">CREDIT INTELLIGENCE</div>',
        unsafe_allow_html=True,
    )

    # These are real Streamlit components, not a displayed HTML snippet.
    st.subheader("Lending Risk AI")
    st.caption("Loan Credit Analysis Platform")

    st.divider()

    st.markdown("### About the Project")
    st.write(
        "A Python-based machine learning application for analysing "
        "historical loan data and comparing credit-risk model scores."
    )

    st.divider()

    st.markdown("### Assessment Information")
    st.write(
        "This application compares Logistic Regression and Random Forest. "
        "Its preliminary screening result is for educational purposes "
        "and is not an official bank loan approval or rejection."
    )

    st.divider()

    st.markdown(
        """
        <div style="color: #FBF6EF; font-size: 0.80rem; line-height: 1.6;">
            <span style="color: #D6AE61; font-weight: 750;">
                MODEL NOTE
            </span>
            <br>
            Model scores are not calibrated real-world default probabilities.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# APPLICANT INPUT FORM
# ============================================================

st.markdown(
    '<div class="section-kicker">APPLICANT PROFILE</div>'
    '<div class="section-title">Applicant Information</div>',
    unsafe_allow_html=True,
)

st.write(
    "Enter applicant details to compare model scores and view the "
    "preliminary screening result. Use the same currency for annual "
    "income and requested loan amount."
)

with st.form("applicant_form", clear_on_submit=False):
    st.markdown("### Personal and financial details")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Applicant age", min_value=18, max_value=100,
            value=30, step=1,
        )

    with col2:
        annual_income = st.number_input(
            "Annual income", min_value=1, value=600000, step=10000,
        )

    with col3:
        employment_length = st.number_input(
            "Employment length (years)",
            min_value=0.0, max_value=100.0, value=5.0, step=1.0,
        )

    col4, col5, col6 = st.columns(3)

    with col4:
        loan_amount = st.number_input(
            "Requested loan amount", min_value=1, value=200000, step=10000,
        )

    with col5:
        home_ownership = st.selectbox(
            "Home ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"],
        )

    with col6:
        loan_intent = st.selectbox(
            "Loan purpose",
            [
                "PERSONAL", "EDUCATION", "MEDICAL", "VENTURE",
                "HOMEIMPROVEMENT", "DEBTCONSOLIDATION",
            ],
        )

    st.markdown("### Credit history")

    col7, col8 = st.columns(2)

    with col7:
        previous_default = st.selectbox(
            "Previous default on file",
            ["N", "Y"],
            format_func=lambda value: "No" if value == "N" else "Yes",
        )

    with col8:
        credit_history = st.number_input(
            "Credit history length (years)",
            min_value=0, max_value=100, value=5, step=1,
        )

    st.caption(
        "The dataset does not include an actual numerical CIBIL score. "
        "This form uses the credit-history information available in the dataset."
    )

    submitted = st.form_submit_button(
        "Analyze Credit Risk",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# ANALYSIS AND RESULTS
# ============================================================

if submitted:
    validation_errors = []

    if employment_length > age:
        validation_errors.append(
            "Employment length cannot exceed applicant age."
        )

    if credit_history > age:
        validation_errors.append(
            "Credit history length cannot exceed applicant age."
        )

    if validation_errors:
        for validation_error in validation_errors:
            st.error(validation_error)

    else:
        try:
            logistic_model, forest_model = load_models()

            loan_to_income_ratio = float(loan_amount / annual_income)

            # These feature names match the model training scripts.
            applicant = pd.DataFrame(
                [
                    {
                        "person_age": age,
                        "person_income": annual_income,
                        "person_emp_length": employment_length,
                        "loan_amnt": loan_amount,
                        "loan_percent_income": loan_to_income_ratio,
                        "cb_person_cred_hist_length": credit_history,
                        "person_home_ownership": home_ownership,
                        "loan_intent": loan_intent,
                        "cb_person_default_on_file": previous_default,
                    }
                ]
            )

            # Scores for class 1 (default), not calibrated probabilities.
            logistic_score = float(
                logistic_model.predict_proba(applicant)[0, 1]
            )
            forest_score = float(
                forest_model.predict_proba(applicant)[0, 1]
            )

            score_difference = abs(logistic_score - forest_score)

            logistic_flags_default = logistic_score >= RISK_THRESHOLD
            forest_flags_default = forest_score >= RISK_THRESHOLD

            models_disagree = (
                logistic_flags_default != forest_flags_default
                or score_difference >= MODEL_SCORE_DIFFERENCE_LIMIT
            )

            # Educational demonstration screening rules only.
            screening_reasons = []

            if loan_to_income_ratio > 1.0:
                screening_reasons.append(
                    "Requested loan amount exceeds annual income."
                )
            elif loan_to_income_ratio > LOAN_TO_INCOME_REVIEW_LIMIT:
                screening_reasons.append(
                    "Loan amount exceeds the demonstration "
                    "loan-to-income screening limit."
                )

            if previous_default == "Y":
                screening_reasons.append(
                    "A previous default is recorded and requires "
                    "credit-history verification."
                )

            if models_disagree:
                screening_reasons.append(
                    "The models disagree or their scores differ substantially."
                )

            if logistic_flags_default and forest_flags_default:
                screening_reasons.append(
                    "Both models flag elevated default risk under "
                    "the demonstration threshold."
                )

            if screening_reasons:
                eligibility_status = "NOT ELIGIBLE UNDER DEMO RULES"
            else:
                eligibility_status = "ELIGIBLE UNDER DEMO RULES"

            # Applicant summary
            st.divider()

            st.markdown(
                '<div class="section-kicker">ASSESSMENT OUTPUT</div>'
                '<div class="section-title">Credit Risk Assessment Results</div>',
                unsafe_allow_html=True,
            )

            st.caption(
                "Results are generated by the trained models and this "
                "project's demonstration rules."
            )

            metric1, metric2, metric3 = st.columns(3)
            metric1.metric("Annual income", f"{annual_income:,.0f}")
            metric2.metric("Requested loan", f"{loan_amount:,.0f}")
            metric3.metric("Loan / income ratio", f"{loan_to_income_ratio:.1%}")

            # Model scores
            st.markdown("### Machine Learning Model Results")
            score_col1, score_col2 = st.columns(2)

            with score_col1:
                st.markdown("**Logistic Regression**")
                st.metric("Default-class score", f"{logistic_score:.2%}")
                st.progress(float(np.clip(logistic_score, 0.0, 1.0)))
                st.caption(
                    "Uncalibrated default-class score; not a guaranteed "
                    "real-world probability."
                )

            with score_col2:
                st.markdown("**Random Forest**")
                st.metric("Default-class score", f"{forest_score:.2%}")
                st.progress(float(np.clip(forest_score, 0.0, 1.0)))
                st.caption(
                    "Uncalibrated default-class score; not a guaranteed "
                    "real-world probability."
                )

            st.metric(
                "Difference between model scores",
                f"{score_difference:.2%}",
            )

            # Burgundy & Gold comparison chart
            chart_data = pd.DataFrame(
                {
                    "Model": ["Logistic Regression", "Random Forest"],
                    "Score": [logistic_score, forest_score],
                }
            )

            fig, ax = plt.subplots(figsize=(8, 3.4))
            fig.patch.set_facecolor(IVORY)
            ax.set_facecolor(WHITE)

            bars = ax.barh(
                chart_data["Model"],
                chart_data["Score"],
                color=[BURGUNDY_LIGHT, GOLD],
                height=0.55,
            )

            ax.set_xlim(0, 1)
            ax.set_xlabel("Default-class score (0–1)", color=TEXT, labelpad=10)
            ax.set_title(
                "Model Score Comparison",
                color=BURGUNDY,
                loc="left",
                pad=14,
                fontweight="bold",
            )
            ax.tick_params(axis="x", colors=MUTED)
            ax.tick_params(axis="y", colors=TEXT, length=0, pad=10)
            ax.xaxis.grid(
                True, linestyle="--", linewidth=0.7,
                alpha=0.35, color=GOLD,
            )
            ax.set_axisbelow(True)

            for spine in ax.spines.values():
                spine.set_visible(False)

            for bar, score in zip(bars, chart_data["Score"]):
                ax.text(
                    min(score + 0.025, 0.88),
                    bar.get_y() + bar.get_height() / 2,
                    f"{score:.1%}",
                    va="center",
                    ha="left",
                    color=BURGUNDY,
                    fontweight="bold",
                )

            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            if models_disagree:
                st.warning(
                    "The models disagree or their scores differ substantially. "
                    "Further review is important."
                )
            else:
                st.info(
                    "The configured disagreement rule was not triggered. "
                    "This does not guarantee a correct prediction."
                )

            # Final screening result
            st.divider()

            st.markdown(
                '<div class="section-kicker">DECISION SUPPORT</div>'
                '<div class="section-title">Preliminary Screening Result</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="result-panel">
                    <div class="result-label">
                        Demonstration screening status
                    </div>
                    <div class="result-value">
                        {eligibility_status}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if screening_reasons:
                st.markdown("**Reasons identified by the screening rules**")
                for reason in screening_reasons:
                    st.write(f"- {reason}")
            else:
                st.success(
                    "No configured demonstration screening rules were triggered."
                )

            with st.expander(
                "Important limitations and responsible use",
                expanded=True,
            ):
                st.write(
                    "This is an educational credit-risk prototype. Its screening "
                    "rules are not official bank lending criteria. Model scores "
                    "are not calibrated real-world default probabilities. The "
                    "dataset does not contain an actual numerical CIBIL score. "
                    "Passing the demonstration screen does not guarantee loan "
                    "approval, and failing it does not establish that a borrower "
                    "cannot repay. Real lending decisions require verified "
                    "information, affordability assessment, appropriate lender "
                    "policies, and human review."
                )

        except FileNotFoundError as error:
            st.error(str(error))
            st.info(
                "Run `python src/05_compare_models.py` from the project root "
                "to create the required model files."
            )

        except Exception as error:
            st.error(
                "The assessment could not be completed. Check the model files "
                "and terminal for details."
            )
            st.exception(error)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <strong style="color: #4A102A;">Lending Risk AI</strong>
        &nbsp;|&nbsp; Python, Pandas and Scikit-learn
        <br>
        Educational project for credit-risk analysis.
        Not an official lending decision system.
    </div>
    """,
    unsafe_allow_html=True,
)
