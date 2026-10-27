
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

# =====================================================
# LENDING RISK AI
# Credit Risk Assessment and Demo Eligibility Screening
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"

LOGISTIC_MODEL_PATH = MODEL_DIR / "logistic_regression.joblib"
RANDOM_FOREST_MODEL_PATH = MODEL_DIR / "random_forest.joblib"

# Demonstration thresholds only.
# They are NOT official lending standards.
RISK_THRESHOLD = 0.50
LOAN_TO_INCOME_REVIEW_LIMIT = 0.50
MODEL_SCORE_DIFFERENCE_LIMIT = 0.20


def get_number(message, minimum=None, allow_blank=False):
    """Get a valid numerical input from the user."""

    while True:
        value = input(message).strip()

        if allow_blank and value == "":
            return np.nan

        try:
            number = float(value)

            if not np.isfinite(number):
                print("Please enter a valid number.")
                continue

            if minimum is not None and number < minimum:
                print(f"Value must be at least {minimum}.")
                continue

            return number

        except ValueError:
            print("Invalid input. Please enter a number.")


def get_choice(message, choices):
    """Accept only valid categorical values."""

    while True:
        value = input(message).strip().upper()

        if value in choices:
            return value

        print("Invalid choice. Choose:", ", ".join(choices))


def main():

    print("=" * 60)
    print("                 LENDING RISK AI")
    print("          LOAN CREDIT ANALYSIS SYSTEM")
    print("=" * 60)

    # Check for trained models
    if not LOGISTIC_MODEL_PATH.exists():
        print("\nLogistic Regression model not found.")
        print("Run: python data/05_compare_models.py")
        return

    if not RANDOM_FOREST_MODEL_PATH.exists():
        print("\nRandom Forest model not found.")
        print("Run: python data/05_compare_models.py")
        return

    print("\nEnter the applicant details below.")
    print("Use the same currency for annual income and loan amount.")
    print("Press Enter if employment length is unknown.")

    # -------------------------------------------------
    # 1. Collect applicant details
    # -------------------------------------------------

    age = get_number("Applicant age: ", minimum=18)

    if age > 100:
        print("\nApplicant is outside the model's training age range.")
        print("Further review is required.")
        return

    income = get_number("Annual income: ", minimum=1)

    employment_length = get_number(
        "Employment length in years: ",
        minimum=0,
        allow_blank=True
    )

    if not np.isnan(employment_length) and employment_length > age:
        print("\nInvalid employment length.")
        print("Employment length cannot exceed applicant age.")
        return

    loan_amount = get_number(
        "Requested loan amount: ",
        minimum=1
    )

    home_ownership = get_choice(
        "Home ownership (RENT/OWN/MORTGAGE/OTHER): ",
        ["RENT", "OWN", "MORTGAGE", "OTHER"]
    )

    loan_intent = get_choice(
        "Loan purpose "
        "(PERSONAL/EDUCATION/MEDICAL/VENTURE/"
        "HOMEIMPROVEMENT/DEBTCONSOLIDATION): ",
        [
            "PERSONAL",
            "EDUCATION",
            "MEDICAL",
            "VENTURE",
            "HOMEIMPROVEMENT",
            "DEBTCONSOLIDATION"
        ]
    )

    previous_default = get_choice(
        "Previous default on file? (Y/N): ",
        ["Y", "N"]
    )

    credit_history = get_number(
        "Credit history length in years: ",
        minimum=0
    )

    if credit_history > age:
        print("\nInvalid credit history length.")
        print("Credit history cannot exceed applicant age.")
        return

    # -------------------------------------------------
    # 2. Calculate loan-to-income ratio
    # -------------------------------------------------

    loan_to_income_ratio = loan_amount / income

    # Keep exactly the features used in model training.
    applicant = pd.DataFrame([{
        "person_age": age,
        "person_income": income,
        "person_emp_length": employment_length,
        "loan_amnt": loan_amount,
        "loan_percent_income": loan_to_income_ratio,
        "cb_person_cred_hist_length": credit_history,
        "person_home_ownership": home_ownership,
        "loan_intent": loan_intent,
        "cb_person_default_on_file": previous_default
    }])

    # -------------------------------------------------
    # 3. Load trained Machine Learning models
    # -------------------------------------------------

    logistic_model = joblib.load(LOGISTIC_MODEL_PATH)
    random_forest_model = joblib.load(RANDOM_FOREST_MODEL_PATH)

    # Estimate the model scores for the default class.
    # These are uncalibrated model scores, not guaranteed
    # real-world default probabilities.

    logistic_score = float(
        logistic_model.predict_proba(applicant)[0, 1]
    )

    forest_score = float(
        random_forest_model.predict_proba(applicant)[0, 1]
    )

    # -------------------------------------------------
    # 4. Analyze model outputs
    # -------------------------------------------------

    logistic_predicts_default = logistic_score >= RISK_THRESHOLD
    forest_predicts_default = forest_score >= RISK_THRESHOLD

    model_score_difference = abs(logistic_score - forest_score)

    models_disagree = (
        logistic_predicts_default != forest_predicts_default
        or model_score_difference >= MODEL_SCORE_DIFFERENCE_LIMIT
    )

    # -------------------------------------------------
    # 5. Apply demonstration screening rules
    # -------------------------------------------------

    review_reasons = []

    if loan_to_income_ratio > 1.0:
        review_reasons.append(
            "Requested loan amount exceeds annual income."
        )

    elif loan_to_income_ratio > LOAN_TO_INCOME_REVIEW_LIMIT:
        review_reasons.append(
            "Loan amount exceeds the demonstration loan-to-income limit."
        )

    if previous_default == "Y":
        review_reasons.append(
            "Previous default is recorded; verify the credit history."
        )

    if models_disagree:
        review_reasons.append(
            "The two models disagree or their scores differ substantially."
        )

    if logistic_predicts_default and forest_predicts_default:
        review_reasons.append(
            "Both models flag elevated default risk at the demo threshold."
        )

    # Explicit final binary screening label.
    # "Not eligible" here means not cleared by this demo screen.
    if review_reasons:
        eligibility_status = "NOT ELIGIBLE (DEMO SCREENING)"
        decision_explanation = (
            "The applicant did not pass all demonstration screening rules."
        )
    else:
        eligibility_status = "ELIGIBLE (DEMO SCREENING)"
        decision_explanation = (
            "The applicant passed the initial demonstration screening checks."
        )

    # -------------------------------------------------
    # 6. Display the final assessment report
    # -------------------------------------------------

    print("\n")
    print("=" * 60)
    print("               APPLICANT ANALYSIS REPORT")
    print("=" * 60)

    print("\nAPPLICANT DETAILS")
    print("-" * 60)
    print(f"Age:                       {age:.0f}")
    print(f"Annual income:              {income:,.2f}")
    print(f"Requested loan amount:      {loan_amount:,.2f}")
    print(f"Loan-to-income ratio:       {loan_to_income_ratio:.2%}")
    print(f"Employment length:          {employment_length}")
    print(f"Home ownership:             {home_ownership}")
    print(f"Loan purpose:               {loan_intent}")
    print(f"Previous default on file:   {previous_default}")
    print(f"Credit history length:      {credit_history:.0f} years")

    print("\nMACHINE LEARNING RESULTS")
    print("-" * 60)
    print(f"Logistic Regression score:  {logistic_score:.2%}")
    print(f"Random Forest score:        {forest_score:.2%}")
    print(f"Difference between scores:  {model_score_difference:.2%}")

    print("\nFINAL ELIGIBILITY RESULT")
    print("=" * 60)
    print(f"STATUS: {eligibility_status}")
    print("=" * 60)

    print("\nExplanation:")
    print(decision_explanation)

    if review_reasons:
        print("\nReasons:")
        for number, reason in enumerate(review_reasons, start=1):
            print(f"{number}. {reason}")
    else:
        print("\nNo demonstration screening rules were triggered.")

    print("\nIMPORTANT DISCLAIMER")
    print("-" * 60)
    print("This is an educational Machine Learning prototype.")
    print("These screening rules are not official bank lending policy.")
    print("The model scores are not calibrated real-world probabilities.")
    print("The dataset does not contain an actual numerical CIBIL score.")
    print("Passing this demo does not guarantee loan approval.")
    print("Failing it does not establish that a borrower cannot repay.")
    print("Real lending decisions require verified information,")
    print("appropriate affordability checks, lender policies, and")
    print("human review.")

    print("\n" + "=" * 60)
    print("             ASSESSMENT COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
