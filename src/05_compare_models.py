
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# LENDING RISK AI — MODEL COMPARISON
# Train, evaluate, compare, and save both models.
# Random Forest is saved with compression for GitHub.
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "credit_risk_dataset.csv"
MODEL_DIR = PROJECT_DIR / "models"

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}\n"
        "Download credit_risk_dataset.csv from Kaggle "
        "and place it inside the data folder."
    )

# Load and clean dataset
df = pd.read_csv(DATA_FILE)

df = df.drop_duplicates().copy()
df = df[df["person_age"].between(18, 100)].copy()

# Treat impossible employment lengths as missing
invalid_employment = (
    df["person_emp_length"] > df["person_age"]
)
df.loc[invalid_employment, "person_emp_length"] = np.nan


# Features used by the models
numeric_features = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]

categorical_features = [
    "person_home_ownership",
    "loan_intent",
    "cb_person_default_on_file",
]

feature_columns = numeric_features + categorical_features

X = df[feature_columns]
y = df["loan_status"]


# Training and testing split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


def create_preprocessor():
    """Create preprocessing pipelines for numerical and categorical data."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )


# Models to compare
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}

results = []
trained_models = {}

print("=" * 55)
print("       LENDING RISK AI - MODEL COMPARISON")
print("=" * 55)

print(f"\nRecords after basic cleaning: {len(df)}")
print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")


# Train and evaluate each model
for name, classifier in models.items():

    print("\n" + "=" * 55)
    print(f"TRAINING: {name}")
    print("=" * 55)

    pipeline = Pipeline(
        steps=[
            ("preprocessing", create_preprocessor()),
            ("classifier", classifier),
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Default Precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "Default Recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "Default F1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "ROC-AUC": roc_auc_score(
            y_test,
            probabilities,
        ),
    }

    results.append(metrics)
    trained_models[name] = pipeline

    for metric_name, value in metrics.items():
        if metric_name != "Model":
            print(f"{metric_name}: {value:.4f}")

    print("\nConfusion Matrix:")
    print("Rows = actual outcomes")
    print("Columns = predicted outcomes")

    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["No Default", "Default"],
            zero_division=0,
        )
    )


# Compare results
results_df = pd.DataFrame(results).set_index("Model")

print("\n" + "=" * 65)
print("FINAL MODEL COMPARISON")
print("=" * 65)

print(results_df.round(4).to_string())


# Save models with compression
MODEL_DIR.mkdir(parents=True, exist_ok=True)

logistic_path = (
    MODEL_DIR / "logistic_regression.joblib"
)

forest_path = (
    MODEL_DIR / "random_forest.joblib"
)

joblib.dump(
    trained_models["Logistic Regression"],
    logistic_path,
    compress=3,
)

joblib.dump(
    trained_models["Random Forest"],
    forest_path,
    compress=3,
)

print("\nCompressed model files saved successfully.")
print(f"Logistic Regression: {logistic_path}")
print(f"Random Forest:       {forest_path}")

print("\nMODEL COMPARISON COMPLETED!")
