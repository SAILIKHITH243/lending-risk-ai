
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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# ==========================================
# LENDING RISK AI
# Step 4: Train our first Machine Learning model
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "credit_risk_dataset.csv"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_FILE = MODEL_DIR / "credit_risk_model.joblib"

print("=" * 55)
print("       LENDING RISK AI - MODEL TRAINING")
print("=" * 55)

# 1. Load the dataset
df = pd.read_csv(DATA_FILE)

# 2. Remove duplicate records
df = df.drop_duplicates().copy()

# 3. Exclude records with implausible ages
df = df[df["person_age"].between(18, 100)].copy()

# Treat impossible employment lengths as missing
invalid_employment = df["person_emp_length"] > df["person_age"]
df.loc[invalid_employment, "person_emp_length"] = np.nan

print("\nDataset loaded and basic cleaning completed.")
print("Records available:", len(df))

# 4. Select features available for our initial risk model
# We exclude loan_grade and loan_int_rate because these
# may be assigned by the lender during underwriting.
numeric_features = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_percent_income",
    "cb_person_cred_hist_length"
]

categorical_features = [
    "person_home_ownership",
    "loan_intent",
    "cb_person_default_on_file"
]

feature_columns = numeric_features + categorical_features

X = df[feature_columns]
y = df["loan_status"]

# 5. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# 6. Prepare numerical features
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# 7. Prepare categorical features
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# 8. Combine preprocessing steps
preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# 9. Create the complete Machine Learning pipeline
model = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

# 10. Train the model
print("\nTraining Logistic Regression model...")
model.fit(X_train, y_train)

# 11. Make predictions on unseen test records
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

# 12. Evaluate performance
print("\n" + "=" * 55)
print("MODEL EVALUATION RESULTS")
print("=" * 55)

print(f"\nAccuracy: {accuracy_score(y_test, predictions):.4f}")
print(f"ROC-AUC:  {roc_auc_score(y_test, probabilities):.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["No Default", "Default"],
    zero_division=0
))

print("Confusion Matrix:")
print("Rows = Actual outcomes; Columns = Predicted outcomes")
print("[[True non-defaults, False defaults],")
print(" [Missed defaults, True defaults]]")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

# 13. Save the trained pipeline
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_FILE)

print("\nTrained model saved successfully!")
print("Model location:", MODEL_FILE)

print("\nMODEL TRAINING COMPLETED!")
