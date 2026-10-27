
import pandas as pd
from pathlib import Path

# Locate the project and dataset
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "credit_risk_dataset.csv"

# Load dataset
df = pd.read_csv(DATA_FILE)

print("===== DATA QUALITY CHECK =====")

print("\n1. Original dataset shape:")
print(df.shape)

# Check duplicate records
duplicates = df.duplicated().sum()
print("\n2. Duplicate rows:")
print(duplicates)

# Check missing values
print("\n3. Missing values before cleaning:")
print(df.isnull().sum())

# Inspect unusual values
print("\n4. Unusual employment lengths:")
print(df[df["person_emp_length"] > 60][
    ["person_age", "person_emp_length"]
])

print("\n5. Unusual ages:")
print(df[df["person_age"] > 100][["person_age"]])

# Remove exact duplicate rows
df = df.drop_duplicates().copy()

print("\n6. Dataset shape after removing duplicates:")
print(df.shape)

# Report missing values; we will impute them in our ML pipeline
print("\n7. Missing values after duplicate removal:")
print(df.isnull().sum())

print("\nData quality inspection completed!")
print("Missing values will be handled during model training.")
