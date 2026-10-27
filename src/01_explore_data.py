
import pandas as pd
from pathlib import Path

# ==========================================
# LENDING RISK AI PROJECT
# Step 1: Explore the loan dataset
# ==========================================

# Current file is inside the data folder.
# Its parent folder is the main project folder.
PROJECT_DIR = Path(__file__).resolve().parent.parent

# CSV file location
DATA_FILE = PROJECT_DIR / "data" / "credit_risk_dataset.csv"

print("=" * 50)
print("       LENDING RISK AI PROJECT")
print("=" * 50)

print("\nChecking dataset location...")
print(f"Expected location: {DATA_FILE}")

# Check whether the CSV file exists
if not DATA_FILE.exists():
    print("\nERROR: CSV file not found!")
    print("\nPlease check that your folders look like this:")
    print("Loan Credit Analysis/")
    print("    data/")
    print("        01_explore_data.py")
    print("        credit_risk_dataset.csv")
    print("\nDownload the dataset from:")
    print("https://www.kaggle.com/datasets/laotse/credit-risk-dataset")

else:
    try:
        # Read the CSV dataset
        df = pd.read_csv(DATA_FILE)

        print("\n1. FIRST FIVE LOAN RECORDS")
        print("-" * 50)
        print(df.head().to_string(index=False))

        print("\n2. DATASET INFORMATION")
        print("-" * 50)
        print(f"Total records: {df.shape[0]}")
        print(f"Total columns: {df.shape[1]}")

        print("\n3. COLUMN NAMES")
        print("-" * 50)
        for column in df.columns:
            print(column)

        print("\n4. MISSING VALUES")
        print("-" * 50)
        missing_values = df.isnull().sum()
        print(missing_values.to_string())

        total_missing = int(missing_values.sum())
        print(f"\nTotal missing cells: {total_missing}")

        # Analyze the target column if available
        if "loan_status" in df.columns:
            print("\n5. LOAN STATUS DISTRIBUTION")
            print("-" * 50)
            print(df["loan_status"].value_counts().sort_index())

            print("\n6. LOAN STATUS PERCENTAGES")
            print("-" * 50)
            print(
                (df["loan_status"].value_counts(normalize=True) * 100)
                .sort_index()
                .round(2)
                .to_string()
            )

            # In this dataset, 1 = default and 0 = non-default
            default_rate = (df["loan_status"] == 1).mean() * 100
            print(f"\nHistorical default rate: {default_rate:.2f}%")

        else:
            print("\nWARNING: 'loan_status' column was not found.")
            print("Please verify the dataset columns.")

        print("\n" + "=" * 50)
        print("DATASET EXPLORATION COMPLETED!")
        print("=" * 50)

    except pd.errors.EmptyDataError:
        print("\nERROR: The CSV file is empty.")
        print("Download the original dataset and replace the empty file.")

    except pd.errors.ParserError as error:
        print("\nERROR: The CSV format could not be read.")
        print(error)

    except UnicodeDecodeError:
        print("\nERROR: The CSV encoding could not be read.")
        print("Check that you downloaded the original CSV file.")

    except Exception as error:
        print("\nUNEXPECTED ERROR:")
        print(error)
