
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Locate project and dataset
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "credit_risk_dataset.csv"

# Load dataset
df = pd.read_csv(DATA_FILE)

print("===== LOAN DEFAULT ANALYSIS =====")

# Count each loan status
status_counts = df["loan_status"].value_counts().sort_index()

print("\nLoan Status Counts:")
print(status_counts)

print("\nLoan Status Meaning:")
print("0 = No default recorded")
print("1 = Default recorded")

# Calculate percentages
status_percentages = (
    df["loan_status"].value_counts(normalize=True).sort_index() * 100
)

print("\nLoan Status Percentages:")
print(status_percentages.round(2))

# Plot the chart
status_counts.rename(
    index={0: "No Default", 1: "Default"}
).plot(
    kind="bar",
    xlabel="Historical Loan Outcome",
    ylabel="Number of Records",
    title="Loan Default Distribution",
    rot=0
)

plt.tight_layout()
plt.show()
