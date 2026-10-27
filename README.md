# Lending Risk AI: Loan Credit Analysis Using Machine Learning

## Overview

Lending Risk AI is a Python-based machine learning project that analyzes historical loan data to identify patterns associated with loan defaults.

The project compares Logistic Regression and Random Forest classifiers and evaluates their performance using standard classification metrics. A Streamlit web application allows users to enter example applicant details, review model scores, and view a preliminary screening result.

This project was developed for educational and portfolio purposes. It is not an official loan approval system.

## Problem Statement

Lenders must assess the risk that a borrower may fail to repay a loan. Manual assessment can become time-consuming as application volumes grow.

Machine learning can help identify patterns in historical loan data. However, missed defaults and incorrectly flagged applicants both have consequences, so model performance must be evaluated using more than accuracy alone.

## Project Objectives

- Explore and clean historical loan data.
- Identify missing values, duplicate records, and unusual observations.
- Prepare numerical and categorical features for modelling.
- Train Logistic Regression and Random Forest models.
- Compare model performance using precision, recall, F1-score, accuracy, and ROC-AUC.
- Save and reuse trained models.
- Build a Streamlit interface for applicant-level risk assessment.

## Dataset

Source: [Credit Risk Dataset on Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)

The original dataset contains 32,581 records and 12 columns describing applicants, loan characteristics, and historical loan outcomes.

The target column is `loan_status`:

- `0`: No default recorded.
- `1`: Default recorded.

Initial exploration found 25,473 non-default records and 7,108 default records, corresponding to a historical default rate of 21.82%.

The dataset includes applicant age, annual income, employment length, home ownership, loan purpose, loan amount, interest rate, loan-to-income ratio, a previous-default indicator, and credit history length.

The dataset does not contain an actual numerical CIBIL score. The project uses only the credit-history-related information available in the source data.

## Technology Stack

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Joblib
- Streamlit
- Git and GitHub

## Project Workflow

1. Explore the dataset and inspect its quality.
2. Examine missing values, duplicates, and unusual observations.
3. Prepare features with Scikit-learn preprocessing pipelines.
4. Split data into training and testing sets.
5. Train Logistic Regression and Random Forest models.
6. Evaluate model performance using classification metrics.
7. Save the trained model pipelines.
8. Enter example applicant details through the command-line program or website.
9. Review model scores and preliminary screening rules.

## Model Performance

The following results were obtained during the initial experiment on 6,483 held-out test records.

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 73.99% | 88.66% |
| Default Precision | 44.09% | 91.10% |
| Default Recall | 70.45% | 53.39% |
| Default F1-score | 54.23% | 67.32% |
| ROC-AUC | 0.8017 | 0.8696 |

Random Forest achieved higher accuracy, default precision, F1-score, and ROC-AUC in this experiment. Logistic Regression identified a larger proportion of actual defaults in the test set.

This demonstrates the trade-off between detecting defaults and incorrectly flagging non-default cases. Model selection should consider the consequences of both types of errors rather than relying on accuracy alone.

These results apply to this dataset, feature set, and model configuration. They do not guarantee performance on other lending portfolios.

## Repository Structure

```text
Loan Credit Analysis/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── data/
│   └── credit_risk_dataset.csv
├── models/
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   └── credit_risk_model.joblib
└── src/
    ├── 01_explore_data.py
    ├── 02_analyze_data.py
    ├── 03_prepare_data.py
    ├── 04_train_model.py
    ├── 05_compare_models.py
    └── 06_predict_applicant.py
```

The CSV dataset should be downloaded separately and is excluded from version control. The trained model files are required by the current website configuration.

## Installation and Setup

### Prerequisites

- Python 3.10 or a compatible version
- Git
- Visual Studio Code or another Python editor

### 1. Install Dependencies

From the project root directory, run:

```bash
python -m pip install -r requirements.txt
```

### 2. Download the Dataset

Download the Credit Risk Dataset from Kaggle:

https://www.kaggle.com/datasets/laotse/credit-risk-dataset

Place `credit_risk_dataset.csv` inside the `data` directory.

### 3. Explore the Data

```bash
python src/01_explore_data.py
```

### 4. Analyze Loan Outcomes

```bash
python src/02_analyze_data.py
```

### 5. Inspect Data Quality

```bash
python src/03_prepare_data.py
```

### 6. Train the Baseline Model

```bash
python src/04_train_model.py
```

### 7. Compare Models

```bash
python src/05_compare_models.py
```

This creates the saved Logistic Regression and Random Forest pipelines in the `models` directory.

### 8. Run the Applicant Assessment Script

```bash
python src/06_predict_applicant.py
```

### 9. Launch the Website

```bash
python -m streamlit run app.py
```

Streamlit will display a local URL, normally `http://localhost:8501`.

The website provides an applicant input form, model scores, a score comparison chart, and a preliminary screening result.

## Limitations and Responsible Use

This project is an educational prototype and should not be used to approve or reject real loan applications.

- Model scores have not been calibrated as real-world default probabilities.
- The dataset does not contain an actual numerical CIBIL score.
- The demonstration screening rules are not official bank lending standards.
- The program does not perform a complete affordability or EMI assessment.
- Historical data may contain bias and may not represent future applicants.
- Model disagreement and false predictions require careful review.
- Real lending systems require validated information, affordability checks, appropriate policies, fairness assessments, and human oversight.

A previous-default indicator alone does not establish that a borrower will default on a future loan.

## Future Improvements

- Evaluate additional classification algorithms and model parameters.
- Tune decision thresholds based on business costs.
- Calibrate model scores using appropriate validation data.
- Add feature-importance analysis and model explainability.
- Improve affordability checks and data validation.
- Evaluate performance across relevant applicant groups.
- Deploy the Streamlit website online.
- Add automated tests and model monitoring.

## Conclusion

Lending Risk AI demonstrates an end-to-end introductory machine learning workflow, from dataset exploration and preprocessing to model evaluation and applicant-level assessment.

The comparison between Logistic Regression and Random Forest shows why credit risk analysis requires multiple evaluation metrics and careful interpretation. The project provides a foundation for further work in financial data analysis and responsible machine learning.

## Author

Developed as a personal Python and machine learning project focused on lending risk analysis.
