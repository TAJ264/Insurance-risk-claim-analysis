import pandas as pd  # Library used for loading and processing the insurance claims data
import numpy as np  # Library used for numerical calculations


# ========================================
# INSURANCE RISK & CLAIM ANALYSIS PROJECT
# 02 - DATA CLEANING
# ========================================


# ----------------------------------------
# LOAD INSURANCE CLAIMS DATA
# ----------------------------------------

file_name = "data/insurance_claims.csv.xlsx"

# Load the insurance claims dataset
data = pd.read_excel(file_name)

print("Insurance claims dataset loaded successfully.")
print()


# ----------------------------------------
# REMOVE EMPTY COLUMN (from 01_data_understanding.py)
# ----------------------------------------

# _c39 was an empty column found during data understanding, so it is removed here
data = data.drop(columns=['_c39'])


# ----------------------------------------
# REPLACE "?" WITH PROPER MISSING VALUES
# ----------------------------------------

# Some columns use "?" instead of leaving the cell blank when information is missing.
# Pandas does not recognise "?" as missing on its own, so it needs to be replaced
# with np.nan first before it can be handled properly.

print("Checking for '?' placeholders before cleaning:")
for column in data.columns:
    question_mark_count = (data[column] == '?').sum()
    if question_mark_count > 0:
        print(f"{column}: {question_mark_count} missing values marked as '?'")
print()

# Replace every "?" in the dataset with a real missing value
data = data.replace('?', np.nan)

print("'?' placeholders have been replaced with proper missing values.")
print()


# ----------------------------------------
# HANDLE MISSING VALUES
# ----------------------------------------

# authorities_contacted was already handled in 01_data_understanding.py,
# so the same approach is repeated here to keep this script self-contained

data['authorities_contacted'] = data['authorities_contacted'].fillna('Unknown')

# For these three columns, "Unknown" is used instead of guessing an answer,
# since there is no reliable way to know what the true value should have been
data['collision_type'] = data['collision_type'].fillna('Unknown')
data['property_damage'] = data['property_damage'].fillna('Unknown')
data['police_report_available'] = data['police_report_available'].fillna('Unknown')

print("Missing values after cleaning:")
print(data.isnull().sum().sum(), "missing values remaining in the dataset")
print()


# ----------------------------------------
# CONVERT DATE COLUMNS
# ----------------------------------------

# These two columns are currently stored as plain text, so they need to be
# converted into proper dates before any date-based calculations can be done
data['policy_bind_date'] = pd.to_datetime(data['policy_bind_date'])
data['incident_date'] = pd.to_datetime(data['incident_date'])

print("Date columns converted successfully.")
print(data[['policy_bind_date', 'incident_date']].dtypes)
print()


# ----------------------------------------
# FEATURE ENGINEERING: CUSTOMER TENURE
# ----------------------------------------

# This creates a new column showing how many days passed between the customer
# starting their policy and the incident occurring. This is a useful risk factor,
# since a very new policyholder making a claim could be worth investigating further.
data['customer_tenure_days'] = (data['incident_date'] - data['policy_bind_date']).dt.days

print("New feature created: customer_tenure_days")
print(data['customer_tenure_days'].describe())
print()


# ----------------------------------------
# FIX INVALID TENURE VALUES
# ----------------------------------------

# A negative tenure means the incident happened before the policy even started,
# which is not possible and points to an error in the original data rather than
# a real event. Rather than deleting the whole row and losing the rest of its
# information, these values are flagged and corrected to 0.

invalid_tenure_count = (data['customer_tenure_days'] < 0).sum()
print(f"Found {invalid_tenure_count} row(s) with a negative customer_tenure_days.")

# Set any negative tenure to 0, since it cannot logically be less than 0
data.loc[data['customer_tenure_days'] < 0, 'customer_tenure_days'] = 0

print("Negative tenure values corrected to 0.")
print(data['customer_tenure_days'].describe())
print()


# ----------------------------------------
# SAVE THE CLEANED DATASET
# ----------------------------------------

# Saving the cleaned data means the next script (03_eda.py) can load it directly,
# without needing to repeat all of this cleaning again
output_file = "data/insurance_claims_cleaned.csv"
data.to_csv(output_file, index=False)

print(f"Cleaned dataset saved to {output_file}")
print("Final shape:", data.shape)
