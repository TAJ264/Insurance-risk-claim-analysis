import tkinter as tk # GUI library used to create the user interface
from tkinter import Toplevel, messagebox, Tk, Frame, Button, BOTH, Label, Canvas # Used to create GUI components

import pandas as pd # Library used for loading and processing the insurance claims data
import numpy as np # Library used for numerical calculations
import matplotlib.pyplot as plt # Library used to create graphs and visualisations

from scipy.stats import norm, mode # Libraries used for statistical analysis
from datetime import datetime # Used for working with dates and times
import os # Used for interacting with files and folders

import pandas as pd
import numpy as np


# ========================================
# INSURANCE RISK & CLAIM ANALYSIS PROJECT
# 01 - DATA UNDERSTANDING
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
# DISPLAY FIRST FIVE RECORDS
# ----------------------------------------

print("First five records:")
print(data.head())
print()


# ----------------------------------------
# DATASET SIZE
# ----------------------------------------

print("Dataset size:")
print(data.shape)
print()


# ----------------------------------------
# COLUMN NAMES
# ----------------------------------------

print("Column names:")
print(data.columns.tolist())
print()


# ----------------------------------------
# CHECK FOR MISSING VALUES
# ----------------------------------------

print("Missing values:")
print(data.isnull().sum())
print()


# ----------------------------------------
# DATA CLEANING
# ----------------------------------------

# Remove the empty _c39 column
data = data.drop(columns=['_c39'])

print("Dataset after removing empty columns:")
print(data.shape)
print()


# ----------------------------------------
# HANDLE MISSING VALUES
# ----------------------------------------

# Replace missing values in authorities_contacted
# with "Unknown" rather than removing the entire claim
data['authorities_contacted'] = data['authorities_contacted'].fillna('Unknown')

print("Missing values after cleaning:")
print(data.isnull().sum())
print()


# ----------------------------------------
# FRAUD ANALYSIS
# ----------------------------------------

print("Fraud report results:")
print(data['fraud_reported'].value_counts())
print()


# Calculate the percentage of claims
# reported as fraud
fraud_percentage = (
    data['fraud_reported'].value_counts(normalize=True) * 100
)

print("Fraud percentages:")
print(fraud_percentage)
print()


# ----------------------------------------
# DATA TYPES
# ----------------------------------------

print("Numerical columns:")
print(data.select_dtypes(include=np.number).columns.tolist())
print()

print("Categorical columns:")
print(data.select_dtypes(include='object').columns.tolist())
print()


# ----------------------------------------
# NUMERICAL DATA SUMMARY
# ----------------------------------------

print("Numerical data summary:")
print(data.describe())
print()


# ----------------------------------------
# FINAL DATASET INFORMATION
# ----------------------------------------

print("Final dataset information:")
print(data.info())

