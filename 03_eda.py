import pandas as pd  # Library used for loading and processing the insurance claims data
import matplotlib.pyplot as plt  # Library used to create graphs and visualisations
import os  # Used to check for and create folders


# ----------------------------------------
# MAKE SURE THE OUTPUTS FOLDER EXISTS
# ----------------------------------------

# If the "outputs" folder doesn't exist yet, this creates it.
# This means the script won't crash the first time it's run.
if not os.path.exists('outputs'):
    os.makedirs('outputs')


# ========================================
# INSURANCE RISK & CLAIM ANALYSIS PROJECT
# 03 - EXPLORATORY DATA ANALYSIS (EDA)
# ========================================


# ----------------------------------------
# LOAD THE CLEANED DATASET
# ----------------------------------------

# This loads the file created by 02_cleaning.py, so all the cleaning
# steps (fixing "?" values, dates, tenure) are already applied
file_name = "data/insurance_claims_cleaned.csv"

data = pd.read_csv(file_name)

print("Cleaned insurance claims dataset loaded successfully.")
print("Shape:", data.shape)
print()


# ----------------------------------------
# CHART 1: DISTRIBUTION OF TOTAL CLAIM AMOUNT
# ----------------------------------------

# A histogram shows how the claim amounts are spread out across
# all 1000 claims, not just a handful of rows
plt.figure(figsize=(8, 5))
plt.hist(data['total_claim_amount'], bins=30, color='steelblue', edgecolor='black')

plt.title('Distribution of Total Claim Amount')
plt.xlabel('Total Claim Amount ($)')
plt.ylabel('Number of Claims')

# Save the chart as an image so it can be used in the report
plt.savefig('outputs/chart1_claim_amount_distribution.png', dpi=150, bbox_inches='tight')

print("Chart 1 saved: chart1_claim_amount_distribution.png")

plt.show()


# ----------------------------------------
# CHART 2: FRAUD RATE BY INCIDENT SEVERITY
# ----------------------------------------

# Turn 'Y'/'N' into True/False so we can average it
data['is_fraud'] = data['fraud_reported'] == 'Y'

# Group by severity, then find the average of is_fraud for each group.
# Averaging a True/False column gives the proportion that are True,
# which is exactly the fraud rate for that severity category.
fraud_rate_by_severity = data.groupby('incident_severity')['is_fraud'].mean()

print("Fraud rate by incident severity:")
print(fraud_rate_by_severity)
print()

plt.figure(figsize=(8, 5))
plt.bar(fraud_rate_by_severity.index, fraud_rate_by_severity.values,
        color='steelblue', edgecolor='black')

plt.title('Fraud Rate by Incident Severity')
plt.xlabel('Incident Severity')
plt.ylabel('Fraud Rate')

plt.savefig('outputs/chart2_fraud_by_severity.png', dpi=150, bbox_inches='tight')

print("Chart 2 saved: chart2_fraud_by_severity.png")

plt.show()


# ----------------------------------------
# CHART 3: FRAUD RATE BY INSURED HOBBIES
# ----------------------------------------

# Same approach as Chart 2, but grouped by hobby instead of severity.
# There are 20 different hobbies, so this chart needs to be wider,
# and the labels need to be rotated so they don't overlap.
fraud_rate_by_insured_hobbies = data.groupby('insured_hobbies')['is_fraud'].mean()

print("Fraud rate by insured hobbies:")
print(fraud_rate_by_insured_hobbies)
print()

# Wider figure to fit all 20 hobby labels
plt.figure(figsize=(14, 6))
plt.bar(fraud_rate_by_insured_hobbies.index, fraud_rate_by_insured_hobbies.values,
        color='steelblue', edgecolor='black')

plt.title('Fraud Rate by Insured Hobbies')
plt.xlabel('Insured Hobbies')
plt.ylabel('Fraud Rate')

# Rotate the labels so they don't overlap with 20 categories
plt.xticks(rotation=45, ha='right')

# Shrinks the plot area automatically so nothing gets cut off at the edges
plt.tight_layout()

plt.savefig('outputs/chart3_fraud_by_insured_hobbies.png', dpi=150, bbox_inches='tight')

print("Chart 3 saved: chart3_fraud_by_insured_hobbies.png")

plt.show()


# ----------------------------------------
# CHART 4: NUMBER OF INCIDENTS BY HOUR OF THE DAY
# ----------------------------------------

# .value_counts() counts how many claims fall into each hour (0-23).
# .sort_index() puts them in hour order (0, 1, 2...) rather than
# biggest-count-first, so the bars read as a proper 24-hour timeline.
incident_by_the_hour = data['incident_hour_of_the_day'].value_counts().sort_index()

print("Incidents by hour of the day:")
print(incident_by_the_hour)
print()

plt.figure(figsize=(10, 5))
plt.bar(incident_by_the_hour.index, incident_by_the_hour.values,
        color='steelblue', edgecolor='black')

plt.title('Incidents by Hour of the Day')
plt.xlabel('Hour of the Day (24-hour clock)')
plt.ylabel('Number of Incidents')

# Show every hour (0-23) on the x-axis instead of matplotlib's default spacing
plt.xticks(range(0, 24))

plt.tight_layout()

plt.savefig('outputs/chart4_incidents_by_hour.png', dpi=150, bbox_inches='tight')

print("Chart 4 saved: chart4_incidents_by_hour.png")

plt.show()


# ----------------------------------------
# CHART 5: AVERAGE CLAIM AMOUNT BY INSURED EDUCATION LEVEL
# ----------------------------------------

average_claim_amount = data.groupby('insured_education_level')['total_claim_amount'].mean()

print("Average claim amount by insured education level:")
print(average_claim_amount)
print()

plt.figure(figsize=(10, 6))
plt.bar(average_claim_amount.index, average_claim_amount.values,
        color='steelblue', edgecolor='black')

plt.title('Average Total Claim Amount by Insured Education Level')
plt.xlabel('Insured Education Level')
plt.ylabel('Average Total Claim Amount')

plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.savefig('outputs/chart5_average_claim_by_education.png', dpi=150, bbox_inches='tight')

print("Chart 5 saved: chart5_average_claim_by_education.png")

plt.show()
