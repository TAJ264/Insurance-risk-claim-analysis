import pandas as pd  # Library used for loading and processing the insurance claims data
import numpy as np  # Library used for numerical calculations
import seaborn as sns  # Library used to create the heatmap
import matplotlib.pyplot as plt  # Library used to create graphs and visualisations
from scipy.stats import chi2_contingency, ttest_ind  # Used for the chi-square and t-tests
import os  # Used to check for and create folders


# ========================================
# INSURANCE RISK & CLAIM ANALYSIS PROJECT
# 04 - STATISTICAL ANALYSIS
# ========================================



# ----------------------------------------
# LOAD THE CLEANED DATASET
# ----------------------------------------

file_name = "data/insurance_claims_cleaned.csv"
data = pd.read_csv(file_name)

print("Cleaned insurance claims dataset loaded successfully.")
print("Shape:", data.shape)
print()

# Used by the chi-square test later in this script
data['is_fraud'] = data['fraud_reported'] == 'Y'


# ----------------------------------------
# CORRELATION MATRIX (NUMERIC COLUMNS ONLY)
# ----------------------------------------

# A correlation matrix only makes sense on numeric columns, so text
# columns like incident_severity are filtered out first
print("Numeric columns being used:")
print(data.select_dtypes(include=np.number).columns.tolist())
print()

numeric_data = data.select_dtypes(include=np.number)

# Calculate the correlation between every pair of numeric columns
pearson_corr = numeric_data.corr(method='pearson')

print("Correlation matrix:")
print(pearson_corr)
print()

# Plot the full correlation matrix as a heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', fmt='.2f', annot_kws={'size': 8})
plt.title("Pearson Correlation Heatmap")

# Rotate the x-axis labels and shrink the font so all 19 columns fit
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)

plt.tight_layout()

plt.savefig('outputs/chart6_correlation_heatmap.png', dpi=150, bbox_inches='tight')

print("Chart 6 saved: chart6_correlation_heatmap.png")

plt.show()


# ----------------------------------------
# FIND THE STRONGEST RELATIONSHIP AUTOMATICALLY
# ----------------------------------------

# Turn the grid into a flat list of pairs, so it can be searched
corr_pairs = pearson_corr.unstack()

# Remove the diagonal (a column correlated with itself is always 1.0,
# and isn't an interesting relationship)
corr_pairs = corr_pairs[corr_pairs < 1.0]

# .abs() ignores whether the relationship is positive or negative,
# so a strong negative correlation is treated as equally important
# as a strong positive one when finding the single strongest pair
strongest_pair = corr_pairs.abs().idxmax()
strongest_value = corr_pairs[strongest_pair]

print("Strongest relationship in the dataset:")
print(f"{strongest_pair[0]} & {strongest_pair[1]}: {strongest_value:.4f}")


# ----------------------------------------
# FIND THE WEAKEST RELATIONSHIP AUTOMATICALLY
# ----------------------------------------

# .idxmin() finds the smallest value instead of the largest —
# after .abs(), that means the relationship closest to zero
weakest_pair = corr_pairs.abs().idxmin()
weakest_value = corr_pairs[weakest_pair]

print("Weakest relationship in the dataset:")
print(f"{weakest_pair[0]} & {weakest_pair[1]}: {weakest_value:.4f}")


# ----------------------------------------
# CHI-SQUARE TEST: FRAUD vs INCIDENT SEVERITY
# ----------------------------------------

# A chi-square test checks whether two categorical variables are
# genuinely related, or whether any pattern between them could just
# be due to chance.

# Build a contingency table: counts of fraud/not-fraud for each severity level
contingency_table = pd.crosstab(data['is_fraud'], data['incident_severity'])

print("Contingency table (fraud vs incident severity):")
print(contingency_table)
print()

stat, p, dof, expected = chi2_contingency(contingency_table)

print(f"Chi-square statistic: {stat:.4f}")
print(f"Degrees of freedom: {dof}")
print(f"p-value: {p}")
print()

# A p-value below 0.05 is the usual cutoff for "statistically significant" —
# meaning the relationship is very unlikely to be due to chance
if p < 0.05:
    print("Result: statistically significant relationship between fraud and incident severity.")
else:
    print("Result: no statistically significant relationship found.")


# ----------------------------------------
# CHI-SQUARE TEST: FRAUD vs INSURED HOBBIES
# ----------------------------------------

# Same approach as the severity test above, applied to insured_hobbies instead
contingency_table = pd.crosstab(data['is_fraud'], data['insured_hobbies'])

print("Contingency table (fraud vs insured hobbies):")
print(contingency_table)
print()

stat, p, dof, expected = chi2_contingency(contingency_table)

print(f"Chi-square statistic: {stat:.4f}")
print(f"Degrees of freedom: {dof}")
print(f"p-value: {p}")
print()

if p < 0.05:
    print("Result: statistically significant relationship between fraud and insured hobbies.")
else:
    print("Result: no statistically significant relationship found.")


# ----------------------------------------
# T-TEST: TOTAL CLAIM AMOUNT (FRAUD vs NOT FRAUD)
# ----------------------------------------

# A t-test checks whether the average of a number (here, total_claim_amount)
# is genuinely different between two groups, rather than the difference
# being down to chance.

# Split total_claim_amount into two groups based on fraud status
data['not_fraud'] = data['fraud_reported'] == 'N'

fraud_claims = data[data['is_fraud']]['total_claim_amount']
not_fraud_claims = data[data['not_fraud']]['total_claim_amount']

print("Average claim amount (fraud):", fraud_claims.mean())
print("Average claim amount (not fraud):", not_fraud_claims.mean())
print()

t_statistic, p_value = ttest_ind(fraud_claims, not_fraud_claims)

print(f"t-statistic: {t_statistic:.4f}")
print(f"p-value: {p_value}")
print()

if p_value < 0.05:
    print("Result: statistically significant difference in claim amount between fraud and non-fraud claims.")
else:
    print("Result: no statistically significant difference found.")
