import pandas as pd  # Library used for loading and processing the insurance claims data
from sklearn.model_selection import train_test_split  # Splits data into training and test sets
from sklearn.preprocessing import StandardScaler  # Scales features so models train properly
from sklearn.linear_model import LogisticRegression  # Model 1
from sklearn.ensemble import RandomForestClassifier  # Model 2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report  # Evaluation
import matplotlib.pyplot as plt  # Library used to create graphs and visualisations


# ========================================
# INSURANCE RISK & CLAIM ANALYSIS PROJECT
# 05 - FRAUD PREDICTION MODEL
# ========================================


# ----------------------------------------
# LOAD THE CLEANED DATASET
# ----------------------------------------

file_name = "data/insurance_claims_cleaned.csv"
data = pd.read_csv(file_name)

print("Cleaned insurance claims dataset loaded successfully.")
print("Shape:", data.shape)
print("In plain English: we're working with", data.shape[0], "real insurance claims, each with", data.shape[1], "pieces of information recorded about it.")
print()


# ----------------------------------------
# ENCODE CATEGORICAL COLUMNS
# ----------------------------------------

# One-hot encoding turns each text category into its own 0/1 column,
# since machine learning models can only work with numbers
data_encoded = pd.get_dummies(data, columns=[
    'insured_sex', 'incident_type', 'collision_type',
    'incident_severity', 'police_report_available', 'insured_hobbies'
])

print("Shape after encoding:", data_encoded.shape)
print("In plain English: text categories like 'Major Damage' or 'chess' have been turned into simple yes/no (1/0) columns, since the model can only work with numbers, not words.")
print()
print("New severity columns:", [c for c in data_encoded.columns if 'incident_severity' in c])
print()

# Recreate is_fraud here, since this is a separate script from 04_statistics.py
data_encoded['is_fraud'] = data_encoded['fraud_reported'] == 'Y'


# ----------------------------------------
# BUILD X (FEATURES) AND y (TARGET)
# ----------------------------------------

# Columns that aren't useful as model inputs: IDs, free text, overly
# specific locations, and the target itself (fraud_reported / is_fraud)
columns_to_drop = [
    'months_as_customer', 'policy_number', 'policy_bind_date', 'policy_state',
    'policy_csl', 'policy_deductable', 'umbrella_limit', 'insured_zip',
    'insured_education_level', 'insured_occupation', 'insured_relationship',
    'incident_date', 'authorities_contacted', 'incident_state', 'incident_city',
    'incident_location', 'property_damage', 'injury_claim', 'property_claim',
    'vehicle_claim', 'auto_make', 'auto_model', 'auto_year',
    'fraud_reported', 'is_fraud'
]

X = data_encoded.drop(columns=columns_to_drop)  # Features
y = data_encoded['is_fraud']  # Target

print("X shape (features):", X.shape)
print("y shape (target):", y.shape)
print("In plain English: the model will be given", X.shape[1], "clues about each claim, and its job is to guess one thing — whether that claim was fraud.")
print()


# ----------------------------------------
# SPLIT INTO TRAINING AND TEST SETS
# ----------------------------------------

# 80% train, 20% test. random_state=42 makes the split reproducible,
# so the same split happens every time the script is run.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,        # 20% test data
    random_state=42,      # Fixed seed for reproducibility
    shuffle=True          # Shuffle before splitting
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print()
print("Fraud rate in training set:", y_train.mean())
print("Fraud rate in test set:", y_test.mean())
print("In plain English:", X_train.shape[0], "claims are used to teach the model, and", X_test.shape[0], "claims are kept hidden from it, purely to test whether it actually learned something real rather than just memorising answers.")
print()


# ----------------------------------------
# SCALE THE FEATURES
# ----------------------------------------

# Logistic Regression struggles when features are on very different
# scales (e.g. total_claim_amount in the tens of thousands, vs 0/1
# encoded columns). StandardScaler rescales every numeric column so
# they're all treated fairly by the model.
#
# Important: the scaler is fitted on X_train ONLY, then applied to
# both X_train and X_test. This avoids leaking any information from
# the test set into training.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ----------------------------------------
# TRAIN A LOGISTIC REGRESSION MODEL
# ----------------------------------------

clf = LogisticRegression(max_iter=10000, random_state=0)
clf.fit(X_train_scaled, y_train)

log_reg_predictions = clf.predict(X_test_scaled)

log_reg_accuracy = accuracy_score(y_test, log_reg_predictions) * 100
print()
print(f"Logistic Regression Accuracy: {log_reg_accuracy:.2f}%")
print(f"In plain English: out of every 100 claims this model has never seen before, it correctly guesses fraud or not-fraud for about {log_reg_accuracy:.0f} of them.")


# ----------------------------------------
# TRAIN A RANDOM FOREST MODEL
# ----------------------------------------

# Random Forest is a tree-based model, so it doesn't need scaled
# features the way Logistic Regression did — it splits data based on
# thresholds, not distances, so the original (unscaled) X_train/X_test
# are used here instead.
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

rf_predictions = rf_classifier.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_predictions) * 100
print(f"Random Forest Accuracy: {rf_accuracy:.2f}%")
print(f"In plain English: this model gets about {rf_accuracy:.0f} out of every 100 unseen claims right — slightly fewer than Logistic Regression.")
print()
print("Random Forest Classification Report:")
print(classification_report(y_test, rf_predictions))


# ----------------------------------------
# DETAILED METRICS: PRECISION, RECALL, F1, CONFUSION MATRIX
# ----------------------------------------

# Accuracy alone can be misleading on imbalanced data like this (only
# ~25% of claims are fraud). Precision, recall, and F1 give a fuller
# picture of how well each model actually catches fraud specifically.
#
# Precision: of the claims flagged as fraud, how many really were fraud?
# Recall:    of the real fraud claims, how many did the model catch?
# F1:        a balance between precision and recall.

logistics_precision = precision_score(y_test, log_reg_predictions)
print("Logistic Regression Precision:", logistics_precision)
rf_precision = precision_score(y_test, rf_predictions)
print("Random Forest Precision:", rf_precision)
print()

logistics_recall = recall_score(y_test, log_reg_predictions)
print("Logistic Regression Recall:", logistics_recall)
rf_recall = recall_score(y_test, rf_predictions)
print("Random Forest Recall:", rf_recall)
print()

logistics_f1 = f1_score(y_test, log_reg_predictions)
print("Logistic Regression F1:", logistics_f1)
rf_f1 = f1_score(y_test, rf_predictions)
print("Random Forest F1:", rf_f1)
print()

logistics_confusion_matrix = confusion_matrix(y_test, log_reg_predictions)
print("Logistic Regression Confusion Matrix:")
print(logistics_confusion_matrix)
tn, fp, fn, tp = logistics_confusion_matrix.ravel()
print(f"In plain English: out of {len(y_test)} test claims, the model correctly spotted {tp} real fraud cases and correctly cleared {tn} genuine claims. It wrongly accused {fp} innocent claims of being fraud, and missed {fn} real fraud cases entirely.")
print()

rf_confusion_matrix = confusion_matrix(y_test, rf_predictions)
print("Random Forest Confusion Matrix:")
print(rf_confusion_matrix)
tn, fp, fn, tp = rf_confusion_matrix.ravel()
print(f"In plain English: out of {len(y_test)} test claims, this model correctly spotted {tp} real fraud cases and correctly cleared {tn} genuine claims. It wrongly accused {fp} innocent claims of being fraud, and missed {fn} real fraud cases entirely.")


# ----------------------------------------
# FEATURE IMPORTANCE (RANDOM FOREST)
# ----------------------------------------

# Random Forest can report which features it actually relied on most
# when making predictions — this helps explain WHY it predicts fraud,
# not just how accurate it is.
importances = rf_classifier.feature_importances_
feature_imp = pd.DataFrame({
    'Feature': X_train.columns,
    'Gini Importance': importances
}).sort_values('Gini Importance', ascending=False)

print()
print("Feature importances (Random Forest):")
print(feature_imp)
print("In plain English: the higher the number, the more that clue actually helped the model tell fraud from non-fraud. The top clue by far is whether the damage was severe.")

top15 = feature_imp.head(15)

plt.figure(figsize=(10, 7))
plt.barh(top15['Feature'][::-1], top15['Gini Importance'][::-1], color='steelblue', edgecolor='black')
plt.title("Top 15 Feature Importances — Random Forest")
plt.xlabel("Gini Importance")
plt.tight_layout()

plt.savefig('outputs/chart8_feature_importance.png', dpi=150, bbox_inches='tight')
print()
print("Chart 8 saved: chart8_feature_importance.png")

plt.show()
