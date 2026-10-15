"""
Traditional Programming vs Machine Learning
Subject          : AI ML 1 - CCA2
Class            : S Y BCom FinTech
Assignment       : Traditional Programming Vs Machine Learning Programming 
                   code to work from input taken from an excel file.

Student Name     : <ADMIN / TEAM NAME>
PRN              : <ADMIN PRN>
Branch           : main

Note: This is the base file for Main branch.
Team members should copy this file into their own Dev branch
and update their Name + PRN.
"""

# ============================================================
# 1. IMPORTS
# ============================================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 2. STUDENT DETAILS
# ============================================================
STUDENT_NAME = "<ADMIN / TEAM NAME>"
STUDENT_PRN  = "<ADMIN PRN>"


# ============================================================
# 3. LOAD DATA FROM EXCEL FILE
# ============================================================
print("=" * 70)
print("  TRADITIONAL PROGRAMMING  vs  MACHINE LEARNING")
print(f"  Student Name : {STUDENT_NAME}")
print(f"  PRN          : {STUDENT_PRN}")
print("=" * 70)

print("\nLoading data from Excel file: loan_data.xlsx ...")
df = pd.read_excel("loan_data.xlsx")

print(f"\nTotal records loaded: {len(df)}")
print("\nFirst 5 rows of the data:")
print(df.head())
print("\nColumns:", list(df.columns))


# ============================================================
# 4. TRADITIONAL PROGRAMMING APPROACH
# ============================================================
print("\n" + "=" * 70)
print("PART A: TRADITIONAL PROGRAMMING APPROACH")
print("=" * 70)
print("In Traditional Programming, we write fixed rules ourselves.")
print("Example Rules used:")
print("  1. If Credit_Score >= 700 and Income >= 40000 and Existing_Loans <= 1 → Approved")
print("  2. If Credit_Score >= 750 and Income >= 30000 → Approved")
print("  3. Otherwise → Rejected")
print("-" * 70)

def traditional_predict(row):
    """Traditional rule-based prediction"""
    if row["Credit_Score"] >= 700 and row["Income"] >= 40000 and row["Existing_Loans"] <= 1:
        return "Approved"
    elif row["Credit_Score"] >= 750 and row["Income"] >= 30000:
        return "Approved"
    else:
        return "Rejected"

# Apply traditional rules
df["Traditional_Prediction"] = df.apply(traditional_predict, axis=1)

# Accuracy of traditional rules
traditional_accuracy = (df["Traditional_Prediction"] == df["Loan_Status"]).mean() * 100

print("\nSample Predictions (Traditional Programming):")
print(df[["Applicant_ID", "Credit_Score", "Income", "Existing_Loans", 
          "Loan_Status", "Traditional_Prediction"]].head(10).to_string(index=False))

print(f"\nAccuracy of Traditional Programming rules: {traditional_accuracy:.2f}%")


# ============================================================
# 5. MACHINE LEARNING APPROACH
# ============================================================
print("\n" + "=" * 70)
print("PART B: MACHINE LEARNING APPROACH")
print("=" * 70)
print("In Machine Learning, the model learns rules automatically from data.")
print("We are using Decision Tree Classifier.")
print("-" * 70)

# Features and Target
features = ["Age", "Income", "Credit_Score", "Loan_Amount", "Existing_Loans"]
X = df[features]
y = df["Loan_Status"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"\nTraining samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

# Train model
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
ml_accuracy = accuracy_score(y_test, y_pred) * 100

print(f"\nAccuracy of Machine Learning model: {ml_accuracy:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ============================================================
# 6. COMPARISON
# ============================================================
print("\n" + "=" * 70)
print("COMPARISON: Traditional Programming vs Machine Learning")
print("=" * 70)
print(f"{'Approach':<30} {'Accuracy'}")
print("-" * 45)
print(f"{'Traditional Programming':<30} {traditional_accuracy:.2f}%")
print(f"{'Machine Learning':<30} {ml_accuracy:.2f}%")
print("=" * 70)

print("\nKey Difference:")
print("→ Traditional Programming : Rules are written manually by the programmer.")
print("→ Machine Learning        : Rules are learned automatically from the data.")


# ============================================================
# 7. DEMO - Predict for a New Applicant
# ============================================================
print("\n" + "=" * 70)
print("DEMO: Predict for a New Loan Applicant")
print("=" * 70)

new_applicant = {
    "Age": 34,
    "Income": 55000,
    "Credit_Score": 720,
    "Loan_Amount": 400000,
    "Existing_Loans": 1
}

print("\nNew Applicant Details:")
for key, value in new_applicant.items():
    print(f"  {key}: {value}")

# Traditional prediction
trad_result = traditional_predict(new_applicant)
print(f"\nTraditional Programming Prediction → {trad_result}")

# Machine Learning prediction
ml_result = model.predict(pd.DataFrame([new_applicant]))[0]
print(f"Machine Learning Prediction         → {ml_result}")

print("\n" + "=" * 70)
print("Program completed successfully.")
print("=" * 70)
