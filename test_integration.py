"""
Test script to verify the dataset works with the prediction models
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

print("=" * 60)
print("Testing Student Score Prediction System")
print("=" * 60)

# Load the dataset
print("\n1. Loading dataset...")
df = pd.read_csv("student-dta.csv")
print(f"✓ Dataset loaded successfully: {len(df)} students")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head())

# Check for required columns
print("\n2. Checking required columns...")
required_columns = ["attendance", "hours_studied", "internal_marks", "previous_sem_cgpa", "final_marks"]
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    print(f"✗ Missing columns: {missing_columns}")
    exit(1)
else:
    print("✓ All required columns present")

# Check data statistics
print("\n3. Dataset Statistics:")
print(df.describe())

# Train Linear Regression Model
print("\n4. Training Linear Regression model...")
features = ["attendance", "hours_studied", "internal_marks", "previous_sem_cgpa"]
X = df[features]
y_marks = df["final_marks"]

lin_reg = LinearRegression()
lin_reg.fit(X, y_marks)
print("✓ Linear Regression model trained")
print(f"   Model score (R²): {lin_reg.score(X, y_marks):.4f}")

# Train Logistic Regression Model
print("\n5. Training Logistic Regression model...")
df["pass_fail"] = (df["final_marks"] >= 40).astype(int)
y_pass = df["pass_fail"]

log_reg = LogisticRegression()
log_reg.fit(X, y_pass)
print("✓ Logistic Regression model trained")
print(f"   Model accuracy: {log_reg.score(X, y_pass):.4f}")

# Test prediction with sample data
print("\n6. Testing prediction with sample student...")
sample_student = pd.DataFrame([[85, 4.5, 25, 8.2]], columns=features)
print(f"   Input: Attendance=85%, Hours=4.5, Internal=25, CGPA=8.2")

predicted_marks = lin_reg.predict(sample_student)[0]
prob_pass = log_reg.predict_proba(sample_student)[0][1]

print(f"   Predicted Final Marks: {predicted_marks:.2f}")
print(f"   Probability of Passing: {prob_pass:.2f}")

if prob_pass < 0.4:
    risk = "🔴 High Risk"
elif prob_pass < 0.7:
    risk = "🟠 Medium Risk"
else:
    risk = "🟢 Low Risk"
print(f"   Risk Level: {risk}")

# Check pass/fail distribution
print("\n7. Dataset Distribution:")
pass_count = df["pass_fail"].sum()
fail_count = len(df) - pass_count
print(f"   Passing students (≥40): {pass_count} ({pass_count/len(df)*100:.1f}%)")
print(f"   Failing students (<40): {fail_count} ({fail_count/len(df)*100:.1f}%)")

print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED - Code is ready to run!")
print("=" * 60)
print("\nTo run the Streamlit app, use:")
print("  streamlit run app.py")
