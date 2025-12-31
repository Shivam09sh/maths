
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Load data
df = pd.read_csv('student-dta.csv')
df_eligible = df[df['attendance'] >= 75].copy()

features = ["attendance", "hours_studied", "internal_marks", "previous_sem_cgpa"]
X = df_eligible[features]
y_marks = df_eligible["final_marks"]

# Linear Regression with Positive Constraint
lin_reg = LinearRegression(positive=True)
lin_reg.fit(X, y_marks)

print("--- Linear Regression (Positive=True) Coefficients ---")
print(f"Intercept: {lin_reg.intercept_}")
for f, c in zip(features, lin_reg.coef_):
    print(f"{f}: {c}")

# Test Case
input_data = pd.DataFrame([[85, 1.0, 12, 7.0]], columns=features)
pred_marks = lin_reg.predict(input_data)[0]
print(f"\nPrediction for 85% Att, 1.0 Hrs, 12 Int, 7.0 CGPA: {pred_marks}")
