import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

# -----------------------------
# Title
# -----------------------------
st.title("🎓 Student Score Prediction & Risk Analysis System")

st.write("Upload dataset to analyze and predict student performance.")

# -----------------------------
# Upload CSV
# -----------------------------
uploaded_file = st.file_uploader("Upload your students_data.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # Train Models
    features = ["attendance", "hours_studied", "internal_marks", "previous_sem_cgpa"]
    X = df[features]
    y_marks = df["final_marks"]

    # Train Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(X, y_marks)

    # Pass/Fail column
    df["pass_fail"] = (df["final_marks"] >= 40).astype(int)

    # Logistic Regression
    y_pass = df["pass_fail"]
    log_reg = LogisticRegression()
    log_reg.fit(X, y_pass)

    st.subheader("📈 Enter Student Details for Prediction")

    attendance = st.number_input("Attendance (%)", 0, 100)
    hours = st.number_input("Hours Studied Per Day", 0.0, 10.0)
    internal = st.number_input("Internal Marks", 0, 30)
    cgpa = st.number_input("Previous Semester CGPA", 0.0, 10.0)

    if st.button("Predict"):
        input_data = pd.DataFrame([[attendance, hours, internal, cgpa]], columns=features)

        # Predictions
        predicted_marks = lin_reg.predict(input_data)[0]
        prob_pass = log_reg.predict_proba(input_data)[0][1]

        # Risk level logic
        if prob_pass < 0.4:
            risk = "🔴 High Risk"
        elif prob_pass < 0.7:
            risk = "🟠 Medium Risk"
        else:
            risk = "🟢 Low Risk"

        # Display results
        st.subheader("📌 Prediction Results")
        st.write(f"**Predicted Final Marks:** {predicted_marks:.2f}")
        st.write(f"**Probability of Passing:** {prob_pass:.2f}")
        st.write(f"**Risk Level:** {risk}")