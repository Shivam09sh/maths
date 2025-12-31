import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    
    # Show eligibility statistics
    if 'eligible' in df.columns:
        eligible_count = (df['eligible'] == 'Yes').sum()
        ineligible_count = (df['eligible'] == 'No').sum()
        st.info(f"📋 **Eligibility Status:** {eligible_count} Eligible | {ineligible_count} Ineligible (< 75% attendance)")
    
    # Filter only eligible students for training
    df_eligible = df[df['attendance'] >= 75].copy() if 'attendance' in df.columns else df.copy()

    # Train Models (only on eligible students)
    features = ["attendance", "hours_studied", "internal_marks", "previous_sem_cgpa"]
    X = df_eligible[features]
    y_marks = df_eligible["final_marks"]

    # Train Linear Regression
    lin_reg = LinearRegression(positive=True)
    lin_reg.fit(X, y_marks)

    # Pass/Fail column (only for eligible students)
    df_eligible["pass_fail"] = (df_eligible["final_marks"] >= 40).astype(int)

    # Logistic Regression
    y_pass = df_eligible["pass_fail"]
    log_reg = LogisticRegression()
    log_reg.fit(X, y_pass)

    # -----------------------------
    # TABS: Prediction & Analysis
    # -----------------------------
    tab1, tab2 = st.tabs(["🚀 Prediction", "📊 Analysis"])

    # -----------------------------
    # Tab 1: Prediction
    # -----------------------------
    with tab1:
        st.subheader("📈 Enter Student Details for Prediction")

        attendance = st.number_input("Attendance (%)", 0, 100)
        hours = st.number_input("Hours Studied Per Day", 0.0, 10.0)
        internal = st.number_input("Internal Marks", 0, 30)
        cgpa = st.number_input("Previous Semester CGPA", 0.0, 10.0)

        if st.button("Predict"):
            # Check attendance eligibility (75% minimum required)
            if attendance < 75:
                st.subheader("📌 Prediction Results")
                st.error("❌ **INELIGIBLE FOR FINAL EXAM**")
                st.write(f"**Attendance:** {attendance}%")
                st.write("**Reason:** Minimum 75% attendance required for final exam eligibility")
                st.warning("⚠️ Student must improve attendance to be eligible for final exam")
            else:
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
                st.success(f"✅ **Eligible for Final Exam** (Attendance: {attendance}%)")
                st.write(f"**Predicted Final Marks:** {predicted_marks:.2f}")
                st.write(f"**Probability of Passing:** {prob_pass:.2f}")
                st.write(f"**Risk Level:** {risk}")

    # -----------------------------
    # Tab 2: Analysis
    # -----------------------------
    with tab2:
        st.subheader("📊 Data Analysis & Insights")

        # 1. Key Metrics
        st.markdown("### 📌 Key Average Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg Attendance", f"{df['attendance'].mean():.2f}%")
        col2.metric("Avg Internal Marks", f"{df['internal_marks'].mean():.2f}")
        col3.metric("Avg Final Marks", f"{df['final_marks'].mean():.2f}")
        col4.metric("Pass Rate", f"{(df_eligible['final_marks'] >= 40).mean()*100:.1f}%")

        st.divider()

        # 2. ML Model Architecture
        st.markdown("### 🤖 Dual ML Model Architecture")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Linear Regression Model")
            st.info("""
            **Purpose**: Predict final exam marks (0-100)
            
            **Features**: 4 input variables
            - Attendance (%)
            - Hours Studied (per day)
            - Internal Marks (/30)
            - Previous CGPA (/10)
            
            **Output**: Continuous value (0-100 marks)
            
            **Constraint**: positive=True (no negative predictions)
            """)
            
            # Model coefficients
            st.markdown("**Model Coefficients:**")
            coef_data = pd.DataFrame({
                'Feature': features,
                'Coefficient': lin_reg.coef_,
                'Importance': np.abs(lin_reg.coef_) / np.abs(lin_reg.coef_).sum() * 100
            })
            coef_data = coef_data.sort_values('Importance', ascending=False)
            st.dataframe(coef_data.style.format({'Coefficient': '{:.4f}', 'Importance': '{:.2f}%'}))
        
        with col2:
            st.markdown("#### 📊 Logistic Regression Model")
            st.success("""
            **Purpose**: Predict pass/fail probability
            
            **Features**: Same 4 input variables
            - Attendance (%)
            - Hours Studied (per day)
            - Internal Marks (/30)
            - Previous CGPA (/10)
            
            **Output**: Probability (0.0-1.0)
            
            **Threshold**: 0.5 for pass/fail decision
            """)
            
            # Model accuracy
            y_pred = log_reg.predict(X)
            accuracy = (y_pred == y_pass).mean()
            st.markdown(f"**Model Accuracy**: {accuracy*100:.2f}%")
            
            # Confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_pass, y_pred)
            st.markdown("**Confusion Matrix:**")
            cm_df = pd.DataFrame(cm, 
                                index=['Actual Fail', 'Actual Pass'],
                                columns=['Pred Fail', 'Pred Pass'])
            st.dataframe(cm_df)

        st.divider()

        # 3. Feature Importance Visualization
        st.markdown("### 📊 Feature Importance Analysis")
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Calculate feature importance from Linear Regression coefficients
        importance = np.abs(lin_reg.coef_)
        importance_pct = (importance / importance.sum()) * 100
        
        colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#F9CA24']
        bars = ax.barh(features, importance_pct, color=colors)
        ax.set_xlabel('Importance (%)', fontsize=12)
        ax.set_title('Feature Importance for Final Marks Prediction', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add percentage labels
        for i, (bar, val) in enumerate(zip(bars, importance_pct)):
            ax.text(val + 1, i, f'{val:.1f}%', va='center', fontsize=10)
        
        st.pyplot(fig)

        st.divider()

        # 4. Distributions
        st.markdown("### 📉 Data Distributions")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Attendance Distribution**")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(df["attendance"], kde=True, bins=15, ax=ax, color="#4ECDC4")
            ax.axvline(75, color='red', linestyle='--', linewidth=2, label='Eligibility Threshold (75%)')
            ax.set_xlabel('Attendance (%)')
            ax.set_ylabel('Count')
            ax.legend()
            st.pyplot(fig)

        with col2:
            st.write("**Final Marks Distribution**")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(df["final_marks"], kde=True, bins=15, ax=ax, color="#95E1D3")
            ax.axvline(40, color='red', linestyle='--', linewidth=2, label='Pass Threshold (40)')
            ax.set_xlabel('Final Marks')
            ax.set_ylabel('Count')
            ax.legend()
            st.pyplot(fig)

        st.divider()

        # 5. Pass/Fail Analysis
        st.markdown("### 🎯 Pass/Fail Analysis (Eligible Students)")
        col1, col2 = st.columns(2)
        
        with col1:
            # Pass/Fail pie chart
            fig, ax = plt.subplots(figsize=(6, 6))
            pass_counts = df_eligible['pass_fail'].value_counts()
            colors_pie = ['#FF6B6B', '#4ECDC4']
            labels = ['Fail (<40)', 'Pass (≥40)']
            ax.pie(pass_counts, labels=labels, autopct='%1.1f%%', colors=colors_pie, startangle=90)
            ax.set_title('Pass/Fail Distribution', fontsize=14, fontweight='bold')
            st.pyplot(fig)
        
        with col2:
            # Risk level distribution
            st.markdown("**Risk Level Distribution**")
            
            # Calculate risk levels for all eligible students
            probs = log_reg.predict_proba(X)[:, 1]
            risk_levels = []
            for prob in probs:
                if prob < 0.4:
                    risk_levels.append('🔴 High Risk')
                elif prob < 0.7:
                    risk_levels.append('🟠 Medium Risk')
                else:
                    risk_levels.append('🟢 Low Risk')
            
            risk_counts = pd.Series(risk_levels).value_counts()
            
            fig, ax = plt.subplots(figsize=(6, 6))
            colors_risk = ['#4ECDC4', '#F9CA24', '#FF6B6B']
            ax.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', 
                   colors=colors_risk, startangle=90)
            ax.set_title('Risk Level Distribution', fontsize=14, fontweight='bold')
            st.pyplot(fig)

        st.divider()

        # 6. Relationships
        st.markdown("### 🔗 Feature Relationships")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Attendance vs Final Marks**")
            fig, ax = plt.subplots(figsize=(6, 4))
            eligible_mask = df['attendance'] >= 75
            ax.scatter(df[eligible_mask]['attendance'], df[eligible_mask]['final_marks'], 
                      alpha=0.6, color='#4ECDC4', label='Eligible', s=50)
            ax.scatter(df[~eligible_mask]['attendance'], df[~eligible_mask]['final_marks'], 
                      alpha=0.6, color='#FF6B6B', label='Ineligible', s=50)
            ax.set_xlabel('Attendance (%)')
            ax.set_ylabel('Final Marks')
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)

        with col2:
            st.write("**Hours Studied vs Final Marks**")
            fig, ax = plt.subplots(figsize=(6, 4))
            scatter = ax.scatter(df_eligible['hours_studied'], df_eligible['final_marks'], 
                               c=df_eligible['attendance'], cmap='viridis', alpha=0.6, s=50)
            ax.set_xlabel('Hours Studied (per day)')
            ax.set_ylabel('Final Marks')
            plt.colorbar(scatter, ax=ax, label='Attendance (%)')
            ax.grid(alpha=0.3)
            st.pyplot(fig)

        st.divider()

        # 7. Model Performance Metrics
        st.markdown("### 📈 Model Performance Metrics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Linear Regression Performance")
            from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
            
            y_pred_marks = lin_reg.predict(X)
            r2 = r2_score(y_marks, y_pred_marks)
            mae = mean_absolute_error(y_marks, y_pred_marks)
            rmse = np.sqrt(mean_squared_error(y_marks, y_pred_marks))
            
            metrics_df = pd.DataFrame({
                'Metric': ['R² Score', 'MAE', 'RMSE'],
                'Value': [f'{r2:.4f}', f'{mae:.2f} marks', f'{rmse:.2f} marks'],
                'Interpretation': ['Excellent fit (>0.8)', 'Low error', 'Acceptable variance']
            })
            st.dataframe(metrics_df, hide_index=True)
            
            # Prediction vs Actual plot
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(y_marks, y_pred_marks, alpha=0.5, color='#4ECDC4')
            ax.plot([y_marks.min(), y_marks.max()], [y_marks.min(), y_marks.max()], 
                   'r--', lw=2, label='Perfect Prediction')
            ax.set_xlabel('Actual Marks')
            ax.set_ylabel('Predicted Marks')
            ax.set_title('Prediction Accuracy')
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            st.markdown("#### Logistic Regression Performance")
            from sklearn.metrics import precision_score, recall_score, f1_score
            
            precision = precision_score(y_pass, y_pred)
            recall = recall_score(y_pass, y_pred)
            f1 = f1_score(y_pass, y_pred)
            
            metrics_df2 = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                'Value': [f'{accuracy:.4f}', f'{precision:.4f}', f'{recall:.4f}', f'{f1:.4f}'],
                'Interpretation': ['Highly accurate', 'Few false positives', 'Few false negatives', 'Excellent balance']
            })
            st.dataframe(metrics_df2, hide_index=True)
            
            # ROC Curve
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(y_pass, probs)
            roc_auc = auc(fpr, tpr)
            
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr, tpr, color='#4ECDC4', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Classifier')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)

        st.divider()

        # 8. Correlation Heatmap
        st.markdown("### 🔥 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Select numeric columns for correlation
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", 
                   linewidths=0.5, ax=ax, square=True, cbar_kws={"shrink": 0.8})
        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
        st.pyplot(fig)