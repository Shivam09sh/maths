# 🎓 Student Score Prediction & Risk Analysis System

A Streamlit-based application for predicting student performance and analyzing risk levels with **75% attendance eligibility requirement**.

## 📋 Features

### 1. **Attendance Eligibility Check**
- **Minimum 75% attendance required** for final exam eligibility
- Students below 75% attendance are automatically marked as **INELIGIBLE**
- Ineligible students receive 0 marks (cannot take the final exam)

### 2. **Predictive Analytics**
- **Linear Regression**: Predicts final exam marks
- **Logistic Regression**: Calculates probability of passing
- **Risk Classification**: 
  - 🟢 Low Risk (≥70% pass probability)
  - 🟠 Medium Risk (40-70% pass probability)
  - 🔴 High Risk (<40% pass probability)

### 3. **Dataset Statistics**
- Shows eligibility breakdown (Eligible vs Ineligible students)
- Displays dataset preview with all student information
- Tracks attendance, study hours, internal marks, and previous CGPA

## 📊 Dataset Structure

The dataset (`student-dta.csv` or `students_data.csv`) contains:

| Column | Description | Range/Values |
|--------|-------------|--------------|
| `attendance` | Attendance percentage | 0-100% |
| `hours_studied` | Hours studied per day | 0.0-10.0 |
| `internal_marks` | Internal assessment marks | 0-30 |
| `previous_sem_cgpa` | Previous semester CGPA | 0.0-10.0 |
| `final_marks` | Final exam marks (0 if ineligible) | 0-100 |
| `eligible` | Eligibility status | Yes/No |

### Current Dataset Stats (100 students):
- **Eligible students**: 83 (83%)
- **Ineligible students**: 17 (17%)
- **Passing students** (among eligible): 78 (94%)
- **Failing students** (among eligible): 5 (6%)

## 🚀 How to Run

1. **Install dependencies**:
   ```bash
   pip install streamlit pandas numpy scikit-learn
   ```

2. **Run the application**:
   ```bash
   streamlit run app.py
   ```

3. **Access the app**:
   - Open your browser to `http://localhost:8501`

4. **Upload dataset**:
   - Upload `student-dta.csv` or `students_data.csv`

5. **Make predictions**:
   - Enter student details (attendance, study hours, etc.)
   - Click "Predict" to see results

## 📝 Usage Examples

### Example 1: Eligible Student
**Input:**
- Attendance: 85%
- Hours Studied: 4.5
- Internal Marks: 25
- Previous CGPA: 8.2

**Output:**
```
✅ Eligible for Final Exam (Attendance: 85%)
Predicted Final Marks: 76.30
Probability of Passing: 0.98
Risk Level: 🟢 Low Risk
```

### Example 2: Ineligible Student
**Input:**
- Attendance: 65%
- Hours Studied: 3.0
- Internal Marks: 20
- Previous CGPA: 7.0

**Output:**
```
❌ INELIGIBLE FOR FINAL EXAM
Attendance: 65%
Reason: Minimum 75% attendance required for final exam eligibility
⚠️ Student must improve attendance to be eligible for final exam
```

## 🔍 Model Training

- **Training Data**: Only eligible students (≥75% attendance) are used for training
- Models trained on 83 eligible students from the dataset
- Ensures predictions are based on realistic data from students who actually took the exam

## 📁 Files

- `app.py` - Main Streamlit application
- `student-dta.csv` - Student dataset with eligibility data
- `students_data.csv` - Copy of dataset (same data)
- `test_eligibility.py` - Test script for eligibility system
- `test_integration.py` - Integration test script
- `README.md` - This file

## 🎯 Key Rules

1. **75% minimum attendance** is mandatory for final exam
2. Students below this threshold receive **0 marks** (ineligible)
3. Predictions only work for **eligible students**
4. Models trained on **eligible students only** for accuracy

## 🛠️ Technologies Used

- **Python 3.13**
- **Streamlit 1.52.1** - Web interface
- **Pandas 2.3.3** - Data manipulation
- **NumPy 2.3.4** - Numerical operations
- **Scikit-learn 1.7.2** - Machine learning models

## 📈 Future Enhancements

- Add more features (assignments, quiz scores, etc.)
- Implement attendance improvement suggestions
- Add visualization charts for risk distribution
- Export prediction reports as PDF

## 📄 License

This project is for educational purposes.

---

**Created by**: Student Performance Analysis Team  
**Last Updated**: December 8, 2025
