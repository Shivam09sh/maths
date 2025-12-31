import unittest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
import os
import sys

# Define features strictly
FEATURES = ["attendance", "hours_studied", "internal_marks", "previous_sem_cgpa"]

class TestStudentScoreSystem(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment and mock data"""
        self.mock_data = pd.DataFrame({
            'attendance': [80, 90, 60, 40, 85, 76],
            'hours_studied': [5, 6, 2, 1, 5.5, 2],
            'internal_marks': [25, 28, 15, 10, 26, 12],
            'previous_sem_cgpa': [8.5, 9.0, 6.0, 5.0, 8.8, 6.5],
            'final_marks': [80, 88, 55, 30, 82, 35],
            'eligible': ['Yes', 'Yes', 'No', 'No', 'Yes', 'Yes']
        })
        
        # Prepare data for training
        self.df_eligible = self.mock_data[self.mock_data['attendance'] >= 75].copy()
        
        # Separate features and targets
        self.X = self.df_eligible[FEATURES]
        self.y_marks = self.df_eligible['final_marks']
        self.y_pass = (self.df_eligible['final_marks'] >= 40).astype(int)

    def test_eligibility_logic(self):
        """Test if eligibility filtering works correctly"""
        eligible_count = len(self.df_eligible)
        self.assertEqual(eligible_count, 4, "Should have 4 eligible students (>= 75% attendance)")
        
        # Verify no ineligible students are in the training set
        min_attendance = self.df_eligible['attendance'].min()
        self.assertGreaterEqual(min_attendance, 75, "Training set should not contain attendance < 75%")

    def test_linear_regression_constraints(self):
        """Test Linear Regression model constraints (positive=True)"""
        lin_reg = LinearRegression(positive=True)
        lin_reg.fit(self.X, self.y_marks)
        
        # Check if coefficients are positive
        all_positive = all(coef >= 0 for coef in lin_reg.coef_)
        self.assertTrue(all_positive, "All coefficients should be positive due to positive=True constraint")
        
        # Test prediction logic
        prediction = lin_reg.predict([[85, 6, 25, 8.5]])
        self.assertTrue(0 <= prediction[0] <= 100, "Prediction should be reasonable (0-100 range implied)")

    def test_logistic_regression_probabilities(self):
        """Test Logistic Regression output probabilities"""
        log_reg = LogisticRegression()
        log_reg.fit(self.X, self.y_pass)
        
        # Test probability prediction
        test_input = pd.DataFrame([[85, 6, 25, 8.5]], columns=FEATURES)
        probs = log_reg.predict_proba(test_input)
        
        # probabilities should sum to 1
        self.assertAlmostEqual(probs[0].sum(), 1.0, places=5)
        
        # Check shape
        self.assertEqual(probs.shape, (1, 2), "Should return probabilities for 2 classes (Fail, Pass)")

    def test_risk_classification(self):
        """Test Risk Level Classification Logic"""
        # Mock probabilities
        high_risk_prob = 0.3
        med_risk_prob = 0.6
        low_risk_prob = 0.8
        
        def get_risk(prob):
            if prob < 0.4: return "High Risk"
            elif prob < 0.7: return "Medium Risk"
            return "Low Risk"
            
        self.assertEqual(get_risk(high_risk_prob), "High Risk")
        self.assertEqual(get_risk(med_risk_prob), "Medium Risk")
        self.assertEqual(get_risk(low_risk_prob), "Low Risk")

    def test_feature_integrity(self):
        """Verify feature columns match strict requirements"""
        required_features = ["attendance", "hours_studied", "internal_marks", "previous_sem_cgpa"]
        self.assertEqual(list(self.X.columns), required_features)

if __name__ == '__main__':
    unittest.main()
