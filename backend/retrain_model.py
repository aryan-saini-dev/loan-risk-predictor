import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
import pickle
import joblib

# Generate synthetic training data similar to the original dataset
np.random.seed(42)
n_samples = 1000

# Generate features
age = np.random.randint(18, 75, n_samples)
income = np.random.normal(60000, 30000, n_samples)
income = np.maximum(income, 0)  # Ensure non-negative
loan_amount = np.random.normal(30000, 15000, n_samples)
loan_amount = np.maximum(loan_amount, 0)  # Ensure non-negative
credit_score = np.random.normal(650, 100, n_samples)
credit_score = np.clip(credit_score, 300, 850)
employment_years = np.random.randint(0, 40, n_samples)
education_level = np.random.randint(0, 4, n_samples)
housing_status = np.random.randint(0, 3, n_samples)

# Create target variable based on features (simplified logic)
default = np.zeros(n_samples)
# Higher risk if: low credit score, high loan/income ratio, short employment
risk_score = (
    (850 - credit_score) / 550 * 0.4 +
    (loan_amount / (income + 1)) * 0.3 +
    (1 - employment_years / 40) * 0.2 +
    (3 - education_level) / 3 * 0.1
)
default[risk_score > 0.5] = 1

# Create DataFrame
df = pd.DataFrame({
    'Age': age,
    'Income': income,
    'Loan_Amount': loan_amount,
    'Credit_Score': credit_score,
    'Employment_Years': employment_years,
    'Education_Level': education_level,
    'Housing_Status': housing_status,
    'Default': default
})

# Prepare features and target
X = df[['Age', 'Income', 'Loan_Amount', 'Credit_Score', 'Employment_Years', 'Education_Level', 'Housing_Status']]
y = df['Default']

# Scale features
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
model.fit(X_scaled, y)

print("Model trained successfully")
print(f"Model accuracy: {model.score(X_scaled, y):.4f}")

# Save model and scaler with joblib (more reliable than pickle for sklearn)
joblib.dump(model, 'credit_risk_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved successfully with joblib")
