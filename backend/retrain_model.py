import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
import pickle
import joblib
import os

# Load the real dataset from CSV
# Get the parent directory (project root) to find the CSV file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(BASE_DIR, 'realistic_credit_risk_dataset.csv')

df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} samples from {csv_path}")

# Display basic statistics
print("\nDataset Info:")
print(df.info())
print("\nClass Distribution:")
print(df['Default'].value_counts())

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
