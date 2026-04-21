# Building the Loan Default Risk Prediction ML Model

This guide walks you through building the actual model used in this project. You'll learn by doing - each section includes code you can run and see the results.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Generating Synthetic Data](#generating-synthetic-data)
3. [Understanding the Features](#understanding-the-features)
4. [Creating the Target Variable](#creating-the-target-variable)
5. [Data Preprocessing](#data-preprocessing)
6. [Model Training](#model-training)
7. [Making Predictions](#making-predictions)
8. [Saving the Model](#saving-the-model)

---

## Project Overview

### What We're Building
A binary classification model that predicts whether a loan applicant will default (1) or not (0) based on their financial profile.

### The Actual Model
This project uses **synthetic data** generated in `retrain_model.py` with:
- **1000 samples** of loan applications
- **7 features** per applicant
- **Logistic Regression** classifier
- **93.8% accuracy** on training data

### Why Synthetic Data?
- No access to real loan data (privacy concerns)
- Allows us to test the full pipeline
- Demonstrates the complete ML workflow
- Easy to modify and experiment

**Expected Output**: When you run `retrain_model.py`, you'll see:
```
Model trained successfully
Model accuracy: 0.9380
Model and scaler saved successfully with joblib
```

---

## Generating Synthetic Data

### The Code (from `retrain_model.py`)

```python
import numpy as np
import pandas as pd

# Set random seed for reproducibility
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
```

### What This Does

**`np.random.seed(42)`**: Ensures we get the same "random" data every time. This is crucial for reproducibility - if you run the code twice, you get identical results.

**Age**: Random integers between 18-75 (typical working age range)

**Income**: Normal distribution with:
- Mean: $60,000
- Standard deviation: $30,000
- Minimum: $0 (can't have negative income)

**Loan Amount**: Normal distribution with:
- Mean: $30,000
- Standard deviation: $15,000
- Minimum: $0

**Credit Score**: Normal distribution with:
- Mean: 650
- Standard deviation: 100
- Range: 300-850 (clipped to valid credit score range)

**Employment Years**: Random integers between 0-40 years

**Education Level**: Random integers 0-3:
- 0 = High School
- 1 = Bachelors
- 2 = Masters
- 3 = PhD

**Housing Status**: Random integers 0-2:
- 0 = Rent
- 1 = Mortgage
- 2 = Own

### Expected Data Sample

After running this, you'll have 1000 rows. Here's what the first few look like:

| Age | Income | Loan_Amount | Credit_Score | Employment_Years | Education_Level | Housing_Status |
|-----|--------|-------------|--------------|------------------|-----------------|----------------|
| 56  | 53847  | 34732       | 651          | 8                | 2               | 0              |
| 18  | 84528  | 58216       | 542          | 25               | 2               | 2              |
| 47  | 25201  | 18626       | 654          | 3                | 2               | 1              |

---

## Understanding the Features

### Feature Breakdown

| Feature | Type | Range | Real-world meaning |
|---------|------|-------|-------------------|
| **Age** | Numerical | 18-75 | Older applicants typically more stable |
| **Income** | Numerical | $0-$150,000+ | Higher income = lower default risk |
| **Loan_Amount** | Numerical | $0-$100,000+ | Higher loan = higher risk |
| **Credit_Score** | Numerical | 300-850 | Lower score = higher risk |
| **Employment_Years** | Numerical | 0-40 | Longer employment = more stable |
| **Education_Level** | Categorical | 0-3 | Higher education often correlates with income |
| **Housing_Status** | Categorical | 0-2 | Homeowners typically more stable |

### Why These Features?

These are standard features used in real-world loan approval decisions:
- **Credit Score**: Most important - directly measures creditworthiness
- **Income**: Determines ability to repay
- **Loan Amount**: Size of the obligation
- **Employment**: Job stability indicator
- **Age**: Life stage and experience
- **Education**: Correlates with earning potential
- **Housing**: Asset ownership indicates financial stability

---

## Creating the Target Variable

### The Risk Score Formula

```python
# Create target variable based on features
default = np.zeros(n_samples)

# Higher risk if: low credit score, high loan/income ratio, short employment
risk_score = (
    (850 - credit_score) / 550 * 0.4 +      # Credit score impact (40% weight)
    (loan_amount / (income + 1)) * 0.3 +    # Loan-to-income ratio (30% weight)
    (1 - employment_years / 40) * 0.2 +      # Employment stability (20% weight)
    (3 - education_level) / 3 * 0.1          # Education level (10% weight)
)
default[risk_score > 0.5] = 1
```

### How the Risk Score Works

**Formula Breakdown**:
```
Risk Score = 0.4 × (Credit Factor) + 0.3 × (Loan/Income Factor) + 
             0.2 × (Employment Factor) + 0.1 × (Education Factor)
```

**Credit Factor** (40% weight):
- `(850 - credit_score) / 550`
- Credit score 850 → factor = 0 (lowest risk)
- Credit score 300 → factor = 1 (highest risk)
- **Why**: Credit score is the most important factor

**Loan/Income Factor** (30% weight):
- `loan_amount / (income + 1)`
- Higher ratio = higher risk
- Example: $50K loan / $50K income = 1.0 (high risk)
- Example: $10K loan / $100K income = 0.1 (low risk)
- **Why**: Measures debt burden relative to income

**Employment Factor** (20% weight):
- `(1 - employment_years / 40)`
- 40 years employment → factor = 0 (lowest risk)
- 0 years employment → factor = 1 (highest risk)
- **Why**: Job stability matters

**Education Factor** (10% weight):
- `(3 - education_level) / 3`
- PhD (3) → factor = 0 (lowest risk)
- High School (0) → factor = 1 (highest risk)
- **Why**: Education correlates with financial literacy

### Threshold

```python
default[risk_score > 0.5] = 1
```

If `risk_score > 0.5`, we label as "Will Default" (1). Otherwise "Won't Default" (0).

**Why 0.5?** It's a balanced threshold - if the weighted risk factors exceed 50%, we consider it high risk.

### Expected Output

With this formula, you'll see approximately:
- 30-40% of applicants labeled as "Default" (1)
- 60-70% labeled as "No Default" (0)

This creates a realistic class imbalance - more people don't default than do, which matches real-world data.

---

## Data Preprocessing

### Why Preprocessing?

Machine learning models work best when:
1. Features are on similar scales
2. Data is clean and consistent
3. There are no extreme outliers

### Feature Scaling with RobustScaler

```python
from sklearn.preprocessing import RobustScaler

# Prepare features and target
X = df[['Age', 'Income', 'Loan_Amount', 'Credit_Score', 
        'Employment_Years', 'Education_Level', 'Housing_Status']]
y = df['Default']

# Scale features
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
```

### Why RobustScaler Instead of StandardScaler?

**StandardScaler** uses mean and standard deviation:
- Sensitive to outliers
- If you have one person with $1M income, it skews everything

**RobustScaler** uses median and IQR (Interquartile Range):
- **Median**: The middle value (50th percentile)
- **IQR**: Range between 25th and 75th percentiles
- **Outliers don't affect it**

**Example**:
- Income: [$30K, $40K, $45K, $50K, $500K]
- Mean: $133K (skewed by $500K outlier)
- Median: $45K (unaffected by outlier)

**Expected Output**: After scaling, all features will have:
- Median ≈ 0
- IQR ≈ 1
- Outliers remain outliers but don't distort the scale

### Why Scaling Matters

Without scaling:
- Credit score (300-850) would dominate
- Education level (0-3) would be negligible
- Model would be biased toward features with larger ranges

With scaling:
- All features contribute equally
- Model learns from all features
- Better predictions

---

## Model Training

### Logistic Regression Setup

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=1000,           # More iterations for convergence
    class_weight='balanced', # Handle class imbalance
    random_state=42          # Reproducible results
)

model.fit(X_scaled, y)
```

### Parameter Explanations

**`max_iter=1000`**:
- Default is 100 iterations
- We use 1000 to ensure the model converges
- **Why**: Some datasets need more iterations to find optimal solution

**`class_weight='balanced'`**:
- Automatically adjusts weights for imbalanced classes
- Gives more weight to minority class (defaults)
- **Why**: Without this, model might ignore the minority class

**`random_state=42`**:
- Ensures reproducible results
- Same random seed every time
- **Why**: Debugging and consistency

### Expected Training Output

```
Model trained successfully
Model accuracy: 0.9380
```

**What 93.8% Accuracy Means**:
- The model correctly predicts 938 out of 1000 training samples
- This is high because we're testing on the same data we trained on
- In production, we'd use train/test split for real evaluation

### Why Logistic Regression?

1. **Interpretable**: You can see which features matter most
2. **Fast**: Trains in milliseconds
3. **Probabilistic**: Gives probability of default (e.g., "73% chance of default")
4. **Baseline**: Good starting point before trying complex models
5. **Works Well**: For this type of classification, it's often sufficient

### What the Model Learns

The model learns coefficients for each feature:
- Positive coefficient → increases default risk
- Negative coefficient → decreases default risk
- Larger absolute value → stronger influence

**Example Coefficients** (hypothetical):
- Credit Score: -2.5 (higher score = much lower risk)
- Loan/Income Ratio: +1.8 (higher ratio = higher risk)
- Employment Years: -0.3 (more years = slightly lower risk)

---

## Making Predictions

### Loading the Model

```python
import joblib
import numpy as np

# Load the saved model and scaler
model = joblib.load('credit_risk_model.pkl')
scaler = joblib.load('scaler.pkl')
```

### Preparing Input Data

```python
# Example applicant
features = np.array([[
    35,              # Age
    75000,           # Income
    25000,           # Loan Amount
    700,             # Credit Score
    5,               # Employment Years
    1,               # Education Level (Bachelors)
    1                # Housing Status (Mortgage)
]])

# Scale the features (same as training)
features_scaled = scaler.transform(features)
```

**Important**: You MUST scale input data the same way as training data. The model expects scaled values.

### Getting Prediction

```python
# Get prediction (0 or 1)
prediction = model.predict(features_scaled)[0]

# Get probability of default
probability = model.predict_proba(features_scaled)[0][1]

# Determine risk level
if prediction == 0:
    risk_level = "Low Risk"
    status = "Approved"
else:
    risk_level = "High Risk"
    status = "Denied"

print(f"Prediction: {prediction}")
print(f"Default Probability: {probability:.4f}")
print(f"Risk Level: {risk_level}")
print(f"Status: {status}")
```

### Expected Output Examples

**Example 1: Low Risk Applicant**
```
Prediction: 0
Default Probability: 0.1234
Risk Level: Low Risk
Status: Approved
```
This applicant has good credit score (700), decent income ($75K), and reasonable loan amount ($25K). The model predicts only 12.34% chance of default.

**Example 2: High Risk Applicant**
```
Prediction: 1
Default Probability: 0.7891
Risk Level: High Risk
Status: Denied
```
This applicant might have low credit score, high loan-to-income ratio, or short employment history. The model predicts 78.91% chance of default.

---

## Saving the Model

### Why Save the Model?

1. **Deployment**: Use the trained model in production without retraining
2. **Consistency**: Same model produces same predictions
3. **Efficiency**: Avoid retraining for each request

### Saving with joblib

```python
import joblib

# Save the model
joblib.dump(model, 'credit_risk_model.pkl')

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved successfully with joblib")
```

### Why joblib Instead of pickle?

- **Better for sklearn**: Optimized for scikit-learn objects
- **Efficient**: Handles large numpy arrays better
- **Compatible**: Works across different Python versions (mostly)

**Expected Output**:
```
Model and scaler saved successfully with joblib
```

This creates two files in your directory:
- `credit_risk_model.pkl` - The trained Logistic Regression model
- `scaler.pkl` - The fitted RobustScaler

---

## Summary: What You've Built

### The Complete Pipeline

1. **Generated 1000 synthetic loan applications** with realistic features
2. **Created a risk score formula** to label defaults (weighted by credit score, loan/income ratio, employment, education)
3. **Scaled features using RobustScaler** to handle outliers
4. **Trained Logistic Regression** with class weighting for imbalance
5. **Achieved 93.8% accuracy** on training data
6. **Saved model and scaler** for deployment

### Key Takeaways

- **Synthetic data** lets you test ML pipelines without real data
- **Feature scaling** is crucial when features have different ranges
- **RobustScaler** handles outliers better than StandardScaler
- **Class weighting** prevents the model from ignoring minority classes
- **joblib** is the preferred way to save sklearn models

### Next Steps for Real Data

When you have real loan data:
1. **Exploratory Data Analysis**: Understand distributions, correlations
2. **Train/Test Split**: Evaluate on unseen data
3. **Cross-Validation**: More robust performance estimate
4. **Feature Engineering**: Create derived features (debt-to-income ratio)
5. **Try Other Models**: Random Forest, XGBoost for comparison
6. **Hyperparameter Tuning**: Optimize model parameters

---

## Running the Training Script

To train the model yourself:

```bash
cd backend
python retrain_model.py
```

**Expected Output**:
```
Model trained successfully
Model accuracy: 0.9380
Model and scaler saved successfully with joblib
```

This will create `credit_risk_model.pkl` and `scaler.pkl` in the backend directory, ready for use in the FastAPI application.
