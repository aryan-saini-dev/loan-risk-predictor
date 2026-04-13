from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI(title="Loan Default Risk Prediction API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "https://loan-risk-predictor-q9b4ylwf1-xyaminokiritoxs-projects.vercel.app",
        "https://loan-risk-predictor.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained model and scaler
model = None
scaler = None

class LoanRequest(BaseModel):
    age: int
    income: float
    loan_amount: float
    credit_score: int
    employment_years: float
    education_level: int
    housing_status: int

class LoanResponse(BaseModel):
    prediction: int
    probability: float
    risk_level: str
    status: str

@app.on_event("startup")
async def load_model():
    global model, scaler
    try:
        model_path = os.path.join(BASE_DIR, "credit_risk_model.pkl")
        scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
        print(f"Loading model from: {model_path}")
        print(f"Loading scaler from: {scaler_path}")
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print("Model and scaler loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

@app.get("/")
async def root():
    return {"message": "Loan Default Risk Prediction API", "status": "running"}

@app.post("/predict", response_model=LoanResponse)
async def predict_loan_risk(request: LoanRequest):
    try:
        if model is None or scaler is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Prepare features in the exact order expected by the model
        features = np.array([[
            request.age,
            request.income,
            request.loan_amount,
            request.credit_score,
            request.employment_years,
            request.education_level,
            request.housing_status
        ]])
        
        # Apply scaling transformation
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]
        
        # Determine risk level and status
        if prediction == 0:
            risk_level = "Low Risk"
            status = "Approved"
        else:
            risk_level = "High Risk"
            status = "Denied"
        
        return LoanResponse(
            prediction=int(prediction),
            probability=float(probability),
            risk_level=risk_level,
            status=status
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
