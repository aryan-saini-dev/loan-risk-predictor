# Building the Full-Stack Loan Risk Prediction App

This guide explains how to build a full-stack application with FastAPI backend and Vite frontend, focusing on the architecture, API design, and integration.

## Table of Contents
1. [Project Architecture](#project-architecture)
2. [Setting Up the Backend (FastAPI)](#setting-up-the-backend-fastapi)
3. [API Design](#api-design)
4. [Setting Up the Frontend (Vite)](#setting-up-the-frontend-vite)
5. [Connecting Frontend to Backend](#connecting-frontend-to-backend)
6. [Deployment Strategy](#deployment-strategy)
7. [Environment Configuration](#environment-configuration)
8. [CORS Configuration](#cors-configuration)

---

## Project Architecture

### Tech Stack
- **Backend**: Python FastAPI (high-performance async framework)
- **Frontend**: React with Vite (fast build tool)
- **ML Model**: scikit-learn Logistic Regression
- **Serialization**: joblib for model/scaler persistence

### Project Structure
```
loan-risk-predictor/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── requirements.txt     # Python dependencies
│   ├── credit_risk_model.pkl # Trained ML model
│   ├── scaler.pkl           # Feature scaler
│   └── retrain_model.py     # Model training script
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main React component
│   │   ├── main.jsx         # React entry point
│   │   └── index.css        # Global styles
│   ├── package.json         # Node dependencies
│   ├── vite.config.js       # Vite configuration
│   └── vercel.json          # Vercel deployment config
└── docs/
    ├── 1-ML_MODEL_GUIDE.md
    └── 2-FULLSTACK_APP_GUIDE.md
```

### Data Flow
```
User Input (Frontend) → API Request (Axios) → FastAPI Endpoint → 
Model Prediction → API Response → Frontend Display
```

---

## Setting Up the Backend (FastAPI)

### 1. Create Backend Directory

```bash
mkdir backend
cd backend
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 3. Create requirements.txt

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
python-multipart>=0.0.6
```

**Why these dependencies?**
- `fastapi`: Modern, fast web framework for building APIs
- `uvicorn`: ASGI server to run FastAPI
- `pydantic`: Data validation using Python type hints
- `numpy`: Numerical computing for model predictions
- `scikit-learn`: ML library for model loading
- `joblib`: Efficient serialization for sklearn objects
- `python-multipart`: Handle form data uploads

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Create FastAPI Application (main.py)

```python
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
    allow_origins=["*"],
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
```

**Key Components Explained:**

1. **CORS Middleware**: Allows frontend to make requests to backend
2. **Pydantic Models**: 
   - `LoanRequest`: Validates incoming data
   - `LoanResponse`: Defines API response structure
3. **Startup Event**: Loads model on application startup
4. **POST Endpoint**: `/predict` - Main prediction endpoint
5. **Error Handling**: Graceful error messages for debugging

### 6. Place Model Files

Copy your trained model files to the backend directory:
- `credit_risk_model.pkl`
- `scaler.pkl`

### 7. Run Backend Locally

```bash
uvicorn main:app --reload
```

Backend will run at: `http://localhost:8000`

---

## API Design

### Endpoint: POST /predict

**Purpose**: Predict loan default risk based on applicant features

**Request Body (JSON)**:
```json
{
  "age": 35,
  "income": 75000.0,
  "loan_amount": 25000.0,
  "credit_score": 700,
  "employment_years": 5.0,
  "education_level": 1,
  "housing_status": 1
}
```

**Field Descriptions**:
- `age`: Applicant's age (18-100)
- `income`: Annual income (float)
- `loan_amount`: Requested loan amount (float)
- `credit_score`: Credit score (300-850)
- `employment_years`: Years of employment (float)
- `education_level`: 0=High School, 1=Bachelors, 2=Masters, 3=PhD
- `housing_status`: 0=Rent, 1=Mortgage, 2=Own

**Response (JSON)**:
```json
{
  "prediction": 0,
  "probability": 0.15,
  "risk_level": "Low Risk",
  "status": "Approved"
}
```

**Field Descriptions**:
- `prediction`: 0 (no default) or 1 (default)
- `probability`: Probability of default (0-1)
- `risk_level`: "Low Risk" or "High Risk"
- `status`: "Approved" or "Denied"

### Endpoint: GET /

**Purpose**: Health check endpoint

**Response (JSON)**:
```json
{
  "message": "Loan Default Risk Prediction API",
  "status": "running"
}
```

### API Documentation

FastAPI automatically generates interactive API docs:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## Setting Up the Frontend (Vite)

### 1. Create Frontend Directory

```bash
# Go back to project root
cd ..

# Create frontend directory
mkdir frontend
cd frontend
```

### 2. Initialize Vite React Project

```bash
npm create vite@latest . -- --template react
```

### 3. Install Dependencies

```bash
# Install React dependencies
npm install

# Install additional dependencies
npm install axios
```

**Why Axios?**
- Promise-based HTTP client
- Better error handling than fetch
- Automatic JSON transformation
- Request/response interceptors

### 4. Configure Vite (vite.config.js)

```javascript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true
  }
})
```

### 5. Create Environment Variable File (.env)

```env
VITE_API_URL=http://localhost:8000
```

**Why .env file?**
- Keeps API URL out of source code
- Easy to change between local and production
- Vite automatically loads variables prefixed with `VITE_`

---

## Connecting Frontend to Backend

### 1. Create API Service (api.js)

```javascript
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const predictLoanRisk = async (formData) => {
  try {
    const response = await axios.post(`${API_URL}/predict`, {
      age: parseInt(formData.age),
      income: parseFloat(formData.income),
      loan_amount: parseFloat(formData.loan_amount),
      credit_score: parseInt(formData.credit_score),
      employment_years: parseFloat(formData.employment_years),
      education_level: parseInt(formData.education_level),
      housing_status: parseInt(formData.housing_status)
    });
    return response.data;
  } catch (error) {
    throw error.response?.data?.detail || 'An error occurred';
  }
};
```

### 2. Use API in React Component

```javascript
import { useState } from 'react';
import { predictLoanRisk } from './api';

function App() {
  const [formData, setFormData] = useState({
    age: '',
    income: '',
    loan_amount: '',
    credit_score: '',
    employment_years: '',
    education_level: '',
    housing_status: ''
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setResult(null);

    try {
      const prediction = await predictLoanRisk(formData);
      setResult(prediction);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      {/* Form inputs here */}
      <button type="submit" disabled={loading}>
        {loading ? 'Analyzing...' : 'Predict Risk'}
      </button>
      
      {error && <p className="error">{error}</p>}
      {result && <p>Risk: {result.risk_level}</p>}
    </form>
  );
}
```

### 3. Run Frontend Locally

```bash
npm run dev
```

Frontend will run at: `http://localhost:5173`

---

## Deployment Strategy

### Backend Deployment (Render)

**Why Render?**
- Free tier for Python web services
- Easy GitHub integration
- Automatic SSL
- Built-in health checks

**Steps:**

1. **Push Code to GitHub**
```bash
git add .
git commit -m "Initial commit"
git push
```

2. **Connect to Render**
   - Go to [render.com](https://render.com)
   - Click "New +" → "Web Service"
   - Select your GitHub repository

3. **Configure Render**
   - **Root Directory**: `backend`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

4. **Deploy**
   - Render will build and deploy
   - Get your backend URL (e.g., `https://loan-risk-api.onrender.com`)

**Important Notes:**
- Render provides `$PORT` environment variable automatically
- Free tier spins down after 15 minutes inactivity
- First request after spin-down takes 30-50 seconds

### Frontend Deployment (Vercel)

**Why Vercel?**
- Free tier for static sites
- Instant deployments
- Automatic HTTPS
- Preview deployments

**Steps:**

1. **Push Code to GitHub** (already done)

2. **Connect to Vercel**
   - Go to [vercel.com](https://vercel.com)
   - Click "Add New Project"
   - Select your GitHub repository

3. **Configure Vercel**
   - **Framework Preset**: Vite
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
   - **Install Command**: `npm install`

4. **Add Environment Variable**
   - Go to Settings → Environment Variables
   - Add: `VITE_API_URL` = `https://loan-risk-api.onrender.com`

5. **Deploy**
   - Vercel will build and deploy
   - Get your frontend URL

**Important Notes:**
- Vercel automatically rebuilds on git push
- Static sites stay awake (no spin-down)
- Preview deployments for each branch

---

## Environment Configuration

### Backend Environment Variables

Create `.env` file in backend directory (local development only):

```env
PORT=8000
```

**Production (Render):**
- Render automatically provides `PORT` environment variable
- No manual configuration needed

### Frontend Environment Variables

Create `.env` file in frontend directory:

```env
# Local development
VITE_API_URL=http://localhost:8000

# Production (Vercel)
# VITE_API_URL=https://loan-risk-api.onrender.com
```

**Why VITE_ prefix?**
- Vite only exposes variables starting with `VITE_` to client-side code
- Security: prevents accidental exposure of sensitive variables

**Production (Vercel):**
- Add `VITE_API_URL` in Vercel dashboard
- Settings → Environment Variables
- Set for Production environment

---

## CORS Configuration

### What is CORS?

Cross-Origin Resource Sharing (CORS) is a security feature that restricts web browsers from making requests to a different domain than the one serving the page.

### Why Needed?

Your frontend (Vercel) and backend (Render) are on different domains:
- Frontend: `https://loan-risk-predictor.vercel.app`
- Backend: `https://loan-risk-api.onrender.com`

Without CORS, the browser will block requests from frontend to backend.

### FastAPI CORS Configuration

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)
```

**Configuration Options:**

**Option 1: Allow All Origins (Simple)**
```python
allow_origins=["*"]
```
- Pros: Easy to set up
- Cons: Less secure

**Option 2: Specific Origins (More Secure)**
```python
allow_origins=[
    "http://localhost:5173",
    "https://loan-risk-predictor.vercel.app"
]
```
- Pros: More secure
- Cons: Need to update when domains change

**Recommendation for this project:**
Use `["*"]` for simplicity since this is a public API without authentication.

---

## Testing the Integration

### 1. Test Backend API Directly

Using curl:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "income": 75000.0,
    "loan_amount": 25000.0,
    "credit_score": 700,
    "employment_years": 5.0,
    "education_level": 1,
    "housing_status": 1
  }'
```

### 2. Test with Swagger UI

Navigate to: `http://localhost:8000/docs`

Use the interactive UI to test the `/predict` endpoint.

### 3. Test Frontend Integration

1. Start backend: `uvicorn main:app --reload`
2. Start frontend: `npm run dev`
3. Open browser to `http://localhost:5173`
4. Fill form and submit
5. Check browser console for errors

### 4. Debug Common Issues

**Issue: CORS Error**
- Check CORS configuration in backend
- Ensure backend is running
- Check browser console for exact error

**Issue: Model Not Loaded**
- Check that model files exist in backend directory
- Check backend logs for loading errors
- Ensure joblib is installed

**Issue: Connection Refused**
- Ensure backend is running
- Check API URL in frontend
- Verify port is correct

---

## Summary

### Backend (FastAPI)
- ✅ Fast, modern Python web framework
- ✅ Automatic API documentation
- ✅ Pydantic for data validation
- ✅ Async support for scalability
- ✅ Easy deployment on Render

### Frontend (Vite + React)
- ✅ Fast development server
- ✅ Hot module replacement
- ✅ Optimized production builds
- ✅ Easy deployment on Vercel
- ✅ Environment variable support

### Integration
- ✅ RESTful API design
- ✅ CORS configuration
- ✅ Error handling
- ✅ Environment-based configuration

### Deployment
- ✅ Backend: Render (free tier)
- ✅ Frontend: Vercel (free tier)
- ✅ Separate platforms for scalability
- ✅ Environment variables for configuration

---

## Next Steps

1. **Add Authentication**: Protect API endpoints with API keys or JWT
2. **Rate Limiting**: Prevent abuse of the API
3. **Monitoring**: Track API usage and performance
4. **Logging**: Add structured logging for debugging
5. **Testing**: Add unit and integration tests
6. **CI/CD**: Set up automated testing and deployment
