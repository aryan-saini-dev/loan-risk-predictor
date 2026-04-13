# Loan Default Risk Prediction System

AI-powered web application for predicting loan default risk using a trained Logistic Regression model.

## Project Structure

```
Credit loan project/
├── backend/
│   ├── credit_risk_model.pkl     # Trained model (existing)
│   ├── scaler.pkl               # RobustScaler (existing)
│   ├── main.py                  # FastAPI application
│   └── requirements.txt         # Python dependencies
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       └── index.css
└── README.md
```

## Backend Setup

### Prerequisites
- Python 3.8+
- pip

### Installation

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Run the FastAPI server:
```bash
python main.py
```

The backend will be available at `http://localhost:8000`

### API Endpoint

**POST** `/predict`

Request body:
```json
{
  "age": 35,
  "income": 75000,
  "loan_amount": 25000,
  "credit_score": 720,
  "employment_years": 8,
  "education_level": 1,
  "housing_status": 2
}
```

Response:
```json
{
  "prediction": 0,
  "probability": 0.15,
  "risk_level": "Low Risk",
  "status": "Approved"
}
```

## Frontend Setup

### Prerequisites
- Node.js 18+
- npm

### Installation

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:5173`

### Build for Production

```bash
npm run build
```

## Features

### Backend
- ✅ FastAPI with proper model loading and scaling
- ✅ Critical scaling logic: Uses `scaler.transform()` before prediction
- ✅ Input validation with Pydantic models
- ✅ CORS support for frontend integration
- ✅ Error handling and proper HTTP responses

### Frontend
- ✅ Kinetic Typography Design System
- ✅ Responsive form with all required inputs
- ✅ Real-time validation and loading states
- ✅ Dynamic result display with color-coded risk levels
- ✅ Marquee animations and motion effects
- ✅ Professional financial dashboard aesthetic

### Design System Implementation
- **Color Scheme**: Rich black (#09090B), off-white (#FAFAFA), acid yellow (#DFE104)
- **Typography**: Space Grotesk font with aggressive scale hierarchy
- **Motion**: Infinite marquees, hover transformations, scroll animations
- **Brutalist Styling**: Sharp corners, 2px borders, no shadows
- **Responsive Design**: Mobile-first with dramatic viewport-based typography

## Input Features

- **Age**: Integer (18-100)
- **Annual Income**: Float (USD)
- **Loan Amount**: Float (USD)
- **Credit Score**: Integer (300-850)
- **Employment Years**: Float (years)
- **Education Level**: 
  - 0: High School
  - 1: Bachelors
  - 2: Masters
  - 3: PhD
- **Housing Status**:
  - 0: Rent
  - 1: Mortgage
  - 2: Own

## Running Both Servers

### Terminal 1 (Backend)
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

### Terminal 2 (Frontend)
```bash
cd frontend
npm install
npm run dev
```

## Testing the API

You can test the backend API directly using curl:

```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "age": 35,
  "income": 75000,
  "loan_amount": 25000,
  "credit_score": 720,
  "employment_years": 8,
  "education_level": 1,
  "housing_status": 2
}'
```

## Technology Stack

### Backend
- FastAPI
- scikit-learn
- NumPy
- Uvicorn

### Frontend
- React 18
- Vite
- Tailwind CSS
- Framer Motion
- Axios
- Lucide React

## License

This project is for educational purposes.
