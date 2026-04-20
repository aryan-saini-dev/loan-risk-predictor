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

---

## Deployment Guide

### Overview
This project uses a split deployment:
- **Backend**: Render (Python FastAPI)
- **Frontend**: Vercel (React static site)

### Step 1: Prepare the Code

1. Ensure all changes are committed to GitHub
2. Verify the backend uses `allow_origins=["*"]` for CORS
3. Verify the backend uses `os.environ.get("PORT", 8000)` for port configuration
4. Verify the frontend uses `import.meta.env.VITE_API_URL` for the API URL

### Step 2: Deploy Backend to Render

1. Go to [render.com](https://render.com) and sign up/login
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository
4. Configure the service:

   **Name**: `loan-risk-api` (or any name)

   **Language**: `Python`

   **Branch**: `main`

   **Region**: Choose your preferred region

   **Root Directory**: `backend`

   **Build Command**: `pip install -r requirements.txt`

   **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

5. **Important**: Do NOT manually add a `PORT` environment variable - Render provides `$PORT` automatically
6. Click **"Deploy"**

7. After deployment, copy your backend URL (e.g., `https://loan-risk-api-ai1c.onrender.com`)

**Troubleshooting Render**:
- If you see "no open ports detected", ensure you're using `$PORT` in the start command
- If deployment fails, click "Manual Deploy" → "Deploy latest commit" after pushing changes
- Free tier spins down after 15 minutes of inactivity - first request may take 30-50 seconds

### Step 3: Deploy Frontend to Vercel

1. Go to [vercel.com](https://vercel.com) and sign up/login
2. Click **"Add New Project"**
3. Import your GitHub repository
4. Configure the project:

   **Name**: `loan-risk-frontend` (or any name)

   **Framework Preset**: `Vite`

   **Root Directory**: `frontend`

   **Build Command**: `npm run build`

   **Output Directory**: `dist`

   **Install Command**: `npm install`

5. **Add Environment Variable**:
   - Go to **Settings** → **Environment Variables**
   - Add: `VITE_API_URL` = `https://loan-risk-api-ai1c.onrender.com` (use your actual backend URL)
   - Select **Production** environment
   - Click **Save**

6. Click **"Deploy"**

7. After deployment, copy your frontend URL

**Troubleshooting Vercel**:
- If build fails with "command not found", ensure build command is `npm run build`
- If you see 404 errors, ensure the API URL environment variable is set correctly
- Vercel auto-deploys on push to main branch

### Step 4: Verify Deployment

1. Test the backend directly:
   ```
   curl https://your-backend-url.onrender.com/
   ```
   Should return: `{"message": "Loan Default Risk Prediction API", "status": "running"}`

2. Test the frontend:
   - Open your Vercel URL
   - Fill out the form
   - Click "Predict Risk"
   - Should show loading state, then results

### Step 5: Update CORS (if needed)

If you see CORS errors in the browser console:

1. Update `backend/main.py`:
   ```python
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["*"],  # Allow all origins
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )
   ```

2. Push changes to GitHub
3. Redeploy backend on Render (Manual Deploy)

### Important Notes

**Security**:
- The API URL is visible in browser DevTools (unavoidable for frontend apps)
- Never commit API keys or secrets to GitHub
- Use environment variables for sensitive data

**Free Tier Limitations**:
- **Render**: Spins down after 15 min inactivity - first request takes 30-50 seconds
- **Vercel**: Static sites stay awake - instant loading
- Both have bandwidth limits on free tier

**Updating Deployments**:
- Push changes to GitHub
- Render: Click "Manual Deploy" → "Deploy latest commit"
- Vercel: Auto-deploys on push, or click "Redeploy" in dashboard

## License

This project is for educational purposes.
