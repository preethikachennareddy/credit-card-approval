"""
FastAPI Backend Service — Credit Card Approval Prediction
Endpoints: /predict, /batch-predict, /metrics, /health
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Literal
import joblib
import numpy as np
import pandas as pd
import json
import os
import logging
from datetime import datetime

from preprocessing import CreditApprovalPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Credit Card Approval Prediction API",
    description="ML backend service using SVM and Decision Tree models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Load models at startup ────────────────────────────────────────────────────
MODELS = {}

@app.on_event("startup")
async def load_models():
    global MODELS
    model_paths = {
        'decision_tree': 'models/decision_tree.pkl',
        'svm':           'models/svm.pkl',
        'preprocessor':  'models/preprocessor.pkl',
    }
    for name, path in model_paths.items():
        if os.path.exists(path):
            MODELS[name] = joblib.load(path)
            logger.info(f"Loaded: {name}")
        else:
            logger.warning(f"Model not found: {path}")

    metrics_path = 'reports/model_metrics.json'
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            MODELS['metrics'] = json.load(f)
        logger.info("Loaded model metrics")


# ─── Request/Response schemas ──────────────────────────────────────────────────

class ApplicantInput(BaseModel):
    gender: Optional[str] = Field(None, description="Applicant gender (a/b)")
    age: float = Field(..., ge=18, le=100, description="Age in years")
    debt: float = Field(..., ge=0, description="Debt ratio")
    married: Optional[str] = Field(None, description="Marital status")
    bank_customer: Optional[str] = Field(None, description="Bank customer type")
    education_level: Optional[str] = Field(None, description="Education level code")
    ethnicity: Optional[str] = Field(None, description="Ethnicity code")
    years_employed: float = Field(..., ge=0, description="Years of employment")
    prior_default: int = Field(..., ge=0, le=10, description="Number of prior defaults")
    employed: int = Field(..., ge=0, le=1, description="Currently employed (0/1)")
    credit_score: float = Field(..., ge=0, description="Credit score / prior credit history")
    drivers_license: int = Field(..., ge=0, le=1, description="Has driver's license (0/1)")
    citizen: Optional[str] = Field(None, description="Citizenship status")
    zip_code: int = Field(..., ge=0, description="Zip code")
    income: float = Field(..., ge=0, description="Annual income")
    model: Literal['decision_tree', 'svm', 'both'] = Field('both', description="Model to use")


class PredictionResponse(BaseModel):
    approved: bool
    approval_probability: float
    risk_score: float
    model_used: str
    decision: str
    confidence: str
    timestamp: str


class BatchPredictionResponse(BaseModel):
    total: int
    approved_count: int
    denied_count: int
    approval_rate: float
    predictions: List[dict]


# ─── Helpers ───────────────────────────────────────────────────────────────────

FEATURE_ORDER = [
    'gender', 'age', 'debt', 'married', 'bank_customer', 'education_level',
    'ethnicity', 'years_employed', 'prior_default', 'employed', 'credit_score',
    'drivers_license', 'citizen', 'zip_code', 'income'
]

def prepare_input(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])[FEATURE_ORDER]
    preprocessor: CreditApprovalPreprocessor = MODELS.get('preprocessor')
    if preprocessor:
        df = preprocessor.transform(df)
    return df

def get_confidence(prob: float) -> str:
    if prob >= 0.85 or prob <= 0.15:
        return "Very High"
    elif prob >= 0.70 or prob <= 0.30:
        return "High"
    elif prob >= 0.60 or prob <= 0.40:
        return "Moderate"
    else:
        return "Low"


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "models_loaded": list(MODELS.keys()),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/metrics")
async def get_metrics():
    if 'metrics' not in MODELS:
        raise HTTPException(404, "Model metrics not found. Run training first.")
    return MODELS['metrics']


@app.post("/predict", response_model=PredictionResponse)
async def predict(applicant: ApplicantInput):
    model_name = applicant.model
    input_data = applicant.dict(exclude={'model'})

    try:
        X = prepare_input(input_data)
    except Exception as e:
        raise HTTPException(422, f"Preprocessing error: {str(e)}")

    if model_name == 'both':
        # Ensemble: average probabilities
        probs = []
        for mname in ['decision_tree', 'svm']:
            if mname not in MODELS:
                raise HTTPException(503, f"Model '{mname}' not loaded")
            probs.append(MODELS[mname].predict_proba(X)[0][1])
        prob = float(np.mean(probs))
        used = 'ensemble (DT + SVM)'
    else:
        if model_name not in MODELS:
            raise HTTPException(503, f"Model '{model_name}' not loaded")
        prob = float(MODELS[model_name].predict_proba(X)[0][1])
        used = model_name

    approved = prob >= 0.5
    return PredictionResponse(
        approved=approved,
        approval_probability=round(prob, 4),
        risk_score=round(1 - prob, 4),
        model_used=used,
        decision="APPROVED ✓" if approved else "DENIED ✗",
        confidence=get_confidence(prob),
        timestamp=datetime.now().isoformat()
    )


@app.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(applicants: List[ApplicantInput]):
    if len(applicants) > 1000:
        raise HTTPException(400, "Batch size must be ≤ 1000")

    results = []
    for i, applicant in enumerate(applicants):
        try:
            result = await predict(applicant)
            results.append({
                'id': i + 1,
                'approved': result.approved,
                'probability': result.approval_probability,
                'risk_score': result.risk_score,
                'confidence': result.confidence
            })
        except Exception as e:
            results.append({'id': i + 1, 'error': str(e)})

    approved = [r for r in results if r.get('approved')]
    return BatchPredictionResponse(
        total=len(results),
        approved_count=len(approved),
        denied_count=len(results) - len(approved),
        approval_rate=round(len(approved) / len(results), 4),
        predictions=results
    )


@app.get("/model-info/{model_name}")
async def model_info(model_name: str):
    if model_name not in ('decision_tree', 'svm'):
        raise HTTPException(404, "Model not found")
    if model_name not in MODELS:
        raise HTTPException(503, "Model not loaded")

    model = MODELS[model_name]
    metrics = MODELS.get('metrics', {}).get(model_name, {})

    info = {
        'name': model_name,
        'type': type(model).__name__,
        'parameters': model.get_params(),
        'metrics': {k: v for k, v in metrics.items()
                    if k not in ('confusion_matrix', 'classification_report', 'feature_importance')},
    }

    if model_name == 'decision_tree':
        info['tree_depth'] = model.get_depth()
        info['n_leaves'] = model.get_n_leaves()

    return info
