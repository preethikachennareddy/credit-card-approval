<div align="center">

# Credit Card Approval Prediction

**End-to-end machine learning pipeline predicting credit card approvals using SVM and Decision Trees**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://credit-card-approval-preethika.streamlit.app)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-4169E1?logo=postgresql&logoColor=white)](https://postgresql.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![pytest](https://img.shields.io/badge/Tests-17%20passed-2ecc71?logo=pytest&logoColor=white)](https://pytest.org)

### [View Live Demo](https://credit-card-approval-preethika.streamlit.app)

</div>

---

## Live Demo

**[https://credit-card-approval-preethika.streamlit.app](https://credit-card-approval-preethika.streamlit.app)**

The interactive demo lets you build an applicant profile using readable dropdowns (gender, marital status, citizenship, employment, credit purpose, etc.) and get instant predictions from both models with approval probability, risk level, confidence score, and live charts.

To run locally instead:
```bash
pip install -r requirements.txt
python app/train.py          # train models (~60s)
streamlit run demo/app.py    # opens at localhost:8501
```

---

## Overview

Built on the UCI Credit Approval dataset (690 applicants, 15 anonymized features). The pipeline covers raw data ingestion, preprocessing, model training, a REST API, PostgreSQL storage, and Power BI dashboards.

```
crx_raw.csv  →  Preprocessing  →  SVM / Decision Tree  →  FastAPI  →  PostgreSQL  →  Power BI
                                          ↓
                                  Streamlit Demo (live)
```

---

## Model Performance

Trained with `GridSearchCV` (5-fold cross-validation, ROC-AUC scoring) on an 80/20 train-test split.

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|:--------:|:---------:|:------:|:--------:|:-------:|
| **SVM** (RBF, C=1) | **87.0%** | 89.1% | **80.3%** | **84.5%** | 0.947 |
| Decision Tree (entropy, depth=3) | 82.6% | **91.1%** | 67.2% | 77.4% | **0.954** |

<details>
<summary>Confusion matrices</summary>

**SVM**
```
              Predicted
              Denied  Approved
Actual Denied    71       6
     Approved    12      49
```

**Decision Tree**
```
              Predicted
              Denied  Approved
Actual Denied    73       4
     Approved    20      41
```
</details>

**Top features (Decision Tree)**

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | Prior default on record | 70.7% |
| 2 | Credit history score | 13.0% |
| 3 | Annual income | 8.3% |
| 4 | Debt ratio | 5.6% |
| 5 | Years employed | 1.5% |

---

## Feature Reference

The UCI dataset uses anonymized codes. The demo maps these to readable labels:

| Feature | Raw codes | Display label |
|---------|-----------|--------------| 
| Gender | `a / b` | Female / Male |
| Marital status | `u / y / l` | Married / Single / Other |
| Bank relationship | `g / p / gg` | Existing / New / Former customer |
| Citizenship | `g / p / s` | Citizen / Permanent resident / Temporary resident |
| Prior default | `t / f` | Yes / No |
| Currently employed | `t / f` | Yes / No |
| Driver's licence | `t / f` | Yes / No |
| Credit purpose | `c / cc / m / k...` | Consumer goods / Credit card / Mortgage / Car loan... |
| Applicant group | `v / h / bb...` | Group A / B / C... |

---

## Project Structure

```
credit-card-approval/
├── app/
│   ├── preprocessing.py     # Imputation, encoding, scaling pipeline
│   ├── train.py             # GridSearchCV training + 12 Seaborn plots
│   ├── main.py              # FastAPI service (/predict /batch /metrics /health)
│   └── db_loader.py         # PostgreSQL bulk loader
│
├── demo/
│   └── app.py               # Streamlit interactive demo (live at link above)
│
├── data/
│   └── crx_raw.csv          # UCI Credit Approval dataset (690 records)
│
├── models/
│   ├── decision_tree.pkl    # Trained Decision Tree
│   ├── svm.pkl              # Trained SVM
│   └── preprocessor.pkl     # Fitted preprocessing pipeline
│
├── reports/
│   ├── model_metrics.json   # Full evaluation metrics
│   ├── predictions.csv      # All predictions with probabilities
│   └── plots/               # 12 Seaborn EDA & evaluation charts
│
├── sql/
│   └── schema.sql           # PostgreSQL tables + 4 Power BI views
│
├── tests/
│   └── test_models.py       # 17 pytest unit tests (all passing)
│
├── powerbi/
│   └── POWERBI_GUIDE.md     # Connection setup + DAX measures
│
├── docker-compose.yml        # API + PostgreSQL + pgAdmin + Streamlit
├── Dockerfile
├── Dockerfile.demo
└── requirements.txt
```

---

## Quick Start

### Option A — Demo only (no Docker needed)

```bash
git clone https://github.com/preethikachennareddy/credit-card-approval.git
cd credit-card-approval

pip install -r requirements.txt
python app/train.py
streamlit run demo/app.py
```

### Option B — Full stack with Docker

```bash
docker-compose up --build
```

| Service | URL |
|---------|-----|
| Streamlit demo | http://localhost:8501 |
| FastAPI docs | http://localhost:8000/docs |
| pgAdmin | http://localhost:5050 |

Then load the database:
```bash
python app/db_loader.py
```

---

## API Reference

### `POST /predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35, "debt": 2.5, "years_employed": 8,
    "prior_default": "f", "employed": "t",
    "credit_score": 5, "drivers_license": "t",
    "zip_code": 200, "income": 45000,
    "gender": "b", "married": "u",
    "bank_customer": "g", "education_level": "c",
    "ethnicity": "v", "citizen": "g",
    "model": "both"
  }'
```

**Response**
```json
{
  "approved": true,
  "approval_probability": 0.8741,
  "risk_score": 0.1259,
  "model_used": "ensemble (DT + SVM)",
  "decision": "APPROVED",
  "confidence": "Very High",
  "timestamp": "2024-01-15T10:30:00"
}
```

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Single applicant prediction |
| `POST` | `/batch-predict` | Bulk predictions (up to 1000) |
| `GET`  | `/metrics` | Model evaluation metrics |
| `GET`  | `/health` | Service health check |
| `GET`  | `/model-info/{name}` | Model hyperparameters |

---

## Database

PostgreSQL schema includes:

- `applicants` — all 690 records with features
- `predictions` — model outputs with probabilities
- `model_metrics` — training run history
- `v_approval_by_age_group` — approval rates by age band (Power BI)
- `v_income_approval` — approval rates by income bracket (Power BI)
- `v_model_performance` — accuracy per model (Power BI)
- `v_approval_trends` — daily approval trend (Power BI)

---

## Visualisations Generated

Running `python app/train.py` produces 12 Seaborn plots in `reports/plots/`:

| File | Description |
|------|-------------|
| `01_approval_distribution.png` | Approved vs Denied bar chart |
| `02_age_distribution.png` | Age histograms by outcome |
| `03_correlation_heatmap.png` | Feature correlation matrix |
| `04_income_vs_debt.png` | Scatter coloured by outcome |
| `05_credit_score_boxplot.png` | Credit score by outcome |
| `dt_feature_importance.png` | Decision Tree importances |
| `dt_confusion_matrix.png` | DT confusion matrix heatmap |
| `dt_tree_viz.png` | Tree visualisation (depth 3) |
| `svm_confusion_matrix.png` | SVM confusion matrix |
| `roc_curves.png` | ROC curves — both models |
| `metrics_comparison.png` | Side-by-side metric bars |

---

## Tests

```bash
pytest tests/ -v
# 17 passed in 1.31s
```

Covers: preprocessor fit/transform, missing value imputation, label encoding, unseen category handling, model loading, prediction shapes, probability ranges, and AUC thresholds.

---

## Power BI

See [`powerbi/POWERBI_GUIDE.md`](powerbi/POWERBI_GUIDE.md) for full setup.

4 dashboard pages: Executive Overview · Demographics · Model Performance · Feature Impact
12 ready-to-paste DAX measures included.

---

## Tech Stack

| Category | Technology |
|----------|-----------| 
| Language | Python 3.11 |
| ML | scikit-learn (SVM, Decision Tree, GridSearchCV) |
| Visualisation | Seaborn, Matplotlib |
| Demo | Streamlit |
| API | FastAPI + Uvicorn |
| Database | PostgreSQL 15 + SQLAlchemy |
| Containers | Docker + Docker Compose |
| BI | Microsoft Power BI |
| Tests | pytest (17 unit tests) |

---

## Dataset

UCI Machine Learning Repository — [Credit Approval](https://archive.ics.uci.edu/dataset/27/credit+approval)

> NOTE: All attribute names and values have been changed to meaningless symbols to protect the confidentiality of the data.

690 instances · 15 features · Binary classification · 67 missing values (handled via imputation) · 44.5% approval rate

</div>
