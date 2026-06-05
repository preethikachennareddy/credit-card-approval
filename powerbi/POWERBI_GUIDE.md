# Power BI Dashboard Guide
## Credit Card Approval ML Model: Visual Analytics

---

## 1. Data Connection Setup

### Connect to PostgreSQL
1. Open Power BI Desktop → **Get Data** → **PostgreSQL database**
2. Enter connection details:
   - Server: `localhost` (or Docker host IP)
   - Database: `creditdb`
   - Username: `credituser` / Password: `creditpass`
3. Import these tables/views:
   - `applicants`
   - `predictions`
   - `model_metrics`
   - `v_approval_by_age_group`
   - `v_income_approval`
   - `v_model_performance`

> **Alternatively**: Use `reports/predictions.csv` via **Get Data → Text/CSV**
> for a quick no-database setup.

---

## 2. Recommended Pages

### Page 1: Executive Overview
| Visual | Type | Fields |
|--------|------|--------|
| Approval Rate KPI | Card | `AVG(approved)` |
| Total Applications | Card | `COUNT(record_id)` |
| Approval Donut | Pie/Donut | `approved` (0/1), Count |
| Avg Income by Outcome | Bar | `approved`, `AVG(income)` |
| Avg Credit Score | Gauge | `AVG(credit_score)` |

### Page 2: Applicant Demographics
| Visual | Type | Fields |
|--------|------|--------|
| Approval by Age Group | Stacked Bar | Age group, Count, approved |
| Income Bracket Funnel | Funnel | income_bracket, approval_rate_pct |
| Debt vs Income Scatter | Scatter | income (X), debt (Y), approved (color) |
| Employment Status | Stacked Column | employed, approved |

### Page 3: Model Performance
| Visual | Type | Fields |
|--------|------|--------|
| Accuracy by Model | Clustered Bar | model_name, accuracy_pct |
| AUC Score Card | Multi-row Card | model_name, roc_auc |
| Confusion Matrix | Matrix | predicted, actual, count |
| Probability Distribution | Histogram | probability (bins: 10) |
| Model Comparison | Radar Chart | accuracy, precision, recall, f1, roc_auc |

### Page 4: Feature Impact
| Visual | Type | Fields |
|--------|------|--------|
| Feature Importance Bar | Bar | Feature, Importance (DT) |
| Approval by Credit Score | Line/Area | credit_score bins, approval rate |
| Income Impact | Waterfall | income brackets, delta approval |

---

## 3. DAX Measures

Paste these into the Power BI DAX editor:

```dax
// ─── Core KPIs ──────────────────────────────────────────────────

Approval Rate = 
DIVIDE(
    CALCULATE(COUNTROWS(applicants), applicants[approved] = 1),
    COUNTROWS(applicants),
    0
)

Denial Rate = 1 - [Approval Rate]

Total Applications = COUNTROWS(applicants)

Approved Count = 
CALCULATE(COUNTROWS(applicants), applicants[approved] = 1)

Denied Count = 
CALCULATE(COUNTROWS(applicants), applicants[approved] = 0)


// ─── Model Accuracy ──────────────────────────────────────────────

DT Accuracy = 
CALCULATE(
    DIVIDE(
        CALCULATE(COUNTROWS(predictions), 
            predictions[model_name] = "decision_tree",
            predictions[predicted] = RELATED(applicants[approved])
        ),
        CALCULATE(COUNTROWS(predictions), 
            predictions[model_name] = "decision_tree"
        )
    )
)

SVM Accuracy = 
CALCULATE(
    DIVIDE(
        CALCULATE(COUNTROWS(predictions), 
            predictions[model_name] = "svm",
            predictions[predicted] = RELATED(applicants[approved])
        ),
        CALCULATE(COUNTROWS(predictions), 
            predictions[model_name] = "svm"
        )
    )
)

Avg Approval Probability =
CALCULATE(
    AVERAGE(predictions[probability]),
    predictions[predicted] = 1
)

High Confidence Approvals =
CALCULATE(
    COUNTROWS(predictions),
    predictions[probability] >= 0.80,
    predictions[predicted] = 1
)


// ─── Risk Segmentation ────────────────────────────────────────────

Risk Category = 
SWITCH(
    TRUE(),
    predictions[probability] >= 0.80, "Low Risk",
    predictions[probability] >= 0.60, "Moderate Risk",
    predictions[probability] >= 0.40, "High Risk",
    "Very High Risk"
)

Avg Income Approved = 
CALCULATE(AVERAGE(applicants[income]), applicants[approved] = 1)

Avg Income Denied = 
CALCULATE(AVERAGE(applicants[income]), applicants[approved] = 0)

Income Lift = [Avg Income Approved] - [Avg Income Denied]


// ─── Age Group Segmentation ───────────────────────────────────────

Age Group = 
SWITCH(
    TRUE(),
    applicants[age] < 25, "18-24",
    applicants[age] < 35, "25-34",
    applicants[age] < 45, "35-44",
    applicants[age] < 55, "45-54",
    "55+"
)

Approval Rate by Age =
DIVIDE(
    CALCULATE(COUNTROWS(applicants), applicants[approved] = 1),
    COUNTROWS(applicants)
)
```

---

## 4. Slicers to Add

- **Model Selector**: `predictions[model_name]` (Decision Tree / SVM)
- **Outcome Filter**: `applicants[approved]` (Approved / Denied)
- **Age Range**: Numeric slider on `applicants[age]`
- **Income Range**: Numeric slider on `applicants[income]`
- **Employment Status**: `applicants[employed]`

---

## 5. Color Theme

Use these hex values for consistent branding:

| Element | Color | Hex |
|---------|-------|-----|
| Approved | Green | `#2ECC71` |
| Denied | Red | `#E74C3C` |
| Decision Tree | Blue | `#3498DB` |
| SVM | Orange | `#E67E22` |
| Neutral/BG | Dark | `#2C3E50` |
| KPI Text | White | `#ECF0F1` |

---

## 6. Refresh Setup

For live data, configure **DirectQuery** to PostgreSQL or set up
**Scheduled Refresh** (Power BI Service) pointing to your Docker host.

Recommended refresh: Every 24 hours after retraining.
