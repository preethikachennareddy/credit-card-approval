-- ═══════════════════════════════════════════════════════════════════
-- Credit Card Approval — PostgreSQL Schema
-- ═══════════════════════════════════════════════════════════════════

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ─── Applicants ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS applicants (
    id              SERIAL PRIMARY KEY,
    record_id       INT UNIQUE NOT NULL,
    gender          CHAR(1),
    age             DECIMAL(5,2),
    debt            DECIMAL(10,4),
    married         VARCHAR(4),
    bank_customer   VARCHAR(4),
    education_level VARCHAR(4),
    ethnicity       VARCHAR(4),
    years_employed  DECIMAL(6,2),
    prior_default   SMALLINT,
    employed        SMALLINT,
    credit_score    DECIMAL(8,2),
    drivers_license SMALLINT,
    citizen         CHAR(1),
    zip_code        INT,
    income          DECIMAL(12,2),
    approved        SMALLINT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ─── Model Predictions ───────────────────────────────────────────
CREATE TABLE IF NOT EXISTS predictions (
    id              SERIAL PRIMARY KEY,
    record_id       INT REFERENCES applicants(record_id),
    model_name      VARCHAR(50) NOT NULL,
    predicted       SMALLINT NOT NULL,
    probability     DECIMAL(6,4),
    risk_score      DECIMAL(6,4),
    is_correct      BOOLEAN GENERATED ALWAYS AS (predicted = (SELECT approved FROM applicants a WHERE a.record_id = record_id)) STORED,
    predicted_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ─── Model Metrics ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS model_metrics (
    id              SERIAL PRIMARY KEY,
    model_name      VARCHAR(50) NOT NULL,
    run_date        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    accuracy        DECIMAL(6,4),
    precision_score DECIMAL(6,4),
    recall          DECIMAL(6,4),
    f1_score        DECIMAL(6,4),
    roc_auc         DECIMAL(6,4),
    train_samples   INT,
    test_samples    INT,
    hyperparameters JSONB
);

-- ─── Indexes ─────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_applicants_approved  ON applicants(approved);
CREATE INDEX IF NOT EXISTS idx_applicants_age       ON applicants(age);
CREATE INDEX IF NOT EXISTS idx_applicants_income    ON applicants(income);
CREATE INDEX IF NOT EXISTS idx_predictions_model    ON predictions(model_name);
CREATE INDEX IF NOT EXISTS idx_predictions_record   ON predictions(record_id);

-- ─── Views for PowerBI ───────────────────────────────────────────

-- Approval summary by age group
CREATE OR REPLACE VIEW v_approval_by_age_group AS
SELECT
    CASE
        WHEN age < 25 THEN '18-24'
        WHEN age < 35 THEN '25-34'
        WHEN age < 45 THEN '35-44'
        WHEN age < 55 THEN '45-54'
        ELSE '55+'
    END AS age_group,
    COUNT(*)                                    AS total,
    SUM(approved)                               AS approved_count,
    COUNT(*) - SUM(approved)                    AS denied_count,
    ROUND(AVG(approved::DECIMAL) * 100, 1)      AS approval_rate_pct,
    ROUND(AVG(income), 2)                       AS avg_income,
    ROUND(AVG(credit_score), 2)                 AS avg_credit_score
FROM applicants
GROUP BY age_group
ORDER BY age_group;

-- Model performance comparison
CREATE OR REPLACE VIEW v_model_performance AS
SELECT
    p.model_name,
    COUNT(*)                                         AS total_predictions,
    SUM(p.predicted)                                 AS predicted_approved,
    SUM(CASE WHEN p.is_correct THEN 1 ELSE 0 END)   AS correct_predictions,
    ROUND(AVG(CASE WHEN p.is_correct THEN 1.0 ELSE 0 END) * 100, 2) AS accuracy_pct,
    ROUND(AVG(p.probability), 4)                     AS avg_confidence
FROM predictions p
GROUP BY p.model_name;

-- Income vs approval analysis
CREATE OR REPLACE VIEW v_income_approval AS
SELECT
    CASE
        WHEN income < 20000  THEN '<$20k'
        WHEN income < 50000  THEN '$20k-$50k'
        WHEN income < 80000  THEN '$50k-$80k'
        WHEN income < 120000 THEN '$80k-$120k'
        ELSE '$120k+'
    END AS income_bracket,
    COUNT(*)                                AS total,
    SUM(approved)                           AS approved,
    ROUND(AVG(approved::DECIMAL)*100, 1)    AS approval_rate_pct,
    ROUND(AVG(debt), 2)                     AS avg_debt,
    ROUND(AVG(credit_score), 1)             AS avg_credit_score
FROM applicants
GROUP BY income_bracket
ORDER BY MIN(income);

-- Daily approval trends (simulated dates)
CREATE OR REPLACE VIEW v_approval_trends AS
SELECT
    DATE(created_at)                        AS date,
    COUNT(*)                                AS applications,
    SUM(approved)                           AS approved,
    ROUND(AVG(approved::DECIMAL)*100, 1)    AS approval_rate_pct
FROM applicants
GROUP BY DATE(created_at)
ORDER BY date;
