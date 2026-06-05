"""
Database Loader — Inserts applicants + predictions into PostgreSQL
Run after training: python app/db_loader.py
"""

import pandas as pd
import json
import os
import logging
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_URL = os.getenv(
    'DATABASE_URL',
    'postgresql://credituser:creditpass@localhost:5432/creditdb'
)


def get_engine():
    return create_engine(DB_URL)


def load_applicants(engine, predictions_csv: str = 'reports/predictions.csv'):
    df = pd.read_csv(predictions_csv)

    # Map original column names to DB schema
    col_map = {
        'A1': 'gender', 'A2': 'age', 'A3': 'debt', 'A4': 'married',
        'A5': 'bank_customer', 'A6': 'education_level', 'A7': 'ethnicity',
        'A8': 'years_employed', 'A9': 'prior_default', 'A10': 'employed',
        'A11': 'credit_score', 'A12': 'drivers_license', 'A13': 'citizen',
        'A14': 'zip_code', 'A15': 'income', 'target': 'approved'
    }
    # Handle already-renamed columns
    for old, new in col_map.items():
        if old in df.columns:
            df.rename(columns={old: new}, inplace=True)

    applicant_cols = [
        'record_id', 'gender', 'age', 'debt', 'married', 'bank_customer',
        'education_level', 'ethnicity', 'years_employed', 'prior_default',
        'employed', 'credit_score', 'drivers_license', 'citizen',
        'zip_code', 'income', 'approved'
    ]
    existing = [c for c in applicant_cols if c in df.columns]
    df_applicants = df[existing].copy()

    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE predictions, applicants RESTART IDENTITY CASCADE"))
        conn.commit()

    df_applicants.to_sql('applicants', engine, if_exists='append', index=False)
    logger.info(f"Inserted {len(df_applicants)} applicants")

    return df


def load_predictions(engine, df: pd.DataFrame):
    records = []
    for _, row in df.iterrows():
        for model_name, prob_col, pred_col in [
            ('decision_tree', 'dt_probability', 'dt_prediction'),
            ('svm', 'svm_probability', 'svm_prediction'),
        ]:
            if prob_col in df.columns:
                records.append({
                    'record_id': int(row['record_id']),
                    'model_name': model_name,
                    'predicted': int(row[pred_col]),
                    'probability': float(row[prob_col]),
                    'risk_score': round(1 - float(row[prob_col]), 4)
                })

    df_pred = pd.DataFrame(records)
    df_pred.to_sql('predictions', engine, if_exists='append', index=False)
    logger.info(f"Inserted {len(df_pred)} prediction records")


def load_metrics(engine, metrics_path: str = 'reports/model_metrics.json'):
    with open(metrics_path) as f:
        metrics = json.load(f)

    records = []
    for model_name, m in metrics.items():
        records.append({
            'model_name': model_name,
            'accuracy': m.get('accuracy'),
            'precision_score': m.get('precision'),
            'recall': m.get('recall'),
            'f1_score': m.get('f1_score'),
            'roc_auc': m.get('roc_auc'),
            'train_samples': 552,  # 80% of 690
            'test_samples': 138,
            'hyperparameters': json.dumps({})
        })

    df_metrics = pd.DataFrame(records)
    df_metrics.to_sql('model_metrics', engine, if_exists='append', index=False)
    logger.info(f"Inserted {len(df_metrics)} metric records")


def main():
    logger.info("Connecting to PostgreSQL...")
    engine = get_engine()

    logger.info("Loading applicants...")
    df = load_applicants(engine)

    logger.info("Loading predictions...")
    load_predictions(engine, df)

    logger.info("Loading model metrics...")
    load_metrics(engine)

    logger.info("✓ Database load complete!")

    # Quick check
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM applicants")).fetchone()
        logger.info(f"  Applicants: {result[0]}")
        result = conn.execute(text("SELECT COUNT(*) FROM predictions")).fetchone()
        logger.info(f"  Predictions: {result[0]}")


if __name__ == '__main__':
    main()
