"""
Data Preprocessing Pipeline for Credit Card Approval Model
UCI Credit Approval Dataset (crx.data) — real format with '?' missing values
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COLUMNS = ['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13','A14','A15','target']

FEATURE_NAMES = {
    'A1': 'gender', 'A2': 'age', 'A3': 'debt', 'A4': 'married',
    'A5': 'bank_customer', 'A6': 'education_level', 'A7': 'ethnicity',
    'A8': 'years_employed', 'A9': 'prior_default', 'A10': 'employed',
    'A11': 'credit_score', 'A12': 'drivers_license', 'A13': 'citizen',
    'A14': 'zip_code', 'A15': 'income', 'target': 'approved'
}

CATEGORICAL_COLS = ['gender', 'married', 'bank_customer', 'education_level',
                    'ethnicity', 'prior_default', 'employed', 'drivers_license', 'citizen']
NUMERIC_COLS = ['age', 'debt', 'years_employed', 'credit_score', 'zip_code', 'income']


class CreditApprovalPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.cat_imputer = SimpleImputer(strategy='most_frequent')
        self.num_imputer = SimpleImputer(strategy='median')
        self.feature_columns = None
        self.is_fitted = False

    def load_data(self, filepath: str) -> pd.DataFrame:
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath, header=None, names=COLUMNS, na_values='?')
        df.rename(columns=FEATURE_NAMES, inplace=True)
        # Encode target: + -> 1, - -> 0
        df['approved'] = (df['approved'] == '+').astype(int)
        logger.info(f"Loaded {len(df)} rows | Approval rate: {df['approved'].mean():.1%}")
        logger.info(f"Missing values:\n{df.isnull().sum()[df.isnull().sum()>0]}")
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Fitting and transforming data...")
        df = df.copy()

        cat_present = [c for c in CATEGORICAL_COLS if c in df.columns]
        num_present = [c for c in NUMERIC_COLS if c in df.columns]

        if cat_present:
            df[cat_present] = self.cat_imputer.fit_transform(df[cat_present].astype(str).replace('nan', np.nan))
        if num_present:
            df[num_present] = self.num_imputer.fit_transform(df[num_present])

        for col in cat_present:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le

        if num_present:
            df[num_present] = self.scaler.fit_transform(df[num_present])

        self.feature_columns = [c for c in df.columns if c != 'approved']
        self.is_fitted = True
        logger.info(f"Preprocessing complete. Features: {len(self.feature_columns)}")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit_transform first.")
        df = df.copy()

        cat_present = [c for c in CATEGORICAL_COLS if c in df.columns]
        num_present = [c for c in NUMERIC_COLS if c in df.columns]

        if cat_present:
            df[cat_present] = self.cat_imputer.transform(df[cat_present].astype(str).replace('nan', np.nan))
        if num_present:
            df[num_present] = self.num_imputer.transform(df[num_present])

        for col in cat_present:
            if col in self.label_encoders:
                le = self.label_encoders[col]
                df[col] = df[col].apply(
                    lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1
                )

        if num_present:
            df[num_present] = self.scaler.transform(df[num_present])

        return df

    def save(self, path: str = 'models/preprocessor.pkl'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Preprocessor saved to {path}")

    @classmethod
    def load(cls, path: str = 'models/preprocessor.pkl'):
        return joblib.load(path)


def get_feature_importance_df(feature_names, importances):
    return pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).reset_index(drop=True)
