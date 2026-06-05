"""
Unit Tests — Credit Card Approval Model
Run: pytest tests/ -v
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))
import joblib

from preprocessing import CreditApprovalPreprocessor, get_feature_importance_df

# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    np.random.seed(0)
    n = 100
    return pd.DataFrame({
        'gender': np.random.choice(['a','b',None], n, p=[0.49,0.49,0.02]),
        'age': np.random.uniform(18, 70, n),
        'debt': np.random.uniform(0, 30, n),
        'married': np.random.choice(['u','y',None], n, p=[0.49,0.49,0.02]),
        'bank_customer': np.random.choice(['g','p',None], n, p=[0.49,0.49,0.02]),
        'education_level': np.random.choice(['c','d','cc'], n),
        'ethnicity': np.random.choice(['v','h','bb'], n),
        'years_employed': np.random.uniform(0, 25, n),
        'prior_default': np.random.randint(0, 5, n),
        'employed': np.random.randint(0, 2, n),
        'credit_score': np.random.uniform(300, 850, n),
        'drivers_license': np.random.randint(0, 2, n),
        'citizen': np.random.choice(['g','p','s'], n),
        'zip_code': np.random.randint(100, 9999, n),
        'income': np.random.uniform(0, 100000, n),
        'approved': np.random.randint(0, 2, n),
    })


@pytest.fixture
def preprocessor(sample_df):
    pp = CreditApprovalPreprocessor()
    pp.fit_transform(sample_df.drop(columns=['approved']))
    return pp


@pytest.fixture
def trained_models():
    models = {}
    for name in ['decision_tree', 'svm']:
        path = f'models/{name}.pkl'
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models


# ─── Preprocessing Tests ──────────────────────────────────────────────────────

class TestPreprocessor:
    def test_fit_transform_no_nulls(self, sample_df):
        pp = CreditApprovalPreprocessor()
        df = sample_df.drop(columns=['approved'])
        result = pp.fit_transform(df)
        assert result.isnull().sum().sum() == 0, "Nulls remain after fit_transform"

    def test_fit_transform_shape(self, sample_df):
        pp = CreditApprovalPreprocessor()
        df = sample_df.drop(columns=['approved'])
        result = pp.fit_transform(df)
        assert result.shape[0] == len(df), "Row count changed"
        assert result.shape[1] == df.shape[1], "Column count changed"

    def test_transform_unseen_category(self, sample_df, preprocessor):
        new = sample_df.iloc[:5].copy().drop(columns=['approved'])
        new['gender'] = 'z'  # unseen
        result = preprocessor.transform(new)
        assert result is not None

    def test_feature_columns_set(self, sample_df):
        pp = CreditApprovalPreprocessor()
        df = sample_df.drop(columns=['approved'])
        pp.fit_transform(df)
        assert pp.feature_columns is not None
        assert len(pp.feature_columns) == df.shape[1]

    def test_is_fitted_flag(self, sample_df):
        pp = CreditApprovalPreprocessor()
        assert not pp.is_fitted
        pp.fit_transform(sample_df.drop(columns=['approved']))
        assert pp.is_fitted

    def test_transform_before_fit_raises(self, sample_df):
        pp = CreditApprovalPreprocessor()
        with pytest.raises(RuntimeError):
            pp.transform(sample_df.drop(columns=['approved']))

    def test_numeric_scaling(self, sample_df):
        pp = CreditApprovalPreprocessor()
        df = sample_df.drop(columns=['approved'])
        result = pp.fit_transform(df)
        # Scaled numerics should have mean ~0, std ~1
        assert abs(result['age'].mean()) < 0.5

    def test_label_encoders_populated(self, sample_df):
        pp = CreditApprovalPreprocessor()
        pp.fit_transform(sample_df.drop(columns=['approved']))
        assert len(pp.label_encoders) > 0


# ─── Feature Importance Tests ─────────────────────────────────────────────────

class TestFeatureImportance:
    def test_returns_sorted_df(self):
        features = ['a', 'b', 'c']
        importances = [0.1, 0.5, 0.4]
        fi = get_feature_importance_df(features, importances)
        assert fi['importance'].is_monotonic_decreasing

    def test_correct_length(self):
        features = ['x', 'y']
        importances = [0.6, 0.4]
        fi = get_feature_importance_df(features, importances)
        assert len(fi) == 2

    def test_columns(self):
        fi = get_feature_importance_df(['f1'], [1.0])
        assert 'feature' in fi.columns
        assert 'importance' in fi.columns


# ─── Model Tests ──────────────────────────────────────────────────────────────

class TestModels:
    @pytest.mark.skipif(not os.path.exists('models/decision_tree.pkl'),
                        reason="Model not trained yet")
    def test_decision_tree_loaded(self, trained_models):
        assert 'decision_tree' in trained_models

    @pytest.mark.skipif(not os.path.exists('models/svm.pkl'),
                        reason="Model not trained yet")
    def test_svm_loaded(self, trained_models):
        assert 'svm' in trained_models

    @pytest.mark.skipif(not os.path.exists('models/decision_tree.pkl'),
                        reason="Model not trained yet")
    def test_decision_tree_predict_shape(self, trained_models, sample_df):
        pp = joblib.load('models/preprocessor.pkl')
        df = sample_df.drop(columns=['approved'])
        df_proc = pp.transform(df)
        X = df_proc[pp.feature_columns].values
        preds = trained_models['decision_tree'].predict(X)
        assert len(preds) == len(df)

    @pytest.mark.skipif(not os.path.exists('models/svm.pkl'),
                        reason="Model not trained yet")
    def test_svm_predict_proba_range(self, trained_models, sample_df):
        pp = joblib.load('models/preprocessor.pkl')
        df = sample_df.drop(columns=['approved'])
        df_proc = pp.transform(df)
        X = df_proc[pp.feature_columns].values
        probs = trained_models['svm'].predict_proba(X)[:, 1]
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0

    @pytest.mark.skipif(not os.path.exists('models/decision_tree.pkl'),
                        reason="Model not trained yet")
    def test_dt_accuracy_reasonable(self, trained_models):
        """Decision Tree AUC should be > 0.70"""
        import json
        with open('reports/model_metrics.json') as f:
            metrics = json.load(f)
        assert metrics['decision_tree']['roc_auc'] > 0.70

    @pytest.mark.skipif(not os.path.exists('models/svm.pkl'),
                        reason="Model not trained yet")
    def test_svm_accuracy_reasonable(self, trained_models):
        """SVM AUC should be > 0.80"""
        import json
        with open('reports/model_metrics.json') as f:
            metrics = json.load(f)
        assert metrics['svm']['roc_auc'] > 0.80
