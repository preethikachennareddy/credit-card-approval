"""
ML Training Pipeline — SVM & Decision Tree
Credit Card Approval Prediction
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json
import logging
from datetime import datetime

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.pipeline import Pipeline

from preprocessing import CreditApprovalPreprocessor, get_feature_importance_df

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = 'models'
REPORTS_DIR = 'reports'
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(f'{REPORTS_DIR}/plots', exist_ok=True)


# ─── Seaborn theme ────────────────────────────────────────────────────────────
sns.set_theme(style='darkgrid', palette='deep')
PALETTE = {'Approved': '#2ECC71', 'Denied': '#E74C3C'}
FIG_DPI = 150


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    metrics = {
        'model': model_name,
        'accuracy': round(accuracy_score(y_test, y_pred), 4),
        'precision': round(precision_score(y_test, y_pred), 4),
        'recall': round(recall_score(y_test, y_pred), 4),
        'f1_score': round(f1_score(y_test, y_pred), 4),
        'roc_auc': round(roc_auc_score(y_test, y_prob), 4) if y_prob is not None else None,
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'timestamp': datetime.now().isoformat()
    }

    logger.info(f"\n{'='*50}")
    logger.info(f"  {model_name} Evaluation")
    logger.info(f"{'='*50}")
    for k, v in metrics.items():
        if k not in ('confusion_matrix', 'classification_report', 'timestamp'):
            logger.info(f"  {k:15s}: {v}")

    return metrics


# ─── EDA Plots ────────────────────────────────────────────────────────────────

def plot_eda(df: pd.DataFrame):
    logger.info("Generating EDA plots...")

    # 1. Approval distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df['approved'].value_counts()
    labels = ['Denied', 'Approved']
    colors = ['#E74C3C', '#2ECC71']
    bars = ax.bar(labels, [counts.get(0, 0), counts.get(1, 0)], color=colors,
                  edgecolor='white', linewidth=1.5, width=0.5)
    for bar, cnt in zip(bars, [counts.get(0, 0), counts.get(1, 0)]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{cnt}\n({cnt/len(df)*100:.1f}%)', ha='center', va='bottom', fontweight='bold')
    ax.set_title('Credit Card Approval Distribution', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Count')
    ax.set_ylim(0, max(counts.values) * 1.15)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'{REPORTS_DIR}/plots/01_approval_distribution.png', dpi=FIG_DPI, bbox_inches='tight')
    plt.close()

    # 2. Age distribution by outcome
    fig, ax = plt.subplots(figsize=(9, 5))
    for outcome, label, color in [(0,'Denied','#E74C3C'), (1,'Approved','#2ECC71')]:
        subset = df[df['approved'] == outcome]['age']
        ax.hist(subset, bins=25, alpha=0.65, label=label, color=color, edgecolor='white')
    ax.set_title('Age Distribution by Approval Outcome', fontsize=14, fontweight='bold')
    ax.set_xlabel('Age'); ax.set_ylabel('Count')
    ax.legend()
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'{REPORTS_DIR}/plots/02_age_distribution.png', dpi=FIG_DPI, bbox_inches='tight')
    plt.close()

    # 3. Numeric features correlation heatmap
    num_cols = ['age', 'debt', 'years_employed', 'credit_score', 'income', 'approved']
    num_cols = [c for c in num_cols if c in df.columns]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, ax=ax, square=True, linewidths=0.5,
                cbar_kws={'shrink': 0.8})
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(f'{REPORTS_DIR}/plots/03_correlation_heatmap.png', dpi=FIG_DPI, bbox_inches='tight')
    plt.close()

    # 4. Income vs Debt scatter
    fig, ax = plt.subplots(figsize=(9, 6))
    for outcome, label, color in [(0,'Denied','#E74C3C'), (1,'Approved','#2ECC71')]:
        subset = df[df['approved'] == outcome]
        ax.scatter(subset['income'], subset['debt'], c=color, label=label,
                   alpha=0.5, s=30, edgecolors='none')
    ax.set_title('Income vs Debt by Approval Outcome', fontsize=14, fontweight='bold')
    ax.set_xlabel('Income ($)'); ax.set_ylabel('Debt Ratio')
    ax.legend()
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'{REPORTS_DIR}/plots/04_income_vs_debt.png', dpi=FIG_DPI, bbox_inches='tight')
    plt.close()

    # 5. Credit score boxplot
    fig, ax = plt.subplots(figsize=(8, 5))
    df_plot = df.copy()
    df_plot['Outcome'] = df_plot['approved'].map({0: 'Denied', 1: 'Approved'})
    sns.boxplot(data=df_plot, x='Outcome', y='credit_score',
                palette={'Approved': '#2ECC71', 'Denied': '#E74C3C'}, ax=ax,
                width=0.5, linewidth=1.5)
    ax.set_title('Credit Score Distribution by Approval Outcome', fontsize=14, fontweight='bold')
    ax.set_xlabel(''); ax.set_ylabel('Credit Score')
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'{REPORTS_DIR}/plots/05_credit_score_boxplot.png', dpi=FIG_DPI, bbox_inches='tight')
    plt.close()

    logger.info(f"EDA plots saved to {REPORTS_DIR}/plots/")


# ─── Model Evaluation Plots ────────────────────────────────────────────────────

def plot_confusion_matrix(cm, model_name: str, filename: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ['Denied', 'Approved']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels,
                linewidths=1, linecolor='white', cbar=False)
    ax.set_title(f'Confusion Matrix — {model_name}', fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(filename, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()


def plot_roc_curves(models_data: list):
    """models_data: list of (model, X_test, y_test, name)"""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#3498DB', '#E67E22', '#9B59B6', '#1ABC9C']
    ax.plot([0,1], [0,1], 'k--', lw=1.5, label='Random (AUC=0.50)')

    for i, (model, X_test, y_test, name) in enumerate(models_data):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2.5,
                label=f'{name} (AUC={auc:.3f})')

    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curves — Model Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'{REPORTS_DIR}/plots/roc_curves.png', dpi=FIG_DPI, bbox_inches='tight')
    plt.close()


def plot_feature_importance(fi_df: pd.DataFrame, model_name: str, filename: str):
    top_n = min(15, len(fi_df))
    fi_top = fi_df.head(top_n)
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = sns.color_palette('viridis', top_n)
    bars = ax.barh(fi_top['feature'][::-1], fi_top['importance'][::-1],
                   color=colors[::-1], edgecolor='white', linewidth=0.8)
    ax.set_title(f'Feature Importance — {model_name}', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Importance Score')
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.002, bar.get_y() + bar.get_height()/2,
                f'{w:.3f}', va='center', fontsize=9)
    sns.despine()
    plt.tight_layout()
    plt.savefig(filename, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()


def plot_metrics_comparison(all_metrics: list):
    metrics_of_interest = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    models = [m['model'] for m in all_metrics]
    df_m = pd.DataFrame([{k: m[k] for k in metrics_of_interest} for m in all_metrics])
    df_m['model'] = models
    df_melt = df_m.melt(id_vars='model', var_name='Metric', value_name='Score')

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=df_melt, x='Metric', y='Score', hue='model',
                palette='deep', ax=ax, edgecolor='white')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=12)
    ax.set_ylim(0.5, 1.05)
    ax.set_xlabel(''); ax.set_ylabel('Score')
    ax.legend(title='Model', bbox_to_anchor=(1.01, 1), loc='upper left')
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.3f}',
                    (p.get_x() + p.get_width()/2, p.get_height()),
                    ha='center', va='bottom', fontsize=8)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'{REPORTS_DIR}/plots/metrics_comparison.png', dpi=FIG_DPI, bbox_inches='tight')
    plt.close()


# ─── Training ─────────────────────────────────────────────────────────────────

def train_decision_tree(X_train, y_train, X_test, y_test, feature_names):
    logger.info("\nTraining Decision Tree with GridSearchCV...")
    param_grid = {
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }
    dt = DecisionTreeClassifier(random_state=42)
    grid = GridSearchCV(dt, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)

    best_dt = grid.best_estimator_
    logger.info(f"Best DT params: {grid.best_params_}")

    metrics = evaluate_model(best_dt, X_test, y_test, 'Decision Tree')

    # Feature importance
    fi_df = get_feature_importance_df(feature_names, best_dt.feature_importances_)
    plot_feature_importance(fi_df, 'Decision Tree',
                            f'{REPORTS_DIR}/plots/dt_feature_importance.png')

    # Confusion matrix
    cm = np.array(metrics['confusion_matrix'])
    plot_confusion_matrix(cm, 'Decision Tree',
                          f'{REPORTS_DIR}/plots/dt_confusion_matrix.png')

    # Tree visualization (depth 3 for readability)
    fig, ax = plt.subplots(figsize=(20, 8))
    plot_tree(best_dt, max_depth=3, feature_names=feature_names,
              class_names=['Denied', 'Approved'], filled=True,
              rounded=True, fontsize=9, ax=ax)
    plt.title('Decision Tree (depth=3 preview)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{REPORTS_DIR}/plots/dt_tree_viz.png', dpi=100, bbox_inches='tight')
    plt.close()

    joblib.dump(best_dt, f'{MODELS_DIR}/decision_tree.pkl')
    metrics['feature_importance'] = fi_df.to_dict('records')
    return best_dt, metrics


def train_svm(X_train, y_train, X_test, y_test):
    logger.info("\nTraining SVM with GridSearchCV...")
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto']
    }
    svm = SVC(probability=True, random_state=42)
    grid = GridSearchCV(svm, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)

    best_svm = grid.best_estimator_
    logger.info(f"Best SVM params: {grid.best_params_}")

    metrics = evaluate_model(best_svm, X_test, y_test, 'SVM')

    # For linear kernel, get coefficients as feature importance
    if best_svm.kernel == 'linear':
        importances = np.abs(best_svm.coef_[0])
        fi_df = get_feature_importance_df(
            [f'Feature_{i}' for i in range(len(importances))], importances)
        plot_feature_importance(fi_df, 'SVM (Linear Coefficients)',
                                f'{REPORTS_DIR}/plots/svm_feature_importance.png')

    cm = np.array(metrics['confusion_matrix'])
    plot_confusion_matrix(cm, 'SVM',
                          f'{REPORTS_DIR}/plots/svm_confusion_matrix.png')

    joblib.dump(best_svm, f'{MODELS_DIR}/svm.pkl')
    return best_svm, metrics


def main():
    # ── Load & preprocess ────────────────────────────────────────
    preprocessor = CreditApprovalPreprocessor()
    df_raw = preprocessor.load_data('data/crx_raw.csv')
    df = preprocessor.fit_transform(df_raw)
    preprocessor.save(f'{MODELS_DIR}/preprocessor.pkl')

    # ── EDA ──────────────────────────────────────────────────────
    plot_eda(df_raw.rename(columns={k: v for k, v in {
        'A1':'gender','A2':'age','A3':'debt','A4':'married','A5':'bank_customer',
        'A6':'education_level','A7':'ethnicity','A8':'years_employed','A9':'prior_default',
        'A10':'employed','A11':'credit_score','A12':'drivers_license','A13':'citizen',
        'A14':'zip_code','A15':'income','target':'approved'
    }.items() if k in df_raw.columns}))

    # ── Train/test split ─────────────────────────────────────────
    feature_cols = [c for c in df.columns if c != 'approved']
    X = df[feature_cols].values
    y = df['approved'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

    # ── Train models ─────────────────────────────────────────────
    dt_model, dt_metrics = train_decision_tree(
        X_train, y_train, X_test, y_test, feature_cols)

    svm_model, svm_metrics = train_svm(
        X_train, y_train, X_test, y_test)

    # ── Combined plots ───────────────────────────────────────────
    plot_roc_curves([
        (dt_model, X_test, y_test, 'Decision Tree'),
        (svm_model, X_test, y_test, 'SVM'),
    ])
    plot_metrics_comparison([dt_metrics, svm_metrics])

    # ── Save metrics ─────────────────────────────────────────────
    all_metrics = {'decision_tree': dt_metrics, 'svm': svm_metrics}
    with open(f'{REPORTS_DIR}/model_metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)

    # ── Export predictions for DB / PowerBI ──────────────────────
    df_pred = df_raw.copy()
    df_pred.rename(columns={k: v for k, v in {
        'A2':'age','A3':'debt','A8':'years_employed','A9':'prior_default',
        'A10':'employed','A11':'credit_score','A15':'income','target':'approved'
    }.items() if k in df_pred.columns}, inplace=True)

    X_all = df[feature_cols].values
    df_pred['dt_prediction'] = dt_model.predict(X_all)
    df_pred['dt_probability'] = dt_model.predict_proba(X_all)[:, 1].round(4)
    df_pred['svm_prediction'] = svm_model.predict(X_all)
    df_pred['svm_probability'] = svm_model.predict_proba(X_all)[:, 1].round(4)
    df_pred['record_id'] = range(1, len(df_pred) + 1)
    df_pred.to_csv(f'{REPORTS_DIR}/predictions.csv', index=False)

    logger.info(f"\n{'='*55}")
    logger.info("  TRAINING COMPLETE")
    logger.info(f"{'='*55}")
    logger.info(f"  Decision Tree  — Accuracy: {dt_metrics['accuracy']:.4f} | AUC: {dt_metrics['roc_auc']:.4f}")
    logger.info(f"  SVM            — Accuracy: {svm_metrics['accuracy']:.4f} | AUC: {svm_metrics['roc_auc']:.4f}")
    logger.info(f"\n  Models   → {MODELS_DIR}/")
    logger.info(f"  Plots    → {REPORTS_DIR}/plots/")
    logger.info(f"  Metrics  → {REPORTS_DIR}/model_metrics.json")
    logger.info(f"  Preds    → {REPORTS_DIR}/predictions.csv")

    return dt_model, svm_model, dt_metrics, svm_metrics


if __name__ == '__main__':
    main()
