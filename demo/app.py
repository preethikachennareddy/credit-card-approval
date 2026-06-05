"""
Credit Card Approval — Streamlit Demo
Trains models on first run if not already present.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'app'))

st.set_page_config(
    page_title="Credit Approval ML — Preethika Chennareddy",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0f1117;
        border-right: 1px solid #1e2130;
    }
    section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
    section[data-testid="stSidebar"] .stRadio label { 
        padding: 6px 10px; border-radius: 6px; display: block;
    }

    /* Page background */
    .main { background: #f7f8fa; }

    /* Header bar */
    .page-header {
        background: linear-gradient(135deg, #1a1f36 0%, #2d3561 100%);
        border-radius: 12px;
        padding: 28px 36px;
        margin-bottom: 28px;
        color: white;
    }
    .page-header h1 { color: white; font-size: 1.8rem; font-weight: 600; margin: 0 0 4px; }
    .page-header p  { color: #a0aec0; margin: 0; font-size: 0.95rem; }

    /* Cards */
    .card {
        background: white;
        border-radius: 10px;
        padding: 24px;
        border: 1px solid #e8ecf0;
        margin-bottom: 16px;
    }
    .card-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #8892a0;
        margin-bottom: 4px;
    }
    .card-value {
        font-size: 2rem;
        font-weight: 600;
        color: #1a1f36;
        line-height: 1;
    }
    .card-sub { font-size: 0.8rem; color: #8892a0; margin-top: 4px; }

    /* Decision banners */
    .decision-approved {
        background: #f0fdf4;
        border: 1.5px solid #22c55e;
        border-radius: 10px;
        padding: 18px 24px;
        text-align: center;
        margin-bottom: 16px;
    }
    .decision-approved .label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; color: #16a34a; font-weight: 500; }
    .decision-approved .value { font-size: 1.6rem; font-weight: 700; color: #15803d; margin-top: 2px; }

    .decision-denied {
        background: #fff5f5;
        border: 1.5px solid #ef4444;
        border-radius: 10px;
        padding: 18px 24px;
        text-align: center;
        margin-bottom: 16px;
    }
    .decision-denied .label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; color: #dc2626; font-weight: 500; }
    .decision-denied .value { font-size: 1.6rem; font-weight: 700; color: #b91c1c; margin-top: 2px; }

    /* Metric row */
    .metric-row { display: flex; gap: 12px; margin-bottom: 16px; }
    .metric-box {
        flex: 1;
        background: #f8fafc;
        border: 1px solid #e8ecf0;
        border-radius: 8px;
        padding: 14px 16px;
    }
    .metric-box .m-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em; color: #8892a0; }
    .metric-box .m-value { font-size: 1.2rem; font-weight: 600; color: #1a1f36; margin-top: 2px; }

    /* Section divider */
    .section-title {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #8892a0;
        font-weight: 500;
        margin: 20px 0 10px;
        border-bottom: 1px solid #e8ecf0;
        padding-bottom: 6px;
    }

    /* Footer */
    .footer {
        margin-top: 48px;
        padding: 20px 0;
        border-top: 1px solid #e8ecf0;
        text-align: center;
        color: #8892a0;
        font-size: 0.82rem;
    }
    .footer a { color: #3b5bdb; text-decoration: none; }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Auto-train if models missing ──────────────────────────────────────────────
models_path = os.path.join(ROOT, 'models')
preprocessor_path = os.path.join(models_path, 'preprocessor.pkl')

if not os.path.exists(preprocessor_path):
    with st.spinner("First run — training models, please wait (~60s)..."):
        os.makedirs(models_path, exist_ok=True)
        os.makedirs(os.path.join(ROOT, 'reports', 'plots'), exist_ok=True)
        import subprocess
        result = subprocess.run(
            [sys.executable, os.path.join(ROOT, 'app', 'train.py')],
            cwd=ROOT, capture_output=True, text=True
        )
        if result.returncode != 0:
            st.error(f"Training failed:\n{result.stderr}")
            st.stop()
    st.rerun()

# ── Load artefacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    pp  = joblib.load(os.path.join(ROOT, 'models/preprocessor.pkl'))
    dt  = joblib.load(os.path.join(ROOT, 'models/decision_tree.pkl'))
    svm = joblib.load(os.path.join(ROOT, 'models/svm.pkl'))
    return pp, dt, svm

@st.cache_data
def load_data():
    from preprocessing import CreditApprovalPreprocessor
    pp = CreditApprovalPreprocessor()
    return pp.load_data(os.path.join(ROOT, 'data/crx_raw.csv'))

@st.cache_data
def load_metrics():
    with open(os.path.join(ROOT, 'reports/model_metrics.json')) as f:
        return json.load(f)

@st.cache_data
def load_predictions():
    return pd.read_csv(os.path.join(ROOT, 'reports/predictions.csv'))

try:
    preprocessor, dt_model, svm_model = load_models()
    df_raw   = load_data()
    metrics  = load_metrics()
    df_preds = load_predictions()
except Exception as e:
    st.error(f"Could not load: {e}")
    st.stop()

FEATURE_COLS = preprocessor.feature_columns

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## Credit Approval ML")
st.sidebar.markdown("UCI Credit Approval Dataset")
st.sidebar.markdown("---")

page = st.sidebar.radio("", [
    "Live Predictor", "Model Performance", "Data Explorer", "About"
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset**")
st.sidebar.markdown("690 applicants · 15 features")
st.sidebar.markdown(f"Approval rate: 44.5%")
st.sidebar.markdown("**Models**")
st.sidebar.markdown(f"SVM · AUC {metrics['svm']['roc_auc']:.3f}")
st.sidebar.markdown(f"Decision Tree · AUC {metrics['decision_tree']['roc_auc']:.3f}")
st.sidebar.markdown("---")
st.sidebar.markdown("Preethika Chennareddy")
st.sidebar.markdown("[GitHub](https://github.com/preethikachennareddy/credit-card-approval)")

# ── Shared footer ─────────────────────────────────────────────────────────────
def render_footer():
    st.markdown("""
    <div class="footer">
        Built by <strong>Preethika Chennareddy</strong> &nbsp;|&nbsp;
        UCI Credit Approval Dataset &nbsp;|&nbsp;
        <a href="https://github.com/preethikachennareddy/credit-card-approval" target="_blank">View on GitHub</a>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — LIVE PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Live Predictor":
    st.markdown("""
    <div class="page-header">
        <h1>Credit Card Approval Predictor</h1>
        <p>Adjust the applicant profile and get an instant prediction from trained ML models.</p>
    </div>
    """, unsafe_allow_html=True)

    col_form, col_result = st.columns([1.15, 0.85], gap="large")

    # Lookup maps — display label -> model code
    GENDER_MAP      = {"Male": "b", "Female": "a"}
    MARRIED_MAP     = {"Married": "u", "Single": "y", "Other": "l"}
    BANK_MAP        = {"Existing customer": "g", "New customer": "p", "Former customer": "gg"}
    CITIZEN_MAP     = {"Citizen": "g", "Permanent resident": "p", "Temporary resident": "s"}
    YESNO_TF_MAP    = {"Yes": "t", "No": "f"}
    YESNO_FT_MAP    = {"Yes": "t", "No": "f"}
    EDUCATION_MAP   = {
        "Consumer goods": "c", "Retail / trade": "r", "Credit card": "cc",
        "Investment": "i", "Student loan": "j", "Car loan": "k",
        "Mortgage": "m", "Other debt": "d", "Personal loan": "q",
        "Business": "w", "Mixed credit": "x", "Education loan": "e",
        "Home equity": "aa", "None / unknown": "ff"
    }
    ETHNICITY_MAP   = {
        "Group A": "v", "Group B": "h", "Group C": "bb", "Group D": "j",
        "Group E": "n", "Group F": "z", "Group G": "dd", "Group H": "ff", "Group I": "o"
    }

    with col_form:
        st.markdown('<div class="section-title">Personal information</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            gender_label  = st.selectbox("Gender", list(GENDER_MAP.keys()))
            age           = st.slider("Age", 15, 80, 30)
            married_label = st.selectbox("Marital status", list(MARRIED_MAP.keys()))
        with c2:
            bank_label    = st.selectbox("Bank relationship", list(BANK_MAP.keys()))
            citizen_label = st.selectbox("Citizenship status", list(CITIZEN_MAP.keys()))
            driv_label    = st.selectbox("Driver's licence", ["No", "Yes"])

        st.markdown('<div class="section-title">Financial profile</div>', unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            income        = st.number_input("Annual income ($)", 0, 200000, 35000, step=1000)
            debt          = st.slider("Debt ratio", 0.0, 30.0, 2.0, 0.5)
            credit_score  = st.slider("Credit history score (0 = none, 67 = excellent)", 0, 67, 3)
        with c4:
            years_emp     = st.slider("Years employed", 0.0, 30.0, 4.0, 0.5)
            default_label = st.selectbox("Prior default on record", ["No", "Yes"])
            employed_label= st.selectbox("Currently employed", ["Yes", "No"])

        st.markdown('<div class="section-title">Loan & background details</div>', unsafe_allow_html=True)
        c5, c6 = st.columns(2)
        with c5:
            edu_label     = st.selectbox("Primary credit purpose", list(EDUCATION_MAP.keys()))
            zip_code      = st.number_input("Zip code", 0, 99999, 200)
        with c6:
            eth_label     = st.selectbox("Applicant group", list(ETHNICITY_MAP.keys()))

        # Convert display labels back to model codes
        gender        = GENDER_MAP[gender_label]
        married       = MARRIED_MAP[married_label]
        bank_customer = BANK_MAP[bank_label]
        citizen       = CITIZEN_MAP[citizen_label]
        drivers_lic   = "t" if driv_label == "Yes" else "f"
        prior_default = "t" if default_label == "Yes" else "f"
        employed      = "t" if employed_label == "Yes" else "f"
        education_level = EDUCATION_MAP[edu_label]
        ethnicity     = ETHNICITY_MAP[eth_label]

        model_choice = st.radio("Model", ["SVM", "Decision Tree", "Ensemble (both)"], horizontal=True)

    with col_result:
        input_dict = {
            'gender': gender, 'age': age, 'debt': debt, 'married': married,
            'bank_customer': bank_customer, 'education_level': education_level,
            'ethnicity': ethnicity, 'years_employed': years_emp,
            'prior_default': prior_default, 'employed': employed,
            'credit_score': credit_score, 'drivers_license': drivers_lic,
            'citizen': citizen, 'zip_code': zip_code, 'income': income
        }

        try:
            X = preprocessor.transform(pd.DataFrame([input_dict]))[FEATURE_COLS].values

            if model_choice == "SVM":
                prob = svm_model.predict_proba(X)[0][1]; used = "SVM"
            elif model_choice == "Decision Tree":
                prob = dt_model.predict_proba(X)[0][1]; used = "Decision Tree"
            else:
                prob = np.mean([svm_model.predict_proba(X)[0][1], dt_model.predict_proba(X)[0][1]]); used = "Ensemble"

            approved = prob >= 0.5
            pct = prob * 100

            if approved:
                st.markdown(f"""
                <div class="decision-approved">
                    <div class="label">Decision</div>
                    <div class="value">Approved</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="decision-denied">
                    <div class="label">Decision</div>
                    <div class="value">Denied</div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-box">
                    <div class="m-label">Approval probability</div>
                    <div class="m-value">{pct:.1f}%</div>
                </div>
                <div class="metric-box">
                    <div class="m-label">Risk score</div>
                    <div class="m-value">{100-pct:.1f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            conf = ("Very High" if prob>=0.85 or prob<=0.15 else
                    "High"      if prob>=0.70 or prob<=0.30 else
                    "Moderate"  if prob>=0.60 or prob<=0.40 else "Low")
            risk = ("Low" if prob>=0.75 else "Moderate" if prob>=0.55 else
                    "High" if prob>=0.40 else "Very High")

            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-box">
                    <div class="m-label">Confidence</div>
                    <div class="m-value">{conf}</div>
                </div>
                <div class="metric-box">
                    <div class="m-label">Risk level</div>
                    <div class="m-value">{risk}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f'<div class="section-title">Probability breakdown · Model: {used}</div>', unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(5, 2.2))
            fig.patch.set_facecolor('white')
            ax.set_facecolor('#f8fafc')
            bars = ax.barh(["Denied", "Approved"], [1-prob, prob],
                           color=["#ef4444", "#22c55e"], height=0.45,
                           edgecolor='white', linewidth=1.5)
            ax.set_xlim(0, 1)
            ax.axvline(0.5, color='#94a3b8', ls='--', lw=1.2, label='Decision boundary (0.5)')
            for i, v in enumerate([1-prob, prob]):
                ax.text(min(v + 0.02, 0.92), i, f"{v*100:.1f}%", va='center', fontsize=11, fontweight='500')
            ax.set_xlabel("Probability", fontsize=10, color='#64748b')
            ax.tick_params(colors='#64748b')
            ax.legend(fontsize=8, framealpha=0.8)
            for spine in ax.spines.values():
                spine.set_edgecolor('#e2e8f0')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        except Exception as e:
            st.error(f"Prediction error: {e}")

    render_footer()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.markdown("""
    <div class="page-header">
        <h1>Model Performance</h1>
        <p>Evaluation metrics for SVM and Decision Tree models trained on the UCI Credit Approval dataset.</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Metrics overview", "ROC curves", "Confusion matrices"])

    with tab1:
        col1, col2 = st.columns(2)
        for col, (mname, label) in zip([col1, col2], [("svm","SVM"), ("decision_tree","Decision Tree")]):
            m = metrics[mname]
            with col:
                st.markdown(f"#### {label}")
                st.markdown(f"""
                <div class="metric-row">
                    <div class="metric-box"><div class="m-label">Accuracy</div><div class="m-value">{m['accuracy']*100:.1f}%</div></div>
                    <div class="metric-box"><div class="m-label">ROC-AUC</div><div class="m-value">{m['roc_auc']:.3f}</div></div>
                </div>
                <div class="metric-row">
                    <div class="metric-box"><div class="m-label">Precision</div><div class="m-value">{m['precision']*100:.1f}%</div></div>
                    <div class="metric-box"><div class="m-label">F1 Score</div><div class="m-value">{m['f1_score']*100:.1f}%</div></div>
                </div>
                <div class="metric-row">
                    <div class="metric-box"><div class="m-label">Recall</div><div class="m-value">{m['recall']*100:.1f}%</div></div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        metric_names = ['accuracy','precision','recall','f1_score','roc_auc']
        labels_disp  = ['Accuracy','Precision','Recall','F1','ROC-AUC']
        x = np.arange(len(labels_disp))
        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor('white'); ax.set_facecolor('#f8fafc')
        b1 = ax.bar(x-0.2, [metrics['svm'][k] for k in metric_names], 0.35, label='SVM', color='#3b5bdb', alpha=0.9, edgecolor='white')
        b2 = ax.bar(x+0.2, [metrics['decision_tree'][k] for k in metric_names], 0.35, label='Decision Tree', color='#f76707', alpha=0.9, edgecolor='white')
        ax.set_xticks(x); ax.set_xticklabels(labels_disp); ax.set_ylim(0.6, 1.05)
        ax.legend(); ax.set_ylabel('Score', color='#64748b')
        ax.tick_params(colors='#64748b')
        for bar in list(b1) + list(b2):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f'{bar.get_height():.3f}', ha='center', fontsize=8, color='#475569')
        for sp in ax.spines.values(): sp.set_edgecolor('#e2e8f0')
        sns.despine(); plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

        fi = metrics['decision_tree'].get('feature_importance', [])
        if fi:
            st.markdown("---")
            st.markdown("#### Decision Tree: Feature importance")
            fi_df = pd.DataFrame(fi).head(10)
            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_facecolor('white'); ax.set_facecolor('#f8fafc')
            colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(fi_df)))
            ax.barh(fi_df['feature'][::-1], fi_df['importance'][::-1], color=colors[::-1], edgecolor='white')
            ax.set_xlabel('Importance', color='#64748b'); ax.tick_params(colors='#64748b')
            for sp in ax.spines.values(): sp.set_edgecolor('#e2e8f0')
            sns.despine(); plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

    with tab2:
        target_col = 'approved' if 'approved' in df_preds.columns else 'target'
        if 'dt_probability' in df_preds.columns and target_col in df_preds.columns:
            fig, ax = plt.subplots(figsize=(7, 5))
            fig.patch.set_facecolor('white'); ax.set_facecolor('#f8fafc')
            ax.plot([0,1],[0,1],'--',color='#94a3b8',lw=1.5,label='Random classifier')
            for col_, label_, color_ in [('dt_probability','Decision Tree','#f76707'),('svm_probability','SVM','#3b5bdb')]:
                if col_ in df_preds.columns:
                    fpr,tpr,_ = roc_curve(df_preds[target_col], df_preds[col_])
                    auc = roc_auc_score(df_preds[target_col], df_preds[col_])
                    ax.plot(fpr,tpr,color=color_,lw=2.5,label=f'{label_} (AUC = {auc:.3f})')
            ax.set_xlabel('False Positive Rate', color='#64748b')
            ax.set_ylabel('True Positive Rate', color='#64748b')
            ax.legend(loc='lower right', framealpha=0.9)
            ax.tick_params(colors='#64748b')
            for sp in ax.spines.values(): sp.set_edgecolor('#e2e8f0')
            sns.despine(); plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

    with tab3:
        col1, col2 = st.columns(2)
        for col, (mname, label) in zip([col1,col2],[('svm','SVM'),('decision_tree','Decision Tree')]):
            cm = np.array(metrics[mname]['confusion_matrix'])
            with col:
                fig, ax = plt.subplots(figsize=(4.5, 3.8))
                fig.patch.set_facecolor('white')
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                            xticklabels=['Denied','Approved'],
                            yticklabels=['Denied','Approved'],
                            linewidths=1, linecolor='white', cbar=False,
                            annot_kws={'size': 14, 'weight': 'bold'})
                ax.set_title(label, fontsize=13, fontweight='600', pad=12, color='#1a1f36')
                ax.set_xlabel('Predicted', color='#64748b')
                ax.set_ylabel('Actual', color='#64748b')
                ax.tick_params(colors='#64748b')
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True); plt.close()

    render_footer()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DATA EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Data Explorer":
    st.markdown("""
    <div class="page-header">
        <h1>Data Explorer</h1>
        <p>UCI Credit Approval dataset with 690 applicants, 15 anonymized features.</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Overview", "Distributions", "Correlations"])

    with tab1:
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-box"><div class="m-label">Total records</div><div class="m-value">690</div></div>
            <div class="metric-box"><div class="m-label">Features</div><div class="m-value">15</div></div>
            <div class="metric-box"><div class="m-label">Approved</div><div class="m-value">307 <span style="font-size:0.85rem;color:#8892a0">(44.5%)</span></div></div>
            <div class="metric-box"><div class="m-label">Denied</div><div class="m-value">383 <span style="font-size:0.85rem;color:#8892a0">(55.5%)</span></div></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### Approval distribution")
            counts = df_raw['approved'].value_counts()
            fig, ax = plt.subplots(figsize=(4,3))
            fig.patch.set_facecolor('white'); ax.set_facecolor('#f8fafc')
            bars = ax.bar(['Denied','Approved'],
                          [counts.get(0,0), counts.get(1,0)],
                          color=['#ef4444','#22c55e'], width=0.45,
                          edgecolor='white', linewidth=1.5)
            for bar in bars:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+4,
                        str(int(bar.get_height())), ha='center', fontsize=11, color='#475569')
            ax.tick_params(colors='#64748b')
            for sp in ax.spines.values(): sp.set_edgecolor('#e2e8f0')
            sns.despine(); plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

        with col_b:
            st.markdown("#### Missing values by feature")
            miss = df_raw.isnull().sum()
            miss = miss[miss>0].reset_index()
            miss.columns = ['Feature','Missing']
            miss['Percentage'] = (miss['Missing']/690*100).round(1).astype(str) + '%'
            st.dataframe(miss, hide_index=True, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Raw data sample (first 20 rows)")
        st.dataframe(df_raw.head(20), use_container_width=True)

    with tab2:
        st.markdown("#### Age distribution by outcome")
        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor('white'); ax.set_facecolor('#f8fafc')
        for outcome,label,color in [(0,'Denied','#ef4444'),(1,'Approved','#22c55e')]:
            ax.hist(df_raw[df_raw['approved']==outcome]['age'].dropna(),
                    bins=25, alpha=0.65, label=label, color=color, edgecolor='white')
        ax.set_xlabel('Age', color='#64748b'); ax.set_ylabel('Count', color='#64748b')
        ax.legend(); ax.tick_params(colors='#64748b')
        for sp in ax.spines.values(): sp.set_edgecolor('#e2e8f0')
        sns.despine(); plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

        col1, col2 = st.columns(2)
        df_plot = df_raw.copy()
        df_plot['Outcome'] = df_plot['approved'].map({0:'Denied',1:'Approved'})
        for col_, feat, title in zip([col1,col2],['income','debt'],['Income by outcome','Debt by outcome']):
            with col_:
                st.markdown(f"#### {title}")
                fig,ax = plt.subplots(figsize=(5,3.5))
                fig.patch.set_facecolor('white'); ax.set_facecolor('#f8fafc')
                sns.boxplot(data=df_plot, x='Outcome', y=feat,
                            palette={'Approved':'#22c55e','Denied':'#ef4444'},
                            ax=ax, width=0.45, linewidth=1.2)
                ax.tick_params(colors='#64748b')
                for sp in ax.spines.values(): sp.set_edgecolor('#e2e8f0')
                sns.despine(); plt.tight_layout()
                st.pyplot(fig, use_container_width=True); plt.close()

    with tab3:
        st.markdown("#### Feature correlation matrix")
        num_cols = [c for c in ['age','debt','years_employed','credit_score','zip_code','income','approved'] if c in df_raw.columns]
        corr = df_raw[num_cols].corr()
        fig, ax = plt.subplots(figsize=(7,6))
        fig.patch.set_facecolor('white')
        sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool)),
                    annot=True, fmt='.2f', cmap='RdYlGn',
                    center=0, ax=ax, square=True, linewidths=0.5,
                    linecolor='white', cbar_kws={'shrink':0.8})
        ax.tick_params(colors='#64748b')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    render_footer()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "About":
    st.markdown("""
    <div class="page-header">
        <h1>About this project</h1>
        <p>An end-to-end machine learning pipeline for Credit Card Approval Prediction.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 0.8], gap="large")

    with col1:
        st.markdown("""
#### Overview
This project builds a production-ready ML pipeline to predict credit card approvals
using the UCI Credit Approval dataset. All 15 features are anonymized to protect applicant
confidentiality.

The pipeline covers raw data ingestion, preprocessing, model training with hyperparameter
tuning, a REST API, PostgreSQL storage, Power BI dashboards and this Streamlit app
as an interactive front end.

#### Model results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|:--------:|:---------:|:------:|:--:|:-------:|
| SVM (RBF, C=1) | 87.0% | 89.1% | 80.3% | 84.5% | 0.947 |
| Decision Tree (entropy, depth=3) | 82.6% | 91.1% | 67.2% | 77.4% | 0.954 |

#### Tech stack

| Layer | Technology |
|-------|-----------|
| ML models | SVM (RBF kernel), Decision Tree (entropy), Ensemble |
| Tuning | GridSearchCV, 5-fold cross-validation |
| Visualisation | Seaborn, Matplotlib |
| Demo | Streamlit |
| API | FastAPI + Uvicorn |
| Database | PostgreSQL 15 |
| BI | Microsoft Power BI |
| Containers | Docker + Docker Compose |
| Tests | pytest (17 unit tests) |
""")

    with col2:
        st.markdown("""
#### Dataset
**Source:** [UCI ML Repository — Credit Approval](https://archive.ics.uci.edu/dataset/27/credit+approval)

- 690 applicants
- 15 anonymized features
- Binary target: approved / denied
- 67 missing values handled via imputation
- 44.5% approval rate

#### Repository
[github.com/preethikachennareddy/credit-card-approval](https://github.com/preethikachennareddy/credit-card-approval)
""")

    render_footer()
