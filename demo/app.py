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
    page_title="Credit Approval ML",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.approved { color: #2ecc71; font-weight: 700; font-size: 1.4rem; }
.denied   { color: #e74c3c; font-weight: 700; font-size: 1.4rem; }
</style>
""", unsafe_allow_html=True)

# ── Auto-train if models missing ──────────────────────────────────────────────
models_path = os.path.join(ROOT, 'models')
preprocessor_path = os.path.join(models_path, 'preprocessor.pkl')

if not os.path.exists(preprocessor_path):
    with st.spinner("🔧 First run — training models (this takes ~60 seconds)..."):
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
    st.success("✅ Models trained! Loading demo...")
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
    path = os.path.join(ROOT, 'reports/model_metrics.json')
    with open(path) as f:
        return json.load(f)

@st.cache_data
def load_predictions():
    path = os.path.join(ROOT, 'reports/predictions.csv')
    return pd.read_csv(path)

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
st.sidebar.title("💳 Credit Approval ML")
st.sidebar.caption("UCI Credit Approval · SVM & Decision Tree")

page = st.sidebar.radio("Navigate", [
    "🔍 Live Predictor", "📊 Model Performance", "📈 Data Explorer", "ℹ️ About"
], label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Dataset:** 690 applicants · 15 features · 44.5% approval rate")
st.sidebar.markdown(f"**SVM AUC:** {metrics['svm']['roc_auc']:.3f}")
st.sidebar.markdown(f"**DT AUC:** {metrics['decision_tree']['roc_auc']:.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — LIVE PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Live Predictor":
    st.title("💳 Credit Card Approval Predictor")
    st.caption("Adjust the applicant profile and get instant model predictions.")

    col_form, col_result = st.columns([1.1, 0.9], gap="large")

    with col_form:
        st.markdown("**Personal information**")
        c1, c2 = st.columns(2)
        with c1:
            gender        = st.selectbox("Gender", ["b","a"], format_func=lambda x: "Male" if x=="b" else "Female")
            age           = st.slider("Age", 15, 80, 30)
            married       = st.selectbox("Marital status", ["u","y"], format_func=lambda x: "Married" if x=="u" else "Single")
        with c2:
            bank_customer = st.selectbox("Bank customer", ["g","p","gg"])
            citizen       = st.selectbox("Citizen", ["g","p","s"])
            drivers_lic   = st.selectbox("Driver's licence", ["f","t"], format_func=lambda x: "Yes" if x=="t" else "No")

        st.markdown("**Financial profile**")
        c3, c4 = st.columns(2)
        with c3:
            income        = st.number_input("Income ($)", 0, 200000, 35000, step=1000)
            debt          = st.slider("Debt ratio", 0.0, 30.0, 2.0, 0.5)
            credit_score  = st.slider("Credit history score", 0, 67, 3)
        with c4:
            years_emp     = st.slider("Years employed", 0.0, 30.0, 4.0, 0.5)
            prior_default = st.selectbox("Prior default", ["f","t"], format_func=lambda x: "Yes" if x=="t" else "No")
            employed      = st.selectbox("Currently employed", ["t","f"], format_func=lambda x: "Yes" if x=="t" else "No")

        st.markdown("**Other**")
        c5, c6 = st.columns(2)
        with c5:
            education_level = st.selectbox("Education", ["c","d","cc","i","j","k","m","r","q","w","x","e","aa","ff"])
            zip_code = st.number_input("Zip code", 0, 99999, 200)
        with c6:
            ethnicity = st.selectbox("Ethnicity code", ["v","h","bb","j","n","z","dd","ff","o"])

        model_choice = st.radio("Model", ["SVM","Decision Tree","Ensemble (both)"], horizontal=True)

    with col_result:
        st.markdown("**Prediction**")

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
                st.markdown('<p class="approved">✅ APPROVED</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="denied">❌ DENIED</p>', unsafe_allow_html=True)

            st.progress(prob)

            m1, m2 = st.columns(2)
            m1.metric("Approval probability", f"{pct:.1f}%")
            m2.metric("Risk score", f"{100-pct:.1f}%")

            conf = ("Very High" if prob>=0.85 or prob<=0.15 else "High" if prob>=0.70 or prob<=0.30 else "Moderate" if prob>=0.60 or prob<=0.40 else "Low")
            risk = ("Low" if prob>=0.75 else "Moderate" if prob>=0.55 else "High" if prob>=0.40 else "Very High")

            m3, m4 = st.columns(2)
            m3.metric("Confidence", conf)
            m4.metric("Risk level", risk)
            st.caption(f"Model: {used}")

            fig, ax = plt.subplots(figsize=(5, 2))
            ax.barh(["Denied","Approved"], [1-prob, prob], color=["#e74c3c","#2ecc71"], height=0.4)
            ax.set_xlim(0,1); ax.axvline(0.5, color='gray', ls='--', lw=1)
            for i, v in enumerate([1-prob, prob]):
                ax.text(v+0.01, i, f"{v*100:.1f}%", va='center', fontsize=10)
            sns.despine(); plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

        except Exception as e:
            st.error(f"Prediction error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.title("📊 Model Performance")
    tab1, tab2, tab3 = st.tabs(["Metrics", "ROC Curves", "Confusion Matrices"])

    with tab1:
        col1, col2 = st.columns(2)
        for col, (mname, label) in zip([col1,col2],[("svm","SVM"),("decision_tree","Decision Tree")]):
            m = metrics[mname]
            with col:
                st.subheader(label)
                r1,r2 = st.columns(2)
                r1.metric("Accuracy", f"{m['accuracy']*100:.1f}%")
                r2.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
                r3,r4 = st.columns(2)
                r3.metric("Precision", f"{m['precision']*100:.1f}%")
                r4.metric("F1 Score", f"{m['f1_score']*100:.1f}%")

        st.markdown("---")
        metric_names = ['accuracy','precision','recall','f1_score','roc_auc']
        labels = ['Accuracy','Precision','Recall','F1','ROC-AUC']
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(9,4))
        ax.bar(x-0.2, [metrics['svm'][k] for k in metric_names], 0.35, label='SVM', color='#3266ad', alpha=0.85)
        ax.bar(x+0.2, [metrics['decision_tree'][k] for k in metric_names], 0.35, label='Decision Tree', color='#e67e22', alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0.6,1.05); ax.legend()
        sns.despine(); plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

        fi = metrics['decision_tree'].get('feature_importance', [])
        if fi:
            st.markdown("---")
            st.subheader("Decision Tree — Feature importance")
            fi_df = pd.DataFrame(fi).head(10)
            fig, ax = plt.subplots(figsize=(8,4))
            ax.barh(fi_df['feature'][::-1], fi_df['importance'][::-1],
                    color=plt.cm.viridis(np.linspace(0.3,0.9,len(fi_df)))[::-1])
            ax.set_xlabel('Importance'); sns.despine(); plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

    with tab2:
        target_col = 'approved' if 'approved' in df_preds.columns else 'target'
        if 'dt_probability' in df_preds.columns and target_col in df_preds.columns:
            fig, ax = plt.subplots(figsize=(7,5))
            ax.plot([0,1],[0,1],'k--',lw=1.5,label='Random')
            for col, label, color in [('dt_probability','Decision Tree','#e67e22'),('svm_probability','SVM','#3266ad')]:
                if col in df_preds.columns:
                    fpr,tpr,_ = roc_curve(df_preds[target_col], df_preds[col])
                    auc = roc_auc_score(df_preds[target_col], df_preds[col])
                    ax.plot(fpr,tpr,color=color,lw=2.5,label=f'{label} (AUC={auc:.3f})')
            ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
            ax.legend(loc='lower right'); sns.despine(); plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

    with tab3:
        col1, col2 = st.columns(2)
        for col, (mname, label) in zip([col1,col2],[('svm','SVM'),('decision_tree','Decision Tree')]):
            cm = np.array(metrics[mname]['confusion_matrix'])
            with col:
                fig, ax = plt.subplots(figsize=(4,3.5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                            xticklabels=['Denied','Approved'], yticklabels=['Denied','Approved'],
                            linewidths=1, cbar=False)
                ax.set_title(label); ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True); plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DATA EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Data Explorer":
    st.title("📈 Data Explorer")
    tab1, tab2, tab3 = st.tabs(["Overview","Distributions","Correlations"])

    with tab1:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total records","690"); c2.metric("Features","15")
        c3.metric("Approved","307 (44.5%)"); c4.metric("Denied","383 (55.5%)")
        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Approval distribution")
            counts = df_raw['approved'].value_counts()
            fig, ax = plt.subplots(figsize=(4,3))
            ax.bar(['Denied','Approved'],[counts.get(0,0),counts.get(1,0)],color=['#e74c3c','#2ecc71'],width=0.5)
            sns.despine(); plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()
        with col_b:
            st.subheader("Missing values")
            miss = df_raw.isnull().sum()
            miss = miss[miss>0].reset_index()
            miss.columns = ['Feature','Missing']
            miss['%'] = (miss['Missing']/690*100).round(1)
            st.dataframe(miss, hide_index=True, use_container_width=True)
        st.markdown("---")
        st.subheader("Raw data sample")
        st.dataframe(df_raw.head(20), use_container_width=True)

    with tab2:
        fig, ax = plt.subplots(figsize=(8,4))
        for outcome,label,color in [(0,'Denied','#e74c3c'),(1,'Approved','#2ecc71')]:
            ax.hist(df_raw[df_raw['approved']==outcome]['age'].dropna(),bins=25,alpha=0.6,label=label,color=color)
        ax.set_xlabel('Age'); ax.set_ylabel('Count'); ax.legend(); sns.despine(); plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

        col1,col2 = st.columns(2)
        df_plot = df_raw.copy()
        df_plot['Outcome'] = df_plot['approved'].map({0:'Denied',1:'Approved'})
        for col, feat in zip([col1,col2],['income','debt']):
            with col:
                st.subheader(f"{feat.title()} by outcome")
                fig,ax = plt.subplots(figsize=(5,3.5))
                sns.boxplot(data=df_plot,x='Outcome',y=feat,
                            palette={'Approved':'#2ecc71','Denied':'#e74c3c'},ax=ax,width=0.5)
                sns.despine(); plt.tight_layout()
                st.pyplot(fig, use_container_width=True); plt.close()

    with tab3:
        num_cols = [c for c in ['age','debt','years_employed','credit_score','zip_code','income','approved'] if c in df_raw.columns]
        corr = df_raw[num_cols].corr()
        fig,ax = plt.subplots(figsize=(7,6))
        sns.heatmap(corr,mask=np.triu(np.ones_like(corr,dtype=bool)),annot=True,fmt='.2f',
                    cmap='RdYlGn',center=0,ax=ax,square=True,linewidths=0.5)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.title("ℹ️ About")
    st.markdown("""
## Credit Card Approval Prediction

Built on the UCI Credit Approval dataset (690 applicants, 15 anonymized features).

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| SVM (RBF, C=1) | 87.0% | 0.947 |
| Decision Tree (entropy, depth=3) | 82.6% | 0.954 |

**Stack:** scikit-learn · Streamlit · FastAPI · PostgreSQL · Docker · Power BI

**Repo:** [github.com/preethikachennareddy/credit-card-approval](https://github.com/preethikachennareddy/credit-card-approval)
""")
