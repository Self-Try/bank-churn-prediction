"""
Bank Customer Churn Prediction — Interactive Streamlit App v2
Model : GradientBoostingClassifier  |  AUC-ROC 0.9929  |  F1 0.89
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0f1e;
    color: #e8eaf0;
}
.stApp { background: linear-gradient(135deg, #0a0f1e 0%, #0d1528 50%, #0a0f1e 100%); }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem 2rem; max-width: 1300px; }

.hero {
    background: linear-gradient(120deg, #0d1f3c, #0a2a4a, #0d3b6e);
    border: 1px solid rgba(59,130,246,0.25);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 250px; height: 250px;
    background: radial-gradient(circle, rgba(59,130,246,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #60a5fa, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem 0;
}
.hero-sub { color: #94a3b8; font-size: 1rem; font-weight: 300; margin: 0; }
.badge {
    display: inline-block;
    background: rgba(59,130,246,0.15);
    border: 1px solid rgba(59,130,246,0.35);
    color: #60a5fa;
    padding: 0.2rem 0.75rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
    margin-right: 0.5rem;
    margin-top: 0.8rem;
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #60a5fa;
    margin: 0.5rem 0 1rem 0;
}
.stSlider label, .stSelectbox label, .stNumberInput label {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
}
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #4f46e5) !important;
    color: white !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 1px !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 2.5rem !important;
    width: 100% !important;
    text-transform: uppercase !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1d4ed8, #4338ca) !important;
    box-shadow: 0 8px 25px rgba(59,130,246,0.35) !important;
}
.metric-tile {
    background: rgba(15,23,42,0.9);
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
}
.metric-label {
    font-size: 0.72rem; font-weight: 600;
    letter-spacing: 1.5px; text-transform: uppercase;
    color: #64748b; margin-bottom: 0.4rem;
}
.metric-value { font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 800; color: #f1f5f9; }
.metric-sub   { font-size: 0.72rem; color: #64748b; margin-top: 0.2rem; }
.result-box-high {
    background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(220,38,38,0.06));
    border: 1px solid rgba(239,68,68,0.4);
    border-radius: 16px; padding: 2rem; text-align: center;
}
.result-box-low {
    background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(5,150,105,0.06));
    border: 1px solid rgba(16,185,129,0.4);
    border-radius: 16px; padding: 2rem; text-align: center;
}
.result-title { font-family: 'Syne', sans-serif; font-size: 1.5rem; font-weight: 800; margin-bottom: 0.3rem; }
.result-prob  { font-family: 'Syne', sans-serif; font-size: 3.5rem; font-weight: 800; line-height: 1; }
.result-label { font-size: 0.8rem; color: #94a3b8; margin-top: 0.3rem; }
.risk-pill { display: inline-block; padding: 0.35rem 0.9rem; border-radius: 20px; font-size: 0.78rem; font-weight: 500; margin: 0.25rem; }
.risk-high { background: rgba(239,68,68,0.15); border: 1px solid rgba(239,68,68,0.4); color: #f87171; }
.risk-ok   { background: rgba(16,185,129,0.12); border: 1px solid rgba(16,185,129,0.35); color: #34d399; }
.action-item {
    background: rgba(59,130,246,0.07);
    border-left: 3px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 0.7rem 1rem; margin: 0.5rem 0;
    font-size: 0.88rem; color: #cbd5e1;
}
.prog-wrap { background: rgba(30,41,59,0.8); border-radius: 8px; height: 10px; overflow: hidden; margin-top: 0.5rem; }
.prog-fill-high { height: 100%; border-radius: 8px; background: linear-gradient(90deg, #f59e0b, #ef4444); }
.prog-fill-low  { height: 100%; border-radius: 8px; background: linear-gradient(90deg, #10b981, #06b6d4); }
hr { border-color: rgba(59,130,246,0.12) !important; margin: 1.5rem 0 !important; }
.stSelectbox [data-baseweb="select"] > div {
    background: rgba(15,23,42,0.8) !important;
    border-color: rgba(59,130,246,0.2) !important;
    border-radius: 8px !important;
}
.stNumberInput input {
    background: rgba(15,23,42,0.8) !important;
    border-color: rgba(59,130,246,0.2) !important;
    border-radius: 8px !important;
    color: #e8eaf0 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Load artifacts ────────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)

@st.cache_resource
def load_artifacts():
    model    = joblib.load(os.path.join(BASE, "churn_model.joblib"))
    scaler   = joblib.load(os.path.join(BASE, "scaler.joblib"))
    features = joblib.load(os.path.join(BASE, "feature_names.joblib"))
    return model, scaler, features

model, scaler, FEATURES = load_artifacts()


# ── Feature engineering ───────────────────────────────────────────────────────
def engineer_features(row: dict) -> pd.DataFrame:
    df = pd.DataFrame([row])
    edu_order  = {'Uneducated':0,'High School':1,'College':2,
                  'Graduate':3,'Post-Graduate':4,'Doctorate':5,'Unknown':2}
    inc_order  = {'Less than $40K':0,'$40K - $60K':1,'$60K - $80K':2,
                  '$80K - $120K':3,'$120K +':4,'Unknown':2}
    card_order = {'Blue':0,'Silver':1,'Gold':2,'Platinum':3}
    df['Education_Level'] = df['Education_Level'].map(edu_order)
    df['Income_Category'] = df['Income_Category'].map(inc_order)
    df['Card_Category']   = df['Card_Category'].map(card_order)
    df['Gender']          = (df['Gender'] == 'Male').astype(int)
    df['Marital_Status']  = df['Marital_Status'].map({'Single':0,'Married':1,'Divorced':2,'Unknown':3})
    df['txn_amt_dropped']  = (df['Total_Amt_Chng_Q4_Q1'] < 0.75).astype(int)
    df['txn_ct_dropped']   = (df['Total_Ct_Chng_Q4_Q1']  < 0.75).astype(int)
    df['both_dropped']     = ((df['txn_amt_dropped']==1) & (df['txn_ct_dropped']==1)).astype(int)
    df['avg_txn_value']    = df['Total_Trans_Amt'] / (df['Total_Trans_Ct'] + 1)
    df['revolv_to_limit']  = df['Total_Revolving_Bal'] / (df['Credit_Limit'] + 1)
    df['low_revolving']    = (df['revolv_to_limit'] < 0.05).astype(int)
    df['high_revolving']   = (df['revolv_to_limit'] > 0.90).astype(int)
    df['open_to_buy_ratio']= df['Avg_Open_To_Buy'] / (df['Credit_Limit'] + 1)
    df['engagement_score'] = (df['Total_Trans_Ct']*0.40 +
                              df['Total_Relationship_Count']*0.30 +
                              (12 - df['Months_Inactive_12_mon'])*0.30)
    df['high_inactivity']  = (df['Months_Inactive_12_mon'] >= 3).astype(int)
    df['high_contacts']    = (df['Contacts_Count_12_mon'] >= 4).astype(int)
    df['tenure_segment']   = pd.cut(df['Months_on_book'], bins=[0,24,36,48,999], labels=[0,1,2,3]).astype(int)
    df['age_group']        = pd.cut(df['Customer_Age'], bins=[0,35,45,55,999], labels=[0,1,2,3]).astype(int)
    return df[FEATURES]


# ═══════════════════════════════════════════════════════════════════
#  HERO
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <p class="hero-title">🛡️ ChurnGuard AI</p>
  <p class="hero-sub">Real-time bank customer churn risk prediction · Gradient Boosting · 10,127 customers trained</p>
  <span class="badge">AUC-ROC 0.9929</span>
  <span class="badge">F1-Score 0.89</span>
  <span class="badge">5-Fold CV Validated</span>
  <span class="badge">97% Accuracy</span>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  INPUT FORM
# ═══════════════════════════════════════════════════════════════════
col_left, col_mid, col_right = st.columns(3, gap="medium")

with col_left:
    st.markdown('<p class="section-title">👤 Customer Profile</p>', unsafe_allow_html=True)
    age          = st.slider("Age", 18, 75, 45)
    gender       = st.selectbox("Gender", ["Male", "Female"])
    dependents   = st.slider("Dependents", 0, 5, 2)
    education    = st.selectbox("Education", ["Uneducated","High School","College","Graduate","Post-Graduate","Doctorate","Unknown"])
    marital      = st.selectbox("Marital Status", ["Single","Married","Divorced","Unknown"])
    income       = st.selectbox("Income Bracket", ["Less than $40K","$40K - $60K","$60K - $80K","$80K - $120K","$120K +","Unknown"])

with col_mid:
    st.markdown('<p class="section-title">💳 Account Information</p>', unsafe_allow_html=True)
    card         = st.selectbox("Card Type", ["Blue","Silver","Gold","Platinum"])
    months_book  = st.slider("Months on Book", 12, 56, 36)
    rel_count    = st.slider("Product Relationships", 1, 6, 3, help="Total bank products held")
    months_inact = st.slider("Inactive Months (last 12)", 0, 6, 2)
    contacts_ct  = st.slider("Bank Contact Count (last 12)", 0, 6, 2)

with col_right:
    st.markdown('<p class="section-title">📊 Transaction Behaviour</p>', unsafe_allow_html=True)
    credit_limit = st.number_input("Credit Limit ($)", 1000, 35000, 10000, step=500)
    revolv_bal   = st.number_input("Revolving Balance ($)", 0, 3000, 800, step=100)
    avg_open_buy = st.number_input("Avg Open-to-Buy ($)", 0, 35000, 9000, step=500)
    trans_amt    = st.number_input("Total Transaction Amount ($)", 500, 20000, 4500, step=100)
    trans_ct     = st.slider("Total Transactions", 10, 140, 65)
    amt_chng     = st.slider("Spend Change Ratio Q4/Q1", 0.0, 3.5, 0.76, step=0.01, help="< 0.75 = spend dropped")
    ct_chng      = st.slider("Count Change Ratio Q4/Q1", 0.0, 3.5, 0.71, step=0.01, help="< 0.75 = count dropped")
    util_ratio   = st.slider("Avg Utilization Ratio", 0.0, 1.0, 0.27, step=0.01)


# ═══════════════════════════════════════════════════════════════════
#  PREDICT
# ═══════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
clicked = st.button("⚡  ANALYSE CHURN RISK", type="primary")

if clicked:
    raw = {
        'Customer_Age': age, 'Gender': gender, 'Dependent_count': dependents,
        'Education_Level': education, 'Marital_Status': marital,
        'Income_Category': income, 'Card_Category': card,
        'Months_on_book': months_book, 'Total_Relationship_Count': rel_count,
        'Months_Inactive_12_mon': months_inact, 'Contacts_Count_12_mon': contacts_ct,
        'Credit_Limit': credit_limit, 'Total_Revolving_Bal': revolv_bal,
        'Avg_Open_To_Buy': avg_open_buy, 'Total_Amt_Chng_Q4_Q1': amt_chng,
        'Total_Trans_Amt': trans_amt, 'Total_Trans_Ct': trans_ct,
        'Total_Ct_Chng_Q4_Q1': ct_chng, 'Avg_Utilization_Ratio': util_ratio,
    }

    with st.spinner("Running model inference..."):
        time.sleep(0.5)
        X_in  = engineer_features(raw)
        prob  = model.predict_proba(X_in)[0, 1]
        label = model.predict(X_in)[0]

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Result row ────────────────────────────────────────────────
    res_col, m1, m2, m3, m4 = st.columns([1.6, 1, 1, 1, 1], gap="medium")
    pct        = prob * 100
    is_high    = label == 1
    color      = "#f87171" if is_high else "#34d399"
    box_cls    = "result-box-high" if is_high else "result-box-low"
    bar_cls    = "prog-fill-high"  if is_high else "prog-fill-low"
    verdict    = ("🔴 HIGH CHURN RISK" if is_high else "🟢 LOW CHURN RISK")

    with res_col:
        st.markdown(f"""
        <div class="{box_cls}">
          <p class="result-title" style="color:{color}">{verdict}</p>
          <p class="result-prob" style="color:{color}">{pct:.1f}%</p>
          <p class="result-label">Churn Probability</p>
          <div class="prog-wrap" style="margin-top:1rem">
            <div class="{bar_cls}" style="width:{pct}%"></div>
          </div>
        </div>""", unsafe_allow_html=True)

    eng       = trans_ct*0.40 + rel_count*0.30 + (12 - months_inact)*0.30
    rev_ratio = revolv_bal / (credit_limit + 1)
    spend_drp = amt_chng < 0.75

    with m1:
        st.markdown(f"""<div class="metric-tile">
          <p class="metric-label">Engagement</p>
          <p class="metric-value" style="color:{'#f87171' if eng<30 else '#34d399'}">{eng:.0f}</p>
          <p class="metric-sub">composite score</p></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-tile">
          <p class="metric-label">Spend Drop</p>
          <p class="metric-value" style="color:{'#f87171' if spend_drp else '#34d399'}">{'Yes' if spend_drp else 'No'}</p>
          <p class="metric-sub">Q4 vs Q1</p></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-tile">
          <p class="metric-label">Inactivity</p>
          <p class="metric-value" style="color:{'#f87171' if months_inact>=3 else '#34d399'}">{months_inact}mo</p>
          <p class="metric-sub">last 12 months</p></div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="metric-tile">
          <p class="metric-label">Revolving %</p>
          <p class="metric-value" style="color:{'#f59e0b' if rev_ratio>0.7 else '#60a5fa'}">{rev_ratio*100:.0f}%</p>
          <p class="metric-sub">of credit limit</p></div>""", unsafe_allow_html=True)

    # ── Risk Factors + Actions ────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    rf_col, act_col = st.columns(2, gap="large")

    with rf_col:
        st.markdown('<p class="section-title">🔎 Risk Factor Breakdown</p>', unsafe_allow_html=True)
        factors = [
            ("Spending dropped Q4→Q1",       amt_chng < 0.75,   "Spend declining",         "Spend stable"),
            ("Transaction count dropped",     ct_chng  < 0.75,   "Transactions declining",  "Transactions healthy"),
            ("High inactivity ≥ 3 months",   months_inact >= 3, "Customer disengaging",    "Customer active"),
            ("Excessive contacts ≥ 4",        contacts_ct >= 4,  "Possible dissatisfaction","Normal contact frequency"),
            ("Zero revolving balance",         revolv_bal == 0,   "Card may not be primary", "Credit utilization present"),
            ("Low product relationships ≤ 2", rel_count <= 2,    "Single-product customer", "Multi-product customer"),
            ("Short tenure < 24 months",      months_book < 24,  "Newer / less loyal",      "Long-tenured customer"),
        ]
        for _, is_risk, risk_msg, ok_msg in factors:
            cls = "risk-high" if is_risk else "risk-ok"
            ico = "⚠️" if is_risk else "✅"
            st.markdown(f'<span class="risk-pill {cls}">{ico} {risk_msg if is_risk else ok_msg}</span>',
                        unsafe_allow_html=True)

    with act_col:
        st.markdown('<p class="section-title">💡 Recommended Actions</p>', unsafe_allow_html=True)
        actions = []
        if is_high:
            if months_inact >= 3:
                actions.append("📬 Re-engagement campaign — 3+ inactive months detected")
            if amt_chng < 0.75 and ct_chng < 0.75:
                actions.append("🎁 Offer personalised cashback to reverse spend decline")
            if contacts_ct >= 4:
                actions.append("🛠️ Escalate to senior support — high contacts signal friction")
            if revolv_bal == 0:
                actions.append("💳 Promote EMI / balance-transfer to re-activate card usage")
            if rel_count <= 2:
                actions.append("📦 Cross-sell savings or loan product to deepen relationship")
            if not actions:
                actions.append("📞 Schedule proactive retention call within 7 days")
            actions.append("📊 Flag for 30-day churn watch list")
        else:
            actions.append("✅ Customer is healthy — standard engagement sufficient")
            actions.append("🌟 Consider upsell — customer shows strong loyalty signals")
            if trans_ct > 80:
                actions.append("🏆 Nominate for loyalty rewards tier upgrade")

        for a in actions:
            st.markdown(f'<div class="action-item">{a}</div>', unsafe_allow_html=True)

    # ── Confidence footer ─────────────────────────────────────────
    conf = "Very High" if abs(prob-0.5) > 0.35 else "High" if abs(prob-0.5) > 0.2 else "Moderate"
    st.markdown(f"""
    <br><div style="background:rgba(15,23,42,0.6);border:1px solid rgba(59,130,246,0.12);
         border-radius:10px;padding:0.9rem 1.2rem;font-size:0.82rem;color:#64748b;">
      🤖 <strong style="color:#94a3b8">Model Confidence:</strong> {conf} &nbsp;·&nbsp;
      GradientBoostingClassifier &nbsp;·&nbsp; 300 estimators &nbsp;·&nbsp;
      AUC-ROC 0.9923 &nbsp;·&nbsp; 5-Fold CV 0.9929 ± 0.002
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  BENCHMARK STRIP (always visible)
# ═══════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<p class="section-title">📈 Model Performance Benchmarks</p>', unsafe_allow_html=True)
b1, b2, b3, b4, b5 = st.columns(5)
for col, (lbl, val, sub) in zip([b1,b2,b3,b4,b5], [
    ("AUC-ROC",   "0.9923", "test set"),
    ("F1-Score",  "0.8933", "churned class"),
    ("Recall",    "90.2%",  "churned detected"),
    ("Precision", "88.5%",  "predictions correct"),
    ("CV AUC",    "0.9929", "± 0.002 · 5-fold"),
]):
    with col:
        st.markdown(f"""<div class="metric-tile">
          <p class="metric-label">{lbl}</p>
          <p class="metric-value" style="font-size:1.4rem;color:#60a5fa">{val}</p>
          <p class="metric-sub">{sub}</p></div>""", unsafe_allow_html=True)
