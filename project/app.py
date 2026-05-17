"""
Bank Customer Churn Prediction — Streamlit App
Author  : Your Name
Model   : GradientBoostingClassifier (AUC-ROC 0.9929, F1 0.89)
Dataset : BankChurners (10,127 records, Kaggle)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Churn Predictor",
    page_icon="🏦",
    layout="wide",
)

# ── Load artefacts ───────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)

@st.cache_resource
def load_artifacts():
    model    = joblib.load(os.path.join(BASE, "churn_model.joblib"))
    scaler   = joblib.load(os.path.join(BASE, "scaler.joblib"))
    features = joblib.load(os.path.join(BASE, "feature_names.joblib"))
    return model, scaler, features

model, scaler, FEATURES = load_artifacts()

# ── Feature engineering (mirrors notebook) ──────────────────────────────────
def engineer_features(row: dict) -> pd.DataFrame:
    df = pd.DataFrame([row])

    # Encoding maps
    edu_order  = {'Uneducated':0,'High School':1,'College':2,
                  'Graduate':3,'Post-Graduate':4,'Doctorate':5,'Unknown':2}
    inc_order  = {'Less than $40K':0,'$40K - $60K':1,'$60K - $80K':2,
                  '$80K - $120K':3,'$120K +':4,'Unknown':2}
    card_order = {'Blue':0,'Silver':1,'Gold':2,'Platinum':3}

    df['Education_Level'] = df['Education_Level'].map(edu_order)
    df['Income_Category'] = df['Income_Category'].map(inc_order)
    df['Card_Category']   = df['Card_Category'].map(card_order)
    df['Gender']          = (df['Gender'] == 'Male').astype(int)
    df['Marital_Status']  = df['Marital_Status'].map(
                            {'Single':0,'Married':1,'Divorced':2,'Unknown':3})

    # Feature engineering
    df['txn_amt_dropped']  = (df['Total_Amt_Chng_Q4_Q1'] < 0.75).astype(int)
    df['txn_ct_dropped']   = (df['Total_Ct_Chng_Q4_Q1']  < 0.75).astype(int)
    df['both_dropped']     = ((df['txn_amt_dropped']==1) &
                              (df['txn_ct_dropped']==1)).astype(int)
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
    df['tenure_segment']   = pd.cut(df['Months_on_book'],
                                    bins=[0,24,36,48,999],
                                    labels=[0,1,2,3]).astype(int)
    df['age_group']        = pd.cut(df['Customer_Age'],
                                    bins=[0,35,45,55,999],
                                    labels=[0,1,2,3]).astype(int)
    return df[FEATURES]

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🏦 Bank Customer Churn Predictor")
st.markdown("""
> **GradientBoosting model** trained on 10,127 bank customers  
> AUC-ROC **0.9929** · F1-Score **0.89** · 5-Fold CV AUC **0.9929 ± 0.002**
""")
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Demographics")
    age            = st.slider("Customer Age", 18, 75, 45)
    gender         = st.selectbox("Gender", ["Male", "Female"])
    dependent_ct   = st.slider("Dependent Count", 0, 5, 2)
    education      = st.selectbox("Education Level",
                        ["Uneducated","High School","College",
                         "Graduate","Post-Graduate","Doctorate","Unknown"])
    marital        = st.selectbox("Marital Status",
                        ["Single","Married","Divorced","Unknown"])
    income         = st.selectbox("Income Category",
                        ["Less than $40K","$40K - $60K","$60K - $80K",
                         "$80K - $120K","$120K +","Unknown"])

with col2:
    st.subheader("💳 Account Details")
    card           = st.selectbox("Card Category", ["Blue","Silver","Gold","Platinum"])
    months_book    = st.slider("Months on Book", 12, 56, 36)
    rel_count      = st.slider("Total Relationship Count", 1, 6, 3)
    months_inact   = st.slider("Months Inactive (last 12)", 0, 6, 2)
    contacts_ct    = st.slider("Contacts Count (last 12)", 0, 6, 2)

with col3:
    st.subheader("📊 Transaction Behaviour")
    credit_limit   = st.number_input("Credit Limit ($)", 1000, 35000, 10000, step=500)
    revolving_bal  = st.number_input("Total Revolving Balance ($)", 0, 3000, 800, step=100)
    avg_open_buy   = st.number_input("Avg Open To Buy ($)", 0, 35000, 9000, step=500)
    trans_amt      = st.number_input("Total Transaction Amount ($)", 500, 20000, 4500, step=100)
    trans_ct       = st.slider("Total Transaction Count", 10, 140, 65)
    amt_chng       = st.slider("Amt Change Q4/Q1 ratio", 0.0, 3.5, 0.76, step=0.01)
    ct_chng        = st.slider("Count Change Q4/Q1 ratio", 0.0, 3.5, 0.71, step=0.01)
    util_ratio     = st.slider("Avg Utilization Ratio", 0.0, 1.0, 0.27, step=0.01)

st.divider()
predict_btn = st.button("🔍 Predict Churn Risk", type="primary", use_container_width=True)

if predict_btn:
    raw = {
        'Customer_Age': age, 'Gender': gender, 'Dependent_count': dependent_ct,
        'Education_Level': education, 'Marital_Status': marital,
        'Income_Category': income, 'Card_Category': card,
        'Months_on_book': months_book, 'Total_Relationship_Count': rel_count,
        'Months_Inactive_12_mon': months_inact, 'Contacts_Count_12_mon': contacts_ct,
        'Credit_Limit': credit_limit, 'Total_Revolving_Bal': revolving_bal,
        'Avg_Open_To_Buy': avg_open_buy, 'Total_Amt_Chng_Q4_Q1': amt_chng,
        'Total_Trans_Amt': trans_amt, 'Total_Trans_Ct': trans_ct,
        'Total_Ct_Chng_Q4_Q1': ct_chng, 'Avg_Utilization_Ratio': util_ratio,
    }

    X_input = engineer_features(raw)
    prob    = model.predict_proba(X_input)[0, 1]
    label   = model.predict(X_input)[0]

    r1, r2, r3 = st.columns([1, 1, 1])

    with r2:
        if label == 1:
            st.error(f"## ⚠️ HIGH CHURN RISK\n### Probability: **{prob*100:.1f}%**")
        else:
            st.success(f"## ✅ LOW CHURN RISK\n### Probability: **{prob*100:.1f}%**")

    st.markdown("### 📋 Risk Factor Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Churn Probability", f"{prob*100:.1f}%")
    c2.metric("Transaction Drop",  "Yes" if amt_chng < 0.75 else "No")
    c3.metric("High Inactivity",   "Yes" if months_inact >= 3 else "No")
    c4.metric("High Contacts",     "Yes" if contacts_ct >= 4 else "No")

    st.markdown("### 💡 Recommended Actions")
    actions = []
    if prob > 0.5:
        if months_inact >= 3:
            actions.append("📬 Re-engagement campaign — customer has been inactive for 3+ months.")
        if contacts_ct >= 4:
            actions.append("🛠️ Review service quality — high contact count may signal dissatisfaction.")
        if amt_chng < 0.75:
            actions.append("🎁 Offer personalised rewards to reverse declining spend.")
        if revolving_bal == 0:
            actions.append("💰 Customer shows zero revolving balance — may be disengaging from credit usage.")
        if not actions:
            actions.append("🔁 Schedule a proactive retention call.")
    else:
        actions.append("✅ Customer appears healthy — continue standard engagement.")

    for a in actions:
        st.markdown(f"- {a}")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Model: GradientBoostingClassifier | Dataset: BankChurners (Kaggle) | Built for interview & deployment demo")
