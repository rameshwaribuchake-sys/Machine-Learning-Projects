import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CUSTOMER CHURN PREDICTION SYSTEM
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Load trained model
# -----------------------------
model = pickle.load(open("random_forest_churn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# -----------------------------
# Custom design
# -----------------------------
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: #08111f !important;
}
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 85% 10%, #173b67 0%, transparent 28%),
        radial-gradient(circle at 10% 90%, #102a46 0%, transparent 25%),
        #08111f !important;
}
[data-testid="stHeader"] {
    background: transparent !important;
}
[data-testid="stSidebar"] {
    background: #0b1728 !important;
    border-right: 1px solid #263a52;
}
[data-testid="stSidebar"] * {
    color: #ffffff !important;
}
.block-container {
    max-width: 1250px;
    padding-top: 1.8rem !important;
}
h1, h2, h3, h4, p, label, span {
    color: #ffffff !important;
}
.topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 18px 22px;
    border-radius: 16px;
    background: rgba(17, 31, 51, 0.9);
    border: 1px solid #29425e;
    margin-bottom: 22px;
}
.brand {
    color: #ffffff !important;
    font-size: 27px;
    font-weight: 800;
}
.badge {
    color: #9fd3ff !important;
    background: #102d49;
    border: 1px solid #24537b;
    padding: 7px 13px;
    border-radius: 20px;
    font-size: 13px;
}
.hero {
    padding: 48px;
    border-radius: 24px;
    background: linear-gradient(115deg, #102a46, #163b62);
    border: 1px solid #315778;
    box-shadow: 0 18px 45px rgba(0,0,0,.20);
}
.hero h1 {
    font-size: 43px;
    margin: 0;
    color: #ffffff !important;
}
.hero p {
    color: #c9d8e8 !important;
    font-size: 18px;
    max-width: 850px;
    line-height: 1.65;
}
.section-title {
    margin-top: 25px;
    margin-bottom: 13px;
    color: #ffffff !important;
    font-size: 23px;
    font-weight: 800;
}
.card {
    padding: 22px;
    min-height: 130px;
    border-radius: 18px;
    background: #101d2e;
    border: 1px solid #2a4059;
}
.card small {
    color: #8fb0cc !important;
    font-size: 13px;
    font-weight: 700;
}
.card strong {
    display: block;
    margin-top: 9px;
    color: #ffffff !important;
    font-size: 25px;
}
.info {
    padding: 20px;
    border-radius: 15px;
    background: #0e1d30;
    border-left: 4px solid #4da3ff;
    color: #e8f2fc !important;
}
.result {
    padding: 28px;
    border-radius: 20px;
    text-align: center;
    margin-top: 18px;
}
.stay {
    background: linear-gradient(135deg, #0d3b2a, #14553c);
    border: 1px solid #2c9b6b;
}
.exit {
    background: linear-gradient(135deg, #461d27, #642534);
    border: 1px solid #d35d73;
}
.result-title {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-size: 31px;
    font-weight: 850;
}
.result-sub {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-size: 17px;
    margin-top: 8px;
}
.risk-card {
    padding: 23px;
    border-radius: 18px;
    background: #101d2e;
    border: 1px solid #2a4059;
}
.risk-name {
    color: #ffffff !important;
    font-size: 26px;
    font-weight: 800;
}
.risk-note {
    color: #b9c9d9 !important;
    margin-top: 5px;
}
[data-baseweb="input"] {
    background: #15263a !important;
    border: 1px solid #3a526d !important;
    border-radius: 9px !important;
}
[data-baseweb="input"] input {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}
[data-baseweb="select"] > div {
    background: #15263a !important;
    border: 1px solid #3a526d !important;
}
[data-baseweb="select"] * {
    color: #ffffff !important;
}
.stButton > button {
    width: 100%;
    height: 54px;
    border-radius: 12px;
    background: linear-gradient(90deg, #1976d2, #2997e8) !important;
    color: #ffffff !important;
    border: none !important;
    font-size: 17px;
    font-weight: 800;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #1565b7, #1685d3) !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="topbar">
    <div class="brand">📊 Customer Churn Prediction System</div>
    <div class="badge">MACHINE LEARNING • RANDOM FOREST</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Navigation
# -----------------------------
st.sidebar.markdown("## 📊 Customer Churn")
st.sidebar.caption("Prediction & Customer Risk Analysis")
page = st.sidebar.radio(
    "Pages",
    ["🏠 Home", "🔮 Prediction", "📈 Graphs & Insights"],
    label_visibility="collapsed"
)

# ============================================================
# HOME
# ============================================================
if page == "🏠 Home":
    st.markdown("""
    <div class="hero">
        <h1>Customer Churn Prediction System</h1>
        <p>
            Predict whether a customer is likely to stay with the bank or
            leave. The system uses customer, financial and activity details
            with a trained Random Forest model.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">What this system provides</div>', unsafe_allow_html=True)

    a, b, c = st.columns(3)
    with a:
        st.markdown("""
        <div class="card">
            <small>01 • PREDICTION</small>
            <strong>Stay / Exit</strong>
            <p>Instantly classify the customer.</p>
        </div>
        """, unsafe_allow_html=True)
    with b:
        st.markdown("""
        <div class="card">
            <small>02 • PROBABILITY</small>
            <strong>Risk Score</strong>
            <p>Compare stay and churn probabilities.</p>
        </div>
        """, unsafe_allow_html=True)
    with c:
        st.markdown("""
        <div class="card">
            <small>03 • VISUAL INSIGHTS</small>
            <strong>Graphs</strong>
            <p>Understand risk and important features.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">How it works</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info">
        Enter customer details → preprocess the inputs → Random Forest predicts
        churn → view the prediction, probability comparison and risk level.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# PREDICTION
# ============================================================
elif page == "🔮 Prediction":
    st.markdown("""
    <div class="hero">
        <h1>Customer Risk Prediction</h1>
        <p>Enter customer information and generate a complete churn-risk report.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">👤 Customer Profile</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        credit_score = st.number_input("Credit Score", value=600, step=1)
    with c2:
        age = st.number_input("Age", value=30, step=1)
    with c3:
        tenure = st.number_input("Tenure", value=5, step=1)

    st.markdown('<div class="section-title">💳 Banking & Activity</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        balance = st.number_input("Balance", value=50000.0, step=1000.0)
    with c2:
        num_products = st.number_input("Number of Products", value=2, step=1)
    with c3:
        salary = st.number_input("Estimated Salary", value=50000.0, step=1000.0)

    c1, c2, c3 = st.columns(3)
    with c1:
        has_card = st.selectbox("Has Credit Card", [0, 1])
    with c2:
        active_member = st.selectbox("Active Member", [0, 1])
    with c3:
        country = st.selectbox("Country", ["France", "Germany", "Spain"])

    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Gender", ["Male", "Female"])

    st.markdown("")
    if st.button("🔮 Generate Churn Prediction"):
        germany = 1 if country == "Germany" else 0
        spain = 1 if country == "Spain" else 0
        male = 1 if gender == "Male" else 0

        data = np.array([[
            credit_score, age, tenure, balance, num_products,
            has_card, active_member, salary, germany, spain, male
        ]])

        data_scaled = scaler.transform(data)
        prediction = int(model.predict(data_scaled)[0])

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(data_scaled)[0]
            # Assumes class 0 = stay and class 1 = churn, as in the project.
            stay_prob = float(probs[0])
            churn_prob = float(probs[1])
        else:
            churn_prob = float(prediction)
            stay_prob = 1 - churn_prob

        # Risk level based on churn probability.
        if churn_prob < 0.30:
            risk = "LOW RISK"
            risk_score = churn_prob
        elif churn_prob < 0.60:
            risk = "MEDIUM RISK"
            risk_score = churn_prob
        else:
            risk = "HIGH RISK"
            risk_score = churn_prob

        st.markdown("---")

        if prediction == 1:
            st.markdown("""
            <div class="result exit">
                <div class="result-title">⚠️ Customer Will Exit</div>
                <div class="result-sub">The model predicts that this customer is likely to churn.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result stay">
                <div class="result-title">✅ Customer Will Stay</div>
                <div class="result-sub">The model predicts that this customer is likely to remain with the bank.</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">📊 Prediction Probability Comparison</div>', unsafe_allow_html=True)

        # Probability comparison graph
        prob_df = pd.DataFrame({
            "Outcome": ["Will Stay", "Will Exit"],
            "Probability": [stay_prob, churn_prob]
        })

        fig, ax = plt.subplots(figsize=(9, 3.4))
        bars = ax.barh(prob_df["Outcome"], prob_df["Probability"])
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        ax.set_title("Stay vs Churn Probability")
        for bar, value in zip(bars, prob_df["Probability"]):
            ax.text(
                min(value + 0.02, 0.92),
                bar.get_y() + bar.get_height()/2,
                f"{value:.1%}",
                va="center"
            )
        fig.tight_layout()
        st.pyplot(fig)

        # Risk level section
        st.markdown('<div class="section-title">🚦 Customer Risk Level</div>', unsafe_allow_html=True)

        r1, r2 = st.columns([1, 2])
        with r1:
            st.markdown(f"""
            <div class="risk-card">
                <div class="risk-name">{risk}</div>
                <div class="risk-note">Churn probability: {churn_prob:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            risk_df = pd.DataFrame({
                "Risk Level": ["Low", "Medium", "High"],
                "Churn Probability Range": [0.30, 0.30, 0.40]
            })

            fig2, ax2 = plt.subplots(figsize=(8, 2.4))
            ax2.barh(["Risk scale"], [1])
            ax2.axvline(0.30, linestyle="--")
            ax2.axvline(0.60, linestyle="--")
            ax2.axvline(churn_prob, linewidth=5)
            ax2.set_xlim(0, 1)
            ax2.set_xlabel("Churn Probability")
            ax2.set_yticks([])
            ax2.set_title("Risk Position")
            fig2.tight_layout()
            st.pyplot(fig2)

# ============================================================
# GRAPHS & INSIGHTS
# ============================================================
else:
    st.markdown("""
    <div class="hero">
        <h1>📈 Graphs & Model Insights</h1>
        <p>Visualize the model's decision factors and understand customer risk.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🌲 Random Forest Feature Importance</div>', unsafe_allow_html=True)

    feature_names = [
        "Credit Score", "Age", "Tenure", "Balance", "No. Products",
        "Has Credit Card", "Active Member", "Salary",
        "Germany", "Spain", "Male"
    ]

    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_)

        if len(importance) == len(feature_names):
            imp_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importance
            }).sort_values("Importance", ascending=True)

            fig, ax = plt.subplots(figsize=(9, 5))
            ax.barh(imp_df["Feature"], imp_df["Importance"])
            ax.set_xlabel("Importance")
            ax.set_title("Features Influencing Customer Churn")
            fig.tight_layout()
            st.pyplot(fig)

            st.markdown(f"""
            <div class="info">
                <b>Most influential feature:</b> {imp_df.iloc[-1]["Feature"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("The loaded model contains a different number of features.")
    else:
        st.info("Feature importance is not available for this model.")

    st.markdown('<div class="section-title">📌 Risk Interpretation</div>', unsafe_allow_html=True)

    a, b, c = st.columns(3)
    with a:
        st.markdown("""
        <div class="card">
            <small>LOW RISK</small>
            <strong>&lt; 30%</strong>
            <p>Customer is less likely to churn.</p>
        </div>
        """, unsafe_allow_html=True)
    with b:
        st.markdown("""
        <div class="card">
            <small>MEDIUM RISK</small>
            <strong>30% – 60%</strong>
            <p>Customer needs attention.</p>
        </div>
        """, unsafe_allow_html=True)
    with c:
        st.markdown("""
        <div class="card">
            <small>HIGH RISK</small>
            <strong>&gt; 60%</strong>
            <p>Customer may need retention action.</p>
        </div>
        """, unsafe_allow_html=True)
