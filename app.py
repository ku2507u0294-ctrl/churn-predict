import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="✈️",
    layout="centered"
)

# ── Load model ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ── Encoding maps (must match training) ────────────────────────
FREQUENT_FLYER_MAP       = {"No": 0, "Yes": 1}
ANNUAL_INCOME_MAP        = {"High Income": 0, "Low Income": 1, "Middle Income": 2}
ACCOUNT_SYNCED_MAP       = {"No": 0, "Yes": 1}
BOOKED_HOTEL_MAP         = {"No": 0, "Yes": 1}

# ── Header ─────────────────────────────────────────────────────
st.title("✈️ Customer Churn Predictor")
st.markdown("### Will this customer churn? Find out instantly.")
st.markdown("Fill in the customer details below and click **Predict**.")
st.divider()

# ── Input form ─────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=18, max_value=70, value=30, step=1)
    frequent_flyer = st.selectbox("Frequent Flyer?", options=["No", "Yes"])
    annual_income = st.selectbox(
        "Annual Income Class",
        options=["Low Income", "Middle Income", "High Income"]
    )

with col2:
    services_opted = st.slider("Services Opted", min_value=1, max_value=9, value=3, step=1)
    account_synced = st.selectbox("Account Synced to Social Media?", options=["No", "Yes"])
    booked_hotel = st.selectbox("Booked Hotel or Not?", options=["No", "Yes"])

st.divider()

# ── Prediction ─────────────────────────────────────────────────
if st.button("🔍 Predict Churn", use_container_width=True, type="primary"):

    # Build input dataframe
    input_data = pd.DataFrame([{
        "Age":                        age,
        "FrequentFlyer":              FREQUENT_FLYER_MAP[frequent_flyer],
        "AnnualIncomeClass":          ANNUAL_INCOME_MAP[annual_income],
        "ServicesOpted":              services_opted,
        "AccountSyncedToSocialMedia": ACCOUNT_SYNCED_MAP[account_synced],
        "BookedHotelOrNot":           BOOKED_HOTEL_MAP[booked_hotel],
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ **This customer is likely to CHURN**")
        st.metric("Churn Probability", f"{probability*100:.1f}%")
        st.markdown("""
        **Recommended Actions:**
        - 🎁 Offer a loyalty discount or reward
        - 📞 Schedule a customer satisfaction call
        - 🌟 Suggest premium services tailored to their profile
        """)
    else:
        st.success(f"✅ **This customer is NOT likely to churn**")
        st.metric("Churn Probability", f"{probability*100:.1f}%")
        st.markdown("""
        **Recommended Actions:**
        - 💌 Send a thank-you or appreciation message
        - 🔔 Keep them engaged with new offers
        - 📈 Upsell additional services
        """)

    # Probability bar
    st.markdown("#### Churn Probability Breakdown")
    prob_df = pd.DataFrame({
        "Outcome":     ["Not Churn", "Churn"],
        "Probability": [1 - probability, probability]
    })
    st.bar_chart(prob_df.set_index("Outcome"))

# ── Footer ─────────────────────────────────────────────────────
st.divider()
st.caption("B.Tech Gen AI – 2nd Semester | Customer Churn Prediction using Random Forest")
