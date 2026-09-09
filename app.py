import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

model = joblib.load("models/churn_model.pkl")

st.title("📊 Customer Churn Prediction")
st.caption("Predict customer churn probability and identify high-risk customers.")

st.divider()

st.subheader("Customer Information")

col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure (Months)", 0, 100, 12)
    monthly_charges = st.number_input("Monthly Charges", 0.0, 500.0, 70.0)
    total_charges = st.number_input("Total Charges", 0.0, 50000.0, 840.0)
    support_calls = st.number_input("Support Calls", 0, 50, 2)

with col2:
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        ["Credit", "Debit", "Cash", "UPI"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["Fiber", "DSL", "Unknown"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No"]
    )

st.divider()

if st.button("🔮 Predict Churn", use_container_width=True):

    customer = pd.DataFrame([{
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract": contract,
        "payment_method": payment_method,
        "internet_service": internet_service,
        "tech_support": tech_support,
        "online_security": online_security,
        "support_calls": support_calls
    }])

    probability = model.predict_proba(customer)[0, 1]

    if probability >= 0.70:
        risk = "High Risk"
        recommendation = "Prioritize this customer for a retention campaign."
    elif probability >= 0.40:
        risk = "Medium Risk"
        recommendation = "Monitor the customer and consider proactive engagement."
    else:
        risk = "Low Risk"
        recommendation = "Customer appears stable. Continue regular engagement."

    st.subheader("Prediction Result")

    result1, result2 = st.columns(2)

    with result1:
        st.metric("Churn Probability", f"{probability:.2%}")

    with result2:
        st.metric("Risk Level", risk)

    st.progress(float(probability))

    if probability >= 0.50:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is likely to stay")

    st.info(f"💡 {recommendation}")

    st.subheader("Customer Summary")

    summary = pd.DataFrame({
        "Feature": [
            "Tenure",
            "Monthly Charges",
            "Total Charges",
            "Contract",
            "Payment Method",
            "Internet Service",
            "Tech Support",
            "Online Security",
            "Support Calls"
        ],
        "Value": [
            f"{tenure} months",
            f"{monthly_charges:.2f}",
            f"{total_charges:.2f}",
            contract,
            payment_method,
            internet_service,
            tech_support,
            online_security,
            support_calls
        ]
    })

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

st.divider()

st.caption("Customer Churn Prediction | Machine Learning Portfolio Project")