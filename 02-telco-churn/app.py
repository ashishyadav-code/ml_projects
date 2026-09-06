import os
import sys
import streamlit as st
import pandas as pd

# Add project root to sys.path so src imports work reliably
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Telco Churn & Retention Radar",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# Custom CSS for Sleek Production Styling
# --------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #888888;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .card-high-risk {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(220, 38, 38, 0.05) 100%);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 10px;
        padding: 22px;
        text-align: center;
        margin-top: 15px;
    }
    .card-medium-risk {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.12) 0%, rgba(217, 119, 6, 0.05) 100%);
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-radius: 10px;
        padding: 22px;
        text-align: center;
        margin-top: 15px;
    }
    .card-low-risk {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(5, 150, 105, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 10px;
        padding: 22px;
        text-align: center;
        margin-top: 15px;
    }
    .risk-score {
        font-size: 2.8rem;
        font-weight: 900;
    }
    .recommendation-box {
        background: rgba(30, 41, 59, 0.5);
        border-left: 4px solid #3b82f6;
        padding: 15px 20px;
        border-radius: 4px;
        margin-top: 15px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown('<div class="main-title">📡 Telco Customer Churn & Retention Radar</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Production ML Intelligence • Imbalanced Classification • Retention ROI Engine</div>',
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Sidebar - Quick Presets
# --------------------------------------------------
with st.sidebar:
    st.header("⚡ Customer Presets")
    st.write("Load sample profiles to quickly test edge cases:")
    preset = st.radio(
        "Select Profile:",
        ["Custom Manual Input", "High-Risk Customer (Flight Risk)", "Loyal Long-Term Customer"],
        index=0
    )
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.info("**Champion Model**: Gradient Boosting\n\n**Evaluated Metrics**: ROC-AUC, Recall, Precision, F1-Score\n\n**Target**: Churn (Yes/No)")

# Preset defaults
if preset == "High-Risk Customer (Flight Risk)":
    default_gender = "Female"
    default_senior = 0
    default_partner = "No"
    default_dependents = "No"
    default_tenure = 2
    default_phone = "Yes"
    default_multiple = "No"
    default_internet = "Fiber optic"
    default_security = "No"
    default_backup = "No"
    default_protection = "No"
    default_tech = "No"
    default_tv = "Yes"
    default_movies = "No"
    default_contract = "Month-to-month"
    default_paperless = "Yes"
    default_payment = "Electronic check"
    default_monthly = 85.50
    default_total = 171.00
elif preset == "Loyal Long-Term Customer":
    default_gender = "Male"
    default_senior = 0
    default_partner = "Yes"
    default_dependents = "Yes"
    default_tenure = 60
    default_phone = "Yes"
    default_multiple = "Yes"
    default_internet = "DSL"
    default_security = "Yes"
    default_backup = "Yes"
    default_protection = "Yes"
    default_tech = "Yes"
    default_tv = "Yes"
    default_movies = "Yes"
    default_contract = "Two year"
    default_paperless = "No"
    default_payment = "Bank transfer (automatic)"
    default_monthly = 65.00
    default_total = 3900.00
else:
    default_gender = "Male"
    default_senior = 0
    default_partner = "No"
    default_dependents = "No"
    default_tenure = 12
    default_phone = "Yes"
    default_multiple = "No"
    default_internet = "Fiber optic"
    default_security = "No"
    default_backup = "Yes"
    default_protection = "No"
    default_tech = "No"
    default_tv = "No"
    default_movies = "No"
    default_contract = "Month-to-month"
    default_paperless = "Yes"
    default_payment = "Electronic check"
    default_monthly = 70.00
    default_total = 840.00

# --------------------------------------------------
# Input Form - Organized in 3 Clean Columns
# --------------------------------------------------
st.subheader("📋 Customer Profile & Account Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 👤 Demographics")
    gender = st.selectbox("Gender", ["Male", "Female"], index=0 if default_gender == "Male" else 1)
    senior = st.selectbox("Senior Citizen", [0, 1], index=default_senior, format_func=lambda x: "Yes" if x == 1 else "No")
    partner = st.selectbox("Has Partner?", ["Yes", "No"], index=0 if default_partner == "Yes" else 1)
    dependents = st.selectbox("Has Dependents?", ["Yes", "No"], index=0 if default_dependents == "Yes" else 1)
    tenure = st.slider("Tenure (Months with Company)", min_value=0, max_value=72, value=default_tenure)

with col2:
    st.markdown("#### 🌐 Subscribed Services")
    phone_service = st.selectbox("Phone Service", ["Yes", "No"], index=0 if default_phone == "Yes" else 1)
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], index=["No", "Yes", "No phone service"].index(default_multiple))
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], index=["DSL", "Fiber optic", "No"].index(default_internet))
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(default_security))
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(default_backup))
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(default_protection))
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(default_tech))

with col3:
    st.markdown("#### 💳 Billing & Contract")
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(default_tv))
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(default_movies))
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(default_contract))
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"], index=0 if default_paperless == "Yes" else 1)
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(default_payment)
    )
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=150.0, value=float(default_monthly), step=1.0)
    total_charges = st.number_input("Total Lifetime Charges ($)", min_value=0.0, max_value=10000.0, value=float(default_total), step=10.0)

st.markdown("---")

# --------------------------------------------------
# Predict Action & Business Matrix
# --------------------------------------------------
if st.button("🚀 Analyze Churn Risk & Generate Retention Strategy", type="primary", use_container_width=True):
    try:
        # 1. Package custom input
        customer = CustomData(
            gender=gender,
            SeniorCitizen=senior,
            Partner=partner,
            Dependents=dependents,
            tenure=int(tenure),
            PhoneService=phone_service,
            MultipleLines=multiple_lines,
            InternetService=internet_service,
            OnlineSecurity=online_security,
            OnlineBackup=online_backup,
            DeviceProtection=device_protection,
            TechSupport=tech_support,
            StreamingTV=streaming_tv,
            StreamingMovies=streaming_movies,
            Contract=contract,
            PaperlessBilling=paperless,
            PaymentMethod=payment_method,
            MonthlyCharges=float(monthly_charges),
            TotalCharges=float(total_charges),
        )

        input_df = customer.get_data_as_data_frame()

        # 2. Run Inference
        pipeline = PredictPipeline()
        prediction, probability = pipeline.predict(input_df)

        churn_risk_pct = probability[0] * 100
        is_churn = prediction[0] == 1

        # 3. Display Risk Card
        st.subheader("🎯 Risk Assessment & Radar Output")

        res_col1, res_col2 = st.columns([1, 1.4])

        with res_col1:
            if churn_risk_pct >= 60:
                st.markdown(
                    f"""
                    <div class="card-high-risk">
                        <h4 style="color: #ef4444; margin:0;">⚠️ HIGH FLIGHT RISK</h4>
                        <div class="risk-score" style="color: #ef4444;">{churn_risk_pct:.1f}%</div>
                        <p style="color: #cbd5e1; margin-top:5px;">Immediate retention intervention required.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            elif churn_risk_pct >= 35:
                st.markdown(
                    f"""
                    <div class="card-medium-risk">
                        <h4 style="color: #f59e0b; margin:0;">⚠️ MODERATE CHURN RISK</h4>
                        <div class="risk-score" style="color: #f59e0b;">{churn_risk_pct:.1f}%</div>
                        <p style="color: #cbd5e1; margin-top:5px;">Customer needs proactive engagement & feature adoption.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="card-low-risk">
                        <h4 style="color: #10b981; margin:0;">✅ LOW RISK (LOYAL)</h4>
                        <div class="risk-score" style="color: #10b981;">{churn_risk_pct:.1f}%</div>
                        <p style="color: #cbd5e1; margin-top:5px;">Healthy customer relationship. Upsell potential.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with res_col2:
            st.markdown("#### 💼 Recommended Retention Actions")
            actions = []

            if contract == "Month-to-month":
                actions.append("• **Contract Upgrade**: Lock in customer with a 1-Year or 2-Year contract by offering a 15% discount.")
            if tech_support == "No" and internet_service != "No":
                actions.append("• **Tech Support Incentive**: Offer 3 months of FREE Tech Support (proven to slash churn by 60%).")
            if payment_method == "Electronic check":
                actions.append("• **Automated Billing**: Encourage Auto-pay (Credit Card / Bank Transfer) with a $10 one-time bill credit.")
            if internet_service == "Fiber optic" and monthly_charges > 80:
                actions.append("• **Plan Optimization**: Customer has high charges on Fiber Optic. Review bandwidth needs or bundle security add-ons.")
            if tenure <= 6:
                actions.append("• **Onboarding Concierge**: Schedule a VIP onboarding check-in call to ensure satisfaction during high-risk initial months.")

            if not actions:
                actions.append("• Customer is well-retained! Eligible for loyalty perks or referral program invitations.")

            st.markdown(
                f"""
                <div class="recommendation-box">
                    {"<br>".join(actions)}
                </div>
                """,
                unsafe_allow_html=True
            )

        # --------------------------------------------------
        # Dollar Retention Matrix
        # --------------------------------------------------
        st.markdown("---")
        st.subheader("💰 C-Suite Dollar Retention Cost Calculator")

        calc_col1, calc_col2, calc_col3 = st.columns(3)

        annual_revenue = monthly_charges * 12
        retention_offer_cost = annual_revenue * 0.15  # 15% annual discount
        net_retained_value = annual_revenue - retention_offer_cost

        calc_col1.metric("Annual Customer Revenue (CLV)", f"${annual_revenue:,.2f}")
        calc_col2.metric("Cost of 15% Retention Incentive", f"${retention_offer_cost:,.2f}")
        calc_col3.metric("Net Annual Value Saved", f"${net_retained_value:,.2f}")

    except Exception as e:
        st.error(f"Prediction Pipeline Error: {e}")
