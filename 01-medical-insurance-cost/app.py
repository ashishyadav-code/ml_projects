import os
import sys
import streamlit as st
import pandas as pd

# Add current project root to sys.path so src imports work cleanly
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.Pipeline.predict_pipeline import CustomData, PredictPipeline

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #888888;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(59, 130, 246, 0.1) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        margin-top: 20px;
    }
    .metric-val {
        font-size: 2.5rem;
        font-weight: 900;
        color: #10B981;
    }
    .metric-sub {
        font-size: 0.9rem;
        color: #888888;
        margin-top: 4px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown('<div class="main-title">🏥 Medical Insurance Cost Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Production ML Pipeline • End-to-End Actuarial Health Risk Valuation</div>',
    unsafe_allow_html=True,
)

st.write("Enter the applicant's demographic and lifestyle health details below to get an instant annual insurance premium estimate.")

# --------------------------------------------------
# User Inputs
# --------------------------------------------------
with st.form("insurance_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Applicant Age (Years)", min_value=18, max_value=65, value=30, step=1)
        sex = st.selectbox("Biological Sex", options=["female", "male"], index=0)
        bmi = st.number_input(
            "Body Mass Index (BMI)",
            min_value=15.0,
            max_value=55.0,
            value=26.5,
            step=0.1,
            help="Normal BMI: 18.5 - 24.9 | Overweight: 25.0 - 29.9 | Obese: 30.0+",
        )

        # Dynamic BMI Category Badge
        if bmi < 18.5:
            st.caption("ℹ️ BMI Category: **Underweight** (< 18.5)")
        elif bmi < 25.0:
            st.caption("✅ BMI Category: **Normal Weight** (18.5 – 24.9)")
        elif bmi < 30.0:
            st.caption("⚠️ BMI Category: **Overweight** (25.0 – 29.9)")
        else:
            st.caption("🚨 BMI Category: **Obese** (30.0+)")

    with col2:
        children = st.slider("Number of Dependent Children", min_value=0, max_value=5, value=0, step=1)
        smoker = st.selectbox("Smoking Status", options=["no", "yes"], index=0, help="Smoking significantly impacts medical actuarial charges.")
        region = st.selectbox(
            "Residential Region",
            options=["southwest", "southeast", "northwest", "northeast"],
            index=2,
        )

    submit_button = st.form_submit_button("🔮 Calculate Insurance Premium", use_container_width=True)

# --------------------------------------------------
# Prediction & Results Display
# --------------------------------------------------
if submit_button:
    with st.spinner("Processing features through ML Pipeline..."):
        try:
            # 1. Package input into CustomData object
            applicant_data = CustomData(
                age=age,
                sex=sex,
                bmi=float(bmi),
                children=children,
                smoker=smoker,
                region=region,
            )

            # 2. Convert to DataFrame
            input_df = applicant_data.get_data_as_data_frame()

            # 3. Predict using the PredictPipeline
            pipeline = PredictPipeline()
            predicted_charge = pipeline.predict(input_df)[0]
            monthly_cost = predicted_charge / 12.0

            # 4. Display Results
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="font-size: 0.85rem; font-weight: 700; text-transform: uppercase; color: #888888; letter-spacing: 0.08em;">ESTIMATED ANNUAL PREMIUM</div>
                    <div class="metric-val">${predicted_charge:,.2f}</div>
                    <div class="metric-sub">Approx. <strong>${monthly_cost:,.2f} / month</strong> billed annually</div>
                </div>
            """,
                unsafe_allow_html=True,
            )

            # 5. Risk Factor Insights
            st.markdown("### 📊 Actuarial Risk Analysis")
            if smoker == "yes":
                st.warning("🚬 **High Impact Factor**: Smoking status is the primary cost driver, multiplying baseline risk by ~3x to 4x.")
            else:
                st.info("🚭 **Non-Smoker Benefit**: Applicant qualifies for preferred baseline risk tiers.")

            if bmi >= 30.0 and smoker == "yes":
                st.error("⚠️ **Compounded Risk**: High BMI combined with smoking creates an exponential risk multiplier.")
            elif bmi >= 30.0:
                st.warning("⚖️ **Elevated BMI**: BMI over 30 falls in the clinical obesity range, adding moderate premium loading.")
            else:
                st.success("💪 **Healthy BMI**: BMI is within standard actuarial limits.")

        except Exception as e:
            st.error(f"Prediction Error: {e}")

# Footer
st.markdown("---")
st.caption("Built with Scikit-Learn Pipeline, Random Forest, & Streamlit • Portfolio Project 01")
