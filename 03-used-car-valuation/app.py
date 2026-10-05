import os
import sys
import streamlit as st
import pandas as pd

# Add project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Used Car Dynamic Valuation Engine",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# Custom Styling
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
    .price-card {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.12) 0%, rgba(37, 99, 235, 0.05) 100%);
        border: 1px solid rgba(59, 130, 246, 0.4);
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-top: 15px;
    }
    .price-val {
        font-size: 3rem;
        font-weight: 900;
        color: #3b82f6;
    }
    .price-range {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-top: 6px;
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        background: rgba(16, 185, 129, 0.2);
        color: #10b981;
        margin-bottom: 10px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown('<div class="main-title">🚗 Pre-Owned Car Dynamic Valuation Engine</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Automotive Marketplace Pricing Engine • Advanced Ensemble Regression • Fair Market Band</div>',
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Sidebar - Quick Presets
# --------------------------------------------------
with st.sidebar:
    st.header("⚡ Quick Car Presets")
    st.write("Click to load common market examples:")
    preset = st.radio(
        "Select Vehicle Profile:",
        [
            "Custom Manual Entry",
            "Budget Hatchback (Maruti Swift 2014)",
            "Family SUV (Hyundai Creta 2018)",
            "Luxury Sedan (BMW 3 Series 2016)"
        ],
        index=0
    )
    st.markdown("---")
    st.markdown("### 🏆 Champion ML Model")
    st.info("**Model**: Random Forest Regressor\n\n**Test R²**: 97.12%\n\n**MAPE**: 14.20%\n\n**Features**: Age, KM Usage, Engine CC, BHP, Brand")

# Presets configuration
if preset == "Budget Hatchback (Maruti Swift 2014)":
    default_name = "Maruti Swift Dzire VDI"
    default_year = 2014
    default_km = 145000
    default_fuel = "Diesel"
    default_seller = "Individual"
    default_transmission = "Manual"
    default_owner = "First Owner"
    default_mileage = 23.4
    default_engine = 1248
    default_power = 74.0
    default_seats = 5
elif preset == "Family SUV (Hyundai Creta 2018)":
    default_name = "Hyundai Creta 1.6 CRDi SX"
    default_year = 2018
    default_km = 65000
    default_fuel = "Diesel"
    default_seller = "Dealer"
    default_transmission = "Manual"
    default_owner = "First Owner"
    default_mileage = 19.6
    default_engine = 1582
    default_power = 126.2
    default_seats = 5
elif preset == "Luxury Sedan (BMW 3 Series 2016)":
    default_name = "BMW 3 Series 320d Luxury Line"
    default_year = 2016
    default_km = 45000
    default_fuel = "Diesel"
    default_seller = "Dealer"
    default_transmission = "Automatic"
    default_owner = "First Owner"
    default_mileage = 22.6
    default_engine = 1995
    default_power = 190.0
    default_seats = 5
else:
    default_name = "Maruti Swift VXI"
    default_year = 2017
    default_km = 55000
    default_fuel = "Petrol"
    default_seller = "Individual"
    default_transmission = "Manual"
    default_owner = "First Owner"
    default_mileage = 20.4
    default_engine = 1197
    default_power = 81.8
    default_seats = 5

# --------------------------------------------------
# Car Specification Input Form
# --------------------------------------------------
st.subheader("📋 Enter Vehicle Specifications")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 🚘 Identity & Age")
    name = st.text_input("Car Make & Model Name", value=default_name, help="E.g., Maruti Swift, Honda City, BMW X1")
    year = st.slider("Manufacturing Year", min_value=1995, max_value=2024, value=default_year)
    km_driven = st.number_input("Odometer (Total KM Driven)", min_value=500, max_value=500000, value=default_km, step=5000)
    seats = st.selectbox("Seating Capacity", [4, 5, 6, 7, 8, 9, 10], index=[4, 5, 6, 7, 8, 9, 10].index(int(default_seats)))

with col2:
    st.markdown("#### ⚙️ Technical Specs")
    fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG"], index=["Petrol", "Diesel", "CNG", "LPG"].index(default_fuel))
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"], index=["Manual", "Automatic"].index(default_transmission))
    engine = st.number_input("Engine Displacement (CC)", min_value=600, max_value=6000, value=int(default_engine), step=50)
    max_power = st.number_input("Max Power (BHP)", min_value=30.0, max_value=600.0, value=float(default_power), step=5.0)

with col3:
    st.markdown("#### 📜 Ownership & Economy")
    mileage = st.number_input("Mileage (kmpl or km/kg)", min_value=5.0, max_value=40.0, value=float(default_mileage), step=0.5)
    seller_type = st.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark Dealer"], index=["Individual", "Dealer", "Trustmark Dealer"].index(default_seller))
    owner = st.selectbox(
        "Ownership History",
        ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"],
        index=["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"].index(default_owner)
    )

st.markdown("---")

# --------------------------------------------------
# Prediction & Valuation Display
# --------------------------------------------------
if st.button("🚀 Calculate Fair Market Valuation", type="primary", use_container_width=True):
    try:
        custom_car = CustomData(
            name=name,
            year=int(year),
            km_driven=int(km_driven),
            fuel=fuel,
            seller_type=seller_type,
            transmission=transmission,
            owner=owner,
            mileage=f"{mileage} kmpl",
            engine=f"{engine} CC",
            max_power=f"{max_power} bhp",
            seats=float(seats),
        )

        input_df = custom_car.get_data_as_data_frame()
        pipeline = PredictPipeline()
        prediction = pipeline.predict(input_df)
        fair_price = float(prediction[0])

        # 10% Market Range Band
        min_price = fair_price * 0.92
        max_price = fair_price * 1.08

        # Display Valuation Card
        st.subheader("🎯 Valuation Appraisal Summary")

        st.markdown(
            f"""
            <div class="price-card">
                <span class="badge">FAIR MARKET ESTIMATE</span>
                <div class="price-val">₹{fair_price:,.0f}</div>
                <div class="price-range">Market Valuation Band: <strong>₹{min_price:,.0f}</strong> — <strong>₹{max_price:,.0f}</strong></div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")
        c1, c2, c3 = st.columns(3)
        c1.metric("Recommended Instant Buyout (Dealer)", f"₹{fair_price * 0.90:,.0f}")
        c2.metric("Fair Consumer Retail Price", f"₹{fair_price:,.0f}")
        c3.metric("Max Listing Price (Showroom)", f"₹{fair_price * 1.08:,.0f}")

    except Exception as e:
        st.error(f"Valuation Error: {e}")
