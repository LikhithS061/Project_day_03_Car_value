"""
streamlit_app.py — Minimalist Premium Used Car Price Prediction UI
==================================================================
Launch: streamlit run app/streamlit_app.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from src.predict import predict_price, load_model

# ──────────────────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CarValue",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────
# Minimalist CSS
# ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    /* Global */
    .stApp {
        background-color: #121212;
        font-family: 'Inter', sans-serif;
        color: #E0E0E0;
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Title styling */
    .main-title {
        text-align: center;
        padding: 4rem 0 3rem 0;
    }
    .main-title h1 {
        font-size: 2.2rem;
        font-weight: 300;
        color: #FFFFFF;
        margin-bottom: 0.5rem;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    .main-title p {
        color: #666666;
        font-size: 0.95rem;
        font-weight: 300;
        letter-spacing: 0.5px;
    }

    /* Minimalist card styling overriden by native st.container */
    .st-emotion-cache-1629p8f {
        /* Styling for the native container border if needed */
        background-color: #1A1A1A;
        border: 1px solid #2C2C2C !important;
        border-radius: 4px;
    }

    /* Section label */
    .section-label {
        color: #FFFFFF;
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 2rem;
        border-bottom: 1px solid #1A1A1A;
        padding-bottom: 0.75rem;
    }

    /* Result card */
    .result-card {
        background: #1A1A1A;
        border: 1px solid #333333;
        border-radius: 4px;
        padding: 3.5rem;
        text-align: center;
        margin: 2rem 0;
        animation: fadeIn 0.8s ease-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .result-label {
        color: #666666;
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 1rem;
    }
    .result-price {
        font-size: 4rem;
        font-weight: 300;
        color: #FFFFFF;
        letter-spacing: -2px;
        margin: 0.5rem 0;
    }
    .result-note {
        color: #444444;
        font-size: 0.85rem;
        margin-top: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Metric cards */
    .metrics-row {
        display: flex;
        gap: 1.5rem;
        margin-top: 1.5rem;
    }
    .metric-card {
        flex: 1;
        background: transparent;
        border: 1px solid #1A1A1A;
        border-radius: 4px;
        padding: 1.5rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 300;
        color: #FFFFFF;
    }
    .metric-label {
        color: #555555;
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 0.5rem;
    }

    /* Input styling */
    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #888888 !important;
        font-weight: 400 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    div[data-baseweb="select"] > div, .stNumberInput input {
        background-color: #1A1A1A !important;
        border: 1px solid #2C2C2C !important;
        color: #FFFFFF !important;
        border-radius: 2px !important;
        font-size: 0.95rem !important;
    }

    /* Button styling */
    .stButton > button {
        width: 100%;
        background: #FFFFFF !important;
        color: #000000 !important;
        border: none !important;
        border-radius: 2px !important;
        padding: 1rem 2rem !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        background: #CCCCCC !important;
        color: #000000 !important;
    }

    /* Alert / Info styling */
    .stAlert {
        border-radius: 2px !important;
        border: 1px solid #1A1A1A !important;
        background-color: transparent !important;
        color: #888888 !important;
    }
    
    .stAlert p {
        font-weight: 300 !important;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        color: #333333;
        font-size: 0.70rem;
        padding: 4rem 0 2rem 0;
        font-weight: 400;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    /* Expander & Tabs styling */
    .streamlit-expanderHeader {
        color: #888888 !important;
        font-weight: 400 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .streamlit-expanderContent {
        color: #777777 !important;
        font-weight: 300 !important;
        border: 1px solid #2C2C2C !important;
        border-top: none !important;
        background-color: #1A1A1A !important;
    }
    hr {
        border-color: #2C2C2C !important;
    }
    
    /* Tabs */
    button[data-baseweb="tab"] {
        background-color: transparent !important;
        margin-right: 1rem !important;
        padding-bottom: 0.5rem !important;
        border-bottom: 2px solid transparent !important;
    }
    button[data-baseweb="tab"]:focus {
        outline: none !important;
    }
    button[data-baseweb="tab"] > div {
        color: #666666 !important;
        font-weight: 400 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.85rem !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 2px solid #FFFFFF !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] > div {
        color: #FFFFFF !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────
st.markdown("""
<div class="main-title">
    <h1>CarValue</h1>
    <p>Seeee.. Kar Value from ML</p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# Check if model exists
# ──────────────────────────────────────────────────────────
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "car_price_model.pkl")
if not os.path.exists(model_path):
    st.error("Model not found. Please run the training script.")
    st.stop()

artifact = load_model(model_path)
metrics = artifact.get("metrics", {})


# ──────────────────────────────────────────────────────────
# Application Tabs
# ──────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Valuation Estimator", "Model Evaluation Metrics"])

with tab1:
    # ──────────────────────────────────────────────────────────
    # Input Form
    # ──────────────────────────────────────────────────────────

    # 1. Primary Anchor Section
    with st.container(border=True):
        st.markdown('<div class="section-label">Market Anchor</div>', unsafe_allow_html=True)

        st.info("The original manufacturer list price serves as the primary anchor for our depreciation prediction algorithm.")

        present_price = st.number_input(
            "List Price New (In Lakhs ₹)",
            min_value=0.1,
            max_value=100.0,
            value=8.0,
            step=0.5,
        )

    # 2. Vehicle Details Section
    with st.container(border=True):
        st.markdown('<div class="section-label">Specification & Usage</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

    with col1:
        year = st.slider(
            "Manufacturing Year",
            min_value=2000,
            max_value=2026,
            value=2018,
            step=1,
        )
        fuel_type = st.selectbox(
            "Fuel Type",
            options=["Petrol", "Diesel", "CNG"],
            index=0,
        )
        transmission = st.selectbox(
            "Transmission",
            options=["Manual", "Automatic"],
            index=0,
        )

        with col2:
            kms_driven = st.number_input(
                "Distance Driven (Km)",
                min_value=0,
                max_value=500000,
                value=30000,
                step=1000,
            )
            seller_type = st.selectbox(
                "Seller Type",
                options=["Dealer", "Individual"],
                index=0,
            )
            owner = st.selectbox(
                "Previous Owners",
                options=[0, 1, 2, 3],
                index=0,
            )


    # ──────────────────────────────────────────────────────────
    # Prediction
    # ──────────────────────────────────────────────────────────
    if st.button("Calculate Valuation", use_container_width=True):
        with st.spinner("Executing inference model..."):
            try:
                predicted = predict_price(
                    year=year,
                    present_price=present_price,
                    kms_driven=kms_driven,
                    fuel_type=fuel_type,
                    seller_type=seller_type,
                    transmission=transmission,
                    owner=owner,
                )

                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">Estimated Valuation</div>
                    <div class="result-price">₹{predicted:.2f}L</div>
                    <div class="result-note">
                        {year} • {fuel_type} • {transmission} • {kms_driven:,} km
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Depreciation insight
                if present_price > 0:
                    retention = (predicted / present_price) * 100
                    depreciation = 100 - retention
                    if depreciation > 0:
                        st.info(f"📉 The estimated resale value is **{depreciation:.1f}% less** than buying the car brand new.")
                    else:
                        st.success(f"📈 The estimated resale value is **{abs(depreciation):.1f}% higher** than its original new price.")

            except Exception as e:
                st.error(f"Inference failed: {str(e)}")


with tab2:
    # ──────────────────────────────────────────────────────────
    # Model Metrics Display
    # ──────────────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown('<div class="section-label">Winning Model Metrics (' + artifact.get("model_name", "Best Model") + ')</div>', unsafe_allow_html=True)
        
        if metrics:
            st.markdown("""
            <div class="metrics-row" style="margin-top: 0; margin-bottom: 2rem;">
                <div class="metric-card">
                    <div class="metric-value">{:.3f}</div>
                    <div class="metric-label">Test R² Score</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{:.2f}L</div>
                    <div class="metric-label">Test RMSE</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{:.2f}L</div>
                    <div class="metric-label">Test MAE</div>
                </div>
            </div>
            """.format(
                metrics.get("test_r2", 0),
                metrics.get("test_rmse", 0),
                metrics.get("test_mae", 0),
            ), unsafe_allow_html=True)
    
    # ──────────────────────────────────────────────────────────
    # Comparative Matrix
    # ──────────────────────────────────────────────────────────
    all_results = artifact.get("all_results", {})
    if all_results:
        import pandas as pd
        st.markdown('<div class="section-label">Algorithm Comparison Matrix</div>', unsafe_allow_html=True)
        
        # Build dataframe
        results_list = []
        for model_name, res in all_results.items():
            results_list.append({
                "Algorithm": model_name,
                "Test R² Score": f"{res.get('r2', 0):.4f}",
                "Cross-Val R² (K=5)": f"{res.get('cv_r2', 0):.4f}",
                "RMSE (Lakhs)": f"₹{res.get('rmse', 0):.2f}L",
                "MAE (Lakhs)": f"₹{res.get('mae', 0):.2f}L",
            })
            
        df_comp = pd.DataFrame(results_list)
        
        # Style dataframe for the dark theme
        st.dataframe(
            df_comp, 
            hide_index=True, 
            use_container_width=True,
        )
        
        st.markdown("""
        <div style="font-size: 0.8rem; color: #666; margin-top: 1rem; text-align: center;">
        <i>The application already set to use the best algorithm with the highest Test R² Score.</i>
        </div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# Footer (Removed)
# ──────────────────────────────────────────────────────────

