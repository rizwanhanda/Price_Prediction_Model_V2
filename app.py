import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import numpy as np
from scipy.stats import norm

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="🚀 AI Pricing Intelligence Pro", layout="wide")

# --- LOAD THE SYSTEM BUNDLE ---
@st.cache_resource
def load_system():
    # This reads the bundle containing your stable generalized model
    with open('project_bundle.pkl', 'rb') as f:
        return pickle.load(f)

try:
    bundle = load_system()
    model_v = bundle['model_validator']
    model_i = bundle['model_intel']
    feats_v = bundle['features_validator']
    feats_i = bundle['features_intel']
    cat_options = bundle['category_options']
    brand_options = bundle['brand_options']
    cat_avgs = bundle.get('cat_averages', {})
    brand_avgs = bundle.get('brand_averages', {})
    
    if not cat_avgs:
        st.error("⚠️ Data Sync Error: Please re-run Block 15 in Colab.")
        st.stop()
        
except Exception as e:
    st.error(f"⚠️ System Error: {e}. Ensure 'project_bundle.pkl' is on GitHub.")
    st.stop()

# --- SIDEBAR: THE CONTROL PANEL ---
st.sidebar.title("🎛️ Pricing Control Panel")
engine_mode = st.sidebar.radio("AI Engine:", ["🛡️ Validator (Audit Mode)", "🧠 Intelligence (Simulation Mode)"])

st.sidebar.markdown("---")
input_category = st.sidebar.selectbox("Category", cat_options)
input_brand = st.sidebar.selectbox("Brand", brand_options + ["Other"])
input_rating = st.sidebar.slider("Customer Rating", 1.0, 5.0, 4.2, step=0.1)
input_sales = st.sidebar.number_input("Est. Monthly Sales", min_value=0, value=1000, step=100)

# Hardware Specs UI
st.sidebar.subheader("📐 Hardware Specs")
ram = st.sidebar.number_input("RAM (GB)", 0, 128, 32, step=4) if input_category == "Laptop" else 0
storage = st.sidebar.number_input("Storage (GB)", 0, 2048, 1024, step=64) if input_category == "Laptop" else 0
inches = st.sidebar.number_input("Screen Size (Inches)", 0.0, 100.0, 16.0, step=0.5) if input_category in ["Laptop", "Monitor", "TV"] else 0.0
is_wireless = st.sidebar.checkbox("Wireless Features?", value=True)

if "🛡️" in engine_mode:
    input_price = st.sidebar.number_input("Listed MSRP ($)", value=1200.0)
    input_discount = st.sidebar.slider("Discount (%)", 0, 100, 15)
else:
    input_price = st.sidebar.number_input("Target Launch Price ($)", value=1200.0)

# --- MAIN DASHBOARD ---
st.title("🚀 AI-Powered Pricing Intelligence Dashboard")

if st.button("✨ Generate AI Valuation", type="primary"):
    target_features = feats_v if "🛡️" in engine_mode else feats_i
    input_dict = {col: 0.0 for col in target_features}
    
    # 1. Core Feature Mapping
    input_dict['sales_volume'] = float(input_sales)
    input_dict['rating'] = float(input_rating)
    input_dict['ram_gb'] = float(ram)
    input_dict['storage_gb'] = float(storage)
    input_dict['screen_inches'] = float(inches)
    input_dict['is_wireless'] = 1.0 if is_wireless else 0.0

    # 2. OBJECTIVE SIMULATION LOGIC
    if "🧠" in engine_mode:
        # Intrinsic Floors (Preventing undervaluation)
        floor = 10.0
        if input_category == "TV": floor = 200.0
        elif input_category == "Laptop": floor = 350.0
        input_dict['intrinsic_floor'] = floor

        # Log-Smoothing for Rating Stability
        input_dict['rating_smooth'] = np.log1p(input_rating)
        
        # Premium Hardware Scaling
        if 'premium_score' in target_features:
            input_dict['premium_score'] = (pow(ram, 1.2) * 5) + (np.sqrt(storage) * 8)
            
        # Baseline Clipping ($1000 threshold)
        input_dict['cat_baseline'] = max(min(cat_avgs.get(input_category, 500), 1000), 100)
        input_dict['brand_baseline'] = max(min(brand_avgs.get(input_brand, 500), 1000), 100)

    # 3. Categorical Encoding
    if f"category_{input_category}" in input_dict: input_dict[f"category_{input_category}"] = 1.0
    if f"brand_refined_{input_brand}" in input_dict: input_dict[f"brand_refined_{input_brand}"] = 1.0

    # 4. Inference Engine
    if "🛡️" in engine_mode:
        input_dict['actual_price'] = input_price
        input_dict['discount_percentage'] = input_discount
        final_input = pd.DataFrame([input_dict])[feats_v]
        prediction = model_v.predict(final_input)[0]
    else:
        final_input = pd.DataFrame([input_dict])[feats_i]
        log_pred = model_i.predict(final_input)[0]
        prediction = np.expm1(log_pred)

    # --- DISPLAY METRICS ---
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("AI Market Valuation", f"${prediction:,.2f}")
    with col2:
        if "🧠" in engine_mode:
            delta = ((prediction - input_price) / prediction) * 100
            st.metric("Strategy Delta", f"{delta:+.1f}%")
        else:
            current = input_price * (1 - input_discount/100)
            st.metric("Audit Deviation", f"${(prediction - current):,.2f}")
    with col3:
        status = "Premium / Elite" if prediction > 1000 else "Standard / Mainstream"
        st.metric("Market Tier", status)

    # --- PRICE CONFIDENCE ZONE (The Bell Curve Fix) ---
    st.markdown("---")
    st.subheader("📊 Price Confidence & Market Positioning")
    
    # Gaussian distribution math
    mu, sigma = prediction, prediction * 0.12  # 12% standard deviation
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    y = norm.pdf(x, mu, sigma)
    y = y / max(y)  # Normalize to 0-1 scale

    fig_df = pd.DataFrame({'Price': x, 'Confidence': y})
    fig = px.area(fig_df, x='Price', y='Confidence', 
                  title="Probabilistic AI Valuation Zone",
                  template="plotly_dark",
                  color_discrete_sequence=['#636EFA'])

    # Comparison Lines
    fig.add_vline(x=prediction, line_dash="solid", line_color="#00ff00", 
                  annotation_text="AI Value", annotation_position="top right")
    if "🧠" in engine_mode:
        fig.add_vline(x=input_price, line_dash="dash", line_color="#ff4b4b", 
                      annotation_text="Your Target", annotation_position="top left")

    st.plotly_chart(fig, use_container_width=True)

    # Strategy Insight
    if "🧠" in engine_mode:
        if delta > 10: st.success("🚀 **Aggressive Entry:** Your target is below hardware value. High sales potential.")
        elif delta < -10: st.warning("⚠️ **Premium Strategy:** Your price exceeds hardware value. Requires strong branding.")
        else: st.info("⚖️ **Fair Value:** Target price is perfectly aligned with hardware specifications.")
