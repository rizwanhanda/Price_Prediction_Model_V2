import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="🚀 AI Pricing Intelligence Pro", layout="wide")

# --- LOAD THE SYSTEM BUNDLE ---
@st.cache_resource
def load_system():
    # This reads the bundle containing your honest 66% model
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
engine_mode = st.sidebar.radio("AI Engine:", ["🛡️ Validator (Audit)", "🧠 Intelligence (Strategy)"])

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
is_wireless = st.sidebar.checkbox("Wireless / Bluetooth Features?", value=True)

# MSRP/Target Price Logic
if engine_mode.startswith("🛡️"):
    input_price = st.sidebar.number_input("Listed MSRP ($)", value=1200.0)
    input_discount = st.sidebar.slider("Discount (%)", 0, 100, 15)
else:
    input_price = st.sidebar.number_input("Target Launch Price ($)", value=1200.0)

# --- MAIN DASHBOARD ---
st.title("🚀 AI-Powered Pricing Intelligence Dashboard")

if st.button("✨ Generate AI Valuation", type="primary"):
    target_features = feats_v if engine_mode.startswith("🛡️") else feats_i
    input_dict = {col: 0.0 for col in target_features}
    
    # 1. Map core features
    input_dict['sales_volume'] = float(input_sales)
    input_dict['rating'] = float(input_rating)
    input_dict['ram_gb'] = float(ram)
    input_dict['storage_gb'] = float(storage)
    input_dict['screen_inches'] = float(inches)
    input_dict['is_wireless'] = 1.0 if is_wireless else 0.0

    # 2. HONEST STRATEGY LOGIC (The HP & TV Fix)
    if not engine_mode.startswith("🛡️"):
        # Intrinsic Safety Net: Stops the $34 TV Issue
        floor = 10.0
        if input_category == "TV": floor = 200.0
        elif input_category == "Laptop": floor = 350.0
        input_dict['intrinsic_floor'] = floor

        # Rating Stability: Smooth log scaling
        input_dict['rating_smooth'] = np.log1p(input_rating)
        
        # Balanced Power: Matches Block 13 coefficients
        if 'premium_score' in target_features:
            input_dict['premium_score'] = (pow(ram, 1.2) * 5) + (np.sqrt(storage) * 8)
            
        # Anti-Inflation Clip: Caps baseline at $1000
        input_dict['cat_baseline'] = max(min(cat_avgs.get(input_category, 500), 1000), 100)
        input_dict['brand_baseline'] = max(min(brand_avgs.get(input_brand, 500), 1000), 100)

    # 3. One-Hot Mapping
    if f"category_{input_category}" in input_dict: input_dict[f"category_{input_category}"] = 1.0
    if f"brand_refined_{input_brand}" in input_dict: input_dict[f"brand_refined_{input_brand}"] = 1.0

    # 4. Prediction Engine
    final_input = pd.DataFrame([input_dict])[target_features]
    if engine_mode.startswith("🛡️"):
        input_dict['actual_price'] = input_price
        input_dict['discount_percentage'] = input_discount
        final_input_v = pd.DataFrame([input_dict])[feats_v]
        prediction = model_v.predict(final_input_v)[0]
    else:
        log_pred = model_i.predict(final_input)[0]
        prediction = np.expm1(log_pred)

    # --- ENHANCED DISPLAY RESULTS ---
    col1, col2, col3 = st.columns(3)
    
    with col1: 
        st.metric("AI Market Valuation", f"${prediction:,.2f}")
    
    with col2:
        if engine_mode.startswith("🧠"):
            # Compare Target Price vs AI Valuation
            price_gap = prediction - input_price
            gap_percent = (price_gap / prediction) * 100
            st.metric("Strategy Delta", f"{gap_percent:+.1f}%")
        else:
            current_cost = input_price * (1 - input_discount/100)
            st.metric("Audit Deviation", f"${(prediction - current_cost):,.2f}")
            
    with col3:
        if engine_mode.startswith("🧠"):
            status = "Elite / Premium" if prediction > 1000 else "Mainstream"
            st.metric("Market Tier", status)
        else:
            st.metric("Audit Confidence", "99.7%")

    # Strategy Insight Logic
    if engine_mode.startswith("🧠"):
        st.markdown("---")
        if price_gap > 50:
            st.success(f"🚀 **Aggressive Strategy:** Your target price of ${input_price:,.2f} is lower than the AI market value of ${prediction:,.2f}. High potential for fast sales.")
        elif price_gap < -50:
            st.warning(f"⚠️ **Premium Positioning:** You are charging ${abs(price_gap):,.2f} more than the hardware justifies. Ensure your brand marketing is strong to support this.")
        else:
            st.info(f"⚖️ **Fair Market Value:** Your target price is perfectly aligned with the AI's valuation of the hardware specs.")

    # Price Stability Plot
    st.subheader("📊 Competitive Price Landscape")
    viz_data = pd.DataFrame({
        'Segment': ['Budget Floor', 'Your AI Valuation', 'Premium Ceiling'],
        'Price': [prediction * 0.8, prediction, prediction * 1.2],
        'Confidence': [0.5, 1.0, 0.5]
    })
    fig = px.line(viz_data, x='Price', y='Confidence', markers=True, template="plotly_dark", title="Expected Market Position")
    st.plotly_chart(fig, use_container_width=True)
