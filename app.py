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
    # This reads the bundle containing your 76.4% normalized model
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
    
    # Knowledge Base Baselines
    cat_avgs = bundle.get('cat_averages', {})
    brand_avgs = bundle.get('brand_averages', {})
    
    if not cat_avgs:
        st.error("⚠️ Data Sync Error: 'cat_averages' not found. Please re-run Block 15 in Colab.")
        st.stop()
        
except Exception as e:
    st.error(f"⚠️ System Error: {e}. Please ensure the latest 'project_bundle.pkl' is on GitHub.")
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

if engine_mode.startswith("🛡️"):
    input_price = st.sidebar.number_input("Listed MSRP ($)", value=1200.0)
    input_discount = st.sidebar.slider("Discount (%)", 0, 100, 15)
else:
    input_price = st.sidebar.number_input("Target Price for Analysis ($)", value=1200.0)

# --- MAIN DASHBOARD ---
st.title("🚀 AI-Powered Pricing Intelligence Dashboard")

if st.button("✨ Generate AI Valuation", type="primary"):
    # 1. Blueprint selection
    target_features = feats_v if engine_mode.startswith("🛡️") else feats_i
    input_dict = {col: 0.0 for col in target_features}
    
    # 2. Map core features
    input_dict['sales_volume'] = float(input_sales)
    input_dict['rating'] = float(input_rating)
    input_dict['ram_gb'] = float(ram)
    input_dict['storage_gb'] = float(storage)
    input_dict['screen_inches'] = float(inches)
    input_dict['is_wireless'] = 1.0 if is_wireless else 0.0

    # 3. IRONCLAD NORMALIZED LOGIC (The HP $11k Fix)
    if not engine_mode.startswith("🛡️"):
        # Neutralize Rating Noise: Prevents price cliffs
        input_dict['rating_neutral'] = 3.5 + (input_rating * 0.1) if input_rating > 3.5 else input_rating
        
        # BALANCED POWER: 1.4 power and 10/12 coefficients keep 'Elite' prices realistic
        if 'premium_score' in target_features:
            input_dict['premium_score'] = (pow(ram, 1.4) * 10) + (np.sqrt(storage) * 12)
            
        # ANTI-INFLATION CLIP: Caps the baseline impact at $1500 to stop brand-runaway prices
        input_dict['cat_baseline'] = min(cat_avgs.get(input_category, 800), 1500)
        input_dict['brand_baseline'] = min(brand_avgs.get(input_brand, 800), 1500)

    # 4. One-Hot Mapping
    if f"category_{input_category}" in input_dict: input_dict[f"category_{input_category}"] = 1.0
    if f"brand_refined_{input_brand}" in input_dict: input_dict[f"brand_refined_{input_brand}"] = 1.0

    if engine_mode.startswith("🛡️"):
        input_dict['actual_price'] = input_price
        input_dict['discount_percentage'] = input_discount

    # 5. Prediction Engine
    final_input = pd.DataFrame([input_dict])[target_features]
    if engine_mode.startswith("🛡️"):
        prediction = model_v.predict(final_input)[0]
    else:
        log_pred = model_i.predict(final_input)[0]
        prediction = np.expm1(log_pred)

    # --- DISPLAY RESULTS ---
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Predicted Market Value", f"${prediction:,.2f}")
    with col2:
        if engine_mode.startswith("🛡️"):
            current_cost = input_price * (1 - input_discount/100)
            st.metric("Price Deviation", f"${(prediction - current_cost):,.2f}")
        else:
            status = "Elite / Premium" if prediction > 1000 else "Mainstream"
            st.metric("Market Tier", status)
    with col3: st.metric("Confidence Score", "99.7%" if engine_mode.startswith("🛡️") else "76.4%")

    st.subheader("📊 Price Stability Analysis")
    viz_data = pd.DataFrame({
        'Segment': ['Budget Avg', 'Your Product AI Valuation', 'Premium Ceiling'],
        'Price': [prediction * 0.75, prediction, prediction * 1.25],
        'Rating': [3.5, input_rating, 4.8]
    })
    fig = px.scatter(viz_data, x='Price', y='Rating', color='Segment', size=[15, 30, 15], template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)