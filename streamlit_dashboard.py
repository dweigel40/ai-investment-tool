import streamlit as st
import pandas as pd
from data_loader import load_data, load_etf_data, generate_synthetic_data
from strategy import calculate_tsm_signal, train_msm, adaptive_rebalancing, backtest_strategy
from performance import calculate_performance_metrics
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="📈 AI Investment Tool")

st.title("🧠 AI Investment Tool Dashboard")
st.markdown("Run trend-following and regime-based strategies with adaptive allocation.")

# Sidebar controls
st.sidebar.header("⚙️ Settings")
data_option = st.sidebar.selectbox("Select Data Source", ["CSV: market_data2.csv", "Yahoo Finance (SPY)", "Synthetic"])
lookback = st.sidebar.slider("TSM Lookback (months)", min_value=3, max_value=24, value=12)
regime_count = st.sidebar.selectbox("Number of Regimes (HMM)", [2, 3])
threshold = st.sidebar.slider("Regime Threshold", 0.0, 1.0, 0.6)

# Load data
if data_option == "CSV: market_data2.csv":
    df = load_data()
elif data_option == "Yahoo Finance (SPY)":
    df = load_etf_data('SPY', start='2020-01-01', end='2024-01-01')
else:
    df = generate_synthetic_data()

# Run strategy
df = calculate_tsm_signal(df, lookback=lookback)
df, msm = train_msm(df, n_states=regime_count)
df = adaptive_rebalancing(df, msm, threshold=threshold)
df = backtest_strategy(df)

# Calculate metrics
metrics = calculate_performance_metrics(df)

# Layout
col1, col2 = st.columns(2)

# 📈 Portfolio Chart
with col1:
    st.subheader("📈 Portfolio Performance")
    fig, ax = plt.subplots(figsize=(10, 4))
    df['Portfolio_Value'].plot(ax=ax, title="Portfolio Value Over Time")
    st.pyplot(fig)

# 📊 Metrics
with col2:
    st.subheader("📊 Performance Metrics")
    for k, v in metrics.items():
        st.metric(k, v)

# 📥 Download
st.download_button(
    "📥 Download Strategy Results (CSV)",
    data=df.to_csv().encode('utf-8'),
    file_name="strategy_results.csv",
    mime='text/csv'
)
