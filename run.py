from data_loader import load_data
from strategy import calculate_tsm_signal, train_msm, adaptive_rebalancing, backtest_strategy
from performance import calculate_performance_metrics

print("📊 Starting backtest...")
df = load_data()
df = calculate_tsm_signal(df, lookback=12)
df, msm = train_msm(df)
df = adaptive_rebalancing(df, msm)
df = backtest_strategy(df)

metrics = calculate_performance_metrics(df)
print("\n📈 Performance Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v}")

df.to_csv("strategy_results.csv")
print("\n✅ Results saved to strategy_results.csv")
