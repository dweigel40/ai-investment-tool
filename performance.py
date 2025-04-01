def calculate_performance_metrics(df, return_col='Strategy_Returns'):
    import numpy as np
    returns = df[return_col].dropna()
    sharpe = (returns.mean() / returns.std()) * (252 ** 0.5)
    downside_std = returns[returns < 0].std()
    sortino = (returns.mean() / downside_std) * (252 ** 0.5)
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()
    calmar = returns.mean() * 252 / abs(max_drawdown)
    return {
        "Sharpe Ratio": round(sharpe, 3),
        "Sortino Ratio": round(sortino, 3),
        "Max Drawdown": round(max_drawdown, 3),
        "Calmar Ratio": round(calmar, 3)
    }
