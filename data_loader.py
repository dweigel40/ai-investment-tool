import pandas as pd
import yfinance as yf
import numpy as np

def load_data(file_path='market_data2.csv'):
    df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    df['Returns'] = df['Price'].pct_change()
    return df.dropna()

def load_etf_data(ticker='SPY', start='2020-01-01', end='2024-01-01'):
    df = yf.download(ticker, start=start, end=end)
    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    df = df[[price_col]].rename(columns={price_col: 'Price'})
    df['Returns'] = df['Price'].pct_change()
    df = df.dropna()
    df.to_csv('market_data2.csv')
    print(f"✅ Downloaded ETF data for {ticker}")
    return df

def generate_synthetic_data(days=252):
    np.random.seed(42)
    prices = np.cumprod(1 + np.random.normal(0.0005, 0.01, days)) * 100
    df = pd.DataFrame({'Date': pd.date_range(end=pd.Timestamp.today(), periods=days), 'Price': prices})
    df.set_index('Date', inplace=True)
    df['Returns'] = df['Price'].pct_change()
    return df.dropna()
