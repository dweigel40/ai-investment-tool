import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

def calculate_tsm_signal(df, lookback=12):
    df[f'TSM_{lookback}m'] = np.where(df['Returns'].rolling(lookback).sum() > 0, 1, 0)
    return df

def train_msm(df, n_states=2):
    model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=1000)
    df = df.dropna(subset=['Returns'])
    model.fit(df['Returns'].values.reshape(-1, 1))
    df['Regime'] = model.predict(df['Returns'].values.reshape(-1, 1))
    return df, model

def adaptive_rebalancing(df, msm_model, threshold=0.6):
    probs = msm_model.predict_proba(df['Returns'].values.reshape(-1, 1))
    df['Bull_Prob'] = probs[:, 0]
    df['Bear_Prob'] = probs[:, 1] if probs.shape[1] > 1 else 0
    df['Allocation'] = np.where(df['Bull_Prob'] > threshold, 1.0,
                         np.where(df['Bear_Prob'] > threshold, 0.2, 0.5))
    df['Adaptive_Portfolio'] = df['Allocation'] * df[f'TSM_12m']
    return df

def backtest_strategy(df, signal_column='Adaptive_Portfolio', initial_capital=10000):
    df['Strategy_Returns'] = df[signal_column] * df['Returns']
    df['Portfolio_Value'] = initial_capital * (1 + df['Strategy_Returns']).cumprod()
    return df
