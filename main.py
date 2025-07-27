import os
import json
from functools import reduce
import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from utility import remove_think
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq

# Portfolio optimization imports
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns

load_dotenv()
MODEL_NAME = os.getenv("MODEL_NAME", "qwen/qwen3-32b")
folder_path = 'Predicted_performance'

llm = ChatGroq(model=MODEL_NAME)

def load_returns():
    actual_dfs, pred_dfs = [], []
    for filename in sorted(os.listdir(folder_path)):
        if not filename.lower().endswith('.csv'):
            continue
        symbol = filename.split('_')[0].lower()
        df = pd.read_csv(os.path.join(folder_path, filename), parse_dates=['Date']).sort_values('Date')
        actual_dfs.append(df[['Date', 'Actual']].rename(columns={'Actual': symbol}))
        pred_dfs.append(df[['Date', 'Predicted']].rename(columns={'Predicted': symbol}))
    def merge_on_date(dfs):
        return reduce(lambda L, R: pd.merge(L, R, on='Date', how='outer'), dfs)
    actual_df = merge_on_date(actual_dfs).sort_values('Date').set_index('Date') if actual_dfs else pd.DataFrame()
    pred_df   = merge_on_date(pred_dfs).sort_values('Date').set_index('Date')   if pred_dfs   else pd.DataFrame()
    return actual_df, pred_df

actual_df, pred_df = load_returns()

mean_pred = expected_returns.mean_historical_return(pred_df)
cov_pred = risk_models.sample_cov(pred_df)
mean_act  = expected_returns.mean_historical_return(actual_df)
cov_act  = risk_models.sample_cov(actual_df)

ef_pred = EfficientFrontier(mean_pred, cov_pred)
ef_act  = EfficientFrontier(mean_act,  cov_act)

# Sharpe-optimal weights (for reference)
w_pred_sharpe = ef_pred.max_sharpe()
w_act_sharpe  = ef_act.max_sharpe()
print("Predicted Sharpe Weights:", ef_pred.clean_weights())
print("Actual Sharpe Weights:   ", ef_act.clean_weights())


def optimize_for_risk_capacity(ef: EfficientFrontier, target_vol: float):
    """
    Optimize portfolio to maximize return for a given risk (volatility).
    Returns:
      weights     : dict of asset weights
      exp_return  : expected annual return
      exp_vol     : expected annual volatility
      sharpe      : Sharpe ratio
    """
    # Set target volatility (annualized)
    ef.efficient_risk(target_vol)

    # Clean (round) weights
    cleaned = ef.clean_weights()

    # Compute performance metrics
    exp_ret, exp_vol, sharpe = ef.portfolio_performance()
    return cleaned, exp_ret, exp_vol, sharpe

# Prompt user and handle inputs robustly
def prompt_risk_capacity():
    while True:
        raw = input("Enter your risk capacity (annual volatility, e.g. 0.12 for 12% or 12 for 12%): ")
        try:
            value = float(raw)
        except ValueError:
            print("Invalid format. Please enter a number (e.g., 0.10 or 10).")
            continue
        # Normalize percentage inputs >= 1
        if value >= 1:
            value = value / 100.0
        # Check bounds
        if not (0 < value < 1):
            print("Please enter a volatility between 0 and 1 (e.g., 0.15 for 15%).")
            continue
        return value

# Get user risk target
risk_capacity = prompt_risk_capacity()

# Optimize predicted portfolio for that risk
try:
    w_pred_risk, ret_pred_risk, vol_pred_risk, sr_pred_risk = optimize_for_risk_capacity(
        EfficientFrontier(mean_pred, cov_pred),
        risk_capacity
    )
    print(f"\nPredicted Portfolio for target {risk_capacity*100:.2f}% vol:")
    print("Weights:          ", w_pred_risk)
    print(f"Expected Return:  {ret_pred_risk*100:.2f}%")
    print(f"Expected Volatility: {vol_pred_risk*100:.2f}%")
    print(f"Sharpe Ratio:     {sr_pred_risk:.2f}")
except Exception as e:
    print(f"Optimization error – could not achieve target volatility {risk_capacity*100:.2f}%: {e}")

# You can repeat the above for actual data by swapping in (mean_act, cov_act) to EfficientFrontier
