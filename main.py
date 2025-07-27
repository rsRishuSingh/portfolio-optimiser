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

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
load_dotenv()
MODEL_NAME = os.getenv("MODEL_NAME", "qwen/qwen3-32b")
FOLDER_PATH = 'Predicted_performance'
RISK_FREE_RATE = 0.03  # 3% annual

# Initialize LLM
llm = ChatGroq(model=MODEL_NAME)

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
def load_returns():
    actual_dfs, pred_dfs = [], []
    for filename in sorted(os.listdir(FOLDER_PATH)):
        if not filename.lower().endswith('.csv'):
            continue
        symbol = filename.split('_')[0].lower()
        df = (
            pd.read_csv(os.path.join(FOLDER_PATH, filename), parse_dates=['Date'])
              .sort_values('Date')
        )
        actual_dfs.append(
            df[['Date', 'Actual']].rename(columns={'Actual': symbol})
        )
        pred_dfs.append(
            df[['Date', 'Predicted']].rename(columns={'Predicted': symbol})
        )

    def merge_on_date(dfs):
        return reduce(
            lambda L, R: pd.merge(L, R, on='Date', how='outer'),
            dfs
        )

    actual_df = (
        merge_on_date(actual_dfs)
        .sort_values('Date')
        .set_index('Date')
        if actual_dfs else pd.DataFrame()
    )
    pred_df = (
        merge_on_date(pred_dfs)
        .sort_values('Date')
        .set_index('Date')
        if pred_dfs else pd.DataFrame()
    )
    return actual_df, pred_df

actual_df, pred_df = load_returns()

# -----------------------------------------------------------------------------
# Compute Means & Covariances
# -----------------------------------------------------------------------------
mean_pred = expected_returns.mean_historical_return(pred_df)
cov_pred  = risk_models.sample_cov(pred_df)

mean_act  = expected_returns.mean_historical_return(actual_df)
cov_act   = risk_models.sample_cov(actual_df)

# -----------------------------------------------------------------------------
# Unconstrained Max‑Sharpe Portfolios
# -----------------------------------------------------------------------------
ef_pred = EfficientFrontier(mean_pred, cov_pred)
ef_act  = EfficientFrontier(mean_act,  cov_act)

# Solve for max‑Sharpe with rf=3%
w_pred_sharpe = ef_pred.max_sharpe(risk_free_rate=RISK_FREE_RATE)
w_act_sharpe  = ef_act.max_sharpe(risk_free_rate=RISK_FREE_RATE)

print("Predicted Sharpe Weights:", ef_pred.clean_weights())
print("Actual   Sharpe Weights:", ef_act.clean_weights())

ret_pred, vol_pred, sharpe_pred = ef_pred.portfolio_performance(
    risk_free_rate=RISK_FREE_RATE
)
ret_act, vol_act, sharpe_act = ef_act.portfolio_performance(
    risk_free_rate=RISK_FREE_RATE
)

print(f"Predicted Max Sharpe Ratio: {sharpe_pred:.4f}")
print(f"Actual   Max Sharpe Ratio: {sharpe_act:.4f}")

# -----------------------------------------------------------------------------
# Risk‑Capacity‑Constrained Optimization
# -----------------------------------------------------------------------------
def optimize_for_risk_capacity(
    ef: EfficientFrontier,
    target_vol: float,
    rf: float = RISK_FREE_RATE
):
    """
    Optimize to maximize return for a given target volatility.
    Returns: (weights, expected_return, expected_volatility, sharpe)
    """
    ef.efficient_risk(target_vol)
    cleaned = ef.clean_weights()
    exp_ret, exp_vol, sharpe = ef.portfolio_performance(risk_free_rate=rf)
    return cleaned, exp_ret, exp_vol, sharpe

def prompt_risk_capacity():
    while True:
        raw = input(
            "Enter your risk capacity (annual volatility, e.g. 0.12 for 12% or 12 for 12%): "
        )
        try:
            value = float(raw)
        except ValueError:
            print("Invalid format. Please enter a numeric value.")
            continue
        if value >= 1:
            value = value / 100.0
        if not (0 < value <= 1):
            print("Please enter a volatility between 0 and 1 (e.g., 0.15 for 15%).")
            continue
        return value

# Get and normalize user input
risk_capacity = prompt_risk_capacity()

# Optimize the predicted portfolio under the user's risk target
try:
    w_pred_risk, ret_pr, vol_pr, sr_pr = optimize_for_risk_capacity(
        EfficientFrontier(mean_pred, cov_pred),
        risk_capacity
    )
    print(f"\nPredicted Portfolio for target {risk_capacity*100:.2f}% vol:")
    print("Weights:             ", w_pred_risk)
    print(f"Expected Return:     {ret_pr*100:.2f}%")
    print(f"Expected Volatility: {vol_pr*100:.2f}%")
    print(f"Sharpe Ratio:        {sr_pr:.4f}")
except Exception as e:
    print(
        f"Optimization error – "
        f"could not achieve target volatility {risk_capacity*100:.2f}%: {e}"
    )

# -----------------------------------------------------------------------------
# (You can repeat the same risk‑capacity call for actual data by substituting
# EfficientFrontier(mean_act, cov_act) instead of (mean_pred, cov_pred).)
# -----------------------------------------------------------------------------
