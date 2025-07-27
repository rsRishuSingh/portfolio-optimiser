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
    """
    Load CSV files from a folder and compute actual and predicted returns.

    Each CSV should have columns: Date, Actual, Predicted.
    Filenames should be like SYMBOL_YYYY_predictions.csv (e.g., AAPL_2024_predictions.csv).

    Returns:
        actual_df (pd.DataFrame): DataFrame indexed by Date, with one column per symbol of actual returns.
        pred_df   (pd.DataFrame): DataFrame indexed by Date, with one column per symbol of predicted returns.
    """
    actual_dfs = []
    pred_dfs   = []

    for filename in sorted(os.listdir(folder_path)):
        if not filename.lower().endswith('.csv'):
            continue

        symbol = filename.split('_')[0].lower()
        path   = os.path.join(folder_path, filename)

        df = pd.read_csv(path, parse_dates=['Date'])
        df = df.sort_values('Date')

        actual_dfs.append(df[['Date', 'Actual']].rename(columns={'Actual': symbol}))
        pred_dfs.append(  df[['Date', 'Predicted']].rename(columns={'Predicted': symbol}))

    def merge_on_date(dfs):
        return reduce(lambda L, R: pd.merge(L, R, on='Date', how='outer'), dfs)

    # Merge all symbol‐level DataFrames
    actual_df = merge_on_date(actual_dfs) if actual_dfs else pd.DataFrame()
    pred_df   = merge_on_date(pred_dfs)   if pred_dfs   else pd.DataFrame()

    # Sort by date and then set Date as the index
    actual_df = (actual_df
                 .sort_values('Date')
                 .set_index('Date'))
    pred_df   = (pred_df
                 .sort_values('Date')
                 .set_index('Date'))

    return actual_df, pred_df

actual_df, pred_df = load_returns()
print(actual_df.head())
print(pred_df.head())

meanPredicted = expected_returns.mean_historical_return(pred_df) #expected returns
covariancePredicted = risk_models.sample_cov(pred_df) #Covariance matrix

# Providing expected returns and covariance matrix as input\
ef_Predicted = EfficientFrontier(meanPredicted, covariancePredicted)
# Optimizing weights for Sharpe ratio maximization 
weights_Predicted = ef_Predicted.max_sharpe()
# clean_weights rounds the weights and clips near-zeros
clean_weights_Predicted = ef_Predicted.clean_weights() 
print("Expected : ",clean_weights_Predicted)


meanActual = expected_returns.mean_historical_return(actual_df)
covarianceActual = risk_models.sample_cov(actual_df) 

ef_Actual = EfficientFrontier(meanActual, covarianceActual)
weights_Actual = ef_Actual.max_sharpe()
clean_weights_Actual = ef_Actual.clean_weights() 
print("Actual : ",clean_weights_Actual)


def plot_all_price_comparison(actual_df, pred_df):
    """
    Plot all tickers on a single graph:
      – Actual prices as dotted lines
      – Predicted prices as solid lines
    Each ticker shares a unique color.
    """
    # pull default color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for idx, ticker in enumerate(actual_df.columns):
        color = colors[idx % len(colors)]
        
        # actual dotted
        ax.plot(
            actual_df.index, actual_df[ticker],
            linestyle=':', linewidth=1.5, color=color,
            label=f"{ticker.upper()} Actual"
        )
        # predicted solid
        ax.plot(
            pred_df.index, pred_df[ticker],
            linestyle='-', linewidth=1.5, color=color,
            label=f"{ticker.upper()} Predicted"
        )
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title("Actual vs Predicted Prices for All Tickers")
    ax.legend(ncol=2, fontsize='small')
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


plot_all_price_comparison(actual_df, pred_df)


def compute_weighted_portfolio_returns(returns_df, weights):
    """
    Compute portfolio daily returns given asset returns and weights.

    Parameters:
    -----------
    returns_df : pd.DataFrame
        DataFrame indexed by Date, columns are asset symbols, values are daily returns.
    weights : dict or list or np.ndarray
        If dict: {symbol: weight, ...}
        If list/array: weights aligned in order of returns_df.columns.

    Returns:
    --------
    pd.DataFrame
        Single-column DataFrame (column name 'Portfolio') of weighted portfolio returns.
    """

    # Align weights to columns
    if isinstance(weights, (list, np.ndarray)):
        weights_series = pd.Series(weights, index=returns_df.columns)
    else:
        weights_series = pd.Series(weights)

    # Multiply each column by its weight, then sum across columns
    port_ret = returns_df.mul(weights_series, axis=1).sum(axis=1)

    return port_ret.to_frame(name='Portfolio')


def compute_actual_and_predicted_portfolios(actual_df, pred_df, actual_weights, pred_weights):
    """
    Given actual & predicted returns DataFrames and a weight allocation,
    return two DataFrames of portfolio returns.

    Parameters:
    -----------
    actual_df : pd.DataFrame
        Daily actual returns.
    pred_df : pd.DataFrame
        Daily predicted returns.
    weights : dict or list or np.ndarray
        Portfolio weights per asset.

    Returns:
    --------
    (actual_port_df, pred_port_df)
    Each is a DataFrame with index Date and column 'Portfolio'.
    """
    actual_port = compute_weighted_portfolio_returns(actual_df, actual_weights)
    pred_port   = compute_weighted_portfolio_returns(pred_df,   pred_weights)
    return actual_port, pred_port


actual_port, pred_port  = compute_actual_and_predicted_portfolios(actual_df, pred_df,clean_weights_Actual,clean_weights_Predicted )

print(actual_port.head())
print(pred_port.head())


def plot_portfolio_returns(port_actual, port_predicted):
    """
    Plot the actual and predicted portfolio returns over time.

    Args:
        port_actual (pd.DataFrame): DataFrame with Date index and one column 'Portfolio' for actual returns.
        port_predicted (pd.DataFrame): DataFrame with Date index and one column 'Portfolio' for predicted returns.
    """
    plt.figure(figsize=(10, 6))

    plt.plot(port_actual.index, port_actual['Portfolio'], label='Actual Portfolio Return', linestyle='--', color='blue')
    plt.plot(port_predicted.index, port_predicted['Portfolio'], label='Predicted Portfolio Return', linestyle='-', color='orange')

    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.title('Actual vs Predicted Portfolio Returns')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_portfolio_returns(actual_port, pred_port)
