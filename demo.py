from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

def get_data(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """
    Download adjusted close prices for the given tickers and date range.
    Returns a DataFrame with each column named by ticker.
    """
    data_frames = []
    for ticker in tickers:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        adj_close = df["Adj Close"]
        adj_close.name = ticker
        data_frames.append(adj_close)
    return pd.concat(data_frames, axis=1)

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices).diff().dropna()

def test_stationarity(series: pd.Series, alpha: float = 0.05) -> Tuple[float, bool]:
    pval = adfuller(series)[1]
    return pval, (pval < alpha)

def split_train_test(data: pd.DataFrame, n_forecast: int = 252
                    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = data.iloc[:-n_forecast]
    test  = data.iloc[-n_forecast:]
    return train, test

def select_var_lag(train: pd.DataFrame, maxlags: int = 15) -> int:
    model = VAR(train)
    sel = model.select_order(maxlags=maxlags)
    return sel.bic

def fit_var_model(train: pd.DataFrame, lag_order: int):
    return VAR(train).fit(lag_order)

def forecast_var(var_results, train_index: pd.DatetimeIndex, last_obs: np.ndarray, steps: int
                ) -> pd.DataFrame:
    cols = var_results.names
    last_date = train_index[-1]
    dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=steps)
    fc = var_results.forecast(y=last_obs, steps=steps)
    return pd.DataFrame(fc, index=dates, columns=cols)

def invert_log_returns(forecast_lr: pd.DataFrame, last_price: pd.Series) -> pd.DataFrame:
    cum_lr = forecast_lr.cumsum()
    return np.exp(cum_lr).multiply(last_price, axis=1)

def plot_actual_vs_forecast(actual: pd.Series, forecast: pd.Series, title: str):
    plt.figure(figsize=(10, 5))
    plt.plot(actual, label="Actual")
    plt.plot(forecast, linestyle="--", label="Forecast")
    plt.title(title)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # 1. Load data
    start, end = "2020-01-01", "2025-01-01"
    tickers = ['AAPL', 'TSLA', 'DIS', 'AMD']
    prices = get_data(tickers, start, end)
    
    # Clean the data and handle missing values
    prices = prices.dropna()
    
    # Resample to business day frequency to ensure consistent spacing
    # This fills any gaps and creates a proper business day index
    prices = prices.resample('B').last().dropna()

    # 2. Log-returns
    lr = compute_log_returns(prices)
    
    # Drop any remaining NaN values
    lr = lr.dropna()
    
    # No need to manually set frequency - resample handles this
    print(f"Log returns shape: {lr.shape}")
    print(f"Index frequency: {lr.index.freq}")

    # 3. Stationarity check
    for t in lr.columns:
        pval, is_stat = test_stationarity(lr[t])
        print(f"{t}: p={pval:.4f}, stationary={is_stat}")

    # 4. Train/test split
    train, test = split_train_test(lr, n_forecast=252)
    
    print(f"Training data shape: {train.shape}")
    print(f"Test data shape: {test.shape}")

    # 5. Lag selection (force ≥1)
    p_opt = select_var_lag(train, maxlags=15)
    if p_opt < 1:
        print(f"Selected lag {p_opt} too low; using 1 instead.")
        p_opt = 1
    print("Using lag order:", p_opt)

    # 6. Fit VAR
    var_res = fit_var_model(train, p_opt)

    # 7. Forecast log-returns
    last_obs = train.values[-p_opt:]
    fc_lr = forecast_var(var_res, train.index, last_obs, steps=min(252, len(test)))

    # 8. Price forecasts
    last_price = prices.iloc[len(train)-1]
    fc_prices = invert_log_returns(fc_lr, last_price)

    # 9. Plot
    plot_actual_vs_forecast(
        actual=prices['AAPL'].iloc[-min(252, len(test)):],
        forecast=fc_prices['AAPL'],
        title="AAPL: Actual vs VAR Forecast"
    )