from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from pmdarima import auto_arima


def get_data(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """
    Download adjusted close prices for the given tickers and date range.
    Returns a DataFrame with dates as index and numerical values, formatted as shown in example.
    """
    data_frames = []
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
            if df.empty:
                print(f"Warning: No data found for {ticker}")
                continue
            adj_close = df["Adj Close"]
            adj_close.name = ticker
            data_frames.append(adj_close)
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
            continue
    
    if not data_frames:
        raise ValueError("No data could be downloaded for any ticker")
    
    result = pd.concat(data_frames, axis=1)
    result = result.ffill().dropna()
    
    # Format the data to match the desired output format
    # Reset index to make dates a column, then set proper formatting
    result.index = pd.to_datetime(result.index).strftime('%Y-%m-%d')
    
    # Round values to 6 decimal places to match the format shown
    result = result.round(6)
    
    print("Data sample (first 5 rows):")
    print(result.head())
    print(f"\nData shape: {result.shape}")
    print(f"Date range: {result.index[0]} to {result.index[-1]}")
    
    return result


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log-returns, handling non-positive values.
    """
    if (prices <= 0).any().any():
        print("Warning: Non-positive prices detected; filling zeros for log computation.")
        prices = prices.replace(0, np.nan).fillna(method='ffill')
    return np.log(prices).diff().dropna()


def split_train_test(data: pd.DataFrame, n_forecast: int = 252) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test by reserving last n_forecast points.
    """
    if len(data) <= n_forecast:
        raise ValueError(f"Data length ({len(data)}) must exceed forecast horizon ({n_forecast})")
    return data.iloc[:-n_forecast], data.iloc[-n_forecast:]


def sarima_forecast(
    train_rets: pd.Series,
    h: int,
    m: int = 5
) -> pd.Series:
    """
    Fit SARIMA (with seasonal period m) on train_rets and forecast h periods ahead.
    """
    # Clean and validate
    train_clean = train_rets.replace([np.inf, -np.inf], np.nan).dropna()
    if len(train_clean) < 50:
        raise ValueError(f"Insufficient data ({len(train_clean)}) for SARIMA (min 50)")

    print(f"Training data stats: Mean={train_clean.mean():.6f}, Std={train_clean.std():.6f}")
    
    # Try different approaches in order of preference
    forecast = None
    model_used = None
    
    # Approach 1: SARIMA with seasonal component
    try:
        print("Trying SARIMA model with seasonality...")
        arima = auto_arima(
            train_clean,
            seasonal=True, m=m,
            suppress_warnings=True, error_action='ignore',
            max_p=2, max_q=2, max_d=2,
            max_P=1, max_Q=1, max_D=1,
            start_p=0, start_q=0,
            stepwise=True,
            approximation=False
        )
        print(f"Selected SARIMA order: {arima.order}, seasonal_order: {arima.seasonal_order}")
        forecast = arima.predict(n_periods=h)
        model_used = "SARIMA"
        
        # Check for NaN values
        if np.isnan(forecast).any():
            print("SARIMA forecast contains NaN values, trying fallback...")
            forecast = None
    except Exception as e:
        print(f"SARIMA failed: {e}")
    
    # Approach 2: Non-seasonal ARIMA
    if forecast is None or np.isnan(forecast).any():
        try:
            print("Trying ARIMA model without seasonality...")
            arima = auto_arima(
                train_clean,
                seasonal=False,
                suppress_warnings=True, error_action='ignore',
                max_p=3, max_q=3, max_d=2,
                start_p=0, start_q=0,
                stepwise=True
            )
            print(f"Selected ARIMA order: {arima.order}")
            forecast = arima.predict(n_periods=h)
            model_used = "ARIMA"
            
            if np.isnan(forecast).any():
                print("ARIMA forecast contains NaN values, trying simple forecast...")
                forecast = None
        except Exception as e:
            print(f"ARIMA failed: {e}")
    
    # Approach 3: Simple moving average forecast with trend
    if forecast is None or np.isnan(forecast).any():
        print("Using simple trend-based forecast as fallback...")
        # Calculate recent trend and mean
        recent_window = min(30, len(train_clean) // 4)
        recent_mean = train_clean.tail(recent_window).mean()
        overall_mean = train_clean.mean()
        
        # Use a weighted combination
        forecast_value = 0.7 * recent_mean + 0.3 * overall_mean
        
        # Add slight decay towards long-term mean
        decay_factor = np.exp(-np.arange(h) / (h * 0.5))
        long_term_mean = train_clean.mean()
        forecast = forecast_value * decay_factor + long_term_mean * (1 - decay_factor)
        model_used = "Trend-based"
    
    print(f"Model used: {model_used}")
    print(f"Forecast stats: Mean={np.mean(forecast):.6f}, Std={np.std(forecast):.6f}")
    
    return pd.Series(forecast, index=range(h))


def back_transform_returns(
    forecast_rets: pd.Series,
    last_price: float,
    index: pd.DatetimeIndex
) -> pd.Series:
    """
    Convert forecasted log-returns to price series.
    """
    # Ensure matching lengths
    if len(forecast_rets) != len(index):
        min_len = min(len(forecast_rets), len(index))
        forecast_rets = forecast_rets.iloc[:min_len]
        index = index[:min_len]
    
    # Cap extremes to prevent unrealistic price movements
    forecast_capped = np.clip(forecast_rets, -0.5, 0.5)
    cum_returns = forecast_capped.cumsum()
    prices = last_price * np.exp(cum_returns)
    return pd.Series(prices, index=index)


def plot_actual_vs_forecast(actual: pd.Series, forecast: pd.Series, title: str):
    """
    Plot actual vs forecasted prices.
    """
    # Check for valid data
    if forecast.isna().all():
        print(f"Warning: All forecast values are NaN for {title}")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Plot actual data
    plt.plot(actual.index, actual.values, label="Actual", linewidth=2, color='blue')
    
    # Plot forecast data
    valid_forecast = forecast.dropna()
    if len(valid_forecast) > 0:
        plt.plot(forecast.index, forecast.values, label="Forecast", 
                linestyle='--', linewidth=2, color='red', alpha=0.8)
    
    # Add vertical line at forecast start
    if len(actual) > 0:
        plt.axvline(x=actual.index[-1], color='gray', linestyle=':', 
                   label='Forecast Start', alpha=0.7)
    
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Print value ranges for debugging
    print(f"Actual price range: ${actual.min():.2f} - ${actual.max():.2f}")
    if not forecast.isna().all():
        print(f"Forecast price range: ${forecast.min():.2f} - ${forecast.max():.2f}")
    
    plt.show()


def calculate_forecast_metrics(actual: pd.Series, forecast: pd.Series) -> dict:
    """
    Calculate forecast accuracy metrics.
    """
    # Align series for comparison
    min_len = min(len(actual), len(forecast))
    actual_aligned = actual.iloc[:min_len]
    forecast_aligned = forecast.iloc[:min_len]
    
    # Check for NaN or invalid values
    valid_mask = ~(np.isnan(actual_aligned) | np.isnan(forecast_aligned) | 
                   np.isinf(actual_aligned) | np.isinf(forecast_aligned))
    
    if valid_mask.sum() == 0:
        print("Warning: No valid data points for metric calculation")
        return {
            'MSE': np.nan,
            'RMSE': np.nan,
            'MAE': np.nan,
            'MAPE': np.nan,
            'Valid_Points': 0
        }
    
    actual_clean = actual_aligned[valid_mask]
    forecast_clean = forecast_aligned[valid_mask]
    
    # Calculate metrics
    mse = np.mean((actual_clean - forecast_clean) ** 2)
    rmse = np.sqrt(mse) if mse >= 0 else np.nan
    mae = np.mean(np.abs(actual_clean - forecast_clean))
    
    # MAPE calculation with protection against division by zero
    actual_nonzero = actual_clean[actual_clean != 0]
    forecast_nonzero = forecast_clean[actual_clean != 0]
    if len(actual_nonzero) > 0:
        mape = np.mean(np.abs((actual_nonzero - forecast_nonzero) / actual_nonzero)) * 100
    else:
        mape = np.nan
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'Valid_Points': valid_mask.sum()
    }


if __name__ == "__main__":
    start, end = "2020-01-01", "2025-01-01"
    tickers = ['AAPL', 'TSLA', 'DIS', 'AMD']
    h = 252  # forecast horizon (1 year of trading days)

    # Download and prepare data
    prices = get_data(tickers, start, end)
    rets = compute_log_returns(prices)
    train_rets, test_rets = split_train_test(rets, n_forecast=h)
    train_prices, test_prices = split_train_test(prices, n_forecast=h)

    # Process each ticker
    for tk in tickers:
        if tk not in train_rets:
            continue
        
        print(f"\n{'='*50}")
        print(f"Processing {tk}...")
        print(f"{'='*50}")
        
        try:
            # Generate SARIMA forecast
            forecast_rets = sarima_forecast(train_rets[tk], h)
            
            # Convert back to prices
            last_price = train_prices[tk].iloc[-1]
            forecast_prices = back_transform_returns(forecast_rets, last_price, test_prices[tk].index)
            
            # Calculate and display metrics
            metrics = calculate_forecast_metrics(test_prices[tk], forecast_prices)
            print(f"\nForecast Accuracy Metrics for {tk}:")
            for metric, value in metrics.items():
                if metric == 'MAPE':
                    if np.isnan(value):
                        print(f"{metric}: Unable to calculate")
                    else:
                        print(f"{metric}: {value:.2f}%")
                elif metric == 'Valid_Points':
                    print(f"{metric}: {value}")
                else:
                    if np.isnan(value):
                        print(f"{metric}: Unable to calculate")
                    else:
                        print(f"{metric}: {value:.4f}")
            
            # Plot results
            plot_actual_vs_forecast(test_prices[tk], forecast_prices, f"{tk}: SARIMA Forecast")
            
        except Exception as e:
            print(f"Error processing {tk}: {e}")
            continue

    print(f"\n{'='*50}")
    print("Forecasting completed!")
    print(f"{'='*50}")