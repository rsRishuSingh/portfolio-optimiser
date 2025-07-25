#!/usr/bin/env python3
"""
stock_prediction.py

A Python script to fetch historical stock data of a given ticker,
plot closing prices with moving averages, train an LSTM model to
predict future prices, and evaluate model performance.

Usage:
    python stock_prediction.py --ticker AAPL
"""

import argparse
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

# Define constants for data fetching
START = "2010-01-01"
TODAY = date.today().strftime("%Y-%m-%d")


def load_data(ticker: str) -> pd.DataFrame:
    """
    Download historical data for the given ticker from Yahoo Finance.
    Returns a DataFrame with Date, Open, High, Low, Close, Volume, Adj Close.
    """
    print(f"Fetching data for {ticker} from {START} to {TODAY}...")
    df = yf.download(ticker, START, TODAY, auto_adjust=False)
    df.reset_index(inplace=True)
    print(f"Loaded {len(df)} rows of data for {ticker}.")
    return df


def plot_closing_and_moving_averages(df: pd.DataFrame, ticker: str):
    """
    Plot closing price along with 100-day and 200-day moving averages.
    """
    df = df.copy()
    df['MA100'] = df['Close'].rolling(100).mean()
    df['MA200'] = df['Close'].rolling(200).mean()

    print("Plotting closing price and moving averages...")
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Close'], label='Close Price')
    plt.plot(df['Date'], df['MA100'], 'r', label='100-day MA')
    plt.plot(df['Date'], df['MA200'], 'g', label='200-day MA')
    plt.title(f"{ticker} Closing Price and Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()


def prepare_data(data: pd.DataFrame, split_ratio: float = 0.7):
    """
    Split the data into training and testing sets and scale the Close values.
    Returns raw train/test arrays, scaled train array, and scaler.
    """
    split_index = int(len(data) * split_ratio)
    train = data.iloc[:split_index]
    test = data.iloc[split_index:]
    print(f"Split data: {len(train)} training samples, {len(test)} testing samples.")

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_close = train[['Close']].values
    test_close = test[['Close']].values
    scaled_train = scaler.fit_transform(train_close)
    print("Data scaling complete.")

    return train_close, test_close, scaled_train, scaler


def create_sequences(data_array: np.ndarray, seq_length: int = 100):
    """
    Create sequences of length seq_length for LSTM input.
    Returns x and y arrays.
    """
    x, y = [], []
    for i in range(seq_length, len(data_array)):
        x.append(data_array[i - seq_length:i, 0])
        y.append(data_array[i, 0])
    x_arr, y_arr = np.array(x), np.array(y)
    print(f"Created {x_arr.shape[0]} sequences of length {seq_length}.")
    return x_arr, y_arr


def build_lstm_model(input_shape) -> Sequential:
    """
    Build an LSTM model with multiple layers and dropout.
    """
    print("Building LSTM model...")
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(60, activation='relu', return_sequences=True),
        Dropout(0.3),
        LSTM(80, activation='relu', return_sequences=True),
        Dropout(0.4),
        LSTM(120, activation='relu'),
        Dropout(0.5),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.MeanAbsoluteError()])
    print(model.summary())
    return model


def train_model(model: Sequential, x_train: np.ndarray, y_train: np.ndarray, epochs: int = 100) -> Sequential:
    """
    Train the LSTM model.
    """
    print(f"Training model for {epochs} epochs...")
    history = model.fit(x_train, y_train, epochs=epochs, verbose=1)
    print("Model training complete.")
    return model


def evaluate_and_plot(y_test: np.ndarray, y_pred: np.ndarray, scaler: MinMaxScaler):
    """
    Unscale predictions and actuals, plot comparison, compute MAE% and R2.
    """
    print("Evaluating model...")
    scale_factor = 1 / scaler.scale_[0]
    y_test_unscaled = y_test * scale_factor
    y_pred_unscaled = y_pred * scale_factor

    # Plot actual vs predicted
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_unscaled, 'b', label='Actual Price')
    plt.plot(y_pred_unscaled, 'r', label='Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Metrics
    mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
    mae_pct = (mae / np.mean(y_test_unscaled)) * 100
    r2 = r2_score(y_test_unscaled, y_pred_unscaled)

    print(f"Mean Absolute Error: {mae_pct:.2f}%")
    print(f"R2 Score: {r2:.4f}")

    # Plot R2 bar
    plt.figure()
    plt.barh([0], [r2])
    plt.xlim(-1, 1)
    plt.yticks([])
    plt.xlabel('R2 Score')
    plt.title('Model R2 Score')
    plt.text(r2, 0, f'{r2:.2f}', va='center')
    plt.show()

    # Scatter
    plt.figure()
    plt.scatter(y_test_unscaled, y_pred_unscaled)
    min_val = min(y_test_unscaled.min(), y_pred_unscaled.min())
    max_val = max(y_test_unscaled.max(), y_pred_unscaled.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Prediction vs Actual (R2={r2:.2f})')
    plt.show()


def main(ticker: str, epochs: int = 100):
    # Load and visualize data
    data = load_data(ticker)
    plot_closing_and_moving_averages(data, ticker)

    # Prepare data for model
    train_close, test_close, scaled_train, scaler = prepare_data(data)
    x_train, y_train = create_sequences(scaled_train)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

    # Build and train model
    model = build_lstm_model((x_train.shape[1], 1))
    trained_model = train_model(model, x_train, y_train, epochs)
    trained_model.save('keras_model.h5')
    print(f"Saved trained model to 'keras_model.h5'.")

    # Prepare test sequences
    past_100 = train_close[-100:]
    combined = np.concatenate((past_100, test_close), axis=0)
    scaled_full = scaler.transform(combined)
    x_test, y_test = create_sequences(scaled_full)
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    # Predict and evaluate
    print("Generating predictions on test data...")
    y_pred = trained_model.predict(x_test)
    evaluate_and_plot(y_test, y_pred, scaler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stock Price Prediction using LSTM')
    parser.add_argument('--ticker', type=str, required=True, help='Ticker symbol, e.g. AAPL')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    args = parser.parse_args()
    main(args.ticker, args.epochs)
