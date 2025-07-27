#!/usr/bin/env python3
"""
stock_prediction.py

Fetches historical stock data (2010–2024), trains an LSTM on 2010–2023,
predicts 2024, saves the results to CSV, and then plots.
"""

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

START = "2010-01-01"
END   = "2024-12-31"
TRAIN_END = "2023-12-31"
TEST_START = "2024-01-01"
SEQ_LEN = 100


def load_data(ticker: str) -> pd.DataFrame:
    print(f"Fetching data for {ticker} from {START} to {END}...")
    df = yf.download(ticker, START, END)
    df.reset_index(inplace=True)
    print(f"Loaded {len(df)} rows.")
    return df


def split_by_date(df: pd.DataFrame):
    df['Date'] = pd.to_datetime(df['Date'])
    train_df = df[df['Date'] <= TRAIN_END].copy()
    test_df  = df[df['Date'] >= TEST_START].copy()
    print(f"Training samples: {len(train_df)}, Testing samples: {len(test_df)}")
    return train_df, test_df


def prepare_scaler(train_df: pd.DataFrame):
    scaler = MinMaxScaler((0, 1))
    train_close = train_df[['Close']].values
    scaled_train = scaler.fit_transform(train_close)
    return scaler, scaled_train


def create_sequences(data_arr: np.ndarray, seq_len: int = SEQ_LEN):
    X, y = [], []
    for i in range(seq_len, len(data_arr)):
        X.append(data_arr[i-seq_len:i, 0])
        y.append(data_arr[i, 0])
    return np.array(X), np.array(y)


def build_lstm_model(input_shape):
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
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])
    return model


def main(ticker: str = 'GOOG', epochs: int = 100):
    # 1. Load & split
    df = load_data(ticker)
    train_df, test_df = split_by_date(df)

    # 2. Scale & sequence train
    scaler, scaled_train = prepare_scaler(train_df)
    X_train, y_train = create_sequences(scaled_train)
    X_train = X_train.reshape((*X_train.shape, 1))

    # 3. Build & train
    model = build_lstm_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=epochs, verbose=1)
    model.save(f'{ticker}_model.h5')

    # 4. Prepare test sequences
    last_100 = train_df['Close'].values[-SEQ_LEN:].reshape(-1, 1)
    test_close = test_df[['Close']].values
    combined = np.vstack([last_100, test_close])
    scaled_full = scaler.transform(combined)

    X_test, y_test = create_sequences(scaled_full)
    X_test = X_test.reshape((*X_test.shape, 1))

    # 5. Predict
    y_pred = model.predict(X_test)

    # 6. Unscale
    factor = 1.0 / scaler.scale_[0]
    y_test_un = y_test * factor
    y_pred_un = y_pred.flatten() * factor

    # 7. Save results to CSV
    pred_dates = test_df['Date'].reset_index(drop=True)

    result_df = pd.DataFrame({
        'Date':      pred_dates,
        'Actual':    y_test_un,
        'Predicted': y_pred_un
    })
    csv_name = f"{ticker}_2024_predictions.csv"
    result_df.to_csv(csv_name, index=False)
    print(f"Saved predictions to {csv_name}")

    # 8. Plot
    plt.figure(figsize=(12,6))
    plt.plot(result_df['Date'], result_df['Actual'], label='Actual')
    plt.plot(result_df['Date'], result_df['Predicted'], label='Predicted')
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"{ticker} Actual vs Predicted (2024)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
