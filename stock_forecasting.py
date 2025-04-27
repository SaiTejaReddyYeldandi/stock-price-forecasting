import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models
from statsmodels.tsa.arima.model import ARIMA

# --------------------------------------------
def download_stock_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df = df.reset_index()
    df = df[["Date", "Close"]]
    df.to_csv(f"{ticker}_data.csv", index=False)
    return df

# --------------------------------------------
def plot_stock_price(df, ticker):
    df.index = pd.to_datetime(df.index)
    plt.figure(figsize=(12, 6))
    plt.plot(df["Close"])
    plt.title(f"{ticker} Closing Prices")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --------------------------------------------
def run_arima_forecast(df, ticker):
    df.index = pd.to_datetime(df.index)
    data = df["Close"]
    
    model = ARIMA(data, order=(5, 1, 0)).fit()
    forecast = model.forecast(steps=30)

    last_date = data.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='B')

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(data, label="Actual")
    plt.plot(forecast_dates, forecast, label="ARIMA Forecast", color="red")
    plt.title(f"{ticker} - ARIMA Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Backtest
    train = data[:-30]
    test = data[-30:]
    model_bt = ARIMA(train, order=(5, 1, 0)).fit()
    preds = model_bt.forecast(steps=30)

    rmse = np.sqrt(mean_squared_error(test, preds))
    mae = mean_absolute_error(test, preds)

    print(f"🔹 {ticker} ARIMA RMSE: {rmse:.2f}")
    print(f"🔹 {ticker} ARIMA MAE: {mae:.2f}")

# --------------------------------------------
def run_transformer_forecast(df, ticker):
    df.index = pd.to_datetime(df.index)


    close_prices = df[["Close"]].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_prices)

    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)

    split = int(len(X) * 0.9)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    def transformer_model(shape):
        inputs = layers.Input(shape=shape)
        x = layers.LayerNormalization()(inputs)
        x = layers.MultiHeadAttention(num_heads=2, key_dim=16)(x, x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dense(1)(x)
        model = models.Model(inputs, x)
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return model

    model = transformer_model(X_train.shape[1:])
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=0)

    y_pred = model.predict(X_test)
    y_test_inv = scaler.inverse_transform(y_test)
    y_pred_inv = scaler.inverse_transform(y_pred)

    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mae = mean_absolute_error(y_test_inv, y_pred_inv)

    print(f"🤖 {ticker} Transformer RMSE: {rmse:.2f}")
    print(f"🤖 {ticker} Transformer MAE: {mae:.2f}")

    plt.figure(figsize=(12, 6))
    plt.plot(y_test_inv, label="Actual")
    plt.plot(y_pred_inv, label="Transformer Prediction", alpha=0.7)
    plt.title(f"{ticker} - Transformer Forecast")
    plt.xlabel("Time")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --------------------------------------------
# 🧪 Run Everything Here

if __name__ == "__main__":
    ticker = "MSFT"  # or "TSLA", "MSFT", "AAPL" etc.
    start_date = "2000-01-01"
    end_date = "2024-03-31"

    df = download_stock_data(ticker, start_date, end_date)
    plot_stock_price(df, ticker)
    run_arima_forecast(df.copy(), ticker)
    run_transformer_forecast(df.copy(), ticker)
