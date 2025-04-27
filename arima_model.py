import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

df = pd.read_csv("TSLA_2018_2024.csv", skiprows=2)
df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]  # <-- rename columns properly
df['Date'] = pd.to_datetime(df['Date'])
df.set_index("Date", inplace=True)

data = df["Close"]  # now this will work!



# Plot actual prices
plt.figure(figsize=(12, 6))
plt.plot(data, label='Actual Closing Price')
plt.title("Tesla Closing Prices (2018–2024)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Build ARIMA model (try order=(5, 1, 0))
model = ARIMA(data, order=(5, 1, 0))
model_fit = model.fit()

# Forecast next 30 business days
forecast = model_fit.forecast(steps=30)

# Generate dates for forecast
last_date = data.index[-1]
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='B')

# Plot forecast
plt.figure(figsize=(12, 6))
plt.plot(data, label='Actual')
plt.plot(forecast_dates, forecast, label='ARIMA Forecast (Next 30 Days)', color='red')
plt.title("ARIMA Forecast for Tesla Stock")
plt.xlabel("Date")
plt.ylabel("Closing Price (USD)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Backtest: compare on last 30 known days
train = data[:-30]
test = data[-30:]

model_bt = ARIMA(train, order=(5, 1, 0)).fit()
preds = model_bt.forecast(steps=30)

# Metrics
rmse = np.sqrt(mean_squared_error(test, preds))
mae = mean_absolute_error(test, preds)

print(f"ARIMA RMSE: {rmse:.2f}")
print(f"ARIMA MAE: {mae:.2f}")
