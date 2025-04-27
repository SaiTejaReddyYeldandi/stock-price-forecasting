import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Download Tesla stock data
ticker = "TSLA"
start_date = "2018-01-01"
end_date = "2024-03-31"
tsla_data = yf.download(ticker, start=start_date, end=end_date)

# Save to CSV
tsla_data.to_csv("TSLA_2018_2024.csv")

# Convert 'Close' column to a flat Series
close_prices = tsla_data['Close'].squeeze()

# Plot
plt.figure(figsize=(12, 6))
sns.lineplot(x=tsla_data.index, y=close_prices)
plt.title("Tesla (TSLA) Closing Prices (2018–2024)")
plt.xlabel("Date")
plt.ylabel("Closing Price (USD)")
plt.grid(True)
plt.tight_layout()
plt.show()
