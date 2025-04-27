import yfinance as yf
import pandas as pd

# Download Tesla stock data again, cleanly
tsla_data = yf.download("TSLA", start="2018-01-01", end="2024-03-31")

# Ensure index is datetime
tsla_data.index = pd.to_datetime(tsla_data.index)

# Save clean CSV with Date as column
tsla_data.to_csv("TSLA_2018_2024.csv", index_label="Date")
