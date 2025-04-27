# Stock Price Forecasting: Tesla (TSLA), Apple (AAPL), Microsoft (MSFT)

This project compares the forecasting performance of ARIMA and Transformer models on stock prices for Tesla, Apple, and Microsoft.

## Project Files

- `stock_forecasting.py` : Full Python script to download stock data, train ARIMA and Transformer models, and evaluate forecasting performance.
- `TSLA_2018_2024.csv` : Tesla stock dataset.
- `AAPL_data.csv` : Apple stock dataset.
- `MSFT_data.csv` : Microsoft stock dataset (optional but included if available).

## Models Used

- **ARIMA** : Captures linear patterns.
- **Transformer** : Captures non-linear dependencies.

## Evaluation Metrics

- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**

## How to Run

1. Install required libraries (`pandas`, `tensorflow`, `statsmodels`, `sklearn`, `yfinance`, `matplotlib`).
2. Run `stock_forecasting.py`.
3. Check model evaluation results (RMSE, MAE) and visual graphs.

## Dataset Source

- All datasets downloaded from Yahoo Finance via `yfinance` API.

## License

This project is for academic purposes only.
