import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# Load and prepare data
df = pd.read_csv("TSLA_2018_2024.csv", skiprows=2)
df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

data = df[["Close"]].values

# Normalize data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Create sequences (60 days → 1 day)
X, y = [], []
for i in range(60, len(data_scaled)):
    X.append(data_scaled[i-60:i])
    y.append(data_scaled[i])
X, y = np.array(X), np.array(y)

# Split into train/test (90% train)
split = int(len(X) * 0.9)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Define Transformer model
def transformer_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.LayerNormalization()(inputs)
    x = layers.MultiHeadAttention(num_heads=2, key_dim=16)(x, x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(1)(x)
    model = models.Model(inputs, x)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

# Build and train
model = transformer_model(X_train.shape[1:])
history = model.fit(X_train, y_train, epochs=20, batch_size=32,
                    validation_split=0.1, verbose=1)

# Predict and inverse scale
y_pred = model.predict(X_test)
y_test_inv = scaler.inverse_transform(y_test)
y_pred_inv = scaler.inverse_transform(y_pred)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
mae = mean_absolute_error(y_test_inv, y_pred_inv)

print(f"Transformer RMSE: {rmse:.2f}")
print(f"Transformer MAE: {mae:.2f}")

# Plot
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label="Actual")
plt.plot(y_pred_inv, label="Predicted", alpha=0.7)
plt.title("Transformer Model: Actual vs Predicted (Tesla)")
plt.xlabel("Time")
plt.ylabel("Stock Price (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
