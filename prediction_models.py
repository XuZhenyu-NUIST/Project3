import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

df = pd.read_csv("soil_moisture_data.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df["Time_Ordinal"] = df["Timestamp"].astype("int64") // 10**9

X = df[["Time_Ordinal"]]
y = df["Moisture_Normalized"]

train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

lr_mse = mean_squared_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)
print(f"Linear Regression - MSE: {lr_mse:.4f}, R²: {lr_r2:.4f}")

def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

seq_len = 6
data = df["Moisture_Normalized"].values.reshape(-1, 1)
X_lstm, y_lstm = create_sequences(data, seq_len)

X_train_lstm, X_test_lstm = X_lstm[:train_size-seq_len], X_lstm[train_size-seq_len:]
y_train_lstm, y_test_lstm = y_lstm[:train_size-seq_len], y_lstm[train_size-seq_len:]

lstm_model = Sequential()
lstm_model.add(LSTM(50, input_shape=(seq_len, 1)))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer="adam", loss="mse")

lstm_model.fit(X_train_lstm, y_train_lstm, epochs=20, batch_size=4, verbose=1)

last_seq = data[-seq_len:]
lstm_predictions = []
for _ in range(24):
    next_val = lstm_model.predict(last_seq.reshape(1, seq_len, 1), verbose=0)[0][0]
    lstm_predictions.append(next_val)
    last_seq = np.append(last_seq[1:], next_val).reshape(-1, 1)

lstm_pred_test = lstm_model.predict(X_test_lstm, verbose=0)
lstm_mse = mean_squared_error(y_test_lstm, lstm_pred_test)
lstm_r2 = r2_score(y_test_lstm, lstm_pred_test)
print(f"LSTM Model - MSE: {lstm_mse:.4f}, R²: {lstm_r2:.4f}")

plt.figure(figsize=(12, 6))
plt.plot(df["Timestamp"][train_size:], y_test, label="Actual Moisture", color="blue")
plt.plot(df["Timestamp"][train_size:], lr_pred, label="Linear Regression Prediction", color="orange")
future_times = pd.date_range(start=df["Timestamp"].iloc[-1], periods=25, freq="30min")[1:]
plt.plot(future_times, lstm_predictions, label="LSTM 12-Hour Prediction", color="red", linestyle="--")

plt.xlabel("Time")
plt.ylabel("Normalized Moisture")
plt.title("Soil Moisture Prediction: Linear Regression vs LSTM")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("prediction_results.png")
print("Prediction plot saved as prediction_results.png")
