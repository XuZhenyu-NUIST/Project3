import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("soil_moisture_data.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

df["Moisture_Denoised"] = df["Moisture"].rolling(window=3).mean().fillna(df["Moisture"])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df["Moisture_Normalized"] = scaler.fit_transform(df[["Moisture_Denoised"]])

stats = df[["Moisture", "Moisture_Denoised", "Moisture_Normalized"]].describe()
print("Descriptive Statistics:\n", stats)

plt.figure(figsize=(12, 6))
plt.plot(df["Timestamp"], df["Moisture_Normalized"], label="Normalized Moisture")
plt.xlabel("Time")
plt.ylabel("Normalized Moisture (0=Dry, 1=Moist)")
plt.title("Soil Moisture Trend Over 48 Hours")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("moisture_trend.png")
print("Trend plot saved as moisture_trend.png")
