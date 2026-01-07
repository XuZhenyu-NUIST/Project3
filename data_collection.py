import RPi.GPIO as GPIO
import time
import pandas as pd
from datetime import datetime

channel = 4
GPIO.setmode(GPIO.BCM)
GPIO.setup(channel, GPIO.IN)

sensor_data = []

def collect_data():
    print("Data collection started. Collecting for 48 hours...")
    for _ in range(96):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        moisture = 0 if GPIO.input(channel) else 1
        sensor_data.append({"Timestamp": timestamp, "Moisture": moisture})
        print(f"Collected: {timestamp} - Moisture: {moisture}")
        time.sleep(1800)

try:
    collect_data()
    df = pd.DataFrame(sensor_data)
    df.to_csv("soil_moisture_data.csv", index=False)
    print("Data saved to soil_moisture_data.csv")
except KeyboardInterrupt:
    print("Data collection stopped early.")
finally:
    GPIO.cleanup()
