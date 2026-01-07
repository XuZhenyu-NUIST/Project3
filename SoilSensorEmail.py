import RPi.GPIO as GPIO
import time
import smtplib
from email.message import EmailMessage
from datetime import datetime

from_email_addr = "505183416@qq.com"
from_email_pass = "aeshagobzrmpbgie"
to_email_addr = "715660750@qq.com"
smtp_server = "smtp.office365.com"
smtp_port = 587

channel = 4
GPIO.setmode(GPIO.BCM)
GPIO.setup(channel, GPIO.IN)

def send_email(status):
    msg = EmailMessage()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg.set_content(f"Plant Status Report - {timestamp}\n\nStatus: {status}")
    msg['From'] = from_email_addr
    msg['To'] = to_email_addr
    msg['Subject'] = f'PLANT MOISTURE ALERT - {status.upper()}'

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(from_email_addr, from_email_pass)
        server.send_message(msg)
        print(f'Email sent: {status}')
    except Exception as e:
        print(f'Email error: {e}')
    finally:
        server.quit()

def check_moisture_and_notify():
        check_times = ["08:00", "12:00", "16:00", "20:00"]

    while True:
        current_time = datetime.now().strftime("%H:%M")
        if current_time in check_times:
            if GPIO.input(channel):
                status = "Please water your plant! Soil is dry."
            else:
                status = "Water NOT needed. Soil is moist."
            send_email(status)
            time.sleep(3660)
        time.sleep(60)

try:
    print("Plant Moisture Monitoring System Started...")
    check_moisture_and_notify()
except KeyboardInterrupt:
    print("System stopped manually.")
finally:
    GPIO.cleanup()
