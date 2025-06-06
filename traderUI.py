# streamlit_dashboard.py
import streamlit as st
import time
import os

LOG_FILE = "logs/trading_bot.log"

st.title("📈 MT5 Trading Bot Logs")

log_placeholder = st.empty()

refresh_interval = st.slider("Odświeżanie (sekundy)", 1, 30, 5)

def read_logs(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read()
    return "Brak logów..."

while True:
    logs = read_logs(LOG_FILE)
    log_placeholder.text_area("Logi:", logs, height=600, key="log_output")
    time.sleep(refresh_interval)
