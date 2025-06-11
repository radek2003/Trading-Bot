import streamlit as st
import os
from streamlit_autorefresh import st_autorefresh
import MetaTrader5 as mt5
import sqlite3
from src.data_fetcher import test_trade_history
from src.database import read_logs_from_db

DB_PATH = "logs/trading_logs.db"

st.title("📈 MT5 Trading Bot Logs")

# Odświeżanie co 3 sekundy
st_autorefresh(interval=3000, key="log_refresh")


# mt5.initialize()
# st.dataframe(test_trade_history(days_back=200))
#st.text_area(test_trade_history(days_back=200))

# Czytamy logi z bazy danych
logs = read_logs_from_db(DB_PATH)
logs_joined = "\n".join(logs)

# Przewijalne pole tekstowe
st.text_area("Logi", logs_joined, height=600, key="log_area", disabled=False)
