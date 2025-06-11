# import streamlit as st
# import os
# from streamlit_autorefresh import st_autorefresh
# import MetaTrader5 as mt5
# from src.data_fetcher import test_trade_history
# from main import main

# LOG_FILE = "logs/trading_bot.log"




# st.title("📈 MT5 Trading Bot Logs")



# # Odświeżanie co 3 sekundy
# st_autorefresh(interval=3000, key="log_refresh")

# def read_logs(path):
#     if os.path.exists(path):
#         with open(path, "r") as f:
#             return f.readlines()
#     return ["Brak logów..."]

# # Czytamy logi i odwracamy kolejność (ostatnie na górze)
# logs = read_logs(LOG_FILE)
# logs_reversed = "".join(logs[::-1])

# mt5.initialize()
# st.text(test_trade_history(days_back=200))

# # Przewijalne pole tekstowe z unikalnym kluczem
# st.header("### Ostatnie logi z bota handlowego")
# st.text_area("Logi", logs_reversed, height=600, key="log_area")

# main()
# streamlit_dashboard.py
import streamlit as st
import os
from streamlit_autorefresh import st_autorefresh
import MetaTrader5 as mt5
from src.data_fetcher import test_trade_history

LOG_FILE = "logs/trading_bot.log"

st.title("📈 MT5 Trading Bot Logs")

# Odświeżanie co 3 sekundy
st_autorefresh(interval=3000, key="log_refresh")

def read_logs(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.readlines()
    return ["Brak logów..."]


mt5.initialize()
st.text(test_trade_history(days_back=200))

# Czytamy logi i odwracamy kolejność (ostatnie na górze)
logs = read_logs(LOG_FILE)
logs_reversed = "".join(logs[::-1])

# Przewijalne pole tekstowe z unikalnym kluczem
st.text_area("Logi", logs_reversed, height=600, key="log_area", disabled=True)
