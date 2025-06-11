import streamlit as st
import os
from streamlit_autorefresh import st_autorefresh
import MetaTrader5 as mt5
import sqlite3
import altair as alt
from src.data_fetcher import test_trade_history, fetch_full_trade_history
from src.database import read_logs_from_db
from src.settings_manager import get_setting, set_setting

DB_PATH = "logs/trading_logs.db"

st.title("📈 MT5 Trading Bot Logs")

# Odświeżanie co 3 sekundy
st_autorefresh(interval=3000, key="log_refresh")


st.sidebar.header("⚙️ Ustawienia")


min_candles = st.sidebar.number_input("Minimalna liczba świec", min_value=10, value=int(get_setting("min_candles_for_patterns", "150")))
seq_length = st.sidebar.number_input("Długość sekwencji", min_value=5, value=int(get_setting("seq_len", "30")))

if st.sidebar.button("💾 Zapisz ustawienia"):
    st.text("Ustawiania modelu LSTM")
    set_setting("min_candles_for_patterns", str(min_candles))
    set_setting("seq_len", str(seq_length))
    st.sidebar.success("Zapisano ustawienia!")

if "mt5_initialized" not in st.session_state:
    if not mt5.initialize():
        st.error("❌ Nie udało się zainicjalizować MetaTrader5")
    else:
        mt5.initialize()
        st.session_state["mt5_initialized"] = True# st.dataframe(test_trade_history(days_back=200))
        
deals = fetch_full_trade_history(days_back=14)
#st.dataframe(fetch_full_trade_history(days_back=200))
if deals.empty:
    st.warning("Brak danych o transakcjach.")
else:
    st.subheader("📊 Wykres zysków w czasie")

    deals['color'] = deals['profit'].apply(lambda x: 'Zysk' if x >= 0 else 'Strata')

    chart = alt.Chart(deals).mark_bar(size=20).encode(
        x=alt.X('time:T', title='Czas'),
        y=alt.Y('profit:Q', title='Zysk / Strata'),
        color=alt.Color('color:N', scale=alt.Scale(domain=['Zysk', 'Strata'], range=['green', 'red']), legend=None),
        tooltip=['time', 'profit']
    ).properties(
        width=800,
        height=400,
        title='Zyski i straty z transakcji'
    )

    st.altair_chart(chart, use_container_width=True)

    
# Czytamy logi z bazy danych
logs = read_logs_from_db(limit=200)
logs_joined = "\n".join(logs)

# Przewijalne pole tekstowe
st.text_area("Logi", logs_joined, height=600, key="log_area", disabled=False)
