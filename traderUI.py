import streamlit as st
import os
from streamlit_autorefresh import st_autorefresh
import MetaTrader5 as mt5
import sqlite3
import altair as alt
from src.data_fetcher import test_trade_history, fetch_full_trade_history
from src.database import read_logs_from_db
from src.settings_manager import get_setting, set_setting, set_sentiment, get_sentiment, SENTIMENT_MAPPING
#from loginUI import login_form  # niepotrzebne, bo mamy formę poniżej

DB_PATH = "logs/trading_logs.db"


# Sidebar z logowaniem — tak jak w poprzednim przykładzie
st.sidebar.header("🔐 Logowanie do MetaTrader 5")

if not st.session_state.get("mt5_initialized", False):
    # Formularz logowania
    with st.sidebar.form("login_form"):
        login = st.text_input("Login", value=st.session_state.get("login", ""), key="input_login")
        password = st.text_input("Hasło", type="password", value=st.session_state.get("password", ""), key="input_password")
        server = st.text_input("Serwer", value=st.session_state.get("server", "MetaQuotes-Demo"), key="input_server")
        submitted = st.form_submit_button("Połącz z MT5")

    if submitted:
        st.session_state["login"] = login
        st.session_state["password"] = password
        st.session_state["server"] = server

        if not mt5.initialize():
            st.error(f"❌ Błąd inicjalizacji MT5: {mt5.last_error()}")
            st.session_state["mt5_initialized"] = False
        else:
            if mt5.login(login=int(login), password=password, server=server):
                st.session_state["mt5_initialized"] = True
                st.success("✅ Połączono z MetaTrader 5")
            else:
                st.error(f"❌ Logowanie nieudane. Kod błędu: {mt5.last_error()}")
                st.session_state["mt5_initialized"] = False
else:
    # Pokazujemy info o koncie i wylogowanie
    account_info = mt5.account_info()
    if account_info:
        st.sidebar.write(f"🔐 Zalogowano na konto: {account_info.login}")
        st.sidebar.write(f"💰 Saldo: {account_info.balance}")
    if st.sidebar.button("🔓 Wyloguj"):
        mt5.shutdown()
        st.session_state["mt5_initialized"] = False
        st.session_state.pop("login", None)
        st.session_state.pop("password", None)
        st.session_state.pop("server", None)
        st.experimental_rerun()

# --- TUTAJ cały kod "głównej strony" w warunku ---

if st.session_state.get("mt5_initialized", False):
    # Wszystkie elementy, które chcesz pokazać tylko po zalogowaniu
    st.title("📈 MT5 Trading Bot Logs")

    st_autorefresh(interval=3000, key="log_refresh")

    symbols = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF"]

    st.title("🧠 Wprowadź sentymenty walut")
    st.subheader("Zostaną one wprowadzone jeśli będzie za mało artyków na temat danego symbolu")

    cols = st.columns(len(symbols))  # jedna kolumna na symbol

    for i, symbol in enumerate(symbols):
        with cols[i]:
            choice = st.selectbox(
                f"Sentyment {symbol}",
                options=list(SENTIMENT_MAPPING.keys()),
                key=f"sentiment_{symbol}"
            )
            set_sentiment(symbol, SENTIMENT_MAPPING[choice])

    st.sidebar.header("⚙️ Ustawienia LSTM")
    min_candles = st.sidebar.number_input("Minimalna liczba świec", min_value=10, value=int(get_setting("min_candles_for_patterns", "150")))
    seq_length = st.sidebar.number_input("Długość sekwencji", min_value=5, value=int(get_setting("seq_len", "30")))

    st.sidebar.header("🚫 Ustawienia Ryzyka")

    MAX_RISK_PER_TRADE = st.sidebar.number_input("Maksymalny procent alokacji",step = 0.0001, format="%.6f",
                                                min_value=0.000001, value=float(get_setting("MAX_RISK_PER_TRADE", "0.0001")))

    if st.sidebar.button("💾 Zapisz ustawienia"):
        st.text("Ustawiania modelu LSTM")
        set_setting("min_candles_for_patterns", str(min_candles))
        set_setting("seq_len", str(seq_length))
        st.sidebar.success("Zapisano ustawienia!")

        st.text("Ustawiania ryzyka")
        set_setting("MAX_RISK_PER_TRADE", str(MAX_RISK_PER_TRADE))


    deals = fetch_full_trade_history(days_back=14)
    if deals.empty:
        st.warning("Brak danych o transakcjach.")
    else:
        st.subheader("📊 Wykres zysków w czasie")
        deals['time'] = deals['time'].dt.date
        deals = deals.groupby(['time'])['profit'].agg('sum').reset_index()
        deals['color'] = deals['profit'].apply(lambda x: 'Zysk' if x >= 0 else 'Strata')

        chart = alt.Chart(deals).mark_bar(size=20).encode(
            x=alt.X('time', title='Czas'),
            y=alt.Y('profit:Q', title='Zysk / Strata'),
            color=alt.Color('color:N', scale=alt.Scale(domain=['Zysk', 'Strata'], range=['green', 'red']), legend=None),
            tooltip=['time', 'profit']
        ).properties(
            width=800,
            height=400,
            title='Zyski i straty z transakcji'
        )

        st.altair_chart(chart, use_container_width=True)


    logs = read_logs_from_db(limit=200)
    logs_joined = "\n".join(logs)
    st.text_area("Logi", logs_joined, height=600, key="log_area", disabled=False)
else:
    st.info("🔐 Proszę się zalogować aby zobaczyć dane.")
