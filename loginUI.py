import streamlit as st
import MetaTrader5 as mt5


def login_form():

    st.header("🔐 Logowanie do MetaTrader 5")

    # Formularz logowania
    with st.form("login_form"):
        login = st.text_input("Login", value=st.session_state.get("login", ""), key="input_login")
        password = st.text_input("Hasło", type="password", value=st.session_state.get("password", ""), key="input_password")
        server = st.text_input("Serwer", value=st.session_state.get("server", "MetaQuotes-Demo"), key="input_server")
        submitted = st.form_submit_button("Połącz z MT5")

    # Logowanie tylko jeśli użytkownik kliknie
    if submitted:
        # Zapisz dane w sesji
        st.session_state["login"] = login
        st.session_state["password"] = password
        st.session_state["server"] = server

        # Inicjalizacja
        initialized = mt5.initialize(login=int(login), password=password, server=server)
        if not initialized:
            st.error(f"❌ Błąd inicjalizacji MT5: {mt5.last_error()}")
            return False
        else:
            st.session_state["mt5_initialized"] = True
            st.success("✅ Połączono z MetaTrader 5")
            account_info = mt5.account_info()
            if account_info:
                st.write(f"🔐 Zalogowano na konto: {account_info.name}")
                st.write(f"💰 Saldo: {account_info.balance}")
        # else:
        #     st.error(f"❌ Logowanie nieudane. Kod błędu: {mt5.last_error()}")

    return True