import streamlit as st
import os
from streamlit_autorefresh import st_autorefresh
import MetaTrader5 as mt5
import sqlite3
import altair as alt
import plotly.express as px
from src.data_fetcher import test_trade_history, fetch_full_trade_history,get_open_positions
from src.database import read_logs_from_db
from src.settings_manager import get_setting, set_setting, set_sentiment, get_sentiment, SENTIMENT_MAPPING
#from loginUI import login_form  # niepotrzebne, bo mamy formę poniżej
from dotenv import load_dotenv
from bot_state_managment import is_main_py_running,start_main_py,stop_main_py
import time
import subprocess
import signal
import psutil
import time

DB_PATH = "logs/trading_logs.db"
load_dotenv()

def is_main_py_running():
    """Check if main.py is currently running - optimized version"""
    try:
        if os.name == 'nt':  # Windows
            try:
                result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'], 
                                      capture_output=True, text=True, timeout=2)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')[1:]  # Skip header
                    for line in lines:
                        if 'python.exe' in line:
                            # Extract PID from CSV format
                            parts = line.split('","')
                            if len(parts) >= 2:
                                pid = int(parts[1].replace('"', ''))
                                # Quick check if this PID is running main.py
                                try:
                                    proc = psutil.Process(pid)
                                    cmdline = ' '.join(proc.cmdline())
                                    if 'main.py' in cmdline:
                                        return True, pid
                                except (psutil.NoSuchProcess, psutil.AccessDenied):
                                    continue
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
        
        # Method 3: Cached PID check (fastest for repeated calls)
        if hasattr(st.session_state, 'last_known_pid') and st.session_state.last_known_pid:
            try:
                proc = psutil.Process(st.session_state.last_known_pid)
                if proc.is_running():
                    cmdline = ' '.join(proc.cmdline())
                    if 'main.py' in cmdline:
                        return True, st.session_state.last_known_pid
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                st.session_state.last_known_pid = None
        
        # Method 4: Optimized psutil fallback (only if other methods fail)
        # Only check python processes, not all processes
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if proc.info['name'] and 'python' in proc.info['name'].lower():
                    cmdline = ' '.join(proc.cmdline())
                    if 'main.py' in cmdline:
                        return True, proc.info['pid']
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
                
    except Exception:
        pass
    
    return False, None

def start_main_py():
    """Start main.py in a new terminal window"""
    try:
        import platform
        import sys
        system = platform.system()
        
        if system == "Windows":
            # Windows - use specific Python executable path and & operator
            python_exe = sys.executable  # Gets the current Python executable path
            main_py_path = os.path.join(os.getcwd(), "main.py")
            
            # Use & operator to run in background and open new cmd window
            cmd = f'start cmd /k "{python_exe} {main_py_path}"'
            process = subprocess.Popen(cmd, shell=True, cwd=os.getcwd())
                
        else:  # Linux
            # Try different terminal emulators in order of preference
            python_exe = sys.executable
            main_py_path = os.path.join(os.getcwd(), "main.py")
            
            terminals = [
                ['gnome-terminal', '--', 'bash', '-c', f'cd {os.getcwd()} && {python_exe} {main_py_path}; exec bash'],
                ['xterm', '-e', f'cd {os.getcwd()} && {python_exe} {main_py_path}; exec bash'],
                ['konsole', '-e', f'cd {os.getcwd()} && {python_exe} {main_py_path}; exec bash'],
                ['x-terminal-emulator', '-e', f'cd {os.getcwd()} && {python_exe} {main_py_path}; exec bash']
            ]
            
            process = None
            for terminal_cmd in terminals:
                try:
                    process = subprocess.Popen(terminal_cmd, cwd=os.getcwd())
                    break
                except FileNotFoundError:
                    continue
            
            if process is None:
                # Fallback to background process if no terminal found
                process = subprocess.Popen([python_exe, main_py_path], 
                                         cwd=os.getcwd(),
                                         stdout=subprocess.PIPE, 
                                         stderr=subprocess.PIPE)
        
        
        time.sleep(1)
        
        running, actual_pid = is_main_py_running()
        if running:
            return True, actual_pid
        else:
            return True, process.pid
            
    except Exception as e:
        return False, str(e)

def stop_main_py():
    """Stop main.py process"""
    running, pid = is_main_py_running()
    if running:
        try:
            # Try to terminate gracefully first
            process = psutil.Process(pid)
            process.terminate()
            
            # Wait a bit for graceful shutdown
            try:
                process.wait(timeout=5)
            except psutil.TimeoutExpired:
                # Force kill if graceful shutdown didn't work
                process.kill()
            
            return True, "Process stopped successfully"
        except Exception as e:
            return False, str(e)
    else:
        return False, "Process not running"

    
# Initialize session state variables
if 'bot_should_run' not in st.session_state:
    st.session_state.bot_should_run = False
if 'bot_status_message' not in st.session_state:
    st.session_state.bot_status_message = ""
if 'last_known_pid' not in st.session_state:
    st.session_state.last_known_pid = None


# Sidebar z logowaniem — tak jak w poprzednim przykładzie
st.sidebar.header("🔐 Logowanie do MetaTrader 5")

if not st.session_state.get("mt5_initialized", False):
    # Formularz logowania
    with st.sidebar.form("login_form"):
        login = st.text_input("Login", value=st.session_state.get("login", os.getenv("MT5_LOGIN")), key="input_login")
        password = st.text_input("Hasło", type="password", value=st.session_state.get("password", os.getenv("MT5_PASSWORD")), key="input_password")
        server = st.text_input("Serwer", value=st.session_state.get("server", "MetaQuotes-Demo"), key="input_server")
        submitted = st.form_submit_button("Połącz z MT5")

    if submitted:
        st.session_state["login"] = login
        st.session_state["password"] = password
        st.session_state["server"] = server
        #print(login, password, server)
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



if st.session_state.get("mt5_initialized", False):
    # Wszystkie elementy, które chcesz pokazać tylko po zalogowaniu
    st.header("🤖 Kontrola Trading Bot")

    # Check current process status
    running, current_pid = is_main_py_running()

    # Handle bot control logic
    if st.session_state.bot_should_run and not running:
        # User wants bot to run but it's not running - try to start it
        success, result = start_main_py()
        if success:
            st.session_state.bot_status_message = f"✅ Trading Bot uruchomiony (PID: {result})"
            st.session_state.last_known_pid = result
        else:
            st.session_state.bot_status_message = f"❌ Błąd uruchamiania: {result}"
            st.session_state.bot_should_run = False  # Reset flag on failure

    elif not st.session_state.bot_should_run and running:
        # User wants bot to stop but it's still running - try to stop it
        success, message = stop_main_py()
        if success:
            st.session_state.bot_status_message = "🛑 Trading Bot zatrzymany"
            st.session_state.last_known_pid = None
        else:
            st.session_state.bot_status_message = f"❌ Błąd zatrzymywania: {message}"

    # Update last known PID if process is running
    if running:
        st.session_state.last_known_pid = current_pid

    else:
        st.info("⏹️ Trading Bot nie działa")
        if st.session_state.last_known_pid:
            st.text(f"Ostatni PID: {st.session_state.last_known_pid}")

    # Display status message if exists
    if st.session_state.bot_status_message:
        if "❌" in st.session_state.bot_status_message:
            st.error(st.session_state.bot_status_message)
        else:
            st.success(st.session_state.bot_status_message)

    # Control buttons
    col1, col2 = st.columns(2)

    with col1:
        if st.button("▶️ Start", disabled=running, key="start_bot_btn"):
            st.session_state.bot_should_run = True
            st.experimental_rerun()
            st.session_state.bot_status_message = "🔄 Uruchamianie..."
    with col2:
        if st.button("🛑 Stop", disabled=not running, key="stop_bot_btn"):
            st.session_state.bot_should_run = False
            #running = False
            time.sleep(4)
            st.session_state.bot_status_message = "🔄 Zatrzymywanie..."
            #st.experimental_rerun()

    
    trading_reload_min = int(get_setting("trading_reload", 5))
    trading_reload_min = st.number_input(
        "Czas odświeżania danych (w minutach)", 
        min_value=1, max_value=1440, value=trading_reload_min, step=1, key="trading_reload_min"
    )
    
    # Zapisz ustawienie przy zmianie
    if st.button("💾 Zapisz czas odświeżania"):
        set_setting("trading_reload", str(trading_reload_min))
        st.success("Zapisano czas odświeżania!")
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
            if st.button("Zapisz sentyment", key=symbol):
                set_sentiment(symbol, SENTIMENT_MAPPING[choice])
                st.success("Sentymenty!")

    
    st.subheader("📌 Aktualne wartości sentymentów")

    for symbol in symbols:
        user_sentiment = get_sentiment(symbol)  # zakładam, że taka funkcja istnieje
        st.markdown(f"**{symbol}**: {user_sentiment}")
    
    st.sidebar.header("⚙️ Ustawienia LSTM")
    min_candles = st.sidebar.number_input("Minimalna liczba świec", min_value=10, value=int(get_setting("min_candles_for_patterns", "150")))
    seq_length = st.sidebar.number_input("Długość sekwencji", min_value=5, value=int(get_setting("seq_len", "30")))

    st.sidebar.header("🚫 Ustawienia Ryzyka")

    MAX_RISK_PER_TRADE = st.sidebar.number_input(
        "Maksymalny procent alokacji", step=0.0001, format="%.6f",
        min_value=0.000001, value=float(get_setting("MAX_RISK_PER_TRADE", "0.0001"))
    )

    min_pips = int(get_setting("min_pips", 150))
    max_pips = int(get_setting("max_pips", 200))

    min_pips = st.sidebar.number_input("Minimalna liczba pipsów (SL)", min_value=1, max_value=1000, value=min_pips)
    max_pips = st.sidebar.number_input("Maksymalna liczba pipsów (SL)", min_value=min_pips, max_value=2000, value=max_pips)

    if st.sidebar.button("💾 Zapisz ustawienia"):
        st.text("Ustawiania modelu LSTM")
        set_setting("min_candles_for_patterns", str(min_candles))
        set_setting("seq_len", str(seq_length))
        st.sidebar.success("Zapisano ustawienia!")

        st.text("Ustawiania ryzyka")
        set_setting("MAX_RISK_PER_TRADE", str(MAX_RISK_PER_TRADE))
        set_setting("min_pips", str(min_pips))
        set_setting("max_pips", str(max_pips))
        set_setting("MAX_RISK_PER_TRADE", str(MAX_RISK_PER_TRADE))
        st.sidebar.success("Zapisano ustawienia!")

    open_pos = get_open_positions()
    open_pos_chart = open_pos.groupby(['symbol'])['profit'].agg('sum').reset_index()
    open_pos_chart['color'] = open_pos_chart['profit'].apply(lambda x: 'Zysk' if x >= 0 else 'Strata')
    
    #st.bar_chart(open_pos_chart, x="symbol", y="profit", color = 'color')
    fig = px.bar(open_pos_chart, x="profit", y="symbol", orientation='h', color='color',color_discrete_sequence=[
        "#ff2b2b",
        "#00ff51"
    ])
    

    st.plotly_chart(fig)
    try:
        deals = fetch_full_trade_history(days_back=14)
        deals = deals.iloc[1:]
        
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
    except KeyError:
        st.warning("Brak danych o transakcjach.")

    logs = read_logs_from_db(limit=200)
    logs_joined = "\n".join(logs)
    st.text_area("Logi", logs_joined, height=600, key="log_area", disabled=False)
else:
    st.info("🔐 Proszę się zalogować aby zobaczyć dane.")
