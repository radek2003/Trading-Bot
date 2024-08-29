import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import logging
import joblib
import math
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from ta import add_all_ta_features
from ta.utils import dropna
import tensorflow as tf

# Parametry zarządzania ryzykiem
MAX_RISK_PER_TRADE = 0.01  # 1% kapitału na transakcję
RISK_REWARD_RATIO = 2.0    # Stosunek zysku do ryzyka

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)

def configure_cpu():
    """Konfiguruje użycie CPU."""
    physical_devices = tf.config.list_physical_devices('CPU')
    if physical_devices:
        logging.info(f"Znaleziono CPU: {physical_devices}")

    num_threads = 6
    tf.config.threading.set_intra_op_parallelism_threads(num_threads)
    tf.config.threading.set_inter_op_parallelism_threads(num_threads)

def fetch_historical_data(symbol, timeframe=mt5.TIMEFRAME_M15, bars=1000):
    """Pobiera dane historyczne z MetaTrader 5."""
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        if rates is None or len(rates) == 0:
            logging.error("Brak danych historycznych.")
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'tick_volume']]
        return df
    except Exception as e:
        logging.exception("Problem z pobieraniem danych historycznych.")
        return pd.DataFrame()

def extract_features(data):
    """Ekstrakt cechy z danych i dodaje wskaźniki techniczne."""
    try:
        if data.empty:
            logging.error("Brak danych do ekstrakcji cech.")
            return pd.DataFrame()

        data = dropna(data)
        data = add_all_ta_features(
            data, open="open", high="high", low="low", close="close", volume="tick_volume", fillna=True)

        data['Target'] = np.where(data['close'].shift(-1) > data['close'], 1, 0)

        return data.dropna().drop(['tick_volume'], axis=1)
    except Exception as e:
        logging.exception("Problem z ekstrakcją cech.")
        return pd.DataFrame()

def calculate_moving_averages(data):
    """Oblicza średnie kroczące i generuje sygnały kupna/sprzedaży."""
    if data.empty:
        logging.error("Brak danych do obliczeń.")
        return pd.DataFrame()

    try:
        data['MA20'] = data['close'].rolling(window=20, min_periods=1).mean()
        data['MA60'] = data['close'].rolling(window=60, min_periods=1).mean()
        data['MA100'] = data['close'].rolling(window=100, min_periods=1).mean()
        data['Signal'] = np.where(data['MA20'] > data['MA60'], 1, -1)
        data['Buy_Signal'] = (data['Signal'].shift(1) == -1) & (data['Signal'] == 1)
        data['Sell_Signal'] = (data['Signal'].shift(1) == 1) & (data['Signal'] == -1)
        data['Target'] = (data['close'].shift(-1) > data['close']).astype(int)
        return data.dropna().drop(['tick_volume'], axis=1)
    except Exception as e:
        logging.exception("Problem z obliczaniem średnich kroczących.")
        return pd.DataFrame()

def calculate_macd(df):
    """Oblicza wskaźnik MACD i jego sygnał."""
    df = df.copy()
    df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    return df

def save_model(model, folder_path, filename='best_model.pkl'):
    """Zapisuje model do pliku w określonym folderze."""
    try:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_path = os.path.join(folder_path, filename)
        joblib.dump(model, file_path)
        logging.info(f"Model zapisany jako {file_path}")
    except Exception as e:
        logging.exception("Problem z zapisywaniem modelu.")

def load_model(folder_path, filename='best_model.pkl'):
    """Ładuje model z pliku z określonego folderu."""
    try:
        file_path = os.path.join(folder_path, filename)
        model = joblib.load(file_path)
        logging.info(f"Model załadowany z {file_path}")
        return model
    except Exception as e:
        logging.exception("Problem z ładowaniem modelu.")
        return None

def train_model(data, folder_path='models', model_filename='best_model.pkl'):
    """Trenuje model klasyfikacji z użyciem RandomizedSearchCV i zapisuje najlepszy model."""
    try:
        if data.empty:
            logging.error("Brak danych do trenowania modelu.")
            return None, None

        X = data.drop('Target', axis=1).values
        y = data['Target'].values

        if len(X) < 2:
            logging.error("Za mało danych do trenowania modelu.")
            return None, None

        # Zrównoważenie klas
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_resampled)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42)

        # Ładowanie istniejącego modelu, jeśli dostępny
        existing_model = load_model(folder_path, model_filename)

        if existing_model:
            model = existing_model
            logging.info("Kontynuowanie treningu na podstawie istniejącego modelu.")
        else:
            model = RandomForestClassifier(random_state=42)

        param_dist = {
            'n_estimators': [50, 100, 150, 200, 250, 300, 400, 500],
            'max_depth': [None, 10, 20, 30, 40, 50, 60, 70, 80],
            'min_samples_split': [2, 5, 10, 15, 20, 25, 30],
            'min_samples_leaf': [1, 2, 4, 6, 8, 10, 12],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }

        random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=10, cv=5,
                                           scoring='accuracy', random_state=42, error_score='raise')
        random_search.fit(X_train, y_train)

        best_model = random_search.best_estimator_
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"Najlepsze parametry: {random_search.best_params_}")
        logging.info(f"Dokładność modelu: {accuracy:.2%}")

        # Zapisz najlepszy model
        save_model(best_model, folder_path, model_filename)

        return best_model, scaler

    except Exception as e:
        logging.error(f"Problem z trenowaniem modelu: {e}")
        return None, None


def apply_strategy(model, scaler, new_data, strategy_type='MACD'):
    """Stosuje wybraną strategię na nowych danych."""
    try:
        if strategy_type == 'MACD':
            new_data_with_features = calculate_macd(new_data)
            new_data_with_features = calculate_moving_averages(new_data_with_features)
        else:
            logging.error(f"Nieznana strategia: {strategy_type}")
            return

        if new_data_with_features.empty:
            logging.error("Brak danych z cechami do prognozowania.")
            return

        X_new = scaler.transform(new_data_with_features.drop('Target', axis=1).values)
        predictions = model.predict(X_new)

        return predictions

    except Exception as e:
        logging.exception("Problem z zastosowaniem strategii.")

def calculate_position_size(account_balance, symbol):
    """Oblicza rozmiar pozycji na podstawie kapitału i symbolu."""
    try:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logging.error("Nie można pobrać informacji o symbolu.")
            return 0

        min_lot = symbol_info.volume_min
        step_lot = symbol_info.volume_step
        max_lot = symbol_info.volume_max
        punkt = symbol_info.point
        trade_contract_size = symbol_info.trade_contract_size

        if punkt is None or trade_contract_size is None:
            logging.error("Brak informacji o punktach lub rozmiarze kontraktu dla symbolu.")
            return 0

        punkt_value = punkt * trade_contract_size
        ryzyko_na_transakcję = account_balance * MAX_RISK_PER_TRADE
        stop_loss_value = ryzyko_na_transakcję
        wielkość_pozycji = stop_loss_value / punkt_value

        if wielkość_pozycji < min_lot:
            wielkość_pozycji = min_lot
        if wielkość_pozycji > max_lot:
            wielkość_pozycji = max_lot

        wielkość_pozycji = math.floor(wielkość_pozycji / step_lot) * step_lot

        return wielkość_pozycji
    except Exception as e:
        logging.exception("Problem z obliczaniem wielkości pozycji.")
        return 0

def execute_trade(model_prediction, symbol):
    """Wykonuje transakcję na podstawie predykcji modelu."""
    try:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logging.error("Nie można pobrać informacji o symbolu.")
            return

        punkt = symbol_info.point
        digits = symbol_info.digits

        if punkt is None:
            logging.error("Brak informacji o punktach dla symbolu.")
            return

        # Pobierz aktualne ceny
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logging.error("Nie można pobrać aktualnych cen.")
            return

        current_price = tick.ask if model_prediction == 1 else tick.bid

        # Obliczanie wartości stop loss i take profit w punktach
        stop_loss_value = 0.05 * current_price
        take_profit_value = stop_loss_value * RISK_REWARD_RATIO

        if model_prediction == 1:  # BUY
            stop_loss_price = current_price - stop_loss_value
            take_profit_price = current_price + take_profit_value
            order_type = mt5.ORDER_TYPE_BUY
        else:  # SELL
            stop_loss_price = current_price + stop_loss_value
            take_profit_price = current_price - take_profit_value
            order_type = mt5.ORDER_TYPE_SELL

        stop_loss_price = round(stop_loss_price, digits)
        take_profit_price = round(take_profit_price, digits)

        if stop_loss_price <= 0 or take_profit_price <= 0:
            logging.error("Błąd: Niepoprawne ceny SL/TP.")
            return

        account_info = mt5.account_info()
        if account_info is None:
            logging.error("Nie można pobrać informacji o koncie.")
            return

        account_balance = account_info.balance
        wielkość_pozycji = calculate_position_size(account_balance, symbol)
        if wielkość_pozycji <= 0:
            logging.error("Wielkość pozycji jest równa 0, transakcja nie zostanie przeprowadzona.")
            return

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": wielkość_pozycji,
            "type": order_type,
            "price": current_price,
            "sl": stop_loss_price,
            "tp": take_profit_price,
            "magic": 234000,
            "comment": "Zautomatyzowany handel",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Błąd wykonania transakcji: {result.retcode}, {mt5.last_error()}")
        else:
            logging.info(f"Transakcja przeprowadzona pomyślnie, ID pozycji: {result.order}")

    except Exception as e:
        logging.exception("Problem z realizacją transakcji.")

def main():
    """Główna funkcja programu."""
    configure_cpu()

    if not mt5.initialize():
        logging.error(f"Inicjalizacja MT5 nie powiodła się, kod błędu: {mt5.last_error()}")
        return

    symbol = "EURUSD"
    data = fetch_historical_data(symbol)

    if data.empty:
        logging.error("Brak danych do analizy.")
        mt5.shutdown()
        return

    # Wybór strategii
    strategy_type = 'MACD'  # Można zmienić na 'Heikin-Ashi' w zależności od wybranej strategii

    if strategy_type == 'MACD':
        data_with_features = calculate_moving_averages(data)
        data_with_features = calculate_macd(data_with_features)
    else:
        logging.error(f"Nieznana strategia: {strategy_type}")
        mt5.shutdown()
        return

    if data_with_features.empty:
        logging.error("Brak danych z cechami do trenowania modelu.")
        mt5.shutdown()
        return

    # Ścieżka i nazwa pliku modelu
    folder_path = r"C:\trenowanie modelu do bota"
    model_filename = 'best_model.pkl'

    # Trenowanie modelu
    model, scaler = train_model(data_with_features, folder_path, model_filename)

    if model is None or scaler is None:
        logging.error("Model lub skalowanie nie zostały poprawnie załadowane.")
        mt5.shutdown()
        return

    # Pobranie nowych danych
    new_data = fetch_historical_data(symbol)
    if not new_data.empty:
        predictions = apply_strategy(model, scaler, new_data, strategy_type)
        if predictions is not None:
            latest_data = data_with_features.iloc[-1:]
            latest_data_scaled = scaler.transform(latest_data.drop('Target', axis=1))
            model_prediction = model.predict(latest_data_scaled)[0]
            execute_trade(model_prediction, symbol)

    mt5.shutdown()

if __name__ == "__main__":
    main()
