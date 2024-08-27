import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from ta import add_all_ta_features
from ta.utils import dropna
import requests
import math
import time

# Parametry zarządzania ryzykiem
MAX_RISK_PER_TRADE = 0.01  # 1% kapitału na transakcję
RISK_REWARD_RATIO = 2.0    # Stosunek zysku do ryzyka

def fetch_historical_data(symbol, timeframe=mt5.TIMEFRAME_M1, bars=1000):
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        if rates is None or len(rates) == 0:
            print("Brak danych historycznych.")
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'tick_volume']]
        return df
    except Exception as e:
        print(f"Problem z pobieraniem danych historycznych: {e}")
        return pd.DataFrame()

def extract_features(data):
    try:
        if data.empty:
            print("Brak danych do ekstrakcji cech.")
            return pd.DataFrame()

        data = dropna(data)
        data = add_all_ta_features(
            data, open="open", high="high", low="low", close="close", volume="tick_volume", fillna=True)

        data['Target'] = np.where(data['close'].shift(-1) > data['close'], 1, 0)

        required_columns = ['open', 'high', 'low', 'close', 'tick_volume', 'Target']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            print(f"Brakujące kolumny: {', '.join(missing_columns)}")
            return pd.DataFrame()

        return data.drop(['tick_volume'], axis=1).dropna()
    except Exception as e:
        print(f"Problem z ekstrakcją cech: {e}")
        return pd.DataFrame()

def train_neural_network(data):
    try:
        if data.empty:
            print("Brak danych do trenowania modelu.")
            return None, None

        X = data.drop('Target', axis=1)
        y = data['Target']

        if len(X) < 2:
            print("Za mało danych do trenowania modelu.")
            return None, None

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        neural_network = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            shuffle=True,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            verbose=True
        )

        neural_network.fit(X_train, y_train)

        predictions = neural_network.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        print(f"Dokładność modelu sieci neuronowej: {accuracy:.2%}")

        return neural_network, scaler
    except Exception as e:
        print(f"Problem z trenowaniem modelu: {e}")
        return None, None

def calculate_position_size(account_balance, symbol):
    try:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print("Nie można pobrać informacji o symbolu.")
            return 0

        min_lot = symbol_info.volume_min
        step_lot = symbol_info.volume_step
        max_lot = symbol_info.volume_max
        punkt = symbol_info.point
        trade_contract_size = symbol_info.trade_contract_size

        if punkt is None or trade_contract_size is None:
            print("Brak informacji o punktach lub rozmiarze kontraktu dla symbolu.")
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
        print(f"Problem z obliczaniem wielkości pozycji: {e}")
        return 0

def execute_trade(model_prediction, symbol):
    try:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print("Nie można pobrać informacji o symbolu.")
            return

        punkt = symbol_info.point
        digits = symbol_info.digits

        if punkt is None:
            print("Brak informacji o punktach dla symbolu.")
            return

        # Pobierz aktualne ceny
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            print("Nie można pobrać aktualnych cen.")
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
            print("Błąd: Niepoprawne ceny SL/TP.")
            return

        account_info = mt5.account_info()
        if account_info is None:
            print("Nie można pobrać informacji o koncie.")
            return

        account_balance = account_info.balance
        wielkość_pozycji = calculate_position_size(account_balance, symbol)
        if wielkość_pozycji <= 0:
            print("Wielkość pozycji jest równa 0, transakcja nie zostanie przeprowadzona.")
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
            print(f"Błąd wykonania transakcji: {result.retcode}, {mt5.last_error()}")
        else:
            print(f"Transakcja przeprowadzona pomyślnie, ID pozycji: {result.order}")

    except Exception as e:
        print(f"Problem z realizacją transakcji: {e}")

def main():
    if not mt5.initialize():
        print("Inicjalizacja MT5 nie powiodła się, kod błędu:", mt5.last_error())
        return

    symbol = "EURUSD"
    data = fetch_historical_data(symbol)
    data_with_features = extract_features(data)
    model, scaler = train_neural_network(data_with_features)

    if model is None or scaler is None:
        print("Błąd: Model lub skalowanie nie zostały poprawnie załadowane.")
        mt5.shutdown()
        return

    account_info = mt5.account_info()
    if account_info is None:
        print("Nie można pobrać informacji o koncie.")
        mt5.shutdown()
        return

    account_balance = account_info.balance

    # Pobierz aktualne ceny
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print("Nie można pobrać aktualnych cen.")
        mt5.shutdown()
        return

    last_price = tick.ask

    # Przygotuj dane do predykcji
    latest_data = data_with_features.iloc[-1:]
    latest_data_scaled = scaler.transform(latest_data.drop('Target', axis=1))

    # Wykonaj predykcję
    model_prediction = model.predict(latest_data_scaled)[0]

    # Wykonaj transakcję
    execute_trade(model_prediction, symbol)

    mt5.shutdown()

if __name__ == "__main__":
    main()
