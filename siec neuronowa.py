import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import joblib
import os
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

logging.basicConfig(level=logging.INFO)

def configure_cpu():
    """Konfiguruje użycie CPU."""
    import tensorflow as tf
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
        logging.error(f"Problem z pobieraniem danych historycznych: {e}")
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
        logging.error(f"Problem z obliczaniem średnich kroczących: {e}")
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
        logging.error(f"Problem z zapisywaniem modelu: {e}")

def load_model(folder_path, filename='best_model.pkl'):
    """Ładuje model z pliku z określonego folderu."""
    try:
        file_path = os.path.join(folder_path, filename)
        model = joblib.load(file_path)
        logging.info(f"Model załadowany z {file_path}")
        return model
    except Exception as e:
        logging.error(f"Problem z ładowaniem modelu: {e}")
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
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 10, 20, 30, 40],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4, 6]
        }

        random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=10, cv=5,
                                           scoring='accuracy', random_state=42)
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


    except Exception as e:
        logging.error(f"Problem z zastosowaniem strategii: {e}")

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

    folder_path = r"C:\trenowanie modelu do bota"
    model_filename = 'best_model.pkl'
    model, scaler = train_model(data_with_features, folder_path, model_filename)

    if model is None or scaler is None:
        logging.error("Model lub skalowanie nie zostały poprawnie załadowane.")
        mt5.shutdown()
        return

    new_data = fetch_historical_data(symbol)
    if not new_data.empty:
        apply_strategy(model, scaler, new_data, strategy_type)

    mt5.shutdown()

if __name__ == "__main__":
    main()
