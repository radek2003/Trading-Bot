import MetaTrader5 as mt5
import logging
import pandas as pd
import time
import os
from src.data_fetcher import fetch_historical_data, test_trade_history
from src.model_methods import train_model_with_history
from src.trading import execute_trade, check_for_closed_positions
from src.strategy import apply_strategy, calculate_candlestick_patterns, calculate_macd, calculate_moving_averages

# Wyciszenie ostrzeżeń OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Konfiguracja logowania
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

# Definicja strategii
STRATEGIES = {
    'Candlestick': calculate_candlestick_patterns,
    'MACD': calculate_macd,
    'MovingAverages': calculate_moving_averages,
}

def apply_strategies(data, strategies):
    """Aplikuje wszystkie strategie na dane i łączy ich cechy z jednym Target."""
    data_with_features = data.copy()
    target_col = None  # Przechowuje główny Target

    for strategy_name, strategy_func in strategies.items():
        logging.info(f"Obliczanie cech dla strategii: {strategy_name}")
        try:
            features = strategy_func(data_with_features)
            if features.empty:
                logging.warning(f"Strategia {strategy_name} zwróciła pusty DataFrame.")
                continue

            if 'Target' in features.columns:
                if target_col is None:
                    target_col = features['Target']
                features = features.drop('Target', axis=1)

            new_columns = [col for col in features.columns if col not in data_with_features.columns]
            data_with_features = pd.concat([data_with_features, features[new_columns]], axis=1)
        except Exception as e:
            logging.error(f"Błąd w strategii {strategy_name}: {str(e)}")

    if target_col is not None:
        data_with_features['Target'] = target_col
    else:
        logging.warning("Brak kolumny Target w strategiach, definiuję domyślny.")
        data_with_features['Target'] = (data_with_features['close'].shift(-1) > data_with_features['close']).astype(int)

    return data_with_features

def main():
    """Główna funkcja programu."""
    if not mt5.initialize():
        logging.error(f"Inicjalizacja MT5 nie powiodła się, kod błędu: {mt5.last_error()}")
        return

    symbol = "EURUSD"

    try:
        while True:
            logging.info("Rozpoczynanie nowej iteracji handlu")

            # Pobieranie danych
            data_m5, data_m15 = fetch_historical_data(symbol)
            trade_history = test_trade_history()

            if data_m5.empty or data_m15.empty:
                logging.error("Brak danych 5-minutowych lub 15-minutowych do analizy.")
                continue

            if trade_history.empty:
                logging.warning("Brak historii zleceń, kontynuuję bez dodatkowych danych.")
            else:
                trade_history = trade_history[trade_history['symbol'] == symbol]
                logging.info(f"Pobrano historię zleceń dla {symbol}: {trade_history.shape}")

            data = data_m5

            # Zastosowanie wszystkich strategii
            data_with_features = apply_strategies(data, STRATEGIES)

            if data_with_features.empty:
                logging.error("Brak danych z cechami do analizy.")
                continue

            # Dodanie danych z historii zleceń
            if not trade_history.empty:
                data_with_features = data_with_features.merge(trade_history, on='time', how='left').fillna(0)
            else:
                data_with_features['profit'] = 0
                data_with_features['volume'] = 0
                data_with_features['type'] = 0

            # Dodanie dodatkowych cech
            if 'time' in data_with_features.columns:
                data_with_features['hour'] = pd.to_datetime(data_with_features['time']).dt.hour
            if 'profit' in data_with_features.columns:
                data_with_features['trade_success'] = (data_with_features['profit'] > 0).astype(int)
            if 'type' in data_with_features.columns:
                data_with_features['trade_type'] = data_with_features['type'].map({1: 1, 0: 0}).fillna(0)

            if data_with_features.empty:
                logging.error("Brak danych z cechami do trenowania modelu.")
                continue

            # Trenowanie modelu
            folder_path = r"C:\trenowanie modelu do bota"
            model_filename = 'best_model.pth'
            model, scaler, training_columns = train_model_with_history(data_with_features, folder_path, model_filename)

            if model is None or scaler is None or training_columns is None:
                logging.error("Model, skalowanie lub kolumny treningowe nie zostały poprawnie załadowane.")
                continue

            # Pobieranie nowych danych do predykcji
            new_data_m5, new_data_m15 = fetch_historical_data(symbol)
            if new_data_m5.empty:
                logging.error("Brak nowych danych 5-minutowych do analizy.")
                continue

            latest_data = new_data_m5.iloc[-1:]
            latest_data_with_features = apply_strategies(latest_data, STRATEGIES)

            if latest_data_with_features.empty:
                logging.error("Brak danych z cechami do przewidywań.")
                continue

            # Dodanie danych z historii zleceń dla predykcji
            latest_trade_history = test_trade_history()
            if not latest_trade_history.empty:
                latest_trade_history = latest_trade_history[latest_trade_history['symbol'] == symbol]
                latest_data_with_features = latest_data_with_features.merge(latest_trade_history, on='time', how='left').fillna(0)
            else:
                latest_data_with_features['profit'] = 0
                latest_data_with_features['volume'] = 0
                latest_data_with_features['type'] = 0

            # Dodanie tych samych cech co w danych treningowych
            if 'time' in latest_data_with_features.columns:
                latest_data_with_features['hour'] = pd.to_datetime(latest_data_with_features['time']).dt.hour
            if 'profit' in latest_data_with_features.columns:
                latest_data_with_features['trade_success'] = (latest_data_with_features['profit'] > 0).astype(int)
            if 'type' in latest_data_with_features.columns:
                latest_data_with_features['trade_type'] = latest_data_with_features['type'].map({1: 1, 0: 0}).fillna(0)

            # Dopasowanie kolumn do treningowych
            for col in training_columns:
                if col not in latest_data_with_features.columns:
                    logging.warning(f"Dodaję brakującą kolumnę w predykcji: {col}")
                    latest_data_with_features[col] = 0
            extra_columns = [col for col in latest_data_with_features.columns if col not in training_columns]
            if extra_columns:
                logging.warning(f"Usuwam nadmiarowe kolumny w predykcji: {extra_columns}")
                latest_data_with_features = latest_data_with_features.drop(columns=extra_columns)

            # Wybór cech do skalowania i predykcji
            features_to_scale = latest_data_with_features[training_columns]
            logging.debug(f"Kolumny treningowe: {training_columns}")
            logging.debug(f"Kolumny do predykcji: {list(features_to_scale.columns)}")
            logging.debug(f"Liczba kolumn do predykcji: {features_to_scale.shape[1]}")

            # Skalowanie i predykcja
            try:
                predictions = apply_strategy(model, scaler, features_to_scale, 'Candlestick')
                if predictions is None:
                    logging.error("Predykcja nie powiodła się.")
                    continue

                model_prediction = predictions[0]  # Bierzemy pierwszą predykcję (dla jednego wiersza)
                execute_trade(model_prediction, symbol)
                check_for_closed_positions(symbol)

            except Exception as e:
                logging.error(f"Błąd podczas predykcji lub realizacji transakcji: {str(e)}")
                continue

            logging.info("Oczekiwanie 15 minut przed następną iteracją...")
            time.sleep(900)

    except KeyboardInterrupt:
        logging.info("Przerwano działanie bota przez użytkownika.")
    except Exception as e:
        logging.error(f"Wystąpił nieoczekiwany błąd: {str(e)}")
    finally:
        mt5.shutdown()
        logging.info("Zamknięcie po zakończeniu handlu")

if __name__ == "__main__":
    pd.set_option('future.no_silent_downcasting', True)
    main()