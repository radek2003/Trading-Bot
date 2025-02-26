import MetaTrader5 as mt5
import logging
import pandas as pd
import time
import os
import torch
from src.data_fetcher import fetch_historical_data, test_trade_history
from src.model_methods import train_model_with_history, mc_dropout_predict
from src.trading import execute_trade, check_for_closed_positions
from src.strategy import calculate_candlestick_patterns, calculate_robust_macd, calculate_robust_moving_averages
from src.risk_management import calculate_position_size

# Wyciszenie ostrzeżeń OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Konfiguracja logowania
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

# Definicja strategii
STRATEGIES = {
    'Candlestick': calculate_candlestick_patterns,
    'MACD': calculate_robust_macd,
    'MovingAverages': calculate_robust_moving_averages,
}

def apply_strategies(data, strategies):
    """Aplikuje wszystkie strategie na dane i łączy ich cechy z jednym Target."""
    data_with_features = data.copy()
    target_col = None

    for strategy_name, strategy_func in strategies.items():
        logging.info(f"Obliczanie cech dla strategii: {strategy_name}")
        try:
            # Używamy close_smooth zamiast close dla większej stabilności
            features = strategy_func(data_with_features.assign(close=data_with_features['close_smooth']))
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
        logging.warning("Brak kolumny Target, definiuję domyślny.")
        data_with_features['Target'] = (data_with_features['close'].shift(-1) > data_with_features['close']).astype(int)

    return data_with_features

def main():
    """Główna funkcja programu."""
    if not mt5.initialize():
        logging.error(f"Inicjalizacja MT5 nie powiodła się, kod błędu: {mt5.last_error()}")
        return

    symbol = "EURUSD"
    min_candles_for_patterns = 100  # Minimalna liczba świec do obliczenia formacji

    try:
        while True:
            logging.info("Rozpoczynanie nowej iteracji handlu")

            # Pobieranie danych
            data_m5, data_m15 = fetch_historical_data(symbol)
            trade_history = test_trade_history()

            if data_m5.empty or data_m15.empty:
                logging.error("Brak danych 5-minutowych lub 15-minutowych.")
                continue

            if len(data_m5) < min_candles_for_patterns:
                logging.error(f"Za mało danych historycznych ({len(data_m5)} świec) do obliczenia formacji.")
                continue

            if trade_history.empty:
                logging.warning("Brak historii zleceń, kontynuuję bez dodatkowych danych.")
            else:
                trade_history = trade_history[trade_history['symbol'] == symbol]
                logging.info(f"Pobrano historię zleceń dla {symbol}: {trade_history.shape}")

            data = data_m5

            # Zastosowanie wszystkich strategii z wygładzonymi cenami
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
                logging.error("Model, skalowanie lub kolumny treningowe nie zostały załadowane.")
                continue

            # Pobieranie nowych danych do predykcji
            new_data_m5, new_data_m15 = fetch_historical_data(symbol)
            if new_data_m5.empty:
                logging.error("Brak nowych danych 5-minutowych.")
                continue

            # Bierzemy ostatnie 6 świec do obliczenia formacji, predykcja dla ostatniej
            latest_data = new_data_m5.tail(min_candles_for_patterns)
            if len(latest_data) < min_candles_for_patterns:
                logging.error(f"Za mało danych w latest_data ({len(latest_data)} świec) do obliczenia formacji.")
                continue

            # Używamy close_smooth zamiast close w danych predykcyjnych
            latest_data_with_features = apply_strategies(latest_data, STRATEGIES)

            if latest_data_with_features.empty:
                logging.error("Brak danych z cechami do przewidywań.")
                continue

            # Dodanie danych z historii zleceń
            latest_trade_history = test_trade_history()
            if not latest_trade_history.empty:
                latest_trade_history = latest_trade_history[latest_trade_history['symbol'] == symbol]
                latest_data_with_features = latest_data_with_features.merge(latest_trade_history, on='time', how='left').fillna(0)
            else:
                latest_data_with_features['profit'] = 0
                latest_data_with_features['volume'] = 0
                latest_data_with_features['type'] = 0

            # Dodanie tych samych cech co w treningu
            if 'time' in latest_data_with_features.columns:
                latest_data_with_features['hour'] = pd.to_datetime(latest_data_with_features['time']).dt.hour
            if 'profit' in latest_data_with_features.columns:
                latest_data_with_features['trade_success'] = (latest_data_with_features['profit'] > 0).astype(int)
            if 'type' in latest_data_with_features.columns:
                latest_data_with_features['trade_type'] = latest_data_with_features['type'].map({1: 1, 0: 0}).fillna(0)

            # Dopasowanie kolumn
            for col in training_columns:
                if col not in latest_data_with_features.columns:
                    logging.warning(f"Dodaję brakującą kolumnę: {col}")
                    latest_data_with_features[col] = 0
            extra_columns = [col for col in latest_data_with_features.columns if col not in training_columns]
            if extra_columns:
                logging.warning(f"Usuwam nadmiarowe kolumny: {extra_columns}")
                latest_data_with_features = latest_data_with_features.drop(columns=extra_columns)

            # Wybór cech do predykcji (tylko ostatnia świeca)
            features_to_scale = latest_data_with_features[training_columns].iloc[-1:]
            logging.debug(f"Kolumny do predykcji: {list(features_to_scale.columns)}")

            # Skalowanie i predykcja z Monte Carlo Dropout
            try:
                features_scaled = scaler.transform(features_to_scale.values)
                predictions_mean, predictions_std = mc_dropout_predict(model, torch.tensor(features_scaled, dtype=torch.float32), num_samples=100, device='cuda' if torch.cuda.is_available() else 'cpu')
                if predictions_mean is None:
                    logging.error("Predykcja nie powiodła się.")
                    continue

                model_prediction = torch.argmax(predictions_mean, dim=1)[0].item()
                prediction_confidence = predictions_std[0].max().item()
                logging.debug(f"Predykcja: {model_prediction}, Prawdopodobieństwa: {predictions_mean[0].tolist()}, Niepewność: {prediction_confidence:.4f}")

                confidence_threshold = 0.1
                if prediction_confidence <= confidence_threshold:
                    account_info = mt5.account_info()
                    if account_info is None:
                        logging.error("Nie można pobrać informacji o koncie.")
                        continue
                    account_balance = account_info.balance
                    position_size = calculate_position_size(account_balance, symbol, data_m5)
                    if position_size > 0:
                        execute_trade(model_prediction, symbol, volume=position_size,
                                      historical_data=data_m5, confidence=prediction_confidence)  # Nowe wywołanie
                        check_for_closed_positions(symbol)
                    else:
                        logging.warning("Wielkość pozycji wynosi 0, pomijam transakcję.")
                else:
                    logging.info(
                        f"Robust prediction: brak akcji (predykcja: {model_prediction}, niepewność: {prediction_confidence:.4f})")

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