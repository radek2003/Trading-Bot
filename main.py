import MetaTrader5 as mt5
import logging
import pandas as pd
import time
import os
import torch
from src.data_fetcher import fetch_historical_data, test_trade_history
from src.model_methods import train_model_with_history, mc_dropout_predict, load_model
from src.trading import execute_trade, check_for_closed_positions, integrate_with_main
from src.strategy import apply_strategy, calculate_all_features
from src.risk_management import calculate_position_size

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', force=True)


def main():
    if not mt5.initialize():
        logging.error(f"Inicjalizacja MT5 nie powiodła się, kod błędu: {mt5.last_error()}")
        return

    symbol = "EURUSD"
    min_candles_for_patterns = 200
    seq_len = 30

    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(script_dir, "models")

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        logging.info(f"Utworzono folder models: {folder_path}")

    sl_tp_thread = integrate_with_main(symbol=symbol)

    try:
        while True:
            logging.info("Rozpoczynanie nowej iteracji handlu")

            data_m5, data_m15 = fetch_historical_data(symbol)
            trade_history = test_trade_history()

            if data_m5.empty or data_m15.empty:
                logging.error("Brak danych 5-minutowych lub 15-minutowych.")
                continue

            if len(data_m5) < min_candles_for_patterns:
                logging.error(f"Za mało danych historycznych ({len(data_m5)} świec) do obliczenia formacji.")
                continue

            data = data_m5

            data_with_features = calculate_all_features(data)
            if data_with_features.empty:
                logging.error("Brak danych z cechami do analizy.")
                continue

            if not trade_history.empty:
                data_with_features = data_with_features.merge(trade_history, on='time', how='left').fillna(0)
                data_with_features = data_with_features.drop(columns=['symbol'], errors='ignore')
            else:
                data_with_features['profit'] = 0
                data_with_features['volume'] = 0
                data_with_features['type'] = 0

            if 'time' in data_with_features.columns:
                data_with_features['hour'] = pd.to_datetime(data_with_features['time']).dt.hour
                data_with_features = data_with_features.drop(columns=['time'])
            if 'profit' in data_with_features.columns:
                data_with_features['trade_success'] = (data_with_features['profit'] > 0).astype(int)
            if 'type' in data_with_features.columns:
                data_with_features['trade_type'] = data_with_features['type'].map({1: 1, 0: 0}).fillna(0)

            model_filename = 'best_model.pth'
            model, scaler, training_columns = train_model_with_history(data_with_features, folder_path, model_filename)

            if model is None or scaler is None or training_columns is None:
                logging.error("Model, skalowanie lub kolumny treningowe nie zostały załadowane.")
                continue

            new_data_m5, new_data_m15 = fetch_historical_data(symbol)
            if new_data_m5.empty:
                logging.error("Brak nowych danych 5-minutowych.")
                continue

            latest_data = new_data_m5.tail(min_candles_for_patterns + seq_len - 1)
            if len(latest_data) < seq_len:
                logging.error(
                    f"Za mało danych w latest_data ({len(latest_data)} świec) dla sekwencji LSTM ({seq_len}).")
                continue

            latest_data_with_features = calculate_all_features(latest_data)
            if latest_data_with_features.empty:
                logging.error("Brak danych z cechami do przewidywań.")
                continue

            latest_trade_history = test_trade_history()
            if not latest_trade_history.empty:
                latest_data_with_features = latest_data_with_features.merge(latest_trade_history, on='time',
                                                                            how='left').fillna(0)
                latest_data_with_features = latest_data_with_features.drop(columns=['symbol'], errors='ignore')
            else:
                latest_data_with_features['profit'] = 0
                latest_data_with_features['volume'] = 0
                latest_data_with_features['type'] = 0

            if 'time' in latest_data_with_features.columns:
                latest_data_with_features['hour'] = pd.to_datetime(latest_data_with_features['time']).dt.hour
                latest_data_with_features = latest_data_with_features.drop(columns=['time'])
            if 'profit' in latest_data_with_features.columns:
                latest_data_with_features['trade_success'] = (latest_data_with_features['profit'] > 0).astype(int)
            if 'type' in latest_data_with_features.columns:
                latest_data_with_features['trade_type'] = latest_data_with_features['type'].map({1: 1, 0: 0}).fillna(0)

            try:
                if not hasattr(scaler, 'feature_names'):
                    logging.error("Scaler nie posiada informacji o cechach!")
                    continue

                expected_features = scaler.feature_names
                missing = list(set(expected_features) - set(latest_data_with_features.columns))
                extra = list(set(latest_data_with_features.columns) - set(expected_features))

                # Dodaj brakujące cechy
                for feat in missing:
                    latest_data_with_features[feat] = 0
                    logging.warning(f"Dodano brakującą cechę: {feat}")

                # Usuń nadmiarowe cechy
                latest_data_with_features = latest_data_with_features[expected_features]

            except Exception as e:
                logging.error(f"Błąd przygotowania cech: {str(e)}")
                continue

            features_to_scale = latest_data_with_features.tail(seq_len)
            if len(features_to_scale) < seq_len:
                logging.error(f"Za mało danych po filtracji ({len(features_to_scale)}) dla sekwencji LSTM ({seq_len}).")
                continue

            features_scaled = scaler.transform(features_to_scale.values)
            features_scaled = features_scaled.reshape(1, seq_len, -1)
            logging.debug(f"Shape danych do predykcji: {features_scaled.shape}")

            try:
                features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
                predictions_mean, predictions_std = mc_dropout_predict(
                    model,
                    features_tensor,
                    num_samples=100,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )

                if predictions_mean is None:
                    logging.error("Predykcja nie powiodła się.")
                    continue

                model_prediction = torch.argmax(predictions_mean, dim=1)[0].item()
                prediction_confidence = predictions_std[0].max().item()
                logging.debug(
                    f"Predykcja: {model_prediction}, Prawdopodobieństwa: {predictions_mean[0].tolist()}, Niepewność: {prediction_confidence:.4f}")

                robust_prediction = apply_strategy(model, scaler, latest_data_with_features.tail(seq_len), gamma=0.1,
                                                   num_samples=100)
                if robust_prediction is None:
                    logging.error("Strategia nie zwróciła predykcji.")
                    continue

                logging.info(
                    f"Robust prediction from strategy: {robust_prediction}, Model prediction: {model_prediction}, Uncertainty: {prediction_confidence:.4f}")

                if robust_prediction != 0:
                    account_info = mt5.account_info()
                    if account_info is None:
                        logging.error("Nie można pobrać informacji o koncie.")
                        continue
                    account_balance = account_info.balance
                    position_size = calculate_position_size(account_balance, symbol, data_m5)
                    if position_size > 0:
                        trade_action = 1 if robust_prediction == 1 else 0
                        execute_trade(trade_action, symbol, volume=position_size, historical_data=data_m5,
                                      confidence=prediction_confidence)
                        check_for_closed_positions(symbol)
                    else:
                        logging.warning("Wielkość pozycji wynosi 0, pomijam transakcję.")
                else:
                    logging.info("Robust prediction: brak akcji (trzymaj)")

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