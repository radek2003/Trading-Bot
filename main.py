import MetaTrader5 as mt5
import logging
from config.config import configure_cpu
from src.data_fetcher import fetch_historical_data
from src.feature_extraction import calculate_moving_averages, calculate_macd
from src.model_methods import train_model
from src.strategy import apply_strategy
from src.trading import execute_trade

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
        else:
            logging.error("Brak predykcji z modelu.")
    else:
        logging.error("Brak nowych danych do analizy.")

    mt5.shutdown()

if __name__ == "__main__":
    main()