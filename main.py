import MetaTrader5 as mt5
import logging
import pandas as pd
from src.data_fetcher import fetch_historical_data
from src.model_methods import train_model
from src.strategy import apply_strategy, calculate_candlestick_patterns
from src.trading import execute_trade
import torch


def main():
    """Główna funkcja programu."""

    # Inicjalizacja MetaTrader 5
    if not mt5.initialize():
        logging.error(f"Inicjalizacja MT5 nie powiodła się, kod błędu: {mt5.last_error()}")
        return

    # Definiowanie symbolu waluty
    symbol = "EURUSD"

    # Pobranie danych dla 5-minutowego i 15-minutowego interwału
    data_m5, data_m15 = fetch_historical_data(symbol)

    # Sprawdzenie, czy dane 15-minutowe zostały poprawnie pobrane
    if data_m15.empty:
        logging.error("Brak danych 15-minutowych do analizy.")
        mt5.shutdown()
        return

    # Wykorzystanie danych 15-minutowych do dalszej analizy
    data = data_m15

    # Wybór strategii
    strategy_type = 'Candlestick'  # Można zmienić na 'MACD' w zależności od wybranej strategii

    if strategy_type == 'Candlestick':
        # Obliczenie wzorców świec japońskich
        data_with_features = calculate_candlestick_patterns(data)
    else:
        logging.error(f"Nieznana strategia: {strategy_type}")
        mt5.shutdown()
        return

    # Sprawdzenie, czy dane z cechami zostały poprawnie utworzone
    if data_with_features.empty:
        logging.error("Brak danych z cechami do trenowania modelu.")
        mt5.shutdown()
        return

    # Ścieżka i nazwa pliku do zapisu modelu
    folder_path = r"C:\trenowanie modelu do bota"
    model_filename = 'best_model.pth'

    # Trenowanie modelu
    model, scaler = train_model(data_with_features, folder_path, model_filename)

    # Sprawdzenie, czy model i skalowanie zostały poprawnie załadowane
    if model is None or scaler is None:
        logging.error("Model lub skalowanie nie zostały poprawnie załadowane.")
        mt5.shutdown()
        return

    # Pobranie nowych danych do przewidywań
    new_data_m5, new_data_m15 = fetch_historical_data(symbol)
    if not new_data_m15.empty:
        # Użycie nowo pobranych danych 15-minutowych do przewidywań
        predictions = apply_strategy(model, scaler, new_data_m15, strategy_type)
        if predictions is not None:
            # Przetwarzanie ostatnich danych do przewidywań modelu
            latest_data = new_data_m15.iloc[-1:]
            latest_data_with_features = calculate_candlestick_patterns(latest_data)

            if latest_data_with_features.empty:
                logging.error("Brak danych z cechami do przewidywań.")
                mt5.shutdown()
                return

            latest_data_scaled = scaler.transform(latest_data_with_features.drop('Target', axis=1))
            latest_data_scaled_tensor = torch.tensor(latest_data_scaled, dtype=torch.float32)

            # Przejście modelu w tryb oceny i uzyskanie przewidywań
            model.eval()
            with torch.no_grad():
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                latest_data_scaled_tensor = latest_data_scaled_tensor.to(device)
                outputs = model(latest_data_scaled_tensor)
                _, predicted = torch.max(outputs, 1)
                model_prediction = predicted.item()

            # Wykonanie transakcji na podstawie przewidywań modelu
            execute_trade(model_prediction, symbol)
        else:
            logging.error("Brak predykcji z modelu.")
    else:
        logging.error("Brak nowych danych do analizy.")

    # Zakończenie połączenia z MetaTrader 5
    mt5.shutdown()


if __name__ == "__main__":
    main()
