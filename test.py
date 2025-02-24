import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import logging


def test_trade_history(symbol, days_back=7):
    """Pobieranie historii zleceń dla danego symbolu z MetaTrader 5.

    Args:
        symbol (str): Symbol handlowy (np. "EURUSD").
        days_back (int): Liczba dni wstecz do analizy (domyślnie 7).

    Returns:
        pd.DataFrame: DataFrame z historią zleceń lub pusty DataFrame w przypadku braku danych.
    """
    # Sprawdzanie, czy MT5 jest zainicjalizowane
    if not mt5.terminal_info():
        logging.error("MT5 nie jest zainicjalizowane.")
        return pd.DataFrame()

    # Definiowanie zakresu czasowego
    from_date = datetime.now() - timedelta(days=days_back)
    to_date = datetime.now()

    # Pobranie historii zleceń
    history = mt5.history_orders_get(from_date, to_date)

    # Debugowanie
    logging.debug(f"Zapytanie: from={from_date}, to={to_date}")
    logging.debug(f"Odpowiedź MT5: {history}")

    # Sprawdzanie, czy historia została pobrana
    if history is None or len(history) == 0:
        logging.warning(f"Brak zleceń w historii od {from_date} do {to_date}.")
        return pd.DataFrame()

    # Filtrowanie po symbolu
    filtered_history = [order for order in history if order.symbol == symbol]
    if not filtered_history:
        logging.warning(f"Brak zleceń dla {symbol} w podanym okresie.")
        return pd.DataFrame()

    logging.info(f"Znaleziono {len(filtered_history)} zleceń dla {symbol} w historii.")
    for order in filtered_history:
        logging.debug(f"Zlecenie: {order}")

    # Konwersja danych do DataFrame
    deals = pd.DataFrame([{
        'time': order.time_setup,
        'type': order.type,
        'profit': 0,  # Brak profitu w TradeOrder, ustawiamy 0
        'volume': order.volume_current
    } for order in filtered_history])

    # Konwersja czasu na format pandas datetime
    deals['time'] = pd.to_datetime(deals['time'], unit='s')

    return deals


# Testowanie standalone (opcjonalne)
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    if not mt5.initialize():
        logging.error(f"Nie udało się połączyć z MetaTrader 5, kod błędu: {mt5.last_error()}")
    else:
        df = test_trade_history("EURUSD", days_back=7)
        if not df.empty:
            print(df.head())
        mt5.shutdown()