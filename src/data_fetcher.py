import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import logging

def test_trade_history(days_back=10):
    if not mt5.terminal_info():
        logging.error("MT5 nie jest zainicjalizowane.")
        return pd.DataFrame()

    from_date = datetime.now() - timedelta(days=days_back)
    to_date = datetime.now()

    history = mt5.history_orders_get(from_date, to_date)

    logging.debug(f"Zapytanie: from={from_date}, to={to_date}")

    if history is None or len(history) == 0:
        logging.warning(f"Brak zleceń w historii od {from_date} do {to_date}.")
        return pd.DataFrame()

    logging.info(f"Znaleziono {len(history)} zleceń w historii.")

    deals = pd.DataFrame([{
        'time': order.time_setup,
        'type': order.type,
        'profit': 0,
        'volume': order.volume_current,
        'symbol': order.symbol
    } for order in history])

    deals['time'] = pd.to_datetime(deals['time'], unit='s')
    return deals

def fetch_historical_data(symbol, bars_m5=90000, bars_m15=90000):
    try:
        # Pobieranie danych 5-minutowych
        rates_m5 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, bars_m5)
        if rates_m5 is None or len(rates_m5) == 0:
            logging.error("Brak danych historycznych dla 5m.")
            df_m5 = pd.DataFrame()
        else:
            logging.debug(f"Pobrano {len(rates_m5)} surowych świec M5")
            df_m5 = pd.DataFrame(rates_m5)
            df_m5['time'] = pd.to_datetime(df_m5['time'], unit='s')
            # Wygładzanie cen
            df_m5['close_smooth'] = df_m5['close'].rolling(window=3, min_periods=1).mean()
            # Filtrowanie outliers (3 odchylenia standardowe)
            mean = df_m5['close'].mean()
            std = df_m5['close'].std()
            df_m5 = df_m5[(df_m5['close'] > mean - 3 * std) & (df_m5['close'] < mean + 3 * std)]
            df_m5 = df_m5[['time', 'open', 'high', 'low', 'close', 'close_smooth', 'tick_volume']]
            logging.debug(f"Po filtracji: {len(df_m5)} świec M5")

        # Pobieranie danych 15-minutowych
        rates_m15 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, bars_m15)
        if rates_m15 is None or len(rates_m15) == 0:
            logging.error("Brak danych historycznych dla 15m.")
            df_m15 = pd.DataFrame()
        else:
            logging.debug(f"Pobrano {len(rates_m15)} surowych świec M15")
            df_m15 = pd.DataFrame(rates_m15)
            df_m15['time'] = pd.to_datetime(df_m15['time'], unit='s')
            # Wygładzanie cen
            df_m15['close_smooth'] = df_m15['close'].rolling(window=3, min_periods=1).mean()
            # Filtrowanie outliers
            mean = df_m15['close'].mean()
            std = df_m15['close'].std()
            df_m15 = df_m15[(df_m15['close'] > mean - 3 * std) & (df_m15['close'] < mean + 3 * std)]
            df_m15 = df_m15[['time', 'open', 'high', 'low', 'close', 'close_smooth', 'tick_volume']]
            logging.debug(f"Po filtracji: {len(df_m15)} świec M15")

        return df_m5, df_m15

    except Exception as e:
        logging.exception("Problem z pobieraniem danych historycznych.")
        return pd.DataFrame(), pd.DataFrame()