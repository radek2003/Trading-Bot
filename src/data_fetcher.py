import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import logging

from ta.volatility import AverageTrueRange


def test_trade_history(days_back=200):
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

def fetch_historical_data(symbol, bars_m5=20000, bars_m15=20000):
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
            # Filtracja z użyciem ATR
            atr_m5 = AverageTrueRange(high=df_m5['high'], low=df_m5['low'], close=df_m5['close'], window=14)
            df_m5['atr'] = atr_m5.average_true_range()
            price_changes_m5 = df_m5['close'].diff().abs()
            threshold_m5 = 3 * df_m5['atr']  # Próg: 3-krotność ATR
            before_m5 = len(df_m5)
            df_m5 = df_m5[price_changes_m5 <= threshold_m5]
            logging.debug(f"Po filtracji ATR: {len(df_m5)} świec M5, usunięto {before_m5 - len(df_m5)} rekordów")
            df_m5 = df_m5[['time', 'open', 'high', 'low', 'close', 'close_smooth', 'tick_volume']]

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
            # Filtracja z użyciem ATR
            atr_m15 = AverageTrueRange(high=df_m15['high'], low=df_m15['low'], close=df_m15['close'], window=14)
            df_m15['atr'] = atr_m15.average_true_range()
            price_changes_m15 = df_m15['close'].diff().abs()
            threshold_m15 = 3 * df_m15['atr']  # Próg: 3-krotność ATR
            before_m15 = len(df_m15)
            df_m15 = df_m15[price_changes_m15 <= threshold_m15]
            logging.debug(f"Po filtracji ATR: {len(df_m15)} świec M15, usunięto {before_m15 - len(df_m15)} rekordów")
            df_m15 = df_m15[['time', 'open', 'high', 'low', 'close', 'close_smooth', 'tick_volume']]

        return df_m5, df_m15

    except Exception as e:
        logging.exception("Problem z pobieraniem danych historycznych.")
        return pd.DataFrame(), pd.DataFrame()