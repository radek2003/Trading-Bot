import MetaTrader5 as mt5
import pandas as pd
import logging


def fetch_historical_data(symbol, bars_m5=10000, bars_m15=10000):
    try:
        # Pobieranie danych 5-minutowych
        rates_m5 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, bars_m5)
        if rates_m5 is None or len(rates_m5) == 0:
            logging.error("Brak danych historycznych dla 5m.")
            df_m5 = pd.DataFrame()
        else:
            df_m5 = pd.DataFrame(rates_m5)
            df_m5['time'] = pd.to_datetime(df_m5['time'], unit='s')
            df_m5.set_index('time', inplace=True)
            df_m5 = df_m5[['open', 'high', 'low', 'close', 'tick_volume']]

        # Pobieranie danych 15-minutowych
        rates_m15 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, bars_m15)
        if rates_m15 is None or len(rates_m15) == 0:
            logging.error("Brak danych historycznych dla 15m.")
            df_m15 = pd.DataFrame()
        else:
            df_m15 = pd.DataFrame(rates_m15)
            df_m15['time'] = pd.to_datetime(df_m15['time'], unit='s')
            df_m15.set_index('time', inplace=True)
            df_m15 = df_m15[['open', 'high', 'low', 'close', 'tick_volume']]

        return df_m5, df_m15

    except Exception as e:
        logging.exception("Problem z pobieraniem danych historycznych.")
        return pd.DataFrame(), pd.DataFrame()

