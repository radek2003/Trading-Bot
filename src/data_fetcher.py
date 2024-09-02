import MetaTrader5 as mt5
import pandas as pd
import logging

# Pobieranie danych historycznych z mt5
def fetch_historical_data(symbol, timeframe=mt5.TIMEFRAME_M15, bars=1000):
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
        logging.exception("Problem z pobieraniem danych historycznych.")
        return pd.DataFrame()

print(".")