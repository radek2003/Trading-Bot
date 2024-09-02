import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.utils import dropna
import logging

# Kalkulacje i ekstrakcja cech

def extract_features(data):
    """Ekstrakt cechy z danych i dodaje wskaźniki techniczne."""
    try:
        if data.empty:
            logging.error("Brak danych do ekstrakcji cech.")
            return pd.DataFrame()

        data = dropna(data)
        data = add_all_ta_features(
            data, open="open", high="high", low="low", close="close", volume="tick_volume", fillna=True)

        data['Target'] = np.where(data['close'].shift(-1) > data['close'], 1, 0)

        return data.dropna().drop(['tick_volume'], axis=1)
    except Exception as e:
        logging.exception("Problem z ekstrakcją cech.")
        return pd.DataFrame()

def calculate_moving_averages(data):
    """Oblicza średnie kroczące i generuje sygnały kupna/sprzedaży."""
    if data.empty:
        logging.error("Brak danych do obliczeń.")
        return pd.DataFrame()

    try:
        data['MA20'] = data['close'].rolling(window=20, min_periods=1).mean()
        data['MA60'] = data['close'].rolling(window=60, min_periods=1).mean()
        data['MA100'] = data['close'].rolling(window=100, min_periods=1).mean()
        data['Signal'] = np.where(data['MA20'] > data['MA60'], 1, -1)
        data['Buy_Signal'] = (data['Signal'].shift(1) == -1) & (data['Signal'] == 1)
        data['Sell_Signal'] = (data['Signal'].shift(1) == 1) & (data['Signal'] == -1)
        data['Target'] = (data['close'].shift(-1) > data['close']).astype(int)
        return data.dropna().drop(['tick_volume'], axis=1)
    except Exception as e:
        logging.exception("Problem z obliczaniem średnich kroczących.")
        return pd.DataFrame()

def calculate_macd(df):
    """Oblicza wskaźnik MACD i jego sygnał."""
    df = df.copy()
    df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    return df

