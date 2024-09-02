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



