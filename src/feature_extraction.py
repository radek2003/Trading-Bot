import logging
import numpy as np
import pandas as pd
from ta import add_all_ta_features

def extract_features(data, trade_history):
    """Ekstrakt cechy z danych rynkowych i historii transakcji."""
    try:
        if data.empty:
            logging.error("Brak danych do ekstrakcji cech.")
            return pd.DataFrame()

        data = data.dropna()
        data = add_all_ta_features(
            data, open="open", high="high", low="low", close="close", volume="tick_volume", fillna=True
        )

        # Dodanie cech z historii transakcji
        if not trade_history.empty:
            trade_history = trade_history.groupby('time').agg({
                'type': 'last',  # Ostatni typ transakcji (kupno/sprzedaż)
                'profit': 'sum',  # Łączny zysk/strata
                'volume': 'sum'   # Łączny wolumen
            }).reset_index()
            data = data.merge(trade_history, on='time', how='left').fillna(0)

        data['Target'] = np.where(data['close'].shift(-1) > data['close'], 1, 0)

        return data.dropna().drop(['tick_volume'], axis=1)
    except Exception as e:
        logging.exception("Problem z ekstrakcją cech.")
        return pd.DataFrame()
