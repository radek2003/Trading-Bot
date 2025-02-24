import logging
import pandas as pd
import numpy as np
import torch


def apply_strategy(model, scaler, new_data, strategy_type='Candlestick'):
    """Stosuje wybraną strategię na nowych danych."""
    try:
        if strategy_type == 'MACD':
            new_data_with_features = calculate_macd(new_data)
            new_data_with_features = calculate_moving_averages(new_data_with_features)
        elif strategy_type == 'Candlestick':
            # Stosowanie wzorców świec japońskich jako strategii
            new_data_with_features = calculate_candlestick_patterns(new_data)
        else:
            logging.error(f"Nieznana strategia: {strategy_type}")
            return None

        if new_data_with_features.empty:
            logging.error("Brak danych z cechami do prognozowania.")
            return None

        # Przygotowanie danych do PyTorch
        X_new = scaler.transform(new_data_with_features.drop('Target', axis=1, errors='ignore').values)
        X_new_tensor = torch.tensor(X_new, dtype=torch.float32)

        # Sprawdzenie dostępności GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        # Przewidywanie
        with torch.no_grad():
            X_new_tensor = X_new_tensor.to(device)
            outputs = model(X_new_tensor)
            _, predictions = torch.max(outputs, 1)

        return predictions.cpu().numpy()  # Konwertowanie wyników na numpy array

    except Exception as e:
        logging.exception("Problem z zastosowaniem strategii.")
        return None


def calculate_moving_averages(data):
    """Oblicza średnie kroczące i generuje sygnały kupna/sprzedaży."""
    if data.empty:
        logging.error("Brak danych do obliczeń.")
        return pd.DataFrame()

    try:
        data = data.copy()  # Tworzenie kopii DataFrame
        data['MA20'] = data['close'].rolling(window=20, min_periods=1).mean()
        data['MA60'] = data['close'].rolling(window=60, min_periods=1).mean()
        data['MA100'] = data['close'].rolling(window=100, min_periods=1).mean()
        data['Signal'] = np.where(data['MA20'] > data['MA60'], 1, -1)
        data['Buy_Signal'] = (data['Signal'].shift(1) == -1) & (data['Signal'] == 1)
        data['Sell_Signal'] = (data['Signal'].shift(1) == 1) & (data['Signal'] == -1)
        data['Target'] = (data['close'].shift(-1) > data['close']).astype(int)
        return data.dropna().drop(['tick_volume'], axis=1, errors='ignore')
    except Exception as e:
        logging.exception("Problem z obliczaniem średnich kroczących.")
        return pd.DataFrame()


def calculate_macd(df):
    """Oblicza wskaźnik MACD i jego sygnał."""
    if df.empty:
        logging.error("Brak danych do obliczeń MACD.")
        return pd.DataFrame()

    try:
        df = df.copy()
        df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        return df
    except Exception as e:
        logging.exception("Problem z obliczaniem wskaźnika MACD.")
        return pd.DataFrame()



def calculate_candlestick_patterns(data):
    """Dodaje cechy związane z formacjami świec do DataFrame."""
    data = data.copy()  # Tworzymy kopię DataFrame

    # Cechy dla 5 minut
    data.loc[:, 'Bullish_Engulfing'] = ((data['open'] < data['close'].shift(1)) &
                                        (data['close'] > data['open'].shift(1)) &
                                        (data['open'] <= data['close'].shift(1)) &
                                        (data['close'] >= data['open'].shift(1)))

    data.loc[:, 'Bearish_Engulfing'] = ((data['open'] > data['close'].shift(1)) &
                                        (data['close'] < data['open'].shift(1)) &
                                        (data['open'] >= data['close'].shift(1)) &
                                        (data['close'] <= data['open'].shift(1)))

    data.loc[:, 'Hammer'] = ((data['low'] < data['open']) &
                             ((data['high'] - data['close']) < (data['open'] - data['low']) * 0.3) &
                             ((data['open'] - data['low']) > (data['close'] - data['open']) * 2))

    data.loc[:, 'Inverted_Hammer'] = ((data['high'] > data['open']) &
                                      ((data['close'] - data['low']) < (data['high'] - data['open']) * 0.3) &
                                      ((data['high'] - data['open']) > (data['open'] - data['close']) * 2))

    data.loc[:, 'Doji'] = np.abs(data['close'] - data['open']) < (data['high'] - data['low']) * 0.1

    # Dodanie nowych formacji
    data.loc[:, 'Shooting_Star'] = ((data['high'] > data['open'] + (data['open'] - data['low']) * 0.5) &
                                     (data['low'] > data['open']) &
                                     (data['close'] < data['open']) &
                                     ((data['high'] - data['close']) > (data['close'] - data['low']) * 2))

    data.loc[:, 'Morning_Star'] = ((data['close'].shift(2) < data['open'].shift(2)) &
                                     (data['close'].shift(1) < data['open'].shift(1)) &
                                     (data['close'] > data['open']) &
                                     (data['open'].shift(2) < data['close'].shift(1)) &
                                     (data['close'].shift(1) < data['open']))

    data.loc[:, 'Evening_Star'] = ((data['close'].shift(2) > data['open'].shift(2)) &
                                     (data['close'].shift(1) > data['open'].shift(1)) &
                                     (data['close'] < data['open']) &
                                     (data['open'].shift(2) > data['close'].shift(1)) &
                                     (data['close'].shift(1) > data['open']))

    data.loc[:, 'Spinning_Top'] = ((np.abs(data['close'] - data['open']) < (data['high'] - data['low']) * 0.3) &
                                    ((data['high'] - data['low']) > (data['close'] - data['open']) * 2))

    data.loc[:, 'Target'] = (data['close'].shift(-1) > data['close']).astype(int)
    data.loc[:, 'Zielona_Swieca'] = data['close'] > data['open']
    data.loc[:, 'Czerwona_Swieca'] = data['close'] < data['open']
    data.loc[:, 'Kupuj_5m'] = data['Zielona_Swieca'] & data['Zielona_Swieca'].shift(1)
    data.loc[:, 'Sprzedaj_5m'] = data['Czerwona_Swieca'] & data['Czerwona_Swieca'].shift(1)

    # Cechy dla 15 minut
    data.loc[:, 'Bullish_Engulfing_15m'] = ((data['open'] < data['close'].shift(3)) &
                                            (data['close'] > data['open'].shift(3)) &
                                            (data['open'] <= data['close'].shift(3)) &
                                            (data['close'] >= data['open'].shift(3)))

    data.loc[:, 'Bearish_Engulfing_15m'] = ((data['open'] > data['close'].shift(3)) &
                                            (data['close'] < data['open'].shift(3)) &
                                            (data['open'] >= data['close'].shift(3)) &
                                            (data['close'] <= data['open'].shift(3)))

    data.loc[:, 'Hammer_15m'] = ((data['low'] < data['open']) &
                                 ((data['high'] - data['close']) < (data['open'] - data['low']) * 0.3) &
                                 ((data['open'] - data['low']) > (data['close'] - data['open']) * 2))

    data.loc[:, 'Inverted_Hammer_15m'] = ((data['high'] > data['open']) &
                                          ((data['close'] - data['low']) < (data['high'] - data['open']) * 0.3) &
                                          ((data['high'] - data['open']) > (data['open'] - data['close']) * 2))

    data.loc[:, 'Doji_15m'] = np.abs(data['close'] - data['open']) < (data['high'] - data['low']) * 0.1

    # Dodanie nowych formacji dla 15 minut
    data.loc[:, 'Shooting_Star_15m'] = ((data['high'] > data['open'].shift(3) + (data['open'].shift(3) - data['low'].shift(3)) * 0.5) &
                                         (data['low'].shift(3) > data['open'].shift(3)) &
                                         (data['close'].shift(3) < data['open'].shift(3)) &
                                         ((data['high'].shift(3) - data['close'].shift(3)) > (data['close'].shift(3) - data['low'].shift(3)) * 2))

    data.loc[:, 'Morning_Star_15m'] = ((data['close'].shift(5) < data['open'].shift(5)) &
                                         (data['close'].shift(4) < data['open'].shift(4)) &
                                         (data['close'].shift(3) > data['open'].shift(3)) &
                                         (data['open'].shift(5) < data['close'].shift(4)) &
                                         (data['close'].shift(4) < data['open'].shift(4)))

    data.loc[:, 'Evening_Star_15m'] = ((data['close'].shift(5) > data['open'].shift(5)) &
                                         (data['close'].shift(4) > data['open'].shift(4)) &
                                         (data['close'].shift(3) < data['open'].shift(3)) &
                                         (data['open'].shift(5) > data['close'].shift(4)) &
                                         (data['close'].shift(4) > data['open'].shift(4)))

    data.loc[:, 'Spinning_Top_15m'] = ((np.abs(data['close'].shift(3) - data['open'].shift(3)) < (data['high'].shift(3) - data['low'].shift(3)) * 0.3) &
                                        ((data['high'].shift(3) - data['low'].shift(3)) > (data['close'].shift(3) - data['open'].shift(3)) * 2))

    data.loc[:, 'Target_15m'] = (data['close'].shift(-3) > data['close']).astype(int)
    data.loc[:, 'Zielona_Swieca_15m'] = data['close'] > data['open']
    data.loc[:, 'Czerwona_Swieca_15m'] = data['close'] < data['open']
    data.loc[:, 'Kupuj_15m'] = data['Zielona_Swieca_15m'] & data['Zielona_Swieca_15m'].shift(1)
    data.loc[:, 'Sprzedaj_15m'] = data['Czerwona_Swieca_15m'] & data['Czerwona_Swieca_15m'].shift(1)

    return data
