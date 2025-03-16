import logging
import pandas as pd
import numpy as np
import torch
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange


def apply_strategy(model, scaler, new_data, gamma=0.1, num_samples=500):
    """Stosuje strategię z kryterium Walda i γ-odpornością, analizując wszystkie strategie naraz."""
    try:
        # Obliczenie cech dla wszystkich strategii
        new_data_with_features = calculate_all_features(new_data)

        if new_data_with_features.empty:
            logging.error("Brak danych z cechami do prognozowania.")
            return None

        # Przygotowanie danych do predykcji
        X_new = scaler.transform(new_data_with_features.drop('Target', axis=1, errors='ignore').values)
        X_new_tensor = torch.tensor(X_new, dtype=torch.float32)

        # Ustawienie urządzenia i modelu
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.train()  # Włączamy tryb trenowania dla Monte Carlo Dropout

        # Monte Carlo Dropout: wielokrotne predykcje dla uzyskania rozkładu
        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                output = model(X_new_tensor.to(device))
                probabilities = torch.softmax(output, dim=1)
                predictions.append(probabilities.cpu().numpy())
        predictions = np.array(predictions)  # Kształt: (num_samples, batch_size, num_classes)

        # Zakładamy batch_size == 1 (jedna sekwencja)
        predictions = predictions[:, 0, :]  # Kształt: (num_samples, num_classes)

        # Definicja akcji: 0 - Trzymaj, 1 - Kup, -1 - Sprzedaj
        actions = [0, 1, -1]

        # Obliczenie wyników dla każdej akcji
        action_scores = []
        for action in actions:
            if action == 0:
                # Trzymaj: wynik neutralny
                scores = np.zeros(num_samples)
            elif action == 1:
                # Kup: wynik = P(Kup) - P(Sprzedaj)
                scores = predictions[:, 1] - predictions[:, 2]
            elif action == -1:
                # Sprzedaj: wynik = P(Sprzedaj) - P(Kup)
                scores = predictions[:, 2] - predictions[:, 1]

            # Obliczenie γ-fraktyla
            gamma_fractile = np.percentile(scores, gamma * 100)
            action_scores.append(gamma_fractile)

        # Wybór akcji z najwyższym γ-fraktylem
        best_action_idx = np.argmax(action_scores)
        robust_prediction = actions[best_action_idx]

        return robust_prediction

    except Exception as e:
        logging.exception("Problem z zastosowaniem strategii.")
        return None


def calculate_all_features(data):
    """Oblicza cechy dla wszystkich strategii i łączy je w jeden DataFrame."""
    try:
        data_with_features = data.copy()

        # Obliczenie cech MACD
        data_with_features = calculate_robust_macd(data_with_features)

        # Obliczenie cech formacji świecowych
        data_with_features = calculate_candlestick_patterns(data_with_features)

        # Obliczenie cech średnich kroczących
        data_with_features = calculate_robust_moving_averages(data_with_features)

        return data_with_features

    except Exception as e:
        logging.exception("Problem z obliczaniem wszystkich cech.")
        return pd.DataFrame()


def calculate_robust_moving_averages(data):
    """Oblicza odporne średnie kroczące (mediana) i sygnały."""
    if data.empty:
        logging.error("Brak danych do obliczeń.")
        return pd.DataFrame()

    try:
        data = data.copy()
        # Dynamiczne okna w zależności od liczby dostępnych danych
        ma20_window = min(20, len(data))
        ma60_window = min(60, len(data))
        ma100_window = min(100, len(data))

        data['MA20'] = data['close'].rolling(window=ma20_window, min_periods=1).median()
        data['MA60'] = data['close'].rolling(window=ma60_window, min_periods=1).median()
        data['MA100'] = data['close'].rolling(window=ma100_window, min_periods=1).median()
        data['Signal'] = np.where(data['MA20'] > data['MA60'], 1, -1)
        data['Buy_Signal'] = (data['Signal'].shift(1) == -1) & (data['Signal'] == 1)
        data['Sell_Signal'] = (data['Signal'].shift(1) == 1) & (data['Signal'] == -1)
        data['Target'] = (data['close'].shift(-1) > data['close']).astype(int)
        return data.drop(['tick_volume'], axis=1, errors='ignore')

    except Exception as e:
        logging.exception("Problem z obliczaniem robust średnich kroczących.")
        return pd.DataFrame()


def calculate_robust_macd(df):
    """Oblicza odporny MACD z ważoną medianą zamiast EMA."""
    if df.empty:
        logging.error("Brak danych do obliczeń MACD.")
        return pd.DataFrame()

    try:
        df = df.copy()
        df['EMA12'] = df['close'].rolling(window=12, min_periods=1).median()
        df['EMA26'] = df['close'].rolling(window=26, min_periods=1).median()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_Signal'] = df['MACD'].rolling(window=9, min_periods=1).median()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        return df
    except Exception as e:
        logging.exception("Problem z obliczaniem robust MACD.")
        return pd.DataFrame()


def calculate_candlestick_patterns(data):
    """Dodaje cechy związane z formacjami świec z robust podejściem i nowymi wskaźnikami."""
    try:
        data = data.copy()

        if len(data) <= 5:
            logging.warning("Za mało danych do obliczenia formacji świecowych.")
            return data

        # Podstawowe obliczenia
        body_size = np.abs(data['close'] - data['open'])
        range_size = data['high'] - data['low']
        trend_up = data['close'] > data['close'].shift(1)
        trend_down = data['close'] < data['close'].shift(1)

        # --- Cechy dla 5 minut ---
        data['Bullish_Engulfing'] = ((data['open'] < data['close'].shift(1)) &
                                     (data['close'] > data['open'].shift(1)) &
                                     (data['open'] <= data['close'].shift(1)) &
                                     (data['close'] >= data['open'].shift(1)) &
                                     (body_size > body_size.shift(1) * 1.1))
        data['Bullish_Engulfing_Strength'] = data['Bullish_Engulfing'] * (body_size / range_size)

        data['Bearish_Engulfing'] = ((data['open'] > data['close'].shift(1)) &
                                     (data['close'] < data['open'].shift(1)) &
                                     (data['open'] >= data['close'].shift(1)) &
                                     (data['close'] <= data['open'].shift(1)) &
                                     (body_size > body_size.shift(1) * 1.1))
        data['Bearish_Engulfing_Strength'] = data['Bearish_Engulfing'] * (body_size / range_size)

        data['Hammer'] = ((data['low'] < data['open']) &
                          ((data['high'] - data['close']) < (data['open'] - data['low']) * 0.3) &
                          ((data['open'] - data['low']) > body_size * 2) &
                          trend_down.shift(1))

        data['Inverted_Hammer'] = ((data['high'] > data['open']) &
                                   ((data['close'] - data['low']) < (data['high'] - data['open']) * 0.3) &
                                   ((data['high'] - data['open']) > body_size * 2) &
                                   trend_up.shift(1))

        data['Doji'] = (body_size < range_size * 0.1)

        data['Shooting_Star'] = ((data['high'] > data['open'] + (data['open'] - data['low']) * 0.5) &
                                 (data['low'] > data['open']) &
                                 (data['close'] < data['open']) &
                                 ((data['high'] - data['close']) > body_size * 2) &
                                 trend_up.shift(1))

        data['Morning_Star'] = ((data['close'].shift(2) < data['open'].shift(2)) &
                                (body_size.shift(1) < range_size.shift(1) * 0.3) &
                                (data['close'] > data['open']) &
                                (data['open'].shift(2) < data['close'].shift(1)) &
                                (data['close'].shift(1) < data['open']))

        data['Evening_Star'] = ((data['close'].shift(2) > data['open'].shift(2)) &
                                (body_size.shift(1) < range_size.shift(1) * 0.3) &
                                (data['close'] < data['open']) &
                                (data['open'].shift(2) > data['close'].shift(1)) &
                                (data['close'].shift(1) > data['open']))

        data['Spinning_Top'] = ((body_size < range_size * 0.3) &
                                (range_size > body_size * 2))

        data['Three_White_Soldiers'] = ((data['close'] > data['open']) &
                                        (data['close'].shift(1) > data['open'].shift(1)) &
                                        (data['close'].shift(2) > data['open'].shift(2)) &
                                        (data['close'] > data['close'].shift(1)) &
                                        (data['close'].shift(1) > data['close'].shift(2)))

        data['Three_Black_Crows'] = ((data['close'] < data['open']) &
                                     (data['close'].shift(1) < data['open'].shift(1)) &
                                     (data['close'].shift(2) < data['open'].shift(2)) &
                                     (data['close'] < data['close'].shift(1)) &
                                     (data['close'].shift(1) < data['close'].shift(2)))

        data['Zielona_Swieca'] = data['close'] > data['open']
        data['Czerwona_Swieca'] = data['close'] < data['open']
        data['Kupuj_5m'] = data['Zielona_Swieca'] & data['Zielona_Swieca'].shift(1)
        data['Sprzedaj_5m'] = data['Czerwona_Swieca'] & data['Czerwona_Swieca'].shift(1)
        data['Target'] = (data['close'].shift(-1) > data['close']).astype(int)

        # --- Cechy dla 15 minut (tylko jeśli wystarczająco danych) ---
        if len(data) >= 15:
            data['Bullish_Engulfing_15m'] = ((data['open'] < data['close'].shift(3)) &
                                             (data['close'] > data['open'].shift(3)) &
                                             (data['open'] <= data['close'].shift(3)) &
                                             (data['close'] >= data['open'].shift(3)) &
                                             (body_size > body_size.shift(3) * 1.1))
            data['Bullish_Engulfing_15m_Strength'] = data['Bullish_Engulfing_15m'] * (body_size / range_size)

            data['Bearish_Engulfing_15m'] = ((data['open'] > data['close'].shift(3)) &
                                             (data['close'] < data['open'].shift(3)) &
                                             (data['open'] >= data['close'].shift(3)) &
                                             (data['close'] <= data['open'].shift(3)) &
                                             (body_size > body_size.shift(3) * 1.1))
            data['Bearish_Engulfing_15m_Strength'] = data['Bearish_Engulfing_15m'] * (body_size / range_size)

            data['Hammer_15m'] = ((data['low'] < data['open']) &
                                  ((data['high'] - data['close']) < (data['open'] - data['low']) * 0.3) &
                                  ((data['open'] - data['low']) > body_size * 2) &
                                  trend_down.shift(3))

            data['Inverted_Hammer_15m'] = ((data['high'] > data['open']) &
                                           ((data['close'] - data['low']) < (data['high'] - data['open']) * 0.3) &
                                           ((data['high'] - data['open']) > body_size * 2) &
                                           trend_up.shift(3))

            data['Doji_15m'] = (body_size < range_size * 0.1)

            data['Shooting_Star_15m'] = ((data['high'].shift(3) > data['open'].shift(3) + (
                        data['open'].shift(3) - data['low'].shift(3)) * 0.5) &
                                         (data['low'].shift(3) > data['open'].shift(3)) &
                                         (data['close'].shift(3) < data['open'].shift(3)) &
                                         ((data['high'].shift(3) - data['close'].shift(3)) > body_size.shift(3) * 2) &
                                         trend_up.shift(4))

            data['Morning_Star_15m'] = ((data['close'].shift(5) < data['open'].shift(5)) &
                                        (body_size.shift(4) < range_size.shift(4) * 0.3) &
                                        (data['close'].shift(3) > data['open'].shift(3)) &
                                        (data['open'].shift(5) < data['close'].shift(4)) &
                                        (data['close'].shift(4) < data['open'].shift(4)))

            data['Evening_Star_15m'] = ((data['close'].shift(5) > data['open'].shift(5)) &
                                        (body_size.shift(4) < range_size.shift(4) * 0.3) &
                                        (data['close'].shift(3) < data['open'].shift(3)) &
                                        (data['open'].shift(5) > data['close'].shift(4)) &
                                        (data['close'].shift(4) > data['open'].shift(4)))

            data['Spinning_Top_15m'] = ((body_size.shift(3) < range_size.shift(3) * 0.3) &
                                        (range_size.shift(3) > body_size.shift(3) * 2))

            data['Three_White_Soldiers_15m'] = ((data['close'].shift(3) > data['open'].shift(3)) &
                                                (data['close'].shift(4) > data['open'].shift(4)) &
                                                (data['close'].shift(5) > data['open'].shift(5)) &
                                                (data['close'].shift(3) > data['close'].shift(4)) &
                                                (data['close'].shift(4) > data['close'].shift(5)))

            data['Three_Black_Crows_15m'] = ((data['close'].shift(3) < data['open'].shift(3)) &
                                             (data['close'].shift(4) < data['open'].shift(4)) &
                                             (data['close'].shift(5) < data['open'].shift(5)) &
                                             (data['close'].shift(3) < data['close'].shift(4)) &
                                             (data['close'].shift(4) < data['close'].shift(5)))

            data['Zielona_Swieca_15m'] = data['close'] > data['open']
            data['Czerwona_Swieca_15m'] = data['close'] < data['open']
            data['Kupuj_15m'] = data['Zielona_Swieca_15m'] & data['Zielona_Swieca_15m'].shift(3)
            data['Sprzedaj_15m'] = data['Czerwona_Swieca_15m'] & data['Czerwona_Swieca_15m'].shift(3)
            data['Target_15m'] = (data['close'].shift(-3) > data['close']).astype(int)
        else:
            logging.warning("Za mało danych dla formacji 15-minutowych, ustawiam wartości domyślne.")
            for col in ['Bullish_Engulfing_15m', 'Bullish_Engulfing_15m_Strength', 'Bearish_Engulfing_15m',
                        'Bearish_Engulfing_15m_Strength', 'Hammer_15m', 'Inverted_Hammer_15m', 'Doji_15m',
                        'Shooting_Star_15m', 'Morning_Star_15m', 'Evening_Star_15m', 'Spinning_Top_15m',
                        'Three_White_Soldiers_15m', 'Three_Black_Crows_15m', 'Zielona_Swieca_15m',
                        'Czerwona_Swieca_15m', 'Kupuj_15m', 'Sprzedaj_15m', 'Target_15m']:
                data[col] = 0

        # --- Nowe wskaźniki ---
        # RSI
        rsi = RSIIndicator(data['close'], window=14)
        data['RSI'] = rsi.rsi() if len(data) >= 14 else 0  # Fallback na 0 jeśli za mało danych

        # ATR
        atr = AverageTrueRange(data['high'], data['low'], data['close'], window=14)
        data['ATR'] = atr.average_true_range() if len(data) >= 14 else 0  # Fallback na 0 jeśli za mało danych

        # Filtrowane formacje z ATR
        data['Bullish_Engulfing_Filtered'] = data['Bullish_Engulfing'] & (data['ATR'] > data['ATR'].shift(1))
        data['Bearish_Engulfing_Filtered'] = data['Bearish_Engulfing'] & (data['ATR'] > data['ATR'].shift(1))

        return data

    except Exception as e:
        logging.error(f"Błąd w obliczaniu formacji świecowych: {str(e)}")
        return data