import logging
import pandas as pd
import numpy as np
import torch
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from ta.trend import VortexIndicator

def apply_strategy(model, scaler, new_data, gamma=0.1, num_samples=500):
    """Stosuje strategię z kryterium Walda i γ-odpornością, analizując wszystkie strategie naraz."""
    try:
        # Obliczenie cech dla wszystkich strategii, w tym Vortex Indicator
        new_data_with_features = calculate_all_features(new_data)

        if new_data_with_features.empty:
            logging.error("Brak danych z cechami do prognozowania.")
            return None

        # Przygotowanie danych do predykcji
        X_new = scaler.transform(new_data_with_features.drop('Target', axis=1, errors='ignore').values)
        X_new_tensor = torch.tensor(X_new, dtype=torch.float32)

        # Sprawdzenie i dostosowanie wymiarów
        if X_new_tensor.dim() == 2:
            X_new_tensor = X_new_tensor.unsqueeze(0)  # Dodaj wymiar batcha
            logging.debug(f"Dodano wymiar batcha, nowy kształt X_new_tensor: {X_new_tensor.shape}")

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
        predictions = predictions[:, 0, :]  # Kształt: (num_samples, 2)

        # Definicja akcji: 0 - Wstrzymaj się, 1 - Kup, -1 - Sprzedaj
        actions = [0, 1, -1]

        # Definicja scenariuszy
        scenarios = ['MACD', 'Candlestick', 'Moving_Averages', 'Vortex']

        # Obliczenie wyników dla każdej akcji w każdym scenariuszu
        action_scores = {action: [] for action in actions}
        for scenario in scenarios:
            for action in actions:
                if action == 0:
                    # Wstrzymaj się: wynik neutralny
                    scores = np.zeros(num_samples)
                else:
                    # Dla modelu binarnego: używamy prawdopodobieństwa "handluj" (klasa 1)
                    prob_trade = predictions[:, 1]  # Prawdopodobieństwo "handluj"
                    prob_no_trade = predictions[:, 0]  # Prawdopodobieństwo "nie handluj"
                    base_score = prob_trade - prob_no_trade  # Różnica między "handluj" a "nie handluj"

                    # Kierunek akcji (kup/sprzedaj) na podstawie scenariusza
                    if scenario == 'MACD':
                        macd_signal = new_data_with_features['MACD_Histogram'].iloc[-1]
                        adjustment = 1 if macd_signal > 0 else -1
                    elif scenario == 'Candlestick':
                        bullish_patterns = ['Bullish_Engulfing', 'Hammer', 'Morning_Star']
                        bearish_patterns = ['Bearish_Engulfing', 'Shooting_Star', 'Evening_Star']
                        bullish = any(new_data_with_features[pattern].iloc[-1] for pattern in bullish_patterns)
                        bearish = any(new_data_with_features[pattern].iloc[-1] for pattern in bearish_patterns)
                        adjustment = 1 if bullish else (-1 if bearish else 0)
                    elif scenario == 'Moving_Averages':
                        signal = new_data_with_features['Signal'].iloc[-1]
                        adjustment = signal
                    elif scenario == 'Vortex':
                        vortex_signal = new_data_with_features['Vortex_Diff'].iloc[-1]
                        adjustment = 1 if vortex_signal > 0 else -1

                    # Dopasowanie akcji: jeśli action=1 (kup), ujemny adjustment zmienia na sprzedaj
                    if action == 1 and adjustment < 0:
                        scores = -base_score  # Sprzedaj
                    elif action == -1 and adjustment > 0:
                        scores = -base_score  # Kup (odwrócone)
                    else:
                        scores = base_score * adjustment  # Normalne przypisanie

                # Zapisanie wyników
                action_scores[action].append(scores)

        # Obliczenie γ-fraktyla dla każdej akcji w różnych scenariuszach
        gamma_fractiles = []
        for action in actions:
            all_scores = np.concatenate(action_scores[action], axis=0)
            gamma_fractile = np.percentile(all_scores, gamma * 100)
            gamma_fractiles.append(gamma_fractile)

        # Wybór akcji z najwyższym γ-fraktylem
        best_action_idx = np.argmax(gamma_fractiles)
        robust_prediction = actions[best_action_idx]

        logging.info(f"Wybrana akcja: {robust_prediction} (γ-fraktyle: {gamma_fractiles})")
        return robust_prediction

    except Exception as e:
        logging.exception("Problem z zastosowaniem strategii.")
        return None

def calculate_all_features(data):
    """Oblicza cechy dla wszystkich strategii, w tym Vortex Indicator."""
    try:
        data_with_features = data.copy()

        # Obliczenie cech MACD
        data_with_features = calculate_robust_macd(data_with_features)

        # Obliczenie cech formacji świecowych
        data_with_features = calculate_candlestick_patterns(data_with_features)

        # Obliczenie cech średnich kroczących
        data_with_features = calculate_robust_moving_averages(data_with_features)

        # Obliczenie Vortex Indicator
        data_with_features = calculate_vortex_indicator(data_with_features)

        return data_with_features

    except Exception as e:
        logging.exception("Problem z obliczaniem wszystkich cech.")
        return pd.DataFrame()

def calculate_vortex_indicator(data):
    """Oblicza Vortex Indicator."""
    try:
        vortex = VortexIndicator(high=data['high'], low=data['low'], close=data['close'], window=14)
        data['Vortex_Pos'] = vortex.vortex_indicator_pos()
        data['Vortex_Neg'] = vortex.vortex_indicator_neg()
        data['Vortex_Diff'] = data['Vortex_Pos'] - data['Vortex_Neg']
        return data
    except Exception as e:
        logging.exception("Problem z obliczaniem Vortex Indicator.")
        return data

def calculate_robust_moving_averages(data):
    """Oblicza odporne średnie kroczące (mediana) i sygnały."""
    if data.empty:
        logging.error("Brak danych do obliczeń.")
        return pd.DataFrame()

    try:
        data = data.copy()
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

        # Formacje 5-minutowe
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

        # Formacje 15-minutowe (jeśli wystarczająco danych)
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

        # Nowe wskaźniki
        rsi = RSIIndicator(data['close'], window=14)
        data['RSI'] = rsi.rsi() if len(data) >= 14 else 0

        atr = AverageTrueRange(data['high'], data['low'], data['close'], window=14)
        data['ATR'] = atr.average_true_range() if len(data) >= 14 else 0

        data['Bullish_Engulfing_Filtered'] = data['Bullish_Engulfing'] & (data['ATR'] > data['ATR'].shift(1))
        data['Bearish_Engulfing_Filtered'] = data['Bearish_Engulfing'] & (data['ATR'] > data['ATR'].shift(1))

        return data

    except Exception as e:
        logging.error(f"Błąd w obliczaniu formacji świecowych: {str(e)}")
        return data