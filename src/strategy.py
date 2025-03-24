import logging
import pandas as pd
import numpy as np
import torch
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from ta.trend import VortexIndicator


def apply_strategy(model, scaler, new_data, gamma=0.1, num_samples=5000):
    try:
        new_data_with_features = calculate_all_features(new_data)

        if new_data_with_features.empty:
            logging.error("Brak danych do analizy.")
            return None

        # 1. Pobierz ostatnią cenę
        current_price = new_data_with_features['close'].iloc[-1] if 'close' in new_data_with_features.columns else 0.0

        # 2. Pobierz wartości ze wskaźników jako SKALARY
        try:
            distribution_signal = calculate_price_distribution(new_data_with_features, current_price)
        except Exception as e:
            logging.error(f"Błąd analizy rozkładu cen: {str(e)}")
            distribution_signal = 0

        try:
            bb_pct = new_data_with_features['BB_Pct'].iloc[-1] if 'BB_Pct' in new_data_with_features.columns else 0.5
        except KeyError:
            bb_pct = 0.5
            logging.warning("Brak kolumny BB_Pct - użyto domyślnej wartości 0.5")

        # 3. Sprawdź poprawność skalera
        if not hasattr(scaler, 'feature_names'):
            logging.error("Brak informacji o oczekiwanych cechach!")
            return None

        # 4. Oryginalna logika strategii z poprawkami
        expected_features = scaler.feature_names
        missing = list(set(expected_features) - set(new_data_with_features.columns))
        if missing:
            logging.error(f"Brakujące cechy: {missing}")
            return None

        features = new_data_with_features[expected_features]
        X_new = scaler.transform(features.values)
        X_new_tensor = torch.tensor(X_new, dtype=torch.float32).unsqueeze(0)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        predictions = []
        model.train()
        for _ in range(num_samples):
            with torch.no_grad():
                output = model(X_new_tensor.to(device))
                predictions.append(torch.softmax(output, dim=1).cpu().numpy())

        predictions = np.array(predictions)[:, 0, :]

        actions = [0, 1, -1]
        scenarios = ['MACD', 'Candlestick', 'Moving_Averages', 'Vortex', 'Bollinger', 'PriceDistribution']
        action_scores = {action: [] for action in actions}

        current_sentiment = new_data_with_features['sentiment'].iloc[-1] if 'sentiment' in new_data_with_features.columns else 0.0
        sentiment_factor = 1 + (current_sentiment * 0.3)

        for scenario in scenarios:
            for action in actions:
                if action == 0:
                    scores = np.zeros(num_samples)
                else:
                    prob_trade = predictions[:, 1]
                    prob_no_trade = predictions[:, 0]
                    base_score = (prob_trade - prob_no_trade) * sentiment_factor

                    # Upewnij się, że wszystkie wartości są SKALARAMI
                    if scenario == 'MACD':
                        adj = 1 if new_data_with_features['MACD_Histogram'].iloc[-1] > 0 else -1
                    elif scenario == 'Candlestick':
                        bull = any([
                            new_data_with_features[p].iloc[-1]
                            for p in ['Bullish_Engulfing', 'Hammer']
                            if p in new_data_with_features.columns
                        ])
                        adj = 1 if bull else -1
                    elif scenario == 'Moving_Averages':
                        adj = new_data_with_features['Signal'].iloc[-1] if 'Signal' in new_data_with_features.columns else 0
                    elif scenario == 'Vortex':
                        adj = 1 if new_data_with_features['Vortex_Diff'].iloc[-1] > 0 else -1 if 'Vortex_Diff' in new_data_with_features.columns else 0
                    elif scenario == 'Bollinger':
                        adj = 1.5 if bb_pct > 0.8 else (-1.5 if bb_pct < 0.2 else 1.0)
                    elif scenario == 'PriceDistribution':
                        adj = distribution_signal * 1.2

                    scores = base_score * adj if (action == 1 and adj > 0) or (action == -1 and adj < 0) else -base_score

                action_scores[action].append(scores)

        gamma_percentiles = [np.percentile(np.concatenate(v), gamma * 100) for v in action_scores.values()]
        best_action = actions[np.argmax(gamma_percentiles)]

        logging.info(f"Akcja: {best_action} | BB%: {bb_pct:.2f} | Distrib: {distribution_signal}")
        return best_action

    except Exception as e:
        logging.error(f"Błąd strategii: {str(e)}")
        return None

def calculate_all_features(data):
    try:
        original_time = data['time'].copy() if 'time' in data.columns else None
        data = data.copy()
        data = data.drop(columns=['symbol'], errors='ignore')

        # Dodaj nowe wskaźniki
        data = calculate_robust_macd(data)
        data = calculate_candlestick_patterns(data)
        data = calculate_robust_moving_averages(data)
        data = calculate_vortex_indicator(data)
        data = calculate_bollinger_bands(data)  # Nowy wskaźnik

        if 'sentiment' not in data.columns:
            data['sentiment'] = 0.0
            logging.warning("Brak sentymentu - ustawiono 0.0")

        if original_time is not None:
            data['time'] = original_time

        return data
    except Exception as e:
        logging.error(f"Błąd obliczania cech: {str(e)}")
        return pd.DataFrame()


def calculate_price_distribution(data, current_price):
    """Analizuje rozkład cen z ostatnich 2 tygodni i 3 dni."""
    try:
        if 'time' not in data.columns or len(data) < 15:
            logging.warning("Brak danych czasowych lub za mało rekordów")
            return 0

        data = data.copy()
        data['time'] = pd.to_datetime(data['time'])
        data.set_index('time', inplace=True, drop=False)

        # Analiza 2-tygodniowa
        two_week_data = data.last('14D')['close']
        two_week_low = two_week_data.quantile(0.1) if len(two_week_data) >= 5 else np.nan
        two_week_high = two_week_data.quantile(0.9) if len(two_week_data) >= 5 else np.nan

        # Analiza 3-dniowa
        three_day_data = data.last('3D')['close']
        three_day_low = three_day_data.quantile(0.1) if len(three_day_data) >= 3 else np.nan
        three_day_high = three_day_data.quantile(0.9) if len(three_day_data) >= 3 else np.nan

        # Warunki z zabezpieczeniem przed NaN
        sell_condition = (
            not np.isnan(two_week_high) and
            not np.isnan(three_day_high) and
            (current_price > two_week_high) and
            (current_price > three_day_high)
        )

        buy_condition = (
            not np.isnan(two_week_low) and
            not np.isnan(three_day_low) and
            (current_price < two_week_low) and
            (current_price < three_day_low)
        )

        return -1 if sell_condition else 1 if buy_condition else 0

    except Exception as e:
        logging.error(f"Błąd analizy rozkładu: {str(e)}")
        return 0


def calculate_bollinger_bands(data, window=20, num_std=2):
    """Oblicza Bollinger Bands z robustną obsługą błędów."""
    try:
        if len(data) < window:
            logging.warning(f"Za mało danych ({len(data)} rekordów) do obliczenia Bollinger Bands")
            return data

        data = data.copy()
        data['BB_MA'] = data['close'].rolling(window=window, min_periods=1).mean()
        data['BB_STD'] = data['close'].rolling(window=window, min_periods=1).std()
        data['BB_Upper'] = data['BB_MA'] + (data['BB_STD'] * num_std)
        data['BB_Lower'] = data['BB_MA'] - (data['BB_STD'] * num_std)
        data['BB_Pct'] = (data['close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'] + 1e-8)
        return data
    except Exception as e:
        logging.error(f"Błąd Bollinger Bands: {str(e)}")
        return data



def calculate_vortex_indicator(data):
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
        return df

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