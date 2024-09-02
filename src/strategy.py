import logging
import torch
import pandas as pd
import numpy as np


def apply_strategy(model, scaler, new_data, strategy_type='MACD'):
    """Stosuje wybraną strategię na nowych danych."""
    try:
        if strategy_type == 'MACD':
            new_data_with_features = calculate_macd(new_data)
            new_data_with_features = calculate_moving_averages(new_data_with_features)
        else:
            logging.error(f"Nieznana strategia: {strategy_type}")
            return None

        if new_data_with_features.empty:
            logging.error("Brak danych z cechami do prognozowania.")
            return None

        # Przygotowanie danych do PyTorch
        X_new = scaler.transform(new_data_with_features.drop('Target', axis=1).values)
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