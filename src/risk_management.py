import MetaTrader5 as mt5
import math
import logging
import numpy as np
from config.config import MAX_RISK_PER_TRADE

def calculate_robust_volatility(data, window=20):
    """Oblicza odporną miarę zmienności (MAD) na podstawie danych cenowych."""
    if len(data) < window:
        logging.warning("Za mało danych do obliczenia zmienności.")
        return 0
    price_changes = np.abs(data['close'].pct_change().dropna())
    mad = np.median(np.abs(price_changes - price_changes.median())) * 1.4826  # Skalowanie do odchylenia standardowego
    return mad

def calculate_position_size(account_balance, symbol, historical_data=None):
    """Oblicza rozmiar pozycji z uwzględnieniem odpornej zmienności i optymalizacji."""
    try:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logging.error("Nie można pobrać informacji o symbolu.")
            return 0

        min_lot = symbol_info.volume_min
        step_lot = symbol_info.volume_step
        max_lot = symbol_info.volume_max
        punkt = symbol_info.point
        trade_contract_size = symbol_info.trade_contract_size

        if punkt is None or trade_contract_size is None:
            logging.error("Brak informacji o punktach lub rozmiarze kontraktu.")
            return 0

        punkt_value = punkt * trade_contract_size
        base_risk = account_balance * MAX_RISK_PER_TRADE

        # Robust volatility adjustment
        if historical_data is not None and not historical_data.empty:
            volatility = calculate_robust_volatility(historical_data)
            if volatility > 0:
                # Dynamiczne ryzyko: mniejsze przy większej zmienności
                adjusted_risk = base_risk / (1 + volatility * 100)  # Skalowanie zmienności
            else:
                adjusted_risk = base_risk
        else:
            adjusted_risk = base_risk
            logging.warning("Brak danych historycznych, używam базового ryzyka.")

        # Robust optimization: minimalizacja ryzyka w najgorszym scenariuszu
        stop_loss_pips = 20  # Przykładowy Stop Loss, można dostosować
        worst_case_loss = stop_loss_pips * punkt_value * (1 + volatility)  # Najgorszy scenariusz
        position_size = adjusted_risk / worst_case_loss

        # Dopasowanie do ograniczeń MT5
        if position_size < min_lot:
            position_size = min_lot
        if position_size > max_lot:
            position_size = max_lot
        position_size = math.floor(position_size / step_lot) * step_lot

        logging.info(f"Robust position size: {position_size}, Volatility: {volatility}")
        return position_size

    except Exception as e:
        logging.exception("Problem z obliczaniem robust wielkości pozycji.")
        return 0

if __name__ == "__main__":
    pass