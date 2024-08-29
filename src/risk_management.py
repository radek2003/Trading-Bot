import MetaTrader5 as mt5
import math
import logging
from config.config import MAX_RISK_PER_TRADE

# Zarządzanie ryzykiem

def calculate_position_size(account_balance, symbol):
    """Oblicza rozmiar pozycji na podstawie kapitału i symbolu."""
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
            logging.error("Brak informacji o punktach lub rozmiarze kontraktu dla symbolu.")
            return 0

        punkt_value = punkt * trade_contract_size
        ryzyko_na_transakcję = account_balance * MAX_RISK_PER_TRADE
        stop_loss_value = ryzyko_na_transakcję
        wielkość_pozycji = stop_loss_value / punkt_value

        if wielkość_pozycji < min_lot:
            wielkość_pozycji = min_lot
        if wielkość_pozycji > max_lot:
            wielkość_pozycji = max_lot

        wielkość_pozycji = math.floor(wielkość_pozycji / step_lot) * step_lot

        return wielkość_pozycji
    except Exception as e:
        logging.exception("Problem z obliczaniem wielkości pozycji.")
        return 0
