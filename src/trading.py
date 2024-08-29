import MetaTrader5 as mt5
import logging
from src.risk_management import calculate_position_size
from config.config import RISK_REWARD_RATIO

# Otwieranie pozycji, handlowanie itp.

def execute_trade(model_prediction, symbol):
    """Wykonuje transakcję na podstawie predykcji modelu."""
    try:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logging.error("Nie można pobrać informacji o symbolu.")
            return

        punkt = symbol_info.point
        digits = symbol_info.digits

        if punkt is None:
            logging.error("Brak informacji o punktach dla symbolu.")
            return

        # Pobierz aktualne ceny
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logging.error("Nie można pobrać aktualnych cen.")
            return

        current_price = tick.ask if model_prediction == 1 else tick.bid

        # Obliczanie wartości stop loss i take profit w punktach
        stop_loss_value = 0.05 * current_price
        take_profit_value = stop_loss_value * RISK_REWARD_RATIO

        if model_prediction == 1:  # BUY
            stop_loss_price = current_price - stop_loss_value
            take_profit_price = current_price + take_profit_value
            order_type = mt5.ORDER_TYPE_BUY
        else:  # SELL
            stop_loss_price = current_price + stop_loss_value
            take_profit_price = current_price - take_profit_value
            order_type = mt5.ORDER_TYPE_SELL

        stop_loss_price = round(stop_loss_price, digits)
        take_profit_price = round(take_profit_price, digits)

        if stop_loss_price <= 0 or take_profit_price <= 0:
            logging.error("Błąd: Niepoprawne ceny SL/TP.")
            return

        account_info = mt5.account_info()
        if account_info is None:
            logging.error("Nie można pobrać informacji o koncie.")
            return

        account_balance = account_info.balance
        wielkość_pozycji = calculate_position_size(account_balance, symbol)
        if wielkość_pozycji <= 0:
            logging.error("Wielkość pozycji jest równa 0, transakcja nie zostanie przeprowadzona.")
            return

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": wielkość_pozycji,
            "type": order_type,
            "price": current_price,
            "sl": stop_loss_price,
            "tp": take_profit_price,
            "magic": 234000,
            "comment": "Zautomatyzowany handel",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Błąd wykonania transakcji: {result.retcode}, {mt5.last_error()}")
        else:
            logging.info(f"Transakcja przeprowadzona pomyślnie, ID pozycji: {result.order}")

    except Exception as e:
        logging.exception("Problem z realizacją transakcji.")
