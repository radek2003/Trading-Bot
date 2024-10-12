import MetaTrader5 as mt5
import logging
from src.risk_management import calculate_position_size
from config.config import RISK_REWARD_RATIO


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
        stop_loss_value = 0.01 * current_price  # 5% od bieżącej ceny
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

        # Pobranie informacji o koncie
        account_info = mt5.account_info()
        if account_info is None:
            logging.error("Nie można pobrać informacji o koncie.")
            return

        account_balance = account_info.balance
        wielkość_pozycji = calculate_position_size(account_balance, symbol)
        if wielkość_pozycji <= 0:
            logging.error("Wielkość pozycji jest równa 0, transakcja nie zostanie przeprowadzona.")
            return

        # Przygotowanie zlecenia handlowego z SL i TP
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": wielkość_pozycji,
            "type": order_type,
            "price": current_price,
            "sl": stop_loss_price,  # Ustawienie ceny stop loss
            "tp": take_profit_price,  # Ustawienie ceny take profit
            "magic": 234000,
            "comment": "Zautomatyzowany handel",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # Wysyłanie zlecenia
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Błąd wykonania transakcji: {result.retcode}, {mt5.last_error()}")
        else:
            logging.info(f"Transakcja przeprowadzona pomyślnie, ID pozycji: {result.order}")

    except Exception as e:
        logging.exception("Problem z realizacją transakcji.")


def check_for_closed_positions(symbol):
    """Sprawdza, czy pozycje zostały zamknięte na podstawie SL lub TP."""
    try:
        positions = mt5.positions_get(symbol=symbol)
        if positions is None:
            logging.error(f"Nie udało się pobrać informacji o pozycjach dla {symbol}.")
            return

        for position in positions:
            if position.sl > 0 and position.tp > 0:
                # Logowanie zamknięcia pozycji
                logging.info(f"Pozycja {position.ticket} zamknięta z powodu osiągnięcia SL/TP.")
            else:
                logging.info(f"Pozycja {position.ticket} wciąż otwarta.")

    except Exception as e:
        logging.exception("Problem ze sprawdzeniem pozycji.")


