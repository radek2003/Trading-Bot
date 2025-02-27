import MetaTrader5 as mt5
import logging
import numpy as np
from datetime import datetime, timedelta
import time
from ta.volatility import AverageTrueRange

def execute_trade(model_prediction, symbol, volume, historical_data=None, confidence=None):
    """
    Wykonuje transakcję z robust obliczeniem SL (MAD + ATR) i dynamicznym TP zależnym od pewności predykcji.

    Args:
        model_prediction: 1 (BUY) lub 0 (SELL)
        symbol: Symbol (np. 'EURUSD')
        volume: Wielkość pozycji
        historical_data: Dane historyczne do obliczenia SL
        confidence: Niepewność predykcji (prediction_confidence z Monte Carlo Dropout)
    """
    try:
        # Pobieranie informacji o symbolu
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logging.error("Nie można pobrać informacji o symbolu.")
            return

        punkt = symbol_info.point  # np. 0.00001 dla EURUSD
        digits = symbol_info.digits  # np. 5 dla EURUSD

        # Pobieranie aktualnych cen
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logging.error("Nie można pobrać aktualnych cen.")
            return

        current_price = tick.ask if model_prediction == 1 else tick.bid
        if current_price <= 0:
            logging.error(f"Niepoprawna cena bieżąca: {current_price}")
            return

        # Obliczanie robust Stop Loss (MAD + ATR) - dostosowane do ~150-160 pipsów
        min_pips = 150  # Minimalny SL zwiększony do 150 pipsów
        max_pips = 200  # Maksymalny SL zwiększony do 200 pipsów
        mad_factor = 15000
        atr_window = 14

        if historical_data is not None and not historical_data.empty:
            # MAD (Median Absolute Deviation)
            price_changes = np.abs(historical_data['close'].pct_change().dropna())
            mad = np.median(np.abs(price_changes - price_changes.median())) * 1.4826
            mad_pips = int(mad * mad_factor)

            # ATR (Average True Range)
            atr = AverageTrueRange(high=historical_data['high'], low=historical_data['low'],
                                   close=historical_data['close'], window=atr_window)
            atr_value = atr.average_true_range().iloc[-1]  # Ostatnia wartość ATR
            atr_pips = int(atr_value * 10000)  # Przelicz na pipsy

            # Dostosowana waga MAD i ATR, by SL był bliżej 150-160 pipsów
            stop_loss_pips = int(0.75 * mad_pips + 0.75 * atr_pips)  # Zmniejszono wagi z 1.0 na 0.75
            stop_loss_pips = min(max_pips, max(min_pips, stop_loss_pips))
        else:
            stop_loss_pips = min_pips  # Domyślny SL = 150 pipsów
            logging.warning("Brak danych historycznych, używam domyślnego SL: 150 pipsów.")

        # Obliczanie wartości SL i TP w jednostkach ceny
        stop_loss_value = stop_loss_pips * punkt

        # Dynamiczny Risk-Reward Ratio - dostosowany do ~1:1 lub 1.5:1
        if confidence is not None and confidence < 0.05:  # Wysoka pewność
            risk_reward_ratio = 1.5  # Zamiast 5.0, by TP było bliżej SL
            logging.debug("Wysoka pewność: RR ustawione na 1.5")
        else:
            risk_reward_ratio = 1.0  # Domyślnie 1:1, jak w logach
            logging.debug("Standardowa pewność: RR ustawione na 1.0")
        take_profit_value = stop_loss_value * risk_reward_ratio

        # Ustalanie poziomów SL i TP
        if model_prediction == 1:  # BUY
            stop_loss_price = current_price - stop_loss_value
            take_profit_price = current_price + take_profit_value
        else:  # SELL
            stop_loss_price = current_price + stop_loss_value
            take_profit_price = current_price - take_profit_value

        stop_loss_price = round(stop_loss_price, digits)
        take_profit_price = round(take_profit_price, digits)

        logging.debug(
            f"Current Price: {current_price}, SL Pips: {stop_loss_pips}, SL Value: {stop_loss_value}, TP Value: {take_profit_value}")
        logging.debug(f"SL Price: {stop_loss_price}, TP Price: {take_profit_price}")

        # Walidacja cen
        if stop_loss_price <= 0 or take_profit_price <= 0:
            logging.error(f"Błąd: Niepoprawne ceny SL/TP - SL: {stop_loss_price}, TP: {take_profit_price}")
            return

        if volume <= 0:
            logging.error("Wielkość pozycji jest równa 0, transakcja nie zostanie przeprowadzona.")
            return

        # Tworzenie zlecenia
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY if model_prediction == 1 else mt5.ORDER_TYPE_SELL,
            "price": current_price,
            "sl": stop_loss_price,
            "tp": take_profit_price,
            "magic": 234000,
            "comment": "Zautomatyzowany handel",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # Wysłanie zlecenia
        if not mt5.terminal_info().connected:
            logging.error("Brak połączenia z terminalem MT5.")
            return

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Błąd wykonania transakcji: {result.retcode}, {mt5.last_error()}")
        else:
            logging.info(f"Transakcja przeprowadzona pomyślnie, ID pozycji: {result.order}")

    except Exception as e:
        logging.exception("Problem z realizacją transakcji.")

# Funkcja check_for_closed_positions pozostaje bez zmian
def check_for_closed_positions(symbol):
    try:
        open_positions = mt5.positions_get(symbol=symbol)
        if open_positions is None:
            logging.error(f"Nie udało się pobrać informacji o otwartych pozycjach dla {symbol}. Kod błędu: {mt5.last_error()}")
            return
        if not open_positions:
            logging.debug(f"Brak otwartych pozycji dla {symbol}.")

        to_date = datetime.now()
        from_date = to_date - timedelta(days=10)
        from_date_ts = int(time.mktime(from_date.timetuple()))
        to_date_ts = int(time.mktime(to_date.timetuple()))
        history_deals = mt5.history_deals_get(from_date_ts, to_date_ts, group=f"*{symbol}*")
        if history_deals is None:
            logging.error(f"Nie udało się pobrać historii zleceń dla {symbol}. Kod błędu: {mt5.last_error()}")
            return

        open_position_ids = [pos.ticket for pos in open_positions] if open_positions else []
        for position in open_positions:
            logging.info(f"Pozycja {position.ticket} wciąż otwarta. SL: {position.sl}, TP: {position.tp}, Profit: {position.profit}")

        closed_positions = {}
        for deal in history_deals:
            if deal.type in [mt5.DEAL_TYPE_BUY, mt5.DEAL_TYPE_SELL] and deal.entry == mt5.DEAL_ENTRY_IN:
                closed_positions[deal.position_id] = {"open_price": deal.price, "volume": deal.volume, "type": deal.type}
            elif deal.type in [mt5.DEAL_TYPE_BUY, mt5.DEAL_TYPE_SELL] and deal.entry == mt5.DEAL_ENTRY_OUT:
                if deal.position_id in closed_positions:
                    open_data = closed_positions[deal.position_id]
                    closed_positions[deal.position_id]["close_price"] = deal.price
                    closed_positions[deal.position_id]["profit"] = deal.profit
                    closed_positions[deal.position_id]["reason"] = deal.reason

        for pos_id, data in closed_positions.items():
            if "close_price" in data and pos_id not in open_position_ids:
                reason_str = "Ręczna" if data["reason"] == mt5.DEAL_REASON_CLIENT else \
                            "SL" if abs(data["close_price"] - data["open_price"]) <= 0.0001 else \
                            "TP" if abs(data["close_price"] - data["open_price"]) >= 0.0001 else "Inny"
                logging.info(f"Pozycja {pos_id} zamknięta. Typ: {'BUY' if data['type'] == mt5.DEAL_TYPE_BUY else 'SELL'}, "
                             f"Otwarcie: {data['open_price']:.5f}, Zamknięcie: {data['close_price']:.5f}, "
                             f"Profit: {data['profit']:.2f}, Powód: {reason_str}")

    except Exception as e:
        logging.exception(f"Problem ze sprawdzeniem pozycji dla {symbol}: {str(e)}")