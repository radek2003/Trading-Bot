import MetaTrader5 as mt5
import logging
import numpy as np
from datetime import datetime, timedelta
import time
import pandas as pd
from ta.volatility import AverageTrueRange
import threading
import warnings
from src.settings_manager import get_setting

warnings.filterwarnings("ignore", category=FutureWarning)  


def execute_trade(model_prediction, symbol, volume, historical_data=None, confidence=None):
    try:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logging.error("Nie można pobrać informacji o symbolu.")
            return

        punkt = symbol_info.point
        digits = symbol_info.digits
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logging.error("Nie można pobrać aktualnych cen.")
            return

        current_price = tick.ask if model_prediction == 1 else tick.bid
        if current_price <= 0:
            logging.error(f"Niepoprawna cena bieżąca: {current_price}")
            return

        # Obliczanie spreadu
        spread = (tick.ask - tick.bid) / punkt

        # Dynamiczny SL oparty na ATR
        #min_pips = 150
        #max_pips = 200
        #trading_reload_min = int(get_setting("trading_reload", 30))
        min_pips = int(get_setting("min_pips", 150))
        max_pips = int(get_setting("max_pips", 200))

        
        atr_window = 14
        atr_multiplier = 2.5

        if historical_data is not None and not historical_data.empty:
            atr = AverageTrueRange(high=historical_data['high'], low=historical_data['low'],
                                   close=historical_data['close'], window=atr_window)
            atr_value = atr.average_true_range().iloc[-1]
            stop_loss_pips = int(atr_value * atr_multiplier / punkt)
            stop_loss_pips = min(max_pips, max(min_pips, stop_loss_pips))
        else:
            stop_loss_pips = min_pips
            logging.warning("Brak danych historycznych, używam domyślnego SL: 150 pipsów.")

        stop_loss_value = stop_loss_pips * punkt

        # Elastyczne TP oparte na pewności
        scaling_factor = 1.0
        if confidence is not None:
            risk_reward_ratio = 1 + (1 - confidence) * scaling_factor
        else:
            risk_reward_ratio = 1.0
        take_profit_value = stop_loss_value * risk_reward_ratio

        # Ustalanie SL i TP z uwzględnieniem spreadu
        if model_prediction == 1:  # BUY
            stop_loss_price = current_price - stop_loss_value - spread * punkt
            take_profit_price = current_price + take_profit_value + spread * punkt
        else:  # SELL
            stop_loss_price = current_price + stop_loss_value + spread * punkt
            take_profit_price = current_price - take_profit_value - spread * punkt

        stop_loss_price = round(stop_loss_price, digits)
        take_profit_price = round(take_profit_price, digits)

        # Walidacja
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

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Błąd wykonania transakcji: {result.retcode}, {mt5.last_error()}")
        else:
            logging.info(f"Transakcja przeprowadzona pomyślnie, ID pozycji: {result.order}")

    except Exception as e:
        logging.exception("Problem z realizacją transakcji.")

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

def adjust_sl_tp_robustly(symbol, atr_window=14, atr_multiplier_sl=2.5, atr_multiplier_tp=3.5, min_pips_sl=100, max_pips_sl=200, confidence_threshold=0.15):
    try:
        if not mt5.initialize():
            logging.error(f"Inicjalizacja MT5 nie powiodła się: {mt5.last_error()}")
            return

        logging.info(f"Rozpoczęto monitorowanie i modyfikację SL/TP dla {symbol}")

        while True:
            # Pobieranie otwartych pozycji
            positions = mt5.positions_get(symbol=symbol)
            if positions is None:
                logging.error(f"Nie udało się pobrać pozycji dla {symbol}: {mt5.last_error()}")
                continue
            if not positions:
                logging.debug(f"Brak otwartych pozycji dla {symbol}. Oczekiwanie na nowe pozycje...")
                time.sleep(5)  # Krótkie opóźnienie, jeśli brak pozycji
                continue

            # Pobieranie danych historycznych (5-minutowych) do obliczenia ATR
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, atr_window + 1)
            if rates is None or len(rates) < atr_window:
                logging.warning(f"Za mało danych historycznych dla {symbol} do obliczenia ATR.")
                time.sleep(5)
                continue

            historical_data = pd.DataFrame(rates)
            atr = AverageTrueRange(high=historical_data['high'], low=historical_data['low'],
                                   close=historical_data['close'], window=atr_window)
            atr_value = atr.average_true_range().iloc[-1]

            # Pobieranie aktualnych cen rynkowych
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logging.error(f"Nie można pobrać tick dla {symbol}")
                continue

            current_price = tick.ask if positions[0].type == mt5.ORDER_TYPE_BUY else tick.bid
            symbol_info = mt5.symbol_info(symbol)
            punkt = symbol_info.point
            digits = symbol_info.digits
            spread = (tick.ask - tick.bid) / punkt

            # Robust volatility (MAD) na podstawie ostatnich zmian cen
            price_changes = historical_data['close'].pct_change().dropna()
            mad = np.median(np.abs(price_changes - price_changes.median())) * 1.4826
            volatility_adjustment = 1 + mad * 100

            # Iteracja po wszystkich otwartych pozycjach
            for position in positions:
                position_id = position.ticket
                is_buy = position.type == mt5.ORDER_TYPE_BUY
                original_sl = position.sl
                original_tp = position.tp
                open_price = position.price_open

                # Obliczanie dynamicznego SL opartego na ATR i zmienności
                stop_loss_pips = int(atr_value * atr_multiplier_sl / punkt / volatility_adjustment)
                stop_loss_pips = min(max_pips_sl, max(min_pips_sl, stop_loss_pips))
                stop_loss_value = stop_loss_pips * punkt

                # Obliczanie dynamicznego TP z robust optimization
                take_profit_pips = int(atr_value * atr_multiplier_tp / punkt / volatility_adjustment)
                take_profit_value = take_profit_pips * punkt

                # Ustalanie nowych SL i TP z uwzględnieniem spreadu i typu pozycji
                if is_buy:
                    new_sl = current_price - stop_loss_value - spread * punkt
                    new_tp = current_price + take_profit_value + spread * punkt
                else:
                    new_sl = current_price + stop_loss_value + spread * punkt
                    new_tp = current_price - take_profit_value - spread * punkt

                new_sl = round(new_sl, digits)
                new_tp = round(new_tp, digits)

                # Robust optimization: modyfikacja tylko przy znaczącym ruchu rynkowym
                price_move = abs(current_price - open_price) / punkt
                min_move_threshold = stop_loss_pips * 0.5  # Próg ruchu: 50% obecnego SL

                # Sprawdzanie, czy ruch rynkowy uzasadnia modyfikację
                if price_move > min_move_threshold and abs(new_sl - original_sl) > punkt * 5:
                    # Tworzenie zlecenia modyfikacji
                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": position_id,
                        "sl": new_sl,
                        "tp": new_tp,
                        "symbol": symbol,
                    }

                    result = mt5.order_send(request)
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        logging.info(f"Pozycja {position_id}: SL zmieniono na {new_sl}, TP na {new_tp}")
                    else:
                        logging.error(f"Błąd modyfikacji pozycji {position_id}: {result.retcode}, {mt5.last_error()}")

            # Krótkie opóźnienie, aby uniknąć przeciążenia MT5
            time.sleep(300)

    except KeyboardInterrupt:
        logging.info("Przerwano monitorowanie SL/TP przez użytkownika.")
    except Exception as e:
        logging.error(f"Błąd w adjust_sl_tp_robustly: {str(e)}")
    finally:
        mt5.shutdown()
        logging.info("Zakończono monitorowanie SL/TP.")

def integrate_with_main(symbol="EURUSD"):
    """Integracja z funkcją main() poprzez uruchomienie w osobnym wątku."""
    sl_tp_thread = threading.Thread(target=adjust_sl_tp_robustly, args=(symbol,), daemon=True)
    sl_tp_thread.start()
    logging.info("Uruchomiono wątek do dynamicznej modyfikacji SL/TP")
    return sl_tp_thread

if __name__ == "__main__":
    pass