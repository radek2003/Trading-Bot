import MetaTrader5 as mt5  # Ensure mt5 is imported correctly
import logging
import pandas as pd
import time
import os
import torch
import numpy as np
from src.data_fetcher import fetch_historical_data, test_trade_history
from src.model_methods import train_model_with_history, mc_dropout_predict
from src.trading import execute_trade, check_for_closed_positions, integrate_with_main
from src.strategy import apply_strategy, calculate_all_features
from src.risk_management import calculate_position_size

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

# Cache for news to avoid rate limiting
news_cache = {}
last_news_fetch = {}


def calculate_correlations(data_dict):
    """Calculate correlations between close prices of currency pairs."""
    try:
        close_prices = {}
        for symbol, (data_m5, _) in data_dict.items():
            if not data_m5.empty and 'close' in data_m5.columns:
                close_prices[symbol] = data_m5['close']

        if len(close_prices) < 2:
            logging.warning("Not enough data to calculate correlations.")
            return pd.DataFrame()

        close_df = pd.DataFrame(close_prices)
        returns = close_df.pct_change(fill_method=None).dropna()
        corr_matrix = returns.corr()
        logging.info(f"Correlation matrix:\n{corr_matrix}")
        return corr_matrix
    except Exception as e:
        logging.error(f"Error calculating correlations: {str(e)}")
        return pd.DataFrame()


def add_correlation_features(data, corr_matrix, symbol):
    """Add correlation features to the data for a specific symbol."""
    try:
        if corr_matrix.empty:
            data['avg_correlation'] = 0.0
            return data
        correlations = corr_matrix.get(symbol, pd.Series(0.0))
        other_pairs = [s for s in correlations.index if s != symbol]
        if other_pairs:
            avg_corr = correlations[other_pairs].mean()
        else:
            avg_corr = 0.0
        data['avg_correlation'] = avg_corr
        return data
    except Exception as e:
        logging.error(f"Error adding correlation features for {symbol}: {str(e)}")
        data['avg_correlation'] = 0.0
        return data


def main():
    if not mt5.initialize():
        logging.error(f"MT5 initialization failed, error code: {mt5.last_error()}")
        return

    symbols = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF"]
    min_candles_for_patterns = 200
    seq_len = 30

    # Set script_dir robustly
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))  # Fixed syntax: proper parentheses
    except NameError:
        # Fallback for environments where __file__ is not defined
        script_dir = os.getcwd()  # Use current working directory as fallback
        logging.warning("Could not determine script directory using __file__. Using current working directory: %s",
                        script_dir)

    folder_path = os.path.join(script_dir, "models")

    # Create models directory if it doesn't exist
    try:
        os.makedirs(folder_path, exist_ok=True)
        logging.info(f"Models folder ready at: {folder_path}")
    except Exception as e:
        logging.error(f"Failed to create models folder at {folder_path}: {str(e)}")
        return

    sl_tp_threads = {symbol: integrate_with_main(symbol=symbol) for symbol in symbols}

    try:
        while True:
            logging.info("Starting new trading iteration")

            data_dict = {}
            trade_histories = {}
            for symbol in symbols:
                data_m5, data_m15 = fetch_historical_data(symbol)
                if data_m5.empty or data_m15.empty:
                    logging.error(f"No 5-minute or 15-minute data for {symbol}.")
                    continue
                if len(data_m5) < min_candles_for_patterns:
                    logging.error(f"Insufficient historical data ({len(data_m5)} candles) for {symbol}.")
                    continue
                data_dict[symbol] = (data_m5, data_m15)
                trade_histories[symbol] = test_trade_history()

            corr_matrix = calculate_correlations(data_dict)

            combined_data = []
            for symbol in symbols:
                if symbol not in data_dict:
                    continue
                data_m5, _ = data_dict[symbol]
                trade_history = trade_histories[symbol]
                data_with_features = calculate_all_features(data_m5)
                if data_with_features.empty:
                    logging.error(f"No feature data for {symbol}.")
                    continue

                data_with_features = add_correlation_features(data_with_features, corr_matrix, symbol)

                if not trade_history.empty:
                    data_with_features = data_with_features.merge(trade_history, on='time', how='left').fillna(0)
                    data_with_features = data_with_features.drop(columns=['symbol'], errors='ignore')
                else:
                    data_with_features['profit'] = 0
                    data_with_features['volume'] = 0
                    data_with_features['type'] = 0

                if 'time' in data_with_features.columns:
                    data_with_features['hour'] = pd.to_datetime(data_with_features['time']).dt.hour
                    data_with_features = data_with_features.drop(columns=['time'])
                if 'profit' in data_with_features.columns:
                    data_with_features['trade_success'] = (data_with_features['profit'] > 0).astype(int)
                if 'type' in data_with_features.columns:
                    data_with_features['trade_type'] = data_with_features['type'].map({1: 1, 0: 0}).fillna(0)

                combined_data.append(data_with_features)

            if not combined_data:
                logging.error("No valid data to train model.")
                time.sleep(60)
                continue
            combined_data = pd.concat(combined_data, ignore_index=True)

            model_filename = 'best_model.pth'
            model, scaler, training_columns = train_model_with_history(combined_data, folder_path, model_filename)

            if model is None or scaler is None or training_columns is None:
                logging.error("Model, scaler, or training columns not loaded.")
                time.sleep(60)
                continue

            for symbol in symbols:
                logging.info(f"Processing {symbol}")
                if symbol not in data_dict:
                    logging.error(f"No data available for {symbol}.")
                    continue

                new_data_m5, new_data_m15 = fetch_historical_data(symbol)
                if new_data_m5.empty:
                    logging.error(f"No new 5-minute data for {symbol}.")
                    continue

                latest_data = new_data_m5.tail(min_candles_for_patterns + seq_len - 1)
                if len(latest_data) < seq_len:
                    logging.error(
                        f"Insufficient data ({len(latest_data)} candles) for LSTM sequence ({seq_len}) in {symbol}.")
                    continue

                latest_data_with_features = calculate_all_features(latest_data)
                if latest_data_with_features.empty:
                    logging.error(f"No feature data for predictions in {symbol}.")
                    continue

                latest_data_with_features = add_correlation_features(latest_data_with_features, corr_matrix, symbol)

                latest_trade_history = test_trade_history()
                if not latest_trade_history.empty:
                    latest_data_with_features = latest_data_with_features.merge(latest_trade_history, on='time',
                                                                                how='left').fillna(0)
                    latest_data_with_features = latest_data_with_features.drop(columns=['symbol'], errors='ignore')
                else:
                    latest_data_with_features['profit'] = 0
                    latest_data_with_features['volume'] = 0
                    latest_data_with_features['type'] = 0

                if 'time' in latest_data_with_features.columns:
                    latest_data_with_features['hour'] = pd.to_datetime(latest_data_with_features['time']).dt.hour
                    latest_data_with_features = latest_data_with_features.drop(columns=['time'])
                if 'profit' in latest_data_with_features.columns:
                    latest_data_with_features['trade_success'] = (latest_data_with_features['profit'] > 0).astype(int)
                if 'type' in latest_data_with_features.columns:
                    latest_data_with_features['trade_type'] = latest_data_with_features['type'].map(
                        {1: 1, 0: 0}).fillna(0)

                try:
                    if not hasattr(scaler, 'feature_names'):
                        logging.error(f"Scaler lacks feature information for {symbol}.")
                        continue

                    expected_features = scaler.feature_names
                    missing = list(set(expected_features) - set(latest_data_with_features.columns))
                    extra = list(set(latest_data_with_features.columns) - set(expected_features))

                    for feat in missing:
                        latest_data_with_features[feat] = 0
                        logging.warning(f"Added missing feature: {feat} for {symbol}")

                    latest_data_with_features = latest_data_with_features[expected_features]

                except Exception as e:
                    logging.error(f"Error preparing features for {symbol}: {str(e)}")
                    continue

                features_to_scale = latest_data_with_features.tail(seq_len)
                if len(features_to_scale) < seq_len:
                    logging.error(
                        f"Insufficient data after filtering ({len(features_to_scale)}) for LSTM sequence ({seq_len}) in {symbol}.")
                    continue

                features_scaled = scaler.transform(features_to_scale.values)
                features_scaled = features_scaled.reshape(1, seq_len, -1)
                logging.debug(f"Prediction data shape for {symbol}: {features_scaled.shape}")

                try:
                    features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
                    predictions_mean, predictions_std = mc_dropout_predict(
                        model,
                        features_tensor,
                        num_samples=100,
                        device='cuda' if torch.cuda.is_available() else 'cpu'
                    )

                    if predictions_mean is None:
                        logging.error(f"Prediction failed for {symbol}.")
                        continue

                    model_prediction = torch.argmax(predictions_mean, dim=1)[0].item()
                    prediction_confidence = predictions_std[0].max().item()
                    logging.info(
                        f"Prediction for {symbol}: {model_prediction}, Probabilities: {predictions_mean[0].tolist()}, Uncertainty: {prediction_confidence:.4f}"
                    )

                    robust_prediction = apply_strategy(model, scaler, latest_data_with_features.tail(seq_len),
                                                       gamma=0.1, num_samples=100)
                    if robust_prediction is None:
                        logging.error(f"Strategy returned no prediction for {symbol}.")
                        continue

                    logging.info(
                        f"Robust prediction for {symbol}: {robust_prediction}, Model prediction: {model_prediction}, Uncertainty: {prediction_confidence:.4f}"
                    )

                    if robust_prediction != 0:
                        account_info = mt5.account_info()
                        if account_info is None:
                            logging.error(f"Cannot retrieve account info for {symbol}.")
                            continue
                        account_balance = account_info.balance
                        position_size = calculate_position_size(account_balance, symbol, new_data_m5)
                        if position_size > 0:
                            trade_action = 1 if robust_prediction == 1 else 0
                            execute_trade(trade_action, symbol, volume=position_size, historical_data=new_data_m5,
                                          confidence=prediction_confidence)
                            check_for_closed_positions(symbol)
                        else:
                            logging.warning(f"Position size is 0, skipping trade for {symbol}.")
                    else:
                        logging.info(f"Robust prediction for {symbol}: no action (hold)")

                except Exception as e:
                    logging.error(f"Error during prediction or trade execution for {symbol}: {str(e)}")
                    continue

            logging.info("Waiting 15 minutes before next iteration...")
            time.sleep(900)

    except KeyboardInterrupt:
        logging.info("Bot stopped by user.")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
    finally:
        mt5.shutdown()
        logging.info("Shutdown after trading completion")


if __name__ == "__main__":
    pd.set_option('future.no_silent_downcasting', True)
    main()