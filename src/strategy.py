import logging
from src.feature_extraction import calculate_moving_averages, calculate_macd

# Strategie handlowe

def apply_strategy(model, scaler, new_data, strategy_type='MACD'):
    """Stosuje wybraną strategię na nowych danych."""
    try:
        if strategy_type == 'MACD':
            new_data_with_features = calculate_macd(new_data)
            new_data_with_features = calculate_moving_averages(new_data_with_features)
        else:
            logging.error(f"Nieznana strategia: {strategy_type}")
            return

        if new_data_with_features.empty:
            logging.error("Brak danych z cechami do prognozowania.")
            return

        X_new = scaler.transform(new_data_with_features.drop('Target', axis=1).values)
        predictions = model.predict(X_new)

        return predictions

    except Exception as e:
        logging.exception("Problem z zastosowaniem strategii.")
print(".")