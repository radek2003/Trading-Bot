import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ta import add_all_ta_features
from ta.utils import dropna
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.optimizers import Adam

def configure_cpu():
    # Sprawdź dostępne urządzenia CPU
    physical_devices = tf.config.list_physical_devices('CPU')
    if physical_devices:
        print(f"Znaleziono CPU: {physical_devices}")

    # Ustaw liczbę wątków (CPU) do użycia
    num_threads = 4  # Możesz dostosować w zależności od liczby rdzeni CPU
    tf.config.threading.set_intra_op_parallelism_threads(num_threads)
    tf.config.threading.set_inter_op_parallelism_threads(num_threads)

def fetch_historical_data(symbol, timeframe=mt5.TIMEFRAME_M1, bars=10000):
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        if rates is None or len(rates) == 0:
            print("Brak danych historycznych.")
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'tick_volume']]
        return df
    except Exception as e:
        print(f"Problem z pobieraniem danych historycznych: {e}")
        return pd.DataFrame()

def extract_features(data):
    try:
        if data.empty:
            print("Brak danych do ekstrakcji cech.")
            return pd.DataFrame()

        data = dropna(data)
        data = add_all_ta_features(
            data, open="open", high="high", low="low", close="close", volume="tick_volume", fillna=True)

        data['Target'] = np.where(data['close'].shift(-1) > data['close'], 1, 0)

        required_columns = ['open', 'high', 'low', 'close', 'tick_volume', 'Target']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            print(f"Brakujące kolumny: {', '.join(missing_columns)}")
            return pd.DataFrame()

        return data.drop(['tick_volume'], axis=1).dropna()
    except Exception as e:
        print(f"Problem z ekstrakcją cech: {e}")
        return pd.DataFrame()

def create_advanced_lstm_dense_model(input_shape):
    model = Sequential()

    # Warstwa Input
    model.add(Input(shape=input_shape))

    # Warstwy Bidirectional LSTM
    model.add(Bidirectional(LSTM(512, activation='tanh', return_sequences=True)))
    model.add(Bidirectional(LSTM(256, activation='tanh', return_sequences=True)))
    model.add(Bidirectional(LSTM(128, activation='tanh', return_sequences=True)))
    model.add(Bidirectional(LSTM(64, activation='tanh', return_sequences=True)))
    model.add(Bidirectional(LSTM(32, activation='tanh')))

    # Warstwy Dense
    model.add(Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Wyjście binarne

    # Kompilacja modelu z optymalizatorem Adam
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def scheduler(epoch, lr):
    if epoch % 10 == 0 and epoch != 0:
        lr = lr * 0.9
    return lr

def train_advanced_lstm_dense_model(data, num_trials=5):
    best_accuracy = 0
    best_model = None
    best_scaler = None

    try:
        if data.empty:
            print("Brak danych do trenowania modelu.")
            return None, None

        X = data.drop('Target', axis=1).values
        y = data['Target'].values

        if len(X) < 2:
            print("Za mało danych do trenowania modelu.")
            return None, None

        # Normalizacja danych
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Przygotowanie danych do LSTM (dodanie wymiaru dla sekwencji)
        X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

        # Podział danych
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        for trial in range(num_trials):
            print(f"\nTrening modelu, próba {trial + 1} z {num_trials}...")

            # Stworzenie modelu
            model = create_advanced_lstm_dense_model((X_train.shape[1], X_train.shape[2]))

            # Early Stopping i Learning Rate Scheduler
            early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
            lr_scheduler = LearningRateScheduler(scheduler)

            # Trenowanie modelu
            history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, validation_split=0.2, callbacks=[early_stopping, lr_scheduler])

            # Ocena modelu
            y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Dokładność modelu LSTM z warstwami Dense (próba {trial + 1}): {accuracy:.2%}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_scaler = scaler

        print(f"\nNajlepsza dokładność uzyskana: {best_accuracy:.2%}")

        return best_model, best_scaler

    except Exception as e:
        print(f"Problem z trenowaniem modelu: {e}")
        return None, None

def main():
    # Skonfiguruj optymalizację CPU
    configure_cpu()

    if not mt5.initialize():
        print("Inicjalizacja MT5 nie powiodła się, kod błędu:", mt5.last_error())
        return

    symbol = "EURUSD"
    data = fetch_historical_data(symbol)
    data_with_features = extract_features(data)
    model, scaler = train_advanced_lstm_dense_model(data_with_features, num_trials=5)

    if model is None or scaler is None:
        print("Błąd: Model lub skalowanie nie zostały poprawnie załadowane.")
        mt5.shutdown()
        return

    mt5.shutdown()

if __name__ == "__main__":
    main()
