import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

# Konfiguracja folderów
MODEL_FOLDER = "models"
MODEL_FILENAME = "best_model.pth"

# Definicja sieci LSTM z 3 warstwami
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Sprawdzenie wymiarów wejścia
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Dodaj wymiar batcha, jeśli brakuje

        # Inicjalizacja stanów ukrytych
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Przepuszczenie przez LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Wybór ostatniego kroku czasowego
        out = self.relu(out[:, -1, :])

        # Warstwa wyjściowa
        out = self.fc(out)
        return out

def save_model(model, scaler, folder_path, filename):
    """Zapisuje model i scaler do pliku."""
    try:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_path = os.path.join(folder_path, filename)
        torch.save({'model_state_dict': model.state_dict(), 'scaler': scaler}, file_path)
        logging.info(f"Model zapisany jako {file_path}")
    except Exception as e:
        logging.exception("Problem z zapisywaniem modelu.")

def load_model(model, folder_path, filename):
    """Ładuje model i scaler z pliku."""
    try:
        file_path = os.path.join(folder_path, filename)
        checkpoint = torch.load(file_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        scaler = checkpoint['scaler']
        model.eval()
        logging.info(f"Model załadowany z {file_path}")
        return model, scaler
    except Exception as e:
        logging.exception("Problem z ładowaniem modelu.")
        return None, None

def mc_dropout_predict(model, X, num_samples=500, device='cpu'):
    """Predykcja z Monte Carlo Dropout dla oceny niepewności."""
    model.train()  # Włącz dropout podczas predykcji
    predictions = []
    X = X.to(device)
    for _ in range(num_samples):
        with torch.no_grad():
            output = model(X)
            predictions.append(torch.softmax(output, dim=1))
    predictions = torch.stack(predictions)
    mean_pred = predictions.mean(dim=0)
    std_pred = predictions.std(dim=0)
    return mean_pred, std_pred

def train_model_with_history(data, folder_path, model_filename):
    """Trenuje model klasyfikacji z robust preprocessingiem i walidacją."""
    try:
        # Inżynieria cech zgodna z main
        if 'time' in data.columns:
            data['hour'] = pd.to_datetime(data['time']).dt.hour
            data = data.drop(columns=['time'])
        if data.empty:
            logging.error("Brak danych do trenowania modelu.")
            return None, None, None

        if 'profit' in data.columns:
            data['trade_success'] = (data['profit'] > 0).astype(int)
        if 'type' in data.columns:
            data['trade_type'] = data['type'].map({1: 1, 0: 0}).fillna(0)
            data = data.drop(columns=['type'])
        if 'symbol' in data.columns:
            data = data.drop(columns=['symbol'])

        # Zapisz kolumny treningowe
        training_columns = [col for col in data.columns if col != 'Target']
        logging.debug(f"Kolumny treningowe: {training_columns}")

        # Przygotowanie danych (wypełnianie NaN medianą)
        X = data.drop('Target', axis=1).fillna(data.median()).values
        y = data['Target'].values

        if len(X) < 3:
            logging.error("Za mało danych do trenowania modelu.")
            return None, None, None

        # Robust preprocessing
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        # Przygotowanie sekwencji dla LSTM (seq_len=30)
        seq_len = 30
        X_seq, y_seq = [], []
        for i in range(len(X_scaled) - seq_len + 1):
            X_seq.append(X_scaled[i:i + seq_len])
            y_seq.append(y[i + seq_len - 1])
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        # Podział na train/val/test (70/15/15)
        X_train, X_temp, y_train, y_temp = train_test_split(X_seq, y_seq, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Przygotowanie tensorów
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        # DataLoader
        train_data = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

        # Inicjalizacja modelu LSTM z 3 warstwami
        input_size = X_train.shape[2]  # Liczba cech na krok czasowy
        hidden_size = 256
        num_layers = 3  # Zmieniono z 2 na 3
        output_size = 2  # Binary classification
        model = LSTMModel(input_size, hidden_size, num_layers, output_size)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # Dynamiczne wagi klas
        class_counts = np.bincount(y_train)
        class_weights = torch.tensor([1.5 / class_counts[0], 2.0 / class_counts[1]], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
        scheduler = optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=0.00001,
            max_lr=0.0001,
            step_size_up=200,
            mode='triangular',
            cycle_momentum=False
        )

        # Trening
        num_epochs = 100
        best_val_loss = float('inf')
        early_stop_counter = 0
        patience = 30

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                scheduler.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)

            # Walidacja
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor.to(device))
                val_loss = criterion(val_outputs, y_val_tensor.to(device))

            logging.info(
                f"Epoka [{epoch + 1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                save_model(model, scaler, folder_path, model_filename)
            else:
                early_stop_counter += 1

            if early_stop_counter >= patience:
                logging.info("Wczesne zatrzymanie.")
                break

        # Testowanie z metrykami
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_tensor.to(device))
            _, predicted = torch.max(y_pred, 1)
            accuracy = (predicted == y_test_tensor.to(device)).sum().item() / len(y_test_tensor)
            f1 = f1_score(y_test, predicted.cpu().numpy(), average='weighted')
            cm = confusion_matrix(y_test, predicted.cpu().numpy())
            logging.info(f"Dokładność: {accuracy:.2%}, F1-score: {f1:.2f}, Confusion Matrix: {cm}")

        return model, scaler, training_columns

    except Exception as e:
        logging.error(f"Problem z trenowaniem modelu: {e}")
        return None, None, None