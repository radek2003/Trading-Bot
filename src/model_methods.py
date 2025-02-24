import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

# Konfiguracja folderów
MODEL_FOLDER = "models"  # Folder do zapisywania modeli
MODEL_FILENAME = "best_model.pth"

# Definicja sieci neuronowej z sześcioma warstwami ukrytymi i Batch Normalization
class AdvancedNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, hidden_size5, hidden_size6,
                 output_size):
        super(AdvancedNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)

        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)

        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.bn3 = nn.BatchNorm1d(hidden_size3)

        self.fc4 = nn.Linear(hidden_size3, hidden_size4)
        self.bn4 = nn.BatchNorm1d(hidden_size4)

        self.fc5 = nn.Linear(hidden_size4, hidden_size5)
        self.bn5 = nn.BatchNorm1d(hidden_size5)

        self.fc6 = nn.Linear(hidden_size5, hidden_size6)
        self.bn6 = nn.BatchNorm1d(hidden_size6)

        self.fc7 = nn.Linear(hidden_size6, output_size)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.35)
        self.dropout2 = nn.Dropout(p=0.5)

    def forward(self, x):
        if x.size(0) > 1:  # Jeśli batch ma więcej niż 1 próbkę, użyj BatchNorm
            x = self.relu(self.bn1(self.fc1(x)))
        else:  # Dla batcha z jedną próbką pomiń BatchNorm
            x = self.relu(self.fc1(x))
        x = self.dropout1(x)

        if x.size(0) > 1:
            x = self.relu(self.bn2(self.fc2(x)))
        else:
            x = self.relu(self.fc2(x))
        x = self.dropout2(x)

        if x.size(0) > 1:
            x = self.relu(self.bn3(self.fc3(x)))
        else:
            x = self.relu(self.fc3(x))

        if x.size(0) > 1:
            x = self.relu(self.bn4(self.fc4(x)))
        else:
            x = self.relu(self.fc4(x))
        x = self.dropout2(x)

        if x.size(0) > 1:
            x = self.relu(self.bn5(self.fc5(x)))
        else:
            x = self.relu(self.fc5(x))

        if x.size(0) > 1:
            x = self.relu(self.bn6(self.fc6(x)))
        else:
            x = self.relu(self.fc6(x))

        x = self.fc7(x)
        return x

def save_model(model, scaler, folder_path, filename):
    """Zapisuje model i scaler do pliku w określonym folderze."""
    try:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_path = os.path.join(folder_path, filename)
        torch.save({'model_state_dict': model.state_dict(), 'scaler': scaler}, file_path)
        logging.info(f"Model zapisany jako {file_path}")
    except Exception as e:
        logging.exception("Problem z zapisywaniem modelu.")

def load_model(model, folder_path, filename):
    """Ładuje model i scaler z pliku z określonego folderu."""
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

def train_model_with_history(data, folder_path, model_filename):
    """Trenuje model klasyfikacji z uwzględnieniem cech historii transakcji."""
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
            data['trade_type'] = data['type'].map({1: 1, 0: 0}).fillna(0)  # Zgodne z main
            data = data.drop(columns=['type'])
        if 'symbol' in data.columns:
            data = data.drop(columns=['symbol'])

        # Zapisz kolumny treningowe przed usunięciem 'Target'
        training_columns = [col for col in data.columns if col != 'Target']
        logging.debug(f"Kolumny treningowe w train_model_with_history: {training_columns}")

        # Przygotowanie danych do trenowania
        X = data.drop('Target', axis=1).values
        y = data['Target'].values

        if len(X) < 2:
            logging.error("Za mało danych do trenowania modelu.")
            return None, None, None

        # Zrównoważenie klas przy użyciu SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_resampled)

        # Walidacja danych
        assert not np.isnan(X_scaled).any(), "Dane zawierają NaN!"
        assert not np.isinf(X_scaled).any(), "Dane zawierają nieskończoności!"

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42)

        # Przygotowanie danych do PyTorch
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        # Stworzenie DataLoader
        train_data = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_data, batch_size=max(2, 128), shuffle=True)

        # Inicjalizacja modelu
        input_size = X_train.shape[1]
        hidden_sizes = [1024, 512, 256, 128, 64, 32]
        output_size = len(np.unique(y_train))
        model = AdvancedNN(input_size, *hidden_sizes, output_size)

        # Sprawdzenie dostępności GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # Wagi klas
        class_weights = torch.tensor([1.0, 2.0]).to(device)

        # Optymalizator i funkcja strat z wagami
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0005)

        # Trening modelu
        num_epochs = 70
        best_loss = float('inf')
        early_stop_counter = 0
        patience = 10

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_X)

                if batch_X.size(0) > 1:  # Sprawdzenie, czy batch ma więcej niż 1 próbkę
                    loss = criterion(outputs, batch_y)
                else:
                    logging.warning("Pominięto batch o wielkości 1 z powodu ograniczeń BatchNorm1d.")
                    continue

                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            logging.info(f"Epoka [{epoch + 1}/{num_epochs}], Strata: {avg_loss:.4f}")

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                early_stop_counter = 0
                save_model(model, scaler, folder_path, model_filename)
            else:
                early_stop_counter += 1

            if early_stop_counter >= patience:
                logging.info("Wczesne zatrzymanie - brak poprawy strat walidacyjnych.")
                break

        # Testowanie modelu
        model.eval()
        with torch.no_grad():
            X_test_tensor = X_test_tensor.to(device)
            y_test_tensor = y_test_tensor.to(device)
            y_pred = model(X_test_tensor)
            _, predicted = torch.max(y_pred, 1)
            accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
            logging.info(f"Dokładność modelu: {accuracy:.2%}")

        return model, scaler, training_columns  # Zwracamy także kolumny treningowe

    except Exception as e:
        logging.error(f"Problem z trenowaniem modelu: {e}")
        return None, None, None
