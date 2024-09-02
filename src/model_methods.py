import os
import joblib
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Definicja sieci neuronowej
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def save_model(model, folder_path, filename='best_model.pth'):
    """Zapisuje model do pliku w określonym folderze."""
    try:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_path = os.path.join(folder_path, filename)
        torch.save(model.state_dict(), file_path)
        logging.info(f"Model zapisany jako {file_path}")
    except Exception as e:
        logging.exception("Problem z zapisywaniem modelu.")

def load_model(model, folder_path, filename='best_model.pth'):
    """Ładuje model z pliku z określonego folderu."""
    try:
        file_path = os.path.join(folder_path, filename)
        model.load_state_dict(torch.load(file_path))
        model.eval()
        logging.info(f"Model załadowany z {file_path}")
        return model
    except Exception as e:
        logging.exception("Problem z ładowaniem modelu.")
        return None

def train_model(data, folder_path='models', model_filename='best_model.pth'):
    """Trenuje model klasyfikacji z użyciem PyTorch i zapisuje najlepszy model."""
    try:
        if data.empty:
            logging.error("Brak danych do trenowania modelu.")
            return None, None

        X = data.drop('Target', axis=1).values
        y = data['Target'].values

        if len(X) < 2:
            logging.error("Za mało danych do trenowania modelu.")
            return None, None

        # Zrównoważenie klas
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_resampled)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42)

        # Przygotowanie danych do PyTorch
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        # Inicjalizacja modelu
        input_size = X_train.shape[1]
        hidden_size = 64
        output_size = len(np.unique(y_train))
        model = SimpleNN(input_size, hidden_size, output_size)

        # Sprawdzenie dostępności GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # Optymalizator i funkcja strat
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Trening modelu
        num_epochs = 10
        batch_size = 64

        for epoch in range(num_epochs):
            model.train()
            for i in range(0, len(X_train_tensor), batch_size):
                batch_X = X_train_tensor[i:i + batch_size].to(device)
                batch_y = y_train_tensor[i:i + batch_size].to(device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            logging.info(f"Epoka [{epoch+1}/{num_epochs}], Strata: {loss.item():.4f}")

        # Testowanie modelu
        model.eval()
        with torch.no_grad():
            X_test_tensor = X_test_tensor.to(device)
            y_test_tensor = y_test_tensor.to(device)
            y_pred = model(X_test_tensor)
            _, predicted = torch.max(y_pred, 1)
            accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
            logging.info(f"Dokładność modelu: {accuracy:.2%}")

        # Zapisz model
        save_model(model, folder_path, model_filename)

        return model, scaler

    except Exception as e:
        logging.error(f"Problem z trenowaniem modelu: {e}")
        return None, None
