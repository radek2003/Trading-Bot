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

# Zaktualizowana definicja sieci neuronowej z 6 warstwami ukrytymi
class AdvancedNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, hidden_size5, hidden_size6, output_size):
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
        self.dropout = nn.Dropout(p=0.5)  # Ujednolicony dropout

    def forward(self, x):
        if x.size(0) > 1:
            x = self.relu(self.bn1(self.fc1(x)))
        else:
            x = self.relu(self.fc1(x))
        x = self.dropout(x)
        if x.size(0) > 1:
            x = self.relu(self.bn2(self.fc2(x)))
        else:
            x = self.relu(self.fc2(x))
        x = self.dropout(x)
        if x.size(0) > 1:
            x = self.relu(self.bn3(self.fc3(x)))
        else:
            x = self.relu(self.fc3(x))
        x = self.dropout(x)
        if x.size(0) > 1:
            x = self.relu(self.bn4(self.fc4(x)))
        else:
            x = self.relu(self.fc4(x))
        x = self.dropout(x)
        if x.size(0) > 1:
            x = self.relu(self.bn5(self.fc5(x)))
        else:
            x = self.relu(self.fc5(x))
        x = self.dropout(x)
        if x.size(0) > 1:
            x = self.relu(self.bn6(self.fc6(x)))
        else:
            x = self.relu(self.fc6(x))
        x = self.fc7(x)
        return x

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

        # Podział na train/val/test (70/15/15)
        X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
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
        train_loader = DataLoader(train_data, batch_size=128, shuffle=True)  # Zwiększony batch size

        # Inicjalizacja modelu
        input_size = X_train.shape[1]
        hidden_sizes = [2048, 1024, 512, 256, 128]  # 5 warstw ukrytych
        output_size = 2  # Binary classification
        model = AdvancedNN(input_size, *hidden_sizes, hidden_sizes[-1], output_size)  # Poprawione wywołanie z output_size

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # Dynamiczne wagi klas
        class_counts = np.bincount(y_train)
        class_weights = torch.tensor([1.5 / class_counts[0], 2.0 / class_counts[1]], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)  # Zwiększony learning rate
        scheduler = optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=0.00001,  # Minimalna wartość learning rate
            max_lr=0.0001,    # Maksymalna wartość learning rate
            step_size_up=200,  # Liczba iteracji na cykl w górę (dostosowana wartość)
            mode='triangular',  # Prosty cykl góra-dół
            cycle_momentum=False  # Wyłącz momentum dla Adam
        )

        # Trening
        num_epochs = 100  # Zwiększona liczba epok
        best_val_loss = float('inf')
        early_stop_counter = 0
        patience = 30  # Zwiększona cierpliwość

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
                scheduler.step()  # Aktualizacja lr po każdym batchu
                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)

            # Walidacja
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor.to(device))
                val_loss = criterion(val_outputs, y_val_tensor.to(device))

            logging.info(f"Epoka [{epoch + 1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

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
