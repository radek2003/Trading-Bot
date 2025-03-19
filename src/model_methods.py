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

MODEL_FOLDER = "models"
MODEL_FILENAME = "best_model.pth"


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.5 if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.relu(out[:, -1, :])
        out = self.fc(out)
        return out


def save_model(model, scaler, folder_path, filename, feature_names):
    try:
        os.makedirs(folder_path, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler': scaler,
            'input_size': model.input_size,
            'feature_names': feature_names  # Zapisz cechy osobno
        }, os.path.join(folder_path, filename))
        logging.info(f"Model zapisany: {os.path.join(folder_path, filename)}")
    except Exception as e:
        logging.exception("Błąd zapisu modelu")


def load_model(folder_path, filename, hidden_size=256, num_layers=3):
    try:
        checkpoint = torch.load(os.path.join(folder_path, filename))
        model = LSTMModel(
            input_size=checkpoint['input_size'],
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=2
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        scaler = checkpoint['scaler']
        scaler.feature_names = checkpoint['feature_names']  # Wczytaj cechy

        return model, scaler
    except Exception as e:
        logging.exception("Błąd ładowania modelu")
        return None, None


def mc_dropout_predict(model, X, num_samples=500, device='cpu'):
    model.train()
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
    try:
        if 'sentiment' not in data.columns:
            data['sentiment'] = 0.0
            logging.warning("Dodano domyślny sentyment (0.0)")

        if 'time' in data.columns:
            data['hour'] = pd.to_datetime(data['time']).dt.hour
            data = data.drop(columns=['time'])

        data = data.select_dtypes(include=[np.number])

        if data.empty:
            logging.error("Brak danych do trenowania modelu.")
            return None, None, None

        X = data.drop('Target', axis=1).fillna(data.median()).values
        y = data['Target'].values

        if len(X) < 3:
            logging.error("Za mało danych do trenowania modelu.")
            return None, None, None

        num_features = X.shape[1]
        logging.info(f"Liczba cech: {num_features}")

        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        feature_names = data.drop('Target', axis=1).columns.tolist()
        scaler.feature_names = feature_names

        seq_len = 30
        X_seq, y_seq = [], []
        for i in range(len(X_scaled) - seq_len + 1):
            X_seq.append(X_scaled[i:i + seq_len])
            y_seq.append(y[i + seq_len - 1])

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        X_train, X_temp, y_train, y_temp = train_test_split(X_seq, y_seq, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        train_data = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

        model = LSTMModel(
            input_size=num_features,
            hidden_size=256,
            num_layers=3,
            output_size=2
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        class_counts = np.bincount(y_train)
        class_weights = torch.tensor([
            1.5 / class_counts[0],
            2.0 / class_counts[1]
        ], dtype=torch.float32).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
        scheduler = optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=0.00001,
            max_lr=0.0001,
            step_size_up=200,
            cycle_momentum=False
        )

        best_val_loss = float('inf')
        early_stop_counter = 0
        patience = 30

        for epoch in range(100):
            model.train()
            total_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor.to(device))
                val_loss = criterion(val_outputs, y_val_tensor.to(device))

            logging.info(f"Epoka {epoch + 1} | Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                save_model(model, scaler, folder_path, model_filename, feature_names)
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    logging.info("Early stopping")
                    break

        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_tensor.to(device))
            _, predicted = torch.max(y_pred, 1)
            accuracy = (predicted == y_test_tensor.to(device)).sum().item() / len(y_test_tensor)
            f1 = f1_score(y_test, predicted.cpu().numpy(), average='weighted')
            cm = confusion_matrix(y_test, predicted.cpu().numpy())
            logging.info(f"Dokładność: {accuracy:.2%}, F1-score: {f1:.2f}, Macierz pomyłek:\n{cm}")

        return model, scaler, feature_names

    except Exception as e:
        logging.error(f"Błąd treningu: {str(e)}")
        return None, None, None