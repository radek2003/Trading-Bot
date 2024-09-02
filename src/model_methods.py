import os
import joblib
import logging
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE



# Wszystkie metody/funkcje dotyczące modelu
def save_model(model, folder_path, filename='best_model.pkl'):
    """Zapisuje model do pliku w określonym folderze."""
    try:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_path = os.path.join(folder_path, filename)
        joblib.dump(model, file_path)
        logging.info(f"Model zapisany jako {file_path}")
    except Exception as e:
        logging.exception("Problem z zapisywaniem modelu.")

def load_model(folder_path, filename='best_model.pkl'):
    """Ładuje model z pliku z określonego folderu."""
    try:
        file_path = os.path.join(folder_path, filename)
        model = joblib.load(file_path)
        logging.info(f"Model załadowany z {file_path}")
        return model
    except Exception as e:
        logging.exception("Problem z ładowaniem modelu.")
        return None

def train_model(data, folder_path='models', model_filename='best_model.pkl'):
    """Trenuje model klasyfikacji z użyciem RandomizedSearchCV i zapisuje najlepszy model."""
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

        # Ładowanie istniejącego modelu, jeśli dostępny
        existing_model = load_model(folder_path, model_filename)

        if existing_model:
            model = existing_model
            logging.info("Kontynuowanie treningu na podstawie istniejącego modelu.")
        else:
            model = RandomForestClassifier(random_state=42)

        param_dist = {
            'n_estimators': [50, 100, 150, 200, 250, 300, 400, 500],
            'max_depth': [None, 10, 20, 30, 40, 50, 60, 70, 80],
            'min_samples_split': [2, 5, 10, 15, 20, 25, 30],
            'min_samples_leaf': [1, 2, 4, 6, 8, 10, 12],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }

        random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=10, cv=5,
                                           scoring='accuracy', random_state=42, error_score='raise')
        random_search.fit(X_train, y_train)

        best_model = random_search.best_estimator_
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"Najlepsze parametry: {random_search.best_params_}")
        logging.info(f"Dokładność modelu: {accuracy:.2%}")

        # Zapisz najlepszy model
        save_model(best_model, folder_path, model_filename)

        return best_model, scaler

    except Exception as e:
        logging.error(f"Problem z trenowaniem modelu: {e}")
        return None, None
