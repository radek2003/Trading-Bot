import logging
import sqlite3
import os
from datetime import datetime

class SQLiteHandler(logging.Handler):
    def __init__(self, db_path='logs/trading_logs.db'):
        super().__init__()
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._create_table()

    def _create_table(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    level TEXT,
                    message TEXT
                )
            ''')
            conn.commit()

    def emit(self, record):
        try:
            msg = self.format(record)
            log_time = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO logs (timestamp, level, message) VALUES (?, ?, ?)",
                    (log_time, record.levelname, record.getMessage())
                )
                conn.commit()
        except Exception:
            self.handleError(record)

def read_logs_from_db(path, limit=200):
    if not os.path.exists(path):
        return ["Brak bazy danych logów..."]
    
    try:
        conn = sqlite3.connect(path)
        cursor = conn.cursor()
        cursor.execute("SELECT timestamp, level, message FROM logs ORDER BY id DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        conn.close()
        
        # Formatowanie logów
        formatted = [f"{row[0]} [{row[1]}] {row[2]}" for row in rows]
        return formatted
    except Exception as e:
        return [f"Błąd podczas odczytu logów: {e}"]
