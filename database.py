import logging
import sqlite3
import os
from datetime import datetime

class SQLiteHandler(logging.Handler):
    def __init__(self, db_path='logs/trading_logs.db'):
        super().__init__()
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self._create_table()

    def _create_table(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                level TEXT,
                message TEXT
            )
        ''')
        self.conn.commit()

    def emit(self, record):
        try:
            msg = self.format(record)
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO logs (timestamp, level, message) VALUES (?, ?, ?)",
                (self.formatTime(record), record.levelname, record.getMessage())
            )
            self.conn.commit()
        except Exception:
            self.handleError(record)

    def formatTime(self, record, datefmt=None):
        return datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')

    def close(self):
        self.conn.close()
        super().close()
