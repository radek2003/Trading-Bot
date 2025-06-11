import os
import logging
from datetime import datetime
from src.models import LogEntry, Setting  # Importuj model LogEntry z pliku models.py
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

# Ścieżka do bazy danych
DB_PATH = "logs/trading_logs.db"
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# SQLAlchemy setup
Base = declarative_base()
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)


# Inicjalizacja bazy danych
def init_db():
    Base.metadata.create_all(bind=engine)

# Logger SQLAlchemy
class SQLAlchemyHandler(logging.Handler):
    def __init__(self):
        super().__init__()

    def emit(self, record):
        try:
            session = SessionLocal()
            log_time = datetime.fromtimestamp(record.created)
            log_entry = LogEntry(
                timestamp=log_time,
                level=record.levelname,
                message=record.getMessage()
            )
            session.add(log_entry)
            session.commit()
        except Exception:
            self.handleError(record)
        finally:
            session.close()

# Funkcja do odczytu logów
def read_logs_from_db(limit=200):
    try:
        session = SessionLocal()
        logs = session.query(LogEntry).order_by(LogEntry.id.desc()).limit(limit).all()
        formatted = [f"{log.timestamp.strftime('%Y-%m-%d %H:%M:%S')} [{log.level}] {log.message}" for log in logs]
        return formatted
    except Exception as e:
        return [f"Błąd podczas odczytu logów: {e}"]
    finally:
        session.close()

# Inicjalizacja i konfiguracja loggera (możesz wywołać to w main.py)
def setup_logger():
    init_db()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(SQLAlchemyHandler())
