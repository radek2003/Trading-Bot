from database import engine, SessionLocal
from src.models import Base, Setting

def init_db_and_defaults():
    # 1. Utwórz tabele (jeśli jeszcze nie istnieją)
    Base.metadata.create_all(bind=engine)

    # 2. Dodaj domyślne ustawienia, jeśli nie ma ich w bazie
    session = SessionLocal()
    defaults = {
        "min_candles_for_patterns": "150",
        "seq_len": "30",
        # dodaj kolejne domyślne klucze i wartości tutaj
    }

    for key, value in defaults.items():
        exists = session.query(Setting).filter_by(key=key).first()
        if not exists:
            new_setting = Setting(key=key, value=value)
            session.add(new_setting)
    
    session.commit()
    session.close()
    
init_db_and_defaults()