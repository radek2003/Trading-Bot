from src.database import engine, SessionLocal
from src.models import Base, Setting

from src.models import Setting, Sentiment  # upewnij się, że Sentiment jest zaimportowany

def init_db_and_defaults():
    # 1. Utwórz tabele (jeśli jeszcze nie istnieją)
    Base.metadata.create_all(bind=engine)

    session = SessionLocal()

    # 2. Domyślne ustawienia aplikacji
    default_settings = {
        "min_candles_for_patterns": "150",
        "seq_len": "30",
    }

    for key, value in default_settings.items():
        if not session.query(Setting).filter_by(key=key).first():
            session.add(Setting(key=key, value=value))

    # 3. Domyślne sentymenty dla wybranych symboli
    default_sentiments = {
        "EURUSD": 0.0,
        "USDJPY": 0.0,
        "GBPUSD": 0.0,
        "AUDUSD": 0.0,
        "USDCHF": 0.0,
    }

    for symbol, value in default_sentiments.items():
        if not session.query(Sentiment).filter_by(symbol=symbol).first():
            session.add(Sentiment(symbol=symbol, value=value))

    session.commit()
    session.close()

init_db_and_defaults()