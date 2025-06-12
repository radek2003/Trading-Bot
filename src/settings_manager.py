from src.database import SessionLocal, init_db
from src.models import Setting, Sentiment

SENTIMENT_MAPPING = {
    "weak": -0.5,
    "average": 0.0,
    "high": 0.5
}


def get_setting(key: str, default: str = None) -> str:
    session = SessionLocal()
    setting = session.query(Setting).filter_by(key=key).first()
    session.close()
    return setting.value if setting else default

def set_setting(key: str, value: str):
    session = SessionLocal()
    setting = session.query(Setting).filter_by(key=key).first()
    if setting:
        setting.value = value
    else:
        setting = Setting(key=key, value=value)
        session.add(setting)
    session.commit()
    session.close()
    

def get_sentiment(symbol: str) -> float:
    session = SessionLocal()
    sentiment = session.query(Sentiment).filter_by(symbol=symbol).first()
    session.close()
    return sentiment.value if sentiment else None

def set_sentiment(symbol: str, value: float):
    session = SessionLocal()
    sentiment = session.query(Sentiment).filter_by(symbol=symbol).first()
    if sentiment:
        sentiment.value = value
    else:
        sentiment = Sentiment(symbol=symbol, value=value)
        session.add(sentiment)
    session.commit()
    session.close()