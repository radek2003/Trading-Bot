from sqlalchemy import Column, Integer, String, DateTime, Float
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

class Setting(Base):
    __tablename__ = "settings"

    key = Column(String, primary_key=True)
    value = Column(String)

    def __repr__(self):
        return f"<Setting(key='{self.key}', value='{self.value}')>"
    
class LogEntry(Base):
    __tablename__ = "logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    level = Column(String)
    message = Column(String)
    
class Sentiment(Base):
    __tablename__ = "sentiments"
    symbol = Column(String, primary_key=True)
    value = Column(Float)