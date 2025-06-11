from sqlalchemy import Column, Integer, String, DateTime
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

    id = Column(String, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    level = Column(String)
    message = Column(String)