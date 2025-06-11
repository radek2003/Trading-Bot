from src.database import SessionLocal, init_db
from src.models import Setting


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