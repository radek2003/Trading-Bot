import logging
from src.settings_manager import get_setting

# Parametry. logi i konfiguracja CPU

MAX_RISK_PER_TRADE = float(get_setting("MAX_RISK_PER_TRADE", 0.001)) #0.001  # 1% kapitału na transakcję
RISK_REWARD_RATIO = 2.0    # Stosunek zysku do ryzyka

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("./logs/trading_bot.log"),
        logging.StreamHandler()
    ]
)


