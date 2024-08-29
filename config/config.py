import logging
import tensorflow as tf

# Parametry. logi i konfiguracja CPU

MAX_RISK_PER_TRADE = 0.01  # 1% kapitału na transakcję
RISK_REWARD_RATIO = 2.0    # Stosunek zysku do ryzyka

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("./logs/trading_bot.log"),
        logging.StreamHandler()
    ]
)

def configure_cpu():
    physical_devices = tf.config.list_physical_devices('CPU')
    if physical_devices:
        logging.info(f"Znaleziono CPU: {physical_devices}")

    num_threads = 6
    tf.config.threading.set_intra_op_parallelism_threads(num_threads)
    tf.config.threading.set_inter_op_parallelism_threads(num_threads)
