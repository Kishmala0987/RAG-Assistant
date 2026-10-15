"""
Step 0: shared logger use by every other step

"""

import logging
import os
from datetime import datetime

LOGS_DIRECTORY = "logs"
os.makedirs(LOGS_DIRECTORY, exist_ok=True)

# One log file per run(whenever app.py is executed it will be build) named with the date time, the run started
_run_started_at = datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
RUN_LOG_FILE = os.path.join(LOGS_DIRECTORY, f"run_{_run_started_at}.log")

logging.basicConfig(
    level=logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(RUN_LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

def get_logger(name:str) -> logging.Logger:
    return logging.getLogger(name)