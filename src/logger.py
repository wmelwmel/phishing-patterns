import datetime
from pathlib import Path

from loguru import logger

from src.paths import logs_dir


def setup_logger() -> Path:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"log_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    logger.add(log_file, level="INFO")
    return log_file
