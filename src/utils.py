import logging
import logging.handlers
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_logger(name: str) -> logging.Logger:
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level = logging.INFO
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    fh = logging.handlers.RotatingFileHandler(
        log_dir / "app.log", maxBytes=5_000_000, backupCount=3
    )
    fh.setFormatter(fmt)

    logger.setLevel(level)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


_us_bd = None


def _get_us_bd() -> CustomBusinessDay:
    global _us_bd
    if _us_bd is None:
        cal = USFederalHolidayCalendar()
        _us_bd = CustomBusinessDay(calendar=cal)
    return _us_bd


def is_market_day(date) -> bool:
    """Return True if date is a US equity trading day (excludes weekends and US federal holidays)."""
    import pandas as pd

    d = pd.Timestamp(date)
    # Check if adding 0 business days keeps us on the same day
    bd = _get_us_bd()
    return d == d + 0 * bd


def load_env(dotenv_path: str = ".env") -> None:
    load_dotenv(dotenv_path=dotenv_path, override=False)
