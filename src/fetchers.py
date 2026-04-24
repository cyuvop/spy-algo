import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from .utils import get_logger, load_config

logger = get_logger(__name__)

CACHE_DIR = Path("data/cache")


def _cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{key}.parquet"


def _is_cache_fresh(path: Path, max_staleness_days: int) -> bool:
    if not path.exists():
        return False
    df = pd.read_parquet(path)
    if df.empty:
        return False
    last_date = df.index.max()
    if hasattr(last_date, "date"):
        last_date = last_date.date()
    cutoff = date.today() - timedelta(days=max_staleness_days)
    return last_date >= cutoff


def fetch_prices(
    ticker: str,
    start: str,
    end: str = None,
    max_staleness_days: int = 3,
) -> pd.DataFrame:
    cache_key = f"{ticker}_daily"
    path = _cache_path(cache_key)

    if _is_cache_fresh(path, max_staleness_days):
        logger.debug("Cache hit for %s", ticker)
        df = pd.read_parquet(path)
    else:
        logger.info("Fetching %s from yfinance start=%s", ticker, start)
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if raw.empty:
            raise ValueError(f"yfinance returned no data for {ticker}")
        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        df.index = pd.to_datetime(df.index)
        df.to_parquet(path)
        logger.info("Cached %s to %s (%d rows)", ticker, path, len(df))

    df.index = pd.to_datetime(df.index)
    if end:
        df = df[df.index <= pd.Timestamp(end)]
    return df[df.index >= pd.Timestamp(start)]


def fetch_fred_series(
    series_id: str,
    start: str,
    max_staleness_days: int = 3,
) -> pd.Series:
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "FRED_API_KEY environment variable is not set. "
            "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
        )

    cache_key = f"fred_{series_id}"
    path = _cache_path(cache_key)

    if _is_cache_fresh(path, max_staleness_days):
        logger.debug("Cache hit for FRED %s", series_id)
        df = pd.read_parquet(path)
        series = df.iloc[:, 0]
    else:
        logger.info("Fetching FRED series %s", series_id)
        from fredapi import Fred

        fred = Fred(api_key=api_key)
        series = fred.get_series(series_id, observation_start=start)
        series.name = series_id
        series.index = pd.to_datetime(series.index)
        pd.DataFrame(series).to_parquet(path)
        logger.info("Cached FRED %s to %s (%d rows)", series_id, path, len(series))

    series.index = pd.to_datetime(series.index)
    return series[series.index >= pd.Timestamp(start)]


def get_latest(ticker_or_series: str, is_fred: bool = False) -> dict:
    start = (date.today() - timedelta(days=400)).strftime("%Y-%m-%d")
    if is_fred:
        s = fetch_fred_series(ticker_or_series, start=start)
        s = s.dropna()
        return {"series_id": ticker_or_series, "date": str(s.index[-1].date()), "value": float(s.iloc[-1])}
    else:
        df = fetch_prices(ticker_or_series, start=start)
        last = df.iloc[-1]
        return {
            "ticker": ticker_or_series,
            "date": str(df.index[-1].date()),
            "close": float(last["close"]),
        }
