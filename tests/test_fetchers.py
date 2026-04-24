import os
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.fetchers import fetch_prices, fetch_fred_series, get_latest


def _make_price_df(n=5):
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    return pd.DataFrame(
        {
            "open": [100.0] * n,
            "high": [105.0] * n,
            "low": [99.0] * n,
            "close": [102.0] * n,
            "volume": [1_000_000] * n,
        },
        index=idx,
    )


def _make_fred_series(n=5):
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    s = pd.Series([4.0 + i * 0.1 for i in range(n)], index=idx, name="DGS10")
    return s


# ── fetch_prices ──────────────────────────────────────────────────────────────

class TestFetchPrices:
    def test_returns_dataframe_with_expected_columns(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        expected = _make_price_df()

        with patch("src.fetchers.yf.download") as mock_dl:
            raw = expected.copy()
            raw.columns = ["Open", "High", "Low", "Close", "Volume"]
            mock_dl.return_value = raw
            df = fetch_prices("SPY", start="2024-01-01")

        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert len(df) == 5

    def test_caches_to_parquet_on_first_call(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        raw = _make_price_df()
        raw.columns = ["Open", "High", "Low", "Close", "Volume"]

        with patch("src.fetchers.yf.download", return_value=raw):
            fetch_prices("SPY", start="2024-01-01")

        assert (tmp_path / "data" / "cache" / "SPY_daily.parquet").exists()

    def test_uses_cache_on_second_call(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        raw = _make_price_df()
        raw.columns = ["Open", "High", "Low", "Close", "Volume"]

        with patch("src.fetchers.yf.download", return_value=raw) as mock_dl:
            fetch_prices("SPY", start="2024-01-01")
            # Patch cache freshness to True so second call uses cache
            with patch("src.fetchers._is_cache_fresh", return_value=True):
                fetch_prices("SPY", start="2024-01-01")
            assert mock_dl.call_count == 1  # only called once

    def test_raises_on_empty_response(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with patch("src.fetchers.yf.download", return_value=pd.DataFrame()):
            with pytest.raises(ValueError, match="no data"):
                fetch_prices("FAKE", start="2024-01-01")

    def test_filters_by_start_date(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        raw = _make_price_df(10)
        raw.columns = ["Open", "High", "Low", "Close", "Volume"]

        with patch("src.fetchers.yf.download", return_value=raw):
            df = fetch_prices("SPY", start="2024-01-05")

        assert df.index.min() >= pd.Timestamp("2024-01-05")


# ── fetch_fred_series ─────────────────────────────────────────────────────────

class TestFetchFredSeries:
    def test_raises_if_api_key_missing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="FRED_API_KEY"):
            fetch_fred_series("DGS10", start="2024-01-01")

    def test_returns_series_with_correct_name(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("FRED_API_KEY", "test-key")
        mock_fred = MagicMock()
        mock_fred.get_series.return_value = _make_fred_series()

        with patch("fredapi.Fred", return_value=mock_fred):
            result = fetch_fred_series("DGS10", start="2024-01-01")

        assert result.name == "DGS10"
        assert len(result) == 5

    def test_caches_fred_to_parquet(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("FRED_API_KEY", "test-key")
        mock_fred = MagicMock()
        mock_fred.get_series.return_value = _make_fred_series()

        with patch("fredapi.Fred", return_value=mock_fred):
            fetch_fred_series("DGS10", start="2024-01-01")

        assert (tmp_path / "data" / "cache" / "fred_DGS10.parquet").exists()

    def test_filters_by_start_date(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("FRED_API_KEY", "test-key")
        mock_fred = MagicMock()
        mock_fred.get_series.return_value = _make_fred_series(10)

        with patch("fredapi.Fred", return_value=mock_fred):
            result = fetch_fred_series("DGS10", start="2024-01-05")

        assert result.index.min() >= pd.Timestamp("2024-01-05")


# ── get_latest ────────────────────────────────────────────────────────────────

class TestGetLatest:
    def test_get_latest_price(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        df = _make_price_df()

        with patch("src.fetchers.fetch_prices", return_value=df):
            result = get_latest("SPY")

        assert result["ticker"] == "SPY"
        assert "close" in result
        assert "date" in result

    def test_get_latest_fred(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        s = _make_fred_series()

        with patch("src.fetchers.fetch_fred_series", return_value=s):
            result = get_latest("DGS10", is_fred=True)

        assert result["series_id"] == "DGS10"
        assert "value" in result
        assert "date" in result
