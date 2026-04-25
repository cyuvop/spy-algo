"""Tests for src/run_daily.py — daily runner entrypoint."""

import sys
from contextlib import contextmanager
from datetime import date, timedelta
from unittest.mock import MagicMock, patch, call
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

TEST_CONFIG = {
    "sleeve_a": {
        "trend_sma_period": 200,
        "confirmation_days": 2,
        "structure": "put_credit_spread",
        "target_dte_min": 30,
        "target_dte_max": 45,
        "target_short_delta": 0.25,
        "spread_width": 5,
        "profit_take_pct": 0.50,
        "time_stop_dte": 21,
    },
    "sleeve_b": {
        "entry_sma_short": 20,
        "entry_sma_long": 60,
        "entry_roc_period": 63,
        "exit_confirm_days": 5,
    },
    "capital": {
        "total": 25000,
        "sleeve_a_pct": 0.70,
        "sleeve_b_pct": 0.30,
        "sleeve_a_per_trade_risk_pct": 0.01,
    },
    "alerts": {
        "daily_status_ping": True,
        "discord_webhook_env_var": "DISCORD_WEBHOOK_URL",
    },
    "runtime": {
        "max_data_staleness_days": 3,
        "log_level": "INFO",
    },
}

TODAY = date.today().strftime("%Y-%m-%d")
FRESH_DATE = date.today() - timedelta(days=1)   # 1 day ago → fresh
STALE_DATE = date.today() - timedelta(days=10)  # 10 days ago → stale


def _make_spy_df(last_date: date) -> pd.DataFrame:
    """Return a minimal SPY-shaped DataFrame with a DatetimeIndex."""
    idx = pd.DatetimeIndex([pd.Timestamp(last_date)])
    return pd.DataFrame({"close": [500.0], "open": [499.0], "high": [501.0], "low": [498.0]}, index=idx)


def _make_xlu_df(last_date: date) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(last_date)])
    return pd.DataFrame({"close": [75.0], "open": [74.5], "high": [75.5], "low": [74.0]}, index=idx)


def _make_ten_yr(last_date: date) -> pd.Series:
    idx = pd.DatetimeIndex([pd.Timestamp(last_date)])
    return pd.Series([4.5], index=idx)


SIG_A_ON = {
    "state": "ON",
    "prior_state": "OFF",
    "changed_today": True,
    "days_in_state": 1,
    "spy_close": 500.0,
    "spy_sma_200": 490.0,
    "pct_above_sma": 0.02,
    "suggested_action": "If no open position, sell 30–45 DTE ~0.25 delta SPY put credit spread.",
}

SIG_A_OFF = {
    "state": "OFF",
    "prior_state": "OFF",
    "changed_today": False,
    "days_in_state": 5,
    "spy_close": 480.0,
    "spy_sma_200": 490.0,
    "pct_above_sma": -0.02,
    "suggested_action": "Park capital in SGOV/BIL.",
}

SIG_B_OFF = {
    "state": "OFF",
    "prior_state": "OFF",
    "changed_today": False,
    "days_in_state": 3,
    "ten_yr_yield": 4.5,
    "sma_20": 4.4,
    "sma_60": 4.3,
    "three_month_roc": 0.01,
    "xlu_close": 75.0,
    "suggested_action": "Hold SGOV/BIL. No XLU position.",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_PATCHES = {
    "src.run_daily.load_env": MagicMock(),
    "src.run_daily.load_config": MagicMock(return_value=TEST_CONFIG),
    "src.run_daily.is_market_day": MagicMock(return_value=True),
    "src.run_daily.has_run_today": MagicMock(return_value=False),
    "src.run_daily.fetch_prices": MagicMock(side_effect=lambda ticker, start, **kw: (
        _make_spy_df(FRESH_DATE) if ticker == "SPY" else _make_xlu_df(FRESH_DATE)
    )),
    "src.run_daily.fetch_fred_series": MagicMock(return_value=_make_ten_yr(FRESH_DATE)),
    "src.run_daily.compute_sleeve_a": MagicMock(return_value=pd.DataFrame([SIG_A_OFF])),
    "src.run_daily.compute_sleeve_b": MagicMock(return_value=pd.DataFrame([SIG_B_OFF])),
    "src.run_daily.get_last_state": MagicMock(return_value={"state": "OFF"}),
    "src.run_daily.save_state": MagicMock(),
    "src.run_daily.mark_run": MagicMock(),
    "src.run_daily.format_discord_embed": MagicMock(return_value={"embeds": []}),
    "src.run_daily.send_discord": MagicMock(return_value=True),
    "src.run_daily.send_error_alert": MagicMock(),
}


def _patch_all(**overrides):
    """Return a dict of patches with optional overrides."""
    patches = {**BASE_PATCHES, **overrides}
    return patches


@contextmanager
def _apply_patches(patches: dict):
    """Apply multiple patches and reload the module fresh each time."""
    # Remove cached module so patches take effect cleanly
    for mod in list(sys.modules.keys()):
        if "run_daily" in mod:
            del sys.modules[mod]

    with patch.dict("os.environ", {"DISCORD_WEBHOOK_URL": "https://discord.example.com/webhook"}):
        active = []
        try:
            for target, mock_obj in patches.items():
                p = patch(target, mock_obj)
                p.start()
                active.append(p)
            yield
        finally:
            for p in reversed(active):
                p.stop()
            # Clean up module cache after test
            for mod in list(sys.modules.keys()):
                if "run_daily" in mod:
                    del sys.modules[mod]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExitConditions:
    def test_exits_0_on_non_market_day(self):
        """Should exit 0 immediately when today is not a market day."""
        patches = _patch_all(**{"src.run_daily.is_market_day": MagicMock(return_value=False)})
        with _apply_patches(patches):
            from src.run_daily import main
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_exits_0_if_already_ran_today(self):
        """Should exit 0 when run_log already has today's entry."""
        patches = _patch_all(**{"src.run_daily.has_run_today": MagicMock(return_value=True)})
        with _apply_patches(patches):
            from src.run_daily import main
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_idempotency_second_run_same_day(self):
        """Second run same day: has_run_today=True → exit 0, save_state never called."""
        save_state_mock = MagicMock()
        patches = _patch_all(**{
            "src.run_daily.has_run_today": MagicMock(return_value=True),
            "src.run_daily.save_state": save_state_mock,
        })
        with _apply_patches(patches):
            from src.run_daily import main
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
            save_state_mock.assert_not_called()


class TestStalenessCheck:
    def test_exits_1_on_stale_data(self):
        """Should exit 1 and send error alert when SPY data is 10 days stale."""
        send_error_mock = MagicMock()
        mark_run_mock = MagicMock()
        patches = _patch_all(**{
            "src.run_daily.fetch_prices": MagicMock(side_effect=lambda ticker, start, **kw: (
                _make_spy_df(STALE_DATE) if ticker == "SPY" else _make_xlu_df(STALE_DATE)
            )),
            "src.run_daily.fetch_fred_series": MagicMock(return_value=_make_ten_yr(STALE_DATE)),
            "src.run_daily.send_error_alert": send_error_mock,
            "src.run_daily.mark_run": mark_run_mock,
        })
        with _apply_patches(patches):
            from src.run_daily import main
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
            send_error_mock.assert_called_once()
            # mark_run called with success=False
            mark_run_mock.assert_called_once()
            args = mark_run_mock.call_args
            assert args[1].get("success") is False or args[0][1] is False


class TestAlertBehavior:
    def test_sends_alert_when_state_changes(self):
        """When prior state is OFF but new signal is ON, send_discord should be called."""
        send_discord_mock = MagicMock(return_value=True)
        # sig_a is ON; prior_a is OFF → changed includes "A"
        patches = _patch_all(**{
            "src.run_daily.compute_sleeve_a": MagicMock(return_value=pd.DataFrame([SIG_A_ON])),
            "src.run_daily.get_last_state": MagicMock(side_effect=lambda sleeve, **kw: {"state": "OFF"}),
            "src.run_daily.send_discord": send_discord_mock,
        })
        with _apply_patches(patches):
            from src.run_daily import main
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
            send_discord_mock.assert_called_once()

    def test_sends_alert_on_daily_ping_even_without_change(self):
        """daily_status_ping=True: send_discord called even when no state changed."""
        send_discord_mock = MagicMock(return_value=True)
        config = {**TEST_CONFIG, "alerts": {"daily_status_ping": True, "discord_webhook_env_var": "DISCORD_WEBHOOK_URL"}}
        # Both signals OFF, prior state OFF → no change
        patches = _patch_all(**{
            "src.run_daily.load_config": MagicMock(return_value=config),
            "src.run_daily.get_last_state": MagicMock(return_value={"state": "OFF"}),
            "src.run_daily.compute_sleeve_a": MagicMock(return_value=pd.DataFrame([SIG_A_OFF])),
            "src.run_daily.compute_sleeve_b": MagicMock(return_value=pd.DataFrame([SIG_B_OFF])),
            "src.run_daily.send_discord": send_discord_mock,
        })
        with _apply_patches(patches):
            from src.run_daily import main
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
            send_discord_mock.assert_called_once()

    def test_no_alert_when_no_change_and_ping_off(self):
        """daily_status_ping=False and no state change: send_discord NOT called."""
        send_discord_mock = MagicMock(return_value=True)
        config = {**TEST_CONFIG, "alerts": {"daily_status_ping": False, "discord_webhook_env_var": "DISCORD_WEBHOOK_URL"}}
        patches = _patch_all(**{
            "src.run_daily.load_config": MagicMock(return_value=config),
            "src.run_daily.get_last_state": MagicMock(return_value={"state": "OFF"}),
            "src.run_daily.compute_sleeve_a": MagicMock(return_value=pd.DataFrame([SIG_A_OFF])),
            "src.run_daily.compute_sleeve_b": MagicMock(return_value=pd.DataFrame([SIG_B_OFF])),
            "src.run_daily.send_discord": send_discord_mock,
        })
        with _apply_patches(patches):
            from src.run_daily import main
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
            send_discord_mock.assert_not_called()


class TestStateHandling:
    def test_saves_state_after_successful_run(self):
        """save_state called twice (once per sleeve) and mark_run called with success=True."""
        save_state_mock = MagicMock()
        mark_run_mock = MagicMock()
        patches = _patch_all(**{
            "src.run_daily.save_state": save_state_mock,
            "src.run_daily.mark_run": mark_run_mock,
        })
        with _apply_patches(patches):
            from src.run_daily import main
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
            assert save_state_mock.call_count == 2
            # mark_run called with success=True
            mark_run_mock.assert_called_once()
            args = mark_run_mock.call_args
            success = args[1].get("success") if args[1] else args[0][1]
            assert success is True

    def test_marks_run_failed_on_exception(self):
        """When fetch_prices raises, mark_run should be called with success=False and exit 1."""
        mark_run_mock = MagicMock()
        send_error_mock = MagicMock()
        patches = _patch_all(**{
            "src.run_daily.fetch_prices": MagicMock(side_effect=RuntimeError("network failure")),
            "src.run_daily.mark_run": mark_run_mock,
            "src.run_daily.send_error_alert": send_error_mock,
        })
        with _apply_patches(patches):
            from src.run_daily import main
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
            send_error_mock.assert_called_once()
            mark_run_mock.assert_called_once()
            args = mark_run_mock.call_args
            success = args[1].get("success") if args[1] else args[0][1]
            assert success is False

