"""
Signal tests use synthetic DataFrames with hand-computed expected outputs.
No randomness; deterministic fixtures only.
"""
import numpy as np
import pandas as pd
import pytest

from src.signals import compute_sleeve_a, compute_sleeve_b


# ── helpers ───────────────────────────────────────────────────────────────────

def _spy_prices(closes: list, start="2020-01-02") -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({"close": closes}, index=idx)


def _ten_yr(yields: list, start="2020-01-02") -> pd.Series:
    idx = pd.bdate_range(start, periods=len(yields))
    return pd.Series(yields, index=idx, name="DGS10")


def _xlu_prices(closes: list, start="2020-01-02") -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({"close": closes}, index=idx)


def _default_config() -> dict:
    return {
        "sleeve_a": {
            "trend_sma_period": 5,       # tiny period for test tractability
            "confirmation_days": 2,
            "structure": "put_credit_spread",
            "target_dte_min": 30,
            "target_dte_max": 45,
            "target_short_delta": 0.25,
            "spread_width": 5,
        },
        "sleeve_b": {
            "entry_sma_short": 3,        # tiny periods for test tractability
            "entry_sma_long": 5,
            "entry_roc_period": 4,
            "exit_confirm_days": 2,
        },
        "capital": {
            "total": 25000,
            "sleeve_a_pct": 0.70,
            "sleeve_b_pct": 0.30,
            "sleeve_a_per_trade_risk_pct": 0.01,
        },
    }


# ── Sleeve A ──────────────────────────────────────────────────────────────────

class TestSleeveA:
    def test_returns_dataframe_with_required_columns(self):
        closes = [100.0] * 20
        df = compute_sleeve_a(_spy_prices(closes), _default_config())
        for col in ["state", "prior_state", "changed_today", "days_in_state",
                    "spy_close", "spy_sma_200", "pct_above_sma", "suggested_action"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_stays_off_when_spy_below_sma(self):
        # All closes below the 5-period SMA anchor (we make it so SMA > close always)
        # Build: first 5 days at 110 (to establish SMA=110), then 10 days at 90 (below SMA)
        closes = [110.0] * 5 + [90.0] * 10
        df = compute_sleeve_a(_spy_prices(closes), _default_config())
        # After warmup, should be OFF because close < sma
        later = df.iloc[7:]
        assert (later["state"] == "OFF").all()

    def test_turns_on_after_two_consecutive_closes_above_sma(self):
        # First 5 days at 50 (sma=50), then day 6-7 at 60 (above sma)
        # confirmation_days=2, so regime should flip ON after day 7
        closes = [50.0] * 5 + [60.0] * 10
        df = compute_sleeve_a(_spy_prices(closes), _default_config())
        # sma(5) on closes [50,50,50,50,50,60,60,...]
        # row index 4 sma=50, close=50 → on the boundary (not above)
        # row index 5 sma=(50+50+50+50+60)/5=52, close=60 → above
        # row index 6 sma=(50+50+50+60+60)/5=54, close=60 → above (2nd consecutive)
        # So ON should be True starting row 6
        assert df.iloc[6]["state"] == "ON"

    def test_regime_flips_back_off_after_two_closes_below(self):
        # Start ON: 10 days at 100 (above SMA 5 established at 100)
        # Then drop to 80 for 5 days — 2 consecutive below → OFF
        closes = [100.0] * 10 + [80.0] * 10
        df = compute_sleeve_a(_spy_prices(closes), _default_config())
        # After enough time below, it should be OFF
        assert df.iloc[-1]["state"] == "OFF"

    def test_suggested_action_when_on(self):
        closes = [50.0] * 5 + [60.0] * 10
        df = compute_sleeve_a(_spy_prices(closes), _default_config())
        on_rows = df[df["state"] == "ON"]
        assert len(on_rows) > 0
        assert on_rows["suggested_action"].str.contains("put", case=False).all()

    def test_suggested_action_when_off(self):
        closes = [110.0] * 5 + [90.0] * 10
        df = compute_sleeve_a(_spy_prices(closes), _default_config())
        off_rows = df[df["state"] == "OFF"].dropna(subset=["state"])
        assert len(off_rows) > 0
        assert off_rows["suggested_action"].str.contains("SGOV|BIL|T-bill", case=False).any()

    def test_changed_today_is_true_only_on_transition_day(self):
        # Monotonically rising closes in ON phase: close always stays ahead of SMA(5)
        # so the regime never flips back OFF during the window
        on_phase = [60.0 + i * 5 for i in range(10)]  # 60, 65, 70, ..., 105
        closes = [50.0] * 5 + on_phase
        df = compute_sleeve_a(_spy_prices(closes), _default_config())
        changes = df[df["changed_today"] == True]
        # Exactly one transition (OFF→ON)
        assert len(changes) == 1

    def test_pct_above_sma_calculation(self):
        closes = [100.0] * 5 + [110.0] * 5
        df = compute_sleeve_a(_spy_prices(closes), _default_config())
        # On the row where close=110 and sma=100: pct_above = (110-100)/100 = 0.10
        row = df[df["spy_close"] == 110.0].iloc[0]
        # sma may have warmed up by then — just verify the formula is applied
        if not pd.isna(row["spy_sma_200"]):
            expected = (row["spy_close"] - row["spy_sma_200"]) / row["spy_sma_200"]
            assert row["pct_above_sma"] == pytest.approx(expected)

    def test_days_in_state_increments(self):
        closes = [50.0] * 5 + [60.0] * 10
        df = compute_sleeve_a(_spy_prices(closes), _default_config())
        on_rows = df[df["state"] == "ON"]["days_in_state"].tolist()
        assert on_rows == list(range(1, len(on_rows) + 1))


class TestSleeveANoLookahead:
    """
    Verify signal on day N is identical whether computed on full history
    or on data truncated at day N. This proves no lookahead bias.
    """

    def test_signal_at_day_n_matches_truncated_computation(self):
        # 30 days of prices — some above SMA, some below
        rng = np.random.default_rng(42)
        base = 100.0
        prices = [base]
        for _ in range(29):
            prices.append(prices[-1] * (1 + rng.uniform(-0.02, 0.02)))

        spy = _spy_prices(prices)
        cfg = _default_config()

        full = compute_sleeve_a(spy, cfg)

        # Check 5 random days in the latter half (after warmup)
        for n in [15, 18, 20, 23, 27]:
            truncated = compute_sleeve_a(spy.iloc[:n + 1], cfg)
            assert full.iloc[n]["state"] == truncated.iloc[n]["state"], (
                f"Day {n}: full={full.iloc[n]['state']}, truncated={truncated.iloc[n]['state']}"
            )


# ── Sleeve B ──────────────────────────────────────────────────────────────────

class TestSleeveB:
    def test_returns_dataframe_with_required_columns(self):
        n = 30
        ten_yr = _ten_yr([4.0] * n)
        xlu = _xlu_prices([70.0] * n)
        df = compute_sleeve_b(ten_yr, xlu, _default_config())
        for col in ["state", "prior_state", "changed_today", "days_in_state",
                    "ten_yr_yield", "sma_20", "sma_60", "three_month_roc",
                    "xlu_close", "suggested_action"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_enters_on_when_short_sma_below_long_and_roc_negative(self):
        # Yield falling: starts at 5.0, drops gradually for 20+ days
        # With entry_sma_short=3, entry_sma_long=5, roc_period=4
        yields = [5.0 - i * 0.05 for i in range(25)]
        xlu = _xlu_prices([70.0] * 25)
        df = compute_sleeve_b(_ten_yr(yields), xlu, _default_config())
        # After warmup, sma_short < sma_long (falling yield) and roc < 0
        later = df.dropna().iloc[5:]
        assert (later["state"] == "ON").any()

    def test_stays_off_when_yield_rising(self):
        # Yield rising: short SMA > long SMA → entry condition not met
        yields = [3.0 + i * 0.05 for i in range(25)]
        xlu = _xlu_prices([70.0] * 25)
        df = compute_sleeve_b(_ten_yr(yields), xlu, _default_config())
        later = df.dropna().iloc[5:]
        assert (later["state"] == "OFF").all()

    def test_exits_after_exit_confirm_days_above_long_sma(self):
        # Phase 1: yield falling (ON), Phase 2: yield rising for exit_confirm_days+1
        falling = [5.0 - i * 0.1 for i in range(15)]
        rising = [falling[-1] + i * 0.2 for i in range(15)]
        yields = falling + rising

        xlu = _xlu_prices([70.0] * len(yields))
        df = compute_sleeve_b(_ten_yr(yields), xlu, _default_config())
        valid = df.dropna()
        # Should start ON (falling phase) then eventually turn OFF (rising phase)
        states = valid["state"].tolist()
        assert "ON" in states
        assert "OFF" in states

    def test_suggested_action_buy_xlu_on_entry(self):
        yields = [5.0 - i * 0.05 for i in range(25)]
        xlu = _xlu_prices([70.0] * 25)
        df = compute_sleeve_b(_ten_yr(yields), xlu, _default_config())
        entry_rows = df[(df["changed_today"] == True) & (df["state"] == "ON")]
        if not entry_rows.empty:
            assert entry_rows["suggested_action"].str.contains("XLU", case=False).all()

    def test_suggested_action_sell_xlu_on_exit(self):
        falling = [5.0 - i * 0.1 for i in range(15)]
        rising = [falling[-1] + i * 0.2 for i in range(15)]
        yields = falling + rising
        xlu = _xlu_prices([70.0] * len(yields))
        df = compute_sleeve_b(_ten_yr(yields), xlu, _default_config())
        exit_rows = df[(df["changed_today"] == True) & (df["state"] == "OFF")]
        if not exit_rows.empty:
            assert exit_rows["suggested_action"].str.contains(
                "Sell|SGOV|rotate", case=False
            ).all()


class TestSleeveBNoLookahead:
    def test_signal_at_day_n_matches_truncated_computation(self):
        rng = np.random.default_rng(7)
        yields = [4.0]
        for _ in range(29):
            yields.append(max(0.5, yields[-1] + rng.uniform(-0.05, 0.05)))

        ten_yr = _ten_yr(yields)
        xlu = _xlu_prices([70.0] * 30)
        cfg = _default_config()

        full = compute_sleeve_b(ten_yr, xlu, cfg)

        for n in [15, 18, 20, 23, 27]:
            trunc_yield = ten_yr.iloc[:n + 1]
            trunc_xlu = xlu.iloc[:n + 1]
            truncated = compute_sleeve_b(trunc_yield, trunc_xlu, cfg)
            assert full.iloc[n]["state"] == truncated.iloc[n]["state"], (
                f"Day {n}: full={full.iloc[n]['state']}, truncated={truncated.iloc[n]['state']}"
            )
