import pandas as pd

from .indicators import sma, roc, consecutive_days_true


def compute_sleeve_a(spy_prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    cfg = config["sleeve_a"]
    period = cfg["trend_sma_period"]
    confirm = cfg["confirmation_days"]

    close = spy_prices["close"]
    ma = sma(close, period)

    # Raw condition: close above SMA
    above = (close > ma).fillna(False)

    # Confirmed regime: requires `confirm` consecutive days in new condition
    consec_on = consecutive_days_true(above)
    consec_off = consecutive_days_true(~above)

    state_col = []
    current = "OFF"
    for i in range(len(close)):
        if consec_on.iloc[i] >= confirm:
            current = "ON"
        elif consec_off.iloc[i] >= confirm:
            current = "OFF"
        state_col.append(current)

    state_s = pd.Series(state_col, index=close.index, name="state")
    prior_state = state_s.shift(1).fillna("OFF")
    changed_today = state_s != prior_state

    days_in = []
    count = 0
    prev = None
    for s in state_s:
        if s != prev:
            count = 1
        else:
            count += 1
        days_in.append(count)
        prev = s

    pct_above = ((close - ma) / ma).where(ma.notna(), other=float("nan"))

    structure = cfg.get("structure", "put_credit_spread")
    dte_min = cfg.get("target_dte_min", 30)
    dte_max = cfg.get("target_dte_max", 45)
    delta = cfg.get("target_short_delta", 0.25)

    actions = []
    for s in state_s:
        if s == "ON":
            if structure == "cash_secured_put":
                actions.append(
                    f"If no open position, sell {dte_min}–{dte_max} DTE "
                    f"~{delta} delta cash-secured put."
                )
            else:
                actions.append(
                    f"If no open position, sell {dte_min}–{dte_max} DTE "
                    f"~{delta} delta SPY put credit spread."
                )
        else:
            actions.append(
                "Park capital in SGOV/BIL. Do not open new positions. "
                "Close any existing position at next profitable opportunity or at 21 DTE."
            )

    return pd.DataFrame(
        {
            "state": state_s,
            "prior_state": prior_state,
            "changed_today": changed_today,
            "days_in_state": pd.Series(days_in, index=close.index),
            "spy_close": close,
            "spy_sma_200": ma,
            "pct_above_sma": pct_above,
            "suggested_action": pd.Series(actions, index=close.index),
        }
    )


def compute_sleeve_b(
    ten_yr: pd.Series, xlu_prices: pd.DataFrame, config: dict
) -> pd.DataFrame:
    cfg = config["sleeve_b"]
    sma_short_period = cfg["entry_sma_short"]
    sma_long_period = cfg["entry_sma_long"]
    roc_period = cfg["entry_roc_period"]
    exit_confirm = cfg["exit_confirm_days"]

    yield_s = ten_yr.dropna()
    # Align xlu to yield index
    xlu_close = xlu_prices["close"].reindex(yield_s.index, method="ffill")

    ma_short = sma(yield_s, sma_short_period)
    ma_long = sma(yield_s, sma_long_period)
    yr_roc = roc(yield_s, roc_period)

    # Entry: short SMA < long SMA AND roc < 0
    entry_cond = (ma_short < ma_long) & (yr_roc < 0)
    # Exit candidate: short SMA > long SMA (needs confirm consecutive days)
    exit_cand = (ma_short > ma_long).fillna(False)
    consec_exit = consecutive_days_true(exit_cand)

    state_col = []
    current = "OFF"
    for i in range(len(yield_s)):
        if current == "OFF" and entry_cond.iloc[i] if not pd.isna(entry_cond.iloc[i]) else False:
            current = "ON"
        elif current == "ON" and consec_exit.iloc[i] >= exit_confirm:
            current = "OFF"
        state_col.append(current)

    state_s = pd.Series(state_col, index=yield_s.index, name="state")
    prior_state = state_s.shift(1).fillna("OFF")
    changed_today = state_s != prior_state

    days_in = []
    count = 0
    prev = None
    for s in state_s:
        if s != prev:
            count = 1
        else:
            count += 1
        days_in.append(count)
        prev = s

    actions = []
    for s in state_s:
        if s == "ON":
            actions.append("Buy XLU with sleeve B allocation.")
        else:
            actions.append(
                "Sell XLU entirely. Rotate to SGOV/BIL."
            )

    # Fix ON entries: only "Buy XLU" on state change days; otherwise hold
    action_s = pd.Series(actions, index=yield_s.index)
    hold_on_mask = (state_s == "ON") & (~changed_today)
    action_s[hold_on_mask] = "Hold XLU position."

    hold_off_mask = (state_s == "OFF") & (~changed_today)
    action_s[hold_off_mask] = "Hold SGOV/BIL. No XLU position."

    return pd.DataFrame(
        {
            "state": state_s,
            "prior_state": prior_state,
            "changed_today": changed_today,
            "days_in_state": pd.Series(days_in, index=yield_s.index),
            "ten_yr_yield": yield_s,
            "sma_20": ma_short,
            "sma_60": ma_long,
            "three_month_roc": yr_roc,
            "xlu_close": xlu_close,
            "suggested_action": action_s,
        }
    )
