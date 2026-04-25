"""
Backtest harness for the two-sleeve trading signal system.

Run as:
    python -m src.backtest

Outputs (to reports/backtest/):
    equity_curve.png   — four equity curves
    stats.json         — performance statistics
    trade_log.csv      — state-change log
    summary.md         — human-readable markdown summary
"""
import json
import os
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .fetchers import fetch_fred_series, fetch_prices
from .signals import compute_sleeve_a, compute_sleeve_b
from .utils import get_logger, load_config, load_env

logger = get_logger(__name__)

PUTW_INCEPTION = "2016-04-14"
SGOV_INCEPTION = "2020-05-26"
PUTW_FALLBACK_DAILY = 0.006 / 21  # monthly PUT index long-run avg / trading days per month

REPORTS_DIR = Path("reports/backtest")


# ---------------------------------------------------------------------------
# Core data builder
# ---------------------------------------------------------------------------

def build_backtest_df(
    start: str = "2010-01-01",
    end: str = None,
    config: dict = None,
) -> pd.DataFrame:
    """
    Fetch all data, compute signals, and assemble the backtest DataFrame.

    Columns returned:
        spy_return            — SPY daily return
        putw_return           — PUTW daily return (NaN before inception)
        sgov_return           — SGOV daily return (NaN before inception)
        bil_return            — BIL daily return
        xlu_return            — XLU daily return
        sleeve_a_state        — ON/OFF for sleeve A on this date (signal from T-1, applied to T)
        sleeve_a_changed      — True if sleeve A state changed today
        sleeve_b_state        — ON/OFF for sleeve B on this date (signal from T-1, applied to T)
        sleeve_b_changed      — True if sleeve B state changed today
        return_a              — sleeve A return for this date
        return_b              — sleeve B return for this date
        combined_return       — weighted portfolio return
        cash_proxy_return     — best available cash proxy return (for reference)
    """
    if end is None:
        end = date.today().strftime("%Y-%m-%d")
    if config is None:
        config = load_config("config.yaml")

    wa = config["capital"]["sleeve_a_pct"]
    wb = config["capital"]["sleeve_b_pct"]

    # ---- Fetch price data -------------------------------------------------
    logger.info("Fetching SPY prices %s → %s", start, end)
    spy = fetch_prices("SPY", start=start, end=end)
    spy_index = spy.index  # master trading calendar

    logger.info("Fetching XLU prices")
    xlu = fetch_prices("XLU", start=start, end=end)

    logger.info("Fetching PUTW prices (inception ~%s)", PUTW_INCEPTION)
    try:
        putw = fetch_prices("PUTW", start=start, end=end)
    except Exception:
        putw = pd.DataFrame()

    logger.info("Fetching SGOV prices (inception ~%s)", SGOV_INCEPTION)
    try:
        sgov = fetch_prices("SGOV", start=start, end=end)
    except Exception:
        sgov = pd.DataFrame()

    logger.info("Fetching BIL prices")
    try:
        bil = fetch_prices("BIL", start=start, end=end)
    except Exception:
        bil = pd.DataFrame()

    # ---- Fetch 10Y yield (FRED) or use flat fallback ----------------------
    # Always attempt fetch_fred_series; it raises EnvironmentError if key missing.
    # Fall back to flat 4.0% only when the call fails (no key or network error).
    logger.info("Fetching DGS10 from FRED")
    try:
        ten_yr = fetch_fred_series("DGS10", start=start)
    except EnvironmentError:
        logger.warning("FRED_API_KEY not set; using flat 4.0%% yield for DGS10")
        ten_yr = pd.Series(
            [0.04] * len(spy_index),
            index=spy_index,
            name="DGS10",
        )
    except Exception as exc:
        logger.warning("FRED fetch failed (%s); using flat 4.0%% yield fallback", exc)
        ten_yr = pd.Series(
            [0.04] * len(spy_index),
            index=spy_index,
            name="DGS10",
        )

    # ---- Daily returns ----------------------------------------------------
    def _daily_ret(df: pd.DataFrame) -> pd.Series:
        if df is None or df.empty:
            return pd.Series(dtype=float)
        return (df["close"] / df["close"].shift(1) - 1).reindex(spy_index)

    spy_ret = _daily_ret(spy)
    xlu_ret = _daily_ret(xlu)
    putw_ret = _daily_ret(putw) if not (putw is None or putw.empty) else pd.Series(dtype=float)
    sgov_ret = _daily_ret(sgov) if not (sgov is None or sgov.empty) else pd.Series(dtype=float)
    bil_ret = _daily_ret(bil) if not (bil is None or bil.empty) else pd.Series(dtype=float)

    # ---- Fetch DTB3 (3-month T-bill yield) from FRED — penultimate fallback --
    logger.info("Fetching DTB3 from FRED")
    dtb3_daily_ret: pd.Series | None = None
    try:
        dtb3_raw = fetch_fred_series("DTB3", start=start)
        # DTB3 is an annualised yield (e.g. 5.25 = 5.25%); convert to daily return
        dtb3_ffilled = dtb3_raw.reindex(spy_index).ffill()
        dtb3_daily_ret = (dtb3_ffilled / 100.0) / 252
    except EnvironmentError:
        logger.warning("FRED_API_KEY not set; skipping DTB3 fallback")
    except Exception as exc:
        logger.warning("DTB3 fetch failed (%s); skipping DTB3 fallback", exc)

    # ---- Cash proxy: best available per day -------------------------------
    # Priority: SGOV > BIL > DTB3 > flat 0.04/252
    def _cash_proxy(idx: pd.DatetimeIndex) -> pd.Series:
        """Return per-day best cash proxy daily return."""
        result = pd.Series(index=idx, dtype=float)
        # Flat yield fallback first (always available)
        result[:] = 0.04 / 252
        # DTB3 fills dates where neither SGOV nor BIL have data
        if dtb3_daily_ret is not None:
            dtb3_aligned = dtb3_daily_ret.reindex(idx)
            mask_dtb3 = dtb3_aligned.notna()
            result[mask_dtb3] = dtb3_aligned[mask_dtb3]
        if not bil_ret.empty:
            bil_aligned = bil_ret.reindex(idx)
            mask_bil = bil_aligned.notna()
            result[mask_bil] = bil_aligned[mask_bil]
        if not sgov_ret.empty:
            sgov_aligned = sgov_ret.reindex(idx)
            mask_sgov = sgov_aligned.notna()
            result[mask_sgov] = sgov_aligned[mask_sgov]
        return result

    cash_proxy_ret = _cash_proxy(spy_index)

    # ---- Compute signals over full history --------------------------------
    # Sleeve A — uses SPY prices
    sig_a = compute_sleeve_a(spy, config)

    # Sleeve B — uses 10Y yield aligned to SPY calendar, plus XLU
    ten_yr_aligned = ten_yr.reindex(spy_index).ffill().dropna()
    # compute_sleeve_b expects xlu with the same index as yield_s
    xlu_for_b = xlu.reindex(ten_yr_aligned.index)
    sig_b = compute_sleeve_b(ten_yr_aligned, xlu_for_b, config)

    # Align sig_b back to spy_index (it may be shorter if yield had fewer rows)
    sig_b = sig_b.reindex(spy_index).ffill()

    # ---- Forward-shift signals (T→T+1 to avoid lookahead) ----------------
    # Signal on day T determines the return we capture on day T+1.
    # We shift state forward by 1: the state column represents the signal
    # that was active *before* this day (i.e., computed at end of T-1).
    state_a_raw = sig_a["state"]          # signal as of close of date T
    state_b_raw = sig_b["state"]

    # The return at row T is driven by the signal from row T-1
    state_a_applied = state_a_raw.shift(1)  # applied state for return on day T
    state_b_applied = state_b_raw.shift(1)

    # Track changes in applied state (for trade log — when did we act?)
    # A "change" happens on day T when state_a_raw changed (i.e., yesterday's signal differs)
    # We log changes on the day the signal was computed, not the day of the return.
    changed_a_signal = (state_a_raw != state_a_raw.shift(1))
    changed_b_signal = (state_b_raw != state_b_raw.shift(1))

    # ---- Assemble per-day returns -----------------------------------------
    def _sleeve_a_return(date_idx: pd.DatetimeIndex, state_applied: pd.Series) -> pd.Series:
        result = pd.Series(index=date_idx, dtype=float)
        for dt in date_idx:
            st = state_applied.get(dt, None)
            if pd.isna(st):
                continue
            if st == "ON":
                # Use PUTW if available, else flat fallback
                if not putw_ret.empty and dt in putw_ret.index and not pd.isna(putw_ret.get(dt)):
                    result[dt] = putw_ret[dt]
                else:
                    result[dt] = PUTW_FALLBACK_DAILY
            else:
                # OFF → cash proxy
                result[dt] = cash_proxy_ret[dt]
        return result

    def _sleeve_b_return(date_idx: pd.DatetimeIndex, state_applied: pd.Series) -> pd.Series:
        result = pd.Series(index=date_idx, dtype=float)
        for dt in date_idx:
            st = state_applied.get(dt, None)
            if pd.isna(st):
                continue
            if st == "ON":
                result[dt] = xlu_ret.get(dt, np.nan)
            else:
                result[dt] = cash_proxy_ret[dt]
        return result

    ret_a = _sleeve_a_return(spy_index, state_a_applied)
    ret_b = _sleeve_b_return(spy_index, state_b_applied)
    combined_ret = wa * ret_a + wb * ret_b

    # ---- Build final DataFrame --------------------------------------------
    df = pd.DataFrame(
        {
            "spy_return": spy_ret,
            "putw_return": putw_ret.reindex(spy_index) if not putw_ret.empty else pd.Series(index=spy_index, dtype=float),
            "sgov_return": sgov_ret.reindex(spy_index) if not sgov_ret.empty else pd.Series(index=spy_index, dtype=float),
            "bil_return": bil_ret.reindex(spy_index) if not bil_ret.empty else pd.Series(index=spy_index, dtype=float),
            "xlu_return": xlu_ret,
            "sleeve_a_state": state_a_applied,
            "sleeve_a_changed": changed_a_signal,
            "sleeve_b_state": state_b_applied,
            "sleeve_b_changed": changed_b_signal,
            "return_a": ret_a,
            "return_b": ret_b,
            "combined_return": combined_ret,
            "cash_proxy_return": cash_proxy_ret,
            # raw signal columns (for trade log — signal day)
            "_state_a_signal": state_a_raw,
            "_state_b_signal": state_b_raw,
            "_prior_state_a": state_a_raw.shift(1),
            "_prior_state_b": state_b_raw.shift(1),
            "_spy_close": spy["close"],
            "_ten_yr": ten_yr.reindex(spy_index).ffill(),
        },
        index=spy_index,
    )

    # Drop the first row (NaN because of shift) and any pre-warmup NaN rows
    df = df.dropna(subset=["return_a", "return_b", "combined_return"])
    return df


# ---------------------------------------------------------------------------
# Equity curves
# ---------------------------------------------------------------------------

def compute_equity_curves(df: pd.DataFrame) -> dict:
    """
    Compound daily returns into equity curves starting at 1.0.

    Returns dict with keys: Combined, Sleeve_A, Sleeve_B, SPY
    """
    valid = df.dropna(subset=["combined_return", "return_a", "return_b", "spy_return"])

    def _compound(returns: pd.Series) -> pd.Series:
        curve = (1 + returns).cumprod()
        # Normalize so first value = 1.0
        curve = curve / curve.iloc[0]
        return curve

    return {
        "Combined": _compound(valid["combined_return"]),
        "Sleeve_A": _compound(valid["return_a"]),
        "Sleeve_B": _compound(valid["return_b"]),
        "SPY": _compound(valid["spy_return"]),
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(returns: pd.Series) -> dict:
    """
    Compute performance statistics for a daily return series.

    Parameters
    ----------
    returns : pd.Series of daily returns (not compounded)

    Returns
    -------
    dict with keys: cagr, ann_vol, sharpe, max_drawdown, sortino, calmar
    """
    returns = returns.dropna()
    n = len(returns)
    if n == 0:
        return {k: float("nan") for k in ("cagr", "ann_vol", "sharpe", "max_drawdown", "sortino", "calmar")}

    # CAGR
    total_growth = (1 + returns).prod()
    years = n / 252.0
    cagr = total_growth ** (1.0 / years) - 1.0

    # Annualised volatility
    ann_vol = returns.std() * np.sqrt(252)

    # Sharpe (0% risk-free)
    sharpe = cagr / ann_vol if ann_vol != 0 else float("nan")

    # Max drawdown
    cum = (1 + returns).cumprod()
    rolling_max = cum.cummax()
    drawdown = cum / rolling_max - 1
    max_dd = drawdown.min()

    # Sortino
    neg_returns = returns[returns < 0]
    if len(neg_returns) > 0:
        downside_vol = neg_returns.std() * np.sqrt(252)
        sortino = cagr / downside_vol if downside_vol != 0 else float("nan")
    else:
        sortino = float("nan")

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else float("nan")

    return {
        "cagr": float(cagr),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "sortino": float(sortino),
        "calmar": float(calmar),
    }


# ---------------------------------------------------------------------------
# Trade log
# ---------------------------------------------------------------------------

def build_trade_log(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a trade log capturing every day where a sleeve state changed.

    Uses the raw signal change columns (_state_a_signal, _state_b_signal, etc.)
    so the log date is the day the signal was computed (= the day of the change),
    not the day the return is applied.
    """
    rows = []

    for dt, row in df.iterrows():
        # Sleeve A change
        if row.get("sleeve_a_changed", False):
            prior = row.get("_prior_state_a", "unknown")
            new = row.get("_state_a_signal", "unknown")
            spy_close = row.get("_spy_close", float("nan"))
            rows.append({
                "date": dt.date(),
                "sleeve": "A",
                "prior_state": prior if not pd.isna(prior) else "OFF",
                "new_state": new if not pd.isna(new) else "unknown",
                "spy_close_or_yield": round(float(spy_close), 4) if not pd.isna(spy_close) else None,
                "note": f"SPY regime change: {prior} → {new}",
            })

        # Sleeve B change
        if row.get("sleeve_b_changed", False):
            prior = row.get("_prior_state_b", "unknown")
            new = row.get("_state_b_signal", "unknown")
            ten_yr_val = row.get("_ten_yr", float("nan"))
            rows.append({
                "date": dt.date(),
                "sleeve": "B",
                "prior_state": prior if not pd.isna(prior) else "OFF",
                "new_state": new if not pd.isna(new) else "unknown",
                "spy_close_or_yield": round(float(ten_yr_val), 4) if not pd.isna(ten_yr_val) else None,
                "note": f"Yield regime change: {prior} → {new}",
            })

    return pd.DataFrame(rows, columns=["date", "sleeve", "prior_state", "new_state", "spy_close_or_yield", "note"])


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_equity_curves(curves: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    styles = {
        "Combined": {"color": "#1f77b4", "linewidth": 2.5},
        "Sleeve_A": {"color": "#ff7f0e", "linewidth": 1.5, "linestyle": "--"},
        "Sleeve_B": {"color": "#2ca02c", "linewidth": 1.5, "linestyle": "--"},
        "SPY": {"color": "#d62728", "linewidth": 1.5, "linestyle": ":"},
    }
    for name, series in curves.items():
        kw = styles.get(name, {})
        ax.plot(series.index, series.values, label=name, **kw)

    ax.set_title("Backtest Equity Curve", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved equity curve plot to %s", out_path)


# ---------------------------------------------------------------------------
# Summary markdown
# ---------------------------------------------------------------------------

def build_summary_md(all_stats: dict) -> str:
    def _fmt(val, pct=False, neg_pct=False):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/A"
        if pct or neg_pct:
            return f"{val * 100:.2f}%"
        return f"{val:.2f}"

    lines = [
        "# Backtest Summary",
        "",
        f"**Period:** 2010-01-01 to {date.today().strftime('%Y-%m-%d')}",
        "",
        "## Performance Comparison",
        "",
        "| Metric | Combined | Sleeve A | Sleeve B | SPY B&H |",
        "|--------|----------|----------|----------|---------|",
    ]

    metrics = [
        ("CAGR", "cagr", True),
        ("Ann Vol", "ann_vol", True),
        ("Sharpe", "sharpe", False),
        ("Max Drawdown", "max_drawdown", True),
        ("Sortino", "sortino", False),
        ("Calmar", "calmar", False),
    ]

    for label, key, is_pct in metrics:
        row = f"| {label} |"
        for name in ["Combined", "Sleeve_A", "Sleeve_B", "SPY"]:
            s = all_stats.get(name, {})
            val = s.get(key, float("nan"))
            row += f" {_fmt(val, pct=is_pct)} |"
        lines.append(row)

    lines += [
        "",
        "## Compare to SPY Buy-and-Hold",
        "",
    ]

    spy_s = all_stats.get("SPY", {})
    comb_s = all_stats.get("Combined", {})
    a_s = all_stats.get("Sleeve_A", {})
    b_s = all_stats.get("Sleeve_B", {})

    def _diff(a, b):
        if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in [a, b]):
            return "N/A"
        return f"{(a - b) * 100:+.2f}%"

    spy_cagr = spy_s.get("cagr", float("nan"))
    comb_cagr = comb_s.get("cagr", float("nan"))
    comb_dd = comb_s.get("max_drawdown", float("nan"))
    spy_dd = spy_s.get("max_drawdown", float("nan"))
    comb_sharpe = comb_s.get("sharpe", float("nan"))
    spy_sharpe = spy_s.get("sharpe", float("nan"))

    lines.append(
        f"The combined portfolio (70% Sleeve A / 30% Sleeve B) achieved a CAGR of "
        f"{_fmt(comb_cagr, pct=True)} versus SPY buy-and-hold at {_fmt(spy_cagr, pct=True)} "
        f"({_diff(comb_cagr, spy_cagr)} vs SPY). "
        f"The combined strategy's maximum drawdown was {_fmt(comb_dd, pct=True)} "
        f"compared to SPY's {_fmt(spy_dd, pct=True)}. "
        f"Sharpe ratio: {_fmt(comb_sharpe)} (combined) vs {_fmt(spy_sharpe)} (SPY)."
    )
    lines += [
        "",
        "### Sleeve A vs SPY",
        "",
        f"Sleeve A CAGR: {_fmt(a_s.get('cagr', float('nan')), pct=True)} "
        f"({_diff(a_s.get('cagr', float('nan')), spy_cagr)} vs SPY). "
        f"Max drawdown: {_fmt(a_s.get('max_drawdown', float('nan')), pct=True)} vs SPY {_fmt(spy_dd, pct=True)}.",
        "",
        "### Sleeve B vs SPY",
        "",
        f"Sleeve B CAGR: {_fmt(b_s.get('cagr', float('nan')), pct=True)} "
        f"({_diff(b_s.get('cagr', float('nan')), spy_cagr)} vs SPY). "
        f"Max drawdown: {_fmt(b_s.get('max_drawdown', float('nan')), pct=True)} vs SPY {_fmt(spy_dd, pct=True)}.",
        "",
        "---",
        "*Generated by src/backtest.py — human review required before deploying live runner.*",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    load_env()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config("config.yaml")
    end_date = date.today().strftime("%Y-%m-%d")

    logger.info("=== Starting backtest 2010-01-01 → %s ===", end_date)

    df = build_backtest_df(start="2010-01-01", end=end_date, config=config)
    logger.info("Backtest DataFrame: %d rows", len(df))

    # ---- Equity curves ---------------------------------------------------
    curves = compute_equity_curves(df)
    plot_equity_curves(curves, REPORTS_DIR / "equity_curve.png")

    # ---- Stats -----------------------------------------------------------
    valid = df.dropna(subset=["combined_return", "return_a", "return_b", "spy_return"])
    all_stats = {
        "Combined": compute_stats(valid["combined_return"]),
        "Sleeve_A": compute_stats(valid["return_a"]),
        "Sleeve_B": compute_stats(valid["return_b"]),
        "SPY": compute_stats(valid["spy_return"]),
    }

    stats_path = REPORTS_DIR / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    logger.info("Saved stats to %s", stats_path)

    # ---- Trade log -------------------------------------------------------
    trade_log = build_trade_log(df)
    log_path = REPORTS_DIR / "trade_log.csv"
    trade_log.to_csv(log_path, index=False)
    logger.info("Saved trade log to %s (%d state changes)", log_path, len(trade_log))

    # ---- Summary markdown ------------------------------------------------
    summary = build_summary_md(all_stats)
    summary_path = REPORTS_DIR / "summary.md"
    with open(summary_path, "w") as f:
        f.write(summary)
    logger.info("Saved summary to %s", summary_path)

    # ---- Mandatory pause banner ------------------------------------------
    comb = all_stats["Combined"]
    spy = all_stats["SPY"]

    cagr_c = comb["cagr"] * 100
    cagr_s = spy["cagr"] * 100
    sharpe_c = comb["sharpe"]
    sharpe_s = spy["sharpe"]
    dd_c = comb["max_drawdown"] * 100
    dd_s = spy["max_drawdown"] * 100

    print()
    print("=============================================")
    print("BACKTEST COMPLETE — HUMAN REVIEW REQUIRED")
    print("=============================================")
    print(f"Combined CAGR:     {cagr_c:5.2f}%   vs SPY:  {cagr_s:.2f}%")
    print(f"Combined Sharpe:   {sharpe_c:5.2f}    vs SPY:  {sharpe_s:.2f}")
    print(f"Combined MaxDD:    {dd_c:5.1f}%   vs SPY:  {dd_s:.1f}%")
    print()
    print("Review reports/backtest/ before proceeding to live runner.")
    print("Re-run with: python -m src.backtest")
    print("Proceed to daily runner only after explicit user approval.")
    print("=============================================")

    sys.exit(0)


if __name__ == "__main__":
    main()
