# Trading Signal System

A daily rule-based signal generator for two independent trading strategies. Runs every weekday after market close, emits Discord alerts when signals change. **No broker integration — human executes all trades.**

## Quick Start

```bash
cp .env.example .env
# Edit .env: set FRED_API_KEY and DISCORD_WEBHOOK_URL
pip install -r requirements.txt
pytest -v
python -m src.backtest    # review equity curve before going live
python -m src.run_daily   # today's signals to Discord
```

## Strategies

### Sleeve A — SPY Put-Write with Trend Filter
- **ON** when SPY > 200-day SMA for 2 consecutive closes
- **OFF** (park in SGOV) when SPY ≤ 200-day SMA for 2 consecutive closes
- When ON: sell 30–45 DTE ~0.25 delta SPY put credit spread

### Sleeve B — XLU Rates Overlay
- **ON** when 10Y yield 20-day SMA < 60-day SMA AND 3-month rate-of-change < 0
- **OFF** after 5 consecutive days where 20-day SMA > 60-day SMA
- When ON: long XLU; when OFF: park in SGOV

## Setup

1. Get a free FRED API key at https://fred.stlouisfed.org/docs/api/api_key.html
2. Create a Discord webhook in your server settings
3. Add both to `.env`
4. Set `FRED_API_KEY` and `DISCORD_WEBHOOK_URL` as GitHub Actions secrets for automated runs

## Tuning

All parameters are in `config.yaml` — no code edits needed:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `capital.total` | 25000 | Resize everything proportionally |
| `capital.sleeve_a_pct` | 0.70 | Must sum to ≤ 1.0 with sleeve_b_pct |
| `capital.sleeve_b_pct` | 0.30 | Remainder is cash buffer |
| `sleeve_a.enabled` | true | Set false to disable Sleeve A entirely |
| `sleeve_b.enabled` | true | Set false to disable Sleeve B entirely |
| `sleeve_a.trend_sma_period` | 200 | Days for SPY trend SMA |
| `sleeve_a.confirmation_days` | 2 | Consecutive closes to confirm regime flip |
| `sleeve_a.structure` | put_credit_spread | Change to `cash_secured_put` at ~$75k+ |
| `alerts.daily_status_ping` | true | Set false for quiet mode (alerts on changes only) |

## Interpreting Alerts

- **No change, daily ping**: current regime status, no action needed unless managing open positions
- **ON → ON (new)**: regime just started, consider opening a new position per the suggested action
- **ON → OFF**: regime ended, close positions per management rules
- ⚡ NEW badge = state changed today

## DST Note

The GitHub Actions cron (`35 21 * * 1-5`) runs at 4:35 PM EST. During EDT (March–November), this fires at 5:35 PM EDT — one hour late. Adjust the cron to `35 20 * * 1-5` in summer if timeliness matters, or add a second cron entry.

---

> **What this system is:** a disciplined rule-based signal generator with reasonable theoretical grounding (volatility risk premium for Sleeve A; rate-sensitivity of utilities for Sleeve B).
>
> **What it is not:** a guaranteed edge over buy-and-hold SPY. Realistic long-term expectation is SPY-like total return with lower drawdowns and a modestly better risk-adjusted profile. The backtest will show whether this holds in sample; live performance may differ. Past performance does not predict future results.
>
> **Risks:** selling options has unbounded risk in the underlying (mitigated here by spreads). XLU can decouple from rates in unusual regimes. Both sleeves can be whipsawed in choppy markets. Position-size accordingly. This is not financial advice.
