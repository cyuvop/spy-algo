# CLAUDE.md — Trading Signal System

Notes for future Claude Code sessions. Read this before touching any code.

---

## What this project is

A daily rule-based trading signal generator. Two strategies emit Discord alerts after market close. No broker integration — ever. Human executes all trades manually.

- **Sleeve A**: SPY put-write with 200-day SMA trend filter (2-day confirmation)
- **Sleeve B**: XLU long when 10Y yield in confirmed downtrend (20/60-day SMA crossover + ROC)

---

## Architecture

```
fetchers.py  →  indicators.py  →  signals.py  →  state.py  →  alerts.py
   (data)         (math)          (regime)       (SQLite)     (Discord)
                                      ↑
                                  backtest.py (offline replay)
                                  run_daily.py (entrypoint)
```

All signal logic lives in pure functions (`indicators.py`, `signals.py`). No I/O, no side effects. This makes testing and backtest replay trivial.

---

## Key design decisions

### Confirmation window (Sleeve A)
`compute_sleeve_a` uses `consecutive_days_true` to require `confirmation_days` consecutive closes above/below SMA before flipping regime. This is purely rule-based with no lookahead. The loop walks forward through the series and only looks at past data.

### Exit confirmation (Sleeve B)
`compute_sleeve_b` enters ON immediately when entry conditions are met (both SMAs aligned AND ROC negative). Exit requires `exit_confirm_days` consecutive days where short SMA > long SMA. Asymmetric: fast entry, slow exit — intentional to reduce whipsaw.

### Caching
Fetchers cache to `data/cache/{key}.parquet`. Freshness is determined by the last date in the cache vs. today minus `max_data_staleness_days`. Re-fetching refreshes the whole file (not append-only). This is fine for daily runs.

### Config-driven parameters
**Never hardcode SMA periods, confirmation days, or capital allocations.** All come from `config.yaml` via `load_config()`. Both `compute_sleeve_a` and `compute_sleeve_b` accept the full config dict and index into `sleeve_a` / `sleeve_b` keys.

### Test design for signals
Signal tests use small SMA periods (e.g., `trend_sma_period: 5`) so test fixtures don't need 200+ rows. The no-lookahead tests use `np.random.default_rng(fixed_seed)` for reproducibility. When writing new fixtures, be careful: **a constant price series will have `close == SMA` once the SMA warms up**, which trips the `close > SMA` condition. Use a monotonically rising series in the ON phase to keep close strictly above SMA throughout.

---

## File map

| File | Responsibility |
|------|---------------|
| `src/utils.py` | Config loading, logging, `is_market_day`, `load_env` |
| `src/fetchers.py` | yfinance + FRED data with parquet cache |
| `src/indicators.py` | `sma`, `roc`, `consecutive_days_true` — pure functions |
| `src/signals.py` | `compute_sleeve_a`, `compute_sleeve_b` — pure, vectorized |
| `src/state.py` | SQLite read/write for regime state (not yet built) |
| `src/alerts.py` | Discord webhook formatting + sending (not yet built) |
| `src/backtest.py` | Historical replay, equity curve, stats (not yet built) |
| `src/run_daily.py` | Daily entrypoint (not yet built) |
| `config.yaml` | All tunable parameters |
| `tests/` | One test file per source module |

---

## Build status (as of 2026-04-23)

Steps completed (per `CLAUDE_CODE_BUILD_SPEC.md`):
- [x] Step 1: Scaffold
- [x] Step 2: Utils + Fetchers (with mocked tests)
- [x] Step 3: Indicators (100% coverage)
- [x] Step 4: Signals (with no-lookahead tests)

Not yet built:
- [ ] Step 5: State (SQLite persistence)
- [ ] Step 6: Backtest — **STOP after this step for human review**
- [ ] Step 7: Alerts (need Discord webhook to test)
- [ ] Step 8: Daily runner
- [ ] Step 9: GitHub Actions

---

## Running tests

```bash
cd signals
pytest -v              # all 52 tests
pytest tests/test_indicators.py -v   # indicators only
pytest tests/test_signals.py -v      # signals only
```

## Smoke testing fetchers (requires real API keys)

```bash
cd signals
cp .env.example .env   # fill in FRED_API_KEY
python3 - <<'EOF'
from dotenv import load_dotenv; load_dotenv()
from src.fetchers import fetch_prices, fetch_fred_series
spy = fetch_prices("SPY", start="2023-01-01")
print("SPY rows:", len(spy), "last close:", spy["close"].iloc[-1])
dgs10 = fetch_fred_series("DGS10", start="2023-01-01")
print("DGS10 rows:", len(dgs10), "last yield:", dgs10.iloc[-1])
EOF
```

---

## Guardrails (non-negotiable, per spec)

1. Never integrate with a broker API. Never place orders.
2. Secrets (FRED_API_KEY, DISCORD_WEBHOOK_URL) stay in `.env`, never committed.
3. All tests must pass before moving to the next step.
4. **MANDATORY PAUSE after backtest** — print review banner, stop, wait for user approval before building the live runner.
5. No lookahead bias: signal at day T uses only data ≤ day T.
6. Fail loudly on data errors — send Discord error alert, exit non-zero.
7. Idempotency: daily runner checks `has_run_today()` before doing anything.
8. Deterministic: same inputs → same outputs, always.
