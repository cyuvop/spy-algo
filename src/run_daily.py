"""Daily runner entrypoint for the trading signal system. Run as: python3 -m src.run_daily"""

import os
import sys
from datetime import date, timedelta

from .utils import load_env, load_config, get_logger, is_market_day
from .fetchers import fetch_prices, fetch_fred_series
from .signals import compute_sleeve_a, compute_sleeve_b
from .state import get_last_state, save_state, has_run_today, mark_run
from .alerts import format_discord_embed, send_discord, send_error_alert

logger = get_logger(__name__)


def main() -> None:
    load_env()
    config = load_config("config.yaml")
    today = date.today().strftime("%Y-%m-%d")

    if not is_market_day(today):
        logger.info("Not a market day, exiting.")
        sys.exit(0)

    if has_run_today(today):
        logger.info("Already ran today, exiting.")
        sys.exit(0)

    # webhook_url is needed in the error handler below, so resolve it early
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "")

    try:
        two_years_ago = (date.today() - timedelta(days=730)).strftime("%Y-%m-%d")

        logger.info("Fetching SPY prices…")
        spy = fetch_prices("SPY", start=two_years_ago)

        logger.info("Fetching XLU prices…")
        xlu = fetch_prices("XLU", start=two_years_ago)

        logger.info("Fetching DGS10 (10Y Treasury yield)…")
        ten_yr = fetch_fred_series("DGS10", start=two_years_ago)

        max_staleness = config["runtime"]["max_data_staleness_days"]
        staleness_cutoff = date.today() - timedelta(days=max_staleness)
        last_spy_date = spy.index[-1].date()

        if last_spy_date < staleness_cutoff:
            msg = (
                f"SPY data is stale: last date {last_spy_date} is older than "
                f"{max_staleness}-day threshold (cutoff: {staleness_cutoff})"
            )
            logger.error(msg)
            if webhook_url:
                send_error_alert(msg, webhook_url)
            mark_run(today, success=False, notes=msg)
            sys.exit(1)

        logger.info("Computing sleeve A signal…")
        sig_a = compute_sleeve_a(spy, config).iloc[-1].to_dict()

        logger.info("Computing sleeve B signal…")
        sig_b = compute_sleeve_b(ten_yr, xlu, config).iloc[-1].to_dict()

        prior_a = get_last_state("A")
        prior_b = get_last_state("B")

        changed = []
        if prior_a is None or prior_a["state"] != sig_a["state"]:
            changed.append("A")
        if prior_b is None or prior_b["state"] != sig_b["state"]:
            changed.append("B")

        logger.info("State changes: %s", changed if changed else "none")

        if changed or config["alerts"]["daily_status_ping"]:
            if webhook_url:
                payload = format_discord_embed(sig_a, sig_b, changed)
                send_discord(payload, webhook_url)
            else:
                logger.warning("DISCORD_WEBHOOK_URL not set; skipping alert.")

        save_state("A", today, sig_a["state"], sig_a)
        save_state("B", today, sig_b["state"], sig_b)
        mark_run(today, success=True)

        logger.info(
            "Run complete. Sleeve A: %s | Sleeve B: %s | Changed: %s",
            sig_a["state"],
            sig_b["state"],
            changed if changed else "none",
        )
        sys.exit(0)

    except Exception as e:
        logger.exception("Unhandled error in daily runner: %s", e)
        if webhook_url:
            send_error_alert(str(e), webhook_url)
        mark_run(today, success=False, notes=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
