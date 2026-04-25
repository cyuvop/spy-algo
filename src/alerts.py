"""Discord webhook alert formatting and delivery for the trading signal system."""
import time
import logging
from datetime import datetime

import requests
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)

# Discord embed colors
COLOR_GREEN = 0x00AA00
COLOR_RED = 0xCC0000
COLOR_BLUE = 0x0066CC


def _pick_color(sleeve_a: dict, sleeve_b: dict, changed: list[str]) -> int:
    """Return embed color based on which sleeves changed and their direction."""
    if not changed:
        return COLOR_BLUE

    any_on = False
    any_off = False
    for sleeve_name in changed:
        if sleeve_name == "A":
            s = sleeve_a["state"]
        else:
            s = sleeve_b["state"]
        if s == "ON":
            any_on = True
        else:
            any_off = True

    if any_on:
        return COLOR_GREEN
    return COLOR_RED


def _pick_title(sleeve_a: dict, sleeve_b: dict, changed: list[str], date_str: str) -> str:
    """Return embed title based on signal changes."""
    if not changed:
        return f"📊 Daily Status — {date_str}"

    any_on = any(
        (sleeve_a["state"] if name == "A" else sleeve_b["state"]) == "ON"
        for name in changed
    )
    if any_on:
        return f"🟢 NEW SIGNAL — {date_str}"
    return f"🔴 EXIT SIGNAL — {date_str}"


def _format_sleeve_a_field(sleeve_a: dict, changed: list[str]) -> dict:
    """Build the Discord embed field for Sleeve A."""
    a_changed = "A" in changed
    badge = " ⚡NEW" if a_changed else " (no change)"
    prior = sleeve_a["prior_state"]
    state = sleeve_a["state"]
    name = f"SLEEVE A (SPY Put-Write): {prior} → {state}{badge}"

    spy_close = sleeve_a["spy_close"]
    spy_sma = sleeve_a["spy_sma_200"]
    pct = sleeve_a["pct_above_sma"]
    days = sleeve_a["days_in_state"]
    action = sleeve_a["suggested_action"]

    value = (
        f"SPY close: ${spy_close:.2f}\n"
        f"200-day SMA: ${spy_sma:.2f}\n"
        f"Above SMA by: {pct:.1%}\n"
        f"Days in current regime: {days}\n"
        f"ACTION: {action}"
    )
    return {"name": name, "value": value, "inline": False}


def _format_sleeve_b_field(sleeve_b: dict, changed: list[str]) -> dict:
    """Build the Discord embed field for Sleeve B."""
    b_changed = "B" in changed
    badge = " ⚡NEW" if b_changed else " (no change)"
    prior = sleeve_b["prior_state"]
    state = sleeve_b["state"]
    name = f"SLEEVE B (XLU Rates Overlay): {prior} → {state}{badge}"

    ten_yr = sleeve_b["ten_yr_yield"]
    sma_20 = sleeve_b["sma_20"]
    sma_60 = sleeve_b["sma_60"]
    roc = sleeve_b["three_month_roc"]
    xlu = sleeve_b["xlu_close"]
    action = sleeve_b["suggested_action"]

    value = (
        f"10Y yield: {ten_yr:.2f}%\n"
        f"20-day avg: {sma_20:.2f}% | 60-day avg: {sma_60:.2f}%\n"
        f"3-month change: {roc:.2%}\n"
        f"XLU price: ${xlu:.2f}\n"
        f"ACTION: {action}"
    )
    return {"name": name, "value": value, "inline": False}


def format_discord_embed(sleeve_a: dict, sleeve_b: dict, changed: list[str]) -> dict:
    """
    Build a Discord webhook payload with rich embeds.

    sleeve_a and sleeve_b are the signal output dicts (matching the shape from signals.py).
    changed is a list of sleeve names that changed state today, e.g. ["B"] or ["A", "B"] or [].

    Color coding (Discord embed color as integer):
    - Any sleeve changed to ON:  green (0x00AA00)
    - Any sleeve changed to OFF: red   (0xCC0000)
    - No changes:                blue  (0x0066CC)

    If both changed but in different directions, green wins if any went ON.
    """
    # Use local time (America/New_York is acceptable per spec when pytz not installed)
    now = datetime.now()
    date_str = now.strftime("%b %-d, %Y, %-I:%M %p ET")

    color = _pick_color(sleeve_a, sleeve_b, changed)
    title = _pick_title(sleeve_a, sleeve_b, changed, date_str)

    fields = [
        _format_sleeve_a_field(sleeve_a, changed),
        _format_sleeve_b_field(sleeve_b, changed),
    ]

    return {
        "embeds": [
            {
                "title": title,
                "color": color,
                "fields": fields,
                "footer": {"text": "Full state logged."},
            }
        ]
    }


def send_discord(payload: dict, webhook_url: str) -> bool:
    """
    POST payload to webhook_url as JSON.
    Retry up to 3 times with exponential backoff (1s, 2s, 4s).
    Return True on success, False if all retries failed.
    """
    headers = {"Content-Type": "application/json"}
    max_attempts = 3

    for attempt in range(1, max_attempts + 1):
        try:
            logger.info("Discord webhook attempt %d/%d", attempt, max_attempts)
            resp = requests.post(webhook_url, json=payload, headers=headers, timeout=10)
            if resp.status_code in (200, 204):
                logger.info("Discord webhook succeeded (status %d)", resp.status_code)
                return True
            logger.warning(
                "Discord webhook returned unexpected status %d on attempt %d",
                resp.status_code, attempt,
            )
        except RequestException as exc:
            logger.warning("Discord webhook attempt %d failed: %s", attempt, exc)

        if attempt < max_attempts:
            sleep_secs = 2 ** (attempt - 1)  # 1s, 2s, 4s
            logger.info("Retrying in %ds…", sleep_secs)
            time.sleep(sleep_secs)

    logger.error("Discord webhook failed after %d attempts", max_attempts)
    return False


def send_error_alert(error_message: str, webhook_url: str) -> None:
    """
    Send a simpler red embed for runtime failures.
    Title: "⚠️ Signal System Error"
    Description: error_message
    Color: red (0xCC0000)
    """
    payload = {
        "embeds": [
            {
                "title": "⚠️ Signal System Error",
                "description": error_message,
                "color": COLOR_RED,
            }
        ]
    }
    send_discord(payload, webhook_url)
