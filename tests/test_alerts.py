"""Tests for src/alerts.py — Discord webhook formatting and sending."""
import pytest
from unittest.mock import patch, MagicMock
from requests.exceptions import RequestException

from src.alerts import format_discord_embed, send_discord, send_error_alert


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SLEEVE_A = {
    "sleeve": "A",
    "state": "ON",
    "prior_state": "OFF",
    "changed_today": True,
    "days_in_state": 1,
    "spy_close": 582.14,
    "spy_sma_200": 561.38,
    "pct_above_sma": 0.037,
    "suggested_action": "Sell 30-45 DTE put credit spread.",
}

SLEEVE_B = {
    "sleeve": "B",
    "state": "OFF",
    "prior_state": "OFF",
    "changed_today": False,
    "days_in_state": 5,
    "ten_yr_yield": 4.12,
    "sma_20": 4.18,
    "sma_60": 4.29,
    "three_month_roc": -0.024,
    "xlu_close": 78.42,
    "suggested_action": "Hold SGOV/BIL. No XLU position.",
}

WEBHOOK_URL = "https://discord.com/api/webhooks/test/fake"


# ---------------------------------------------------------------------------
# format_discord_embed tests
# ---------------------------------------------------------------------------


def test_format_embed_no_changes_is_blue():
    payload = format_discord_embed(SLEEVE_A, SLEEVE_B, changed=[])
    color = payload["embeds"][0]["color"]
    assert color == 0x0066CC


def test_format_embed_on_change_is_green():
    sleeve_a_on = {**SLEEVE_A, "state": "ON", "changed_today": True}
    payload = format_discord_embed(sleeve_a_on, SLEEVE_B, changed=["A"])
    color = payload["embeds"][0]["color"]
    assert color == 0x00AA00


def test_format_embed_off_change_is_red():
    sleeve_b_off = {**SLEEVE_B, "state": "OFF", "changed_today": True}
    payload = format_discord_embed(SLEEVE_A, sleeve_b_off, changed=["B"])
    # sleeve_b went OFF (changed, and state is OFF) → red
    color = payload["embeds"][0]["color"]
    assert color == 0xCC0000


def test_format_embed_title_has_daily_status_when_no_changes():
    payload = format_discord_embed(SLEEVE_A, SLEEVE_B, changed=[])
    title = payload["embeds"][0]["title"]
    assert "Daily Status" in title


def test_format_embed_title_has_new_signal_when_on():
    sleeve_a_on = {**SLEEVE_A, "state": "ON", "changed_today": True}
    payload = format_discord_embed(sleeve_a_on, SLEEVE_B, changed=["A"])
    title = payload["embeds"][0]["title"]
    assert "NEW SIGNAL" in title


def test_format_embed_sleeve_a_field_has_new_badge():
    sleeve_a_changed = {**SLEEVE_A, "state": "ON", "changed_today": True}
    payload = format_discord_embed(sleeve_a_changed, SLEEVE_B, changed=["A"])
    fields = payload["embeds"][0]["fields"]
    a_field = next(f for f in fields if f["name"].startswith("SLEEVE A"))
    assert "⚡NEW" in a_field["name"]


def test_format_embed_sleeve_b_field_no_badge_when_unchanged():
    payload = format_discord_embed(SLEEVE_A, SLEEVE_B, changed=[])
    fields = payload["embeds"][0]["fields"]
    b_field = next(f for f in fields if f["name"].startswith("SLEEVE B"))
    assert "(no change)" in b_field["name"]


def test_format_embed_payload_structure():
    payload = format_discord_embed(SLEEVE_A, SLEEVE_B, changed=[])
    assert "embeds" in payload
    embed = payload["embeds"][0]
    assert "title" in embed
    assert "color" in embed
    assert "fields" in embed
    assert "footer" in embed
    # Ensure two fields (one per sleeve)
    assert len(embed["fields"]) == 2
    for field in embed["fields"]:
        assert "name" in field
        assert "value" in field
        assert field.get("inline") is False


# ---------------------------------------------------------------------------
# send_discord tests
# ---------------------------------------------------------------------------


def test_send_discord_returns_true_on_204():
    mock_resp = MagicMock()
    mock_resp.status_code = 204
    with patch("src.alerts.requests.post", return_value=mock_resp) as mock_post:
        result = send_discord({"embeds": []}, WEBHOOK_URL)
    assert result is True
    mock_post.assert_called_once()


def test_send_discord_returns_false_after_retries():
    with patch("src.alerts.requests.post", side_effect=RequestException("timeout")) as mock_post:
        with patch("src.alerts.time.sleep"):  # skip actual sleeps
            result = send_discord({"embeds": []}, WEBHOOK_URL)
    assert result is False
    assert mock_post.call_count == 3


def test_send_discord_retries_on_failure():
    mock_resp = MagicMock()
    mock_resp.status_code = 204
    side_effects = [RequestException("err"), RequestException("err"), mock_resp]
    with patch("src.alerts.requests.post", side_effect=side_effects) as mock_post:
        with patch("src.alerts.time.sleep"):
            result = send_discord({"embeds": []}, WEBHOOK_URL)
    assert result is True
    assert mock_post.call_count == 3


# ---------------------------------------------------------------------------
# send_error_alert tests
# ---------------------------------------------------------------------------


def test_send_error_alert_sends_red_embed():
    mock_resp = MagicMock()
    mock_resp.status_code = 204
    with patch("src.alerts.requests.post", return_value=mock_resp) as mock_post:
        send_error_alert("Something went wrong", WEBHOOK_URL)
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    # The payload is passed as the json= kwarg
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    embed = payload["embeds"][0]
    assert embed["color"] == 0xCC0000
    assert "Error" in embed["title"]
