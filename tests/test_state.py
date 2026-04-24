"""
Tests for src/state.py — SQLite state persistence.

All tests use pytest's tmp_path fixture to avoid touching data/state.db.
"""
import pytest

from src.state import get_last_state, save_state, has_run_today, mark_run


# ── get_last_state ────────────────────────────────────────────────────────────

def test_get_last_state_returns_none_when_no_rows(tmp_path):
    db = str(tmp_path / "state.db")
    result = get_last_state("sleeve_a", db_path=db)
    assert result is None


def test_save_and_get_last_state(tmp_path):
    db = str(tmp_path / "state.db")
    payload = {"spy_close": 450.0, "state": "ON"}
    save_state("sleeve_a", "2026-04-24", "ON", payload, db_path=db)
    result = get_last_state("sleeve_a", db_path=db)
    assert result is not None
    assert result["sleeve"] == "sleeve_a"
    assert result["date"] == "2026-04-24"
    assert result["state"] == "ON"
    assert result["payload"] == payload


def test_save_state_is_idempotent(tmp_path):
    db = str(tmp_path / "state.db")
    save_state("sleeve_a", "2026-04-24", "OFF", {"note": "first"}, db_path=db)
    save_state("sleeve_a", "2026-04-24", "ON",  {"note": "second"}, db_path=db)
    result = get_last_state("sleeve_a", db_path=db)
    assert result["state"] == "ON"
    assert result["payload"] == {"note": "second"}


def test_get_last_state_returns_most_recent_date(tmp_path):
    db = str(tmp_path / "state.db")
    save_state("sleeve_a", "2026-04-22", "OFF", {"day": "earlier"}, db_path=db)
    save_state("sleeve_a", "2026-04-24", "ON",  {"day": "later"},   db_path=db)
    result = get_last_state("sleeve_a", db_path=db)
    assert result["date"] == "2026-04-24"
    assert result["state"] == "ON"


# ── has_run_today / mark_run ──────────────────────────────────────────────────

def test_has_run_today_false_when_no_rows(tmp_path):
    db = str(tmp_path / "state.db")
    assert has_run_today("2026-04-24", db_path=db) is False


def test_has_run_today_true_after_mark_run(tmp_path):
    db = str(tmp_path / "state.db")
    mark_run("2026-04-24", success=True, db_path=db)
    assert has_run_today("2026-04-24", db_path=db) is True


def test_has_run_today_false_for_different_date(tmp_path):
    db = str(tmp_path / "state.db")
    mark_run("2026-04-24", success=True, db_path=db)
    assert has_run_today("2026-04-23", db_path=db) is False


def test_mark_run_idempotent(tmp_path):
    db = str(tmp_path / "state.db")
    mark_run("2026-04-24", success=True,  notes="first run",  db_path=db)
    mark_run("2026-04-24", success=False, notes="second run", db_path=db)
    # No error raised; has_run_today still True
    assert has_run_today("2026-04-24", db_path=db) is True


# ── payload JSON roundtrip ────────────────────────────────────────────────────

def test_payload_json_roundtrip(tmp_path):
    db = str(tmp_path / "state.db")
    payload = {
        "spy_close": 451.23,
        "nested": {"sma_200": 430.0, "above": True},
        "tags": ["regime", "ON"],
        "count": 7,
    }
    save_state("sleeve_a", "2026-04-24", "ON", payload, db_path=db)
    result = get_last_state("sleeve_a", db_path=db)
    assert result["payload"] == payload
