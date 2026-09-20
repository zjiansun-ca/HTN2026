import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from backend import db
from backend.main import CASE_BANK, app

client = TestClient(app)
SESSION = "test-session"


@pytest.fixture(autouse=True)
def force_fallback_db(monkeypatch):
    # The test suite must be deterministic and must never touch the real Tiger Data
    # instance (slow, and would pollute the real leaderboard with test events) —
    # regardless of whatever TIGER_DSN happens to be in .env on this machine, every
    # test in this module exercises the graceful-degradation fallback path. Events
    # are intentionally left to accumulate across this module's tests (like the
    # existing /next-case pointer already does) so the leaderboard test below can
    # see what earlier tests logged.
    monkeypatch.setattr(db, "TIGER_DSN", None)


def test_bank_has_16_cases():
    assert len(CASE_BANK) == 16
    ids = {c["case_id"] for c in CASE_BANK}
    assert len(ids) == 16  # all unique


def test_known_under_triaged_case_is_caught():
    # under_triaged-01: bank self-check confirmed CAUGHT (risk 0.830)
    r = client.post("/submit", json={"case_id": "under_triaged-01", "judge_call": "keep"})
    assert r.status_code == 200

    r = client.post("/reveal", json={"case_id": "under_triaged-01", "session_id": SESSION})
    assert r.status_code == 200
    body = r.json()

    assert body["model_flag"] is True
    assert 0.0 <= body["risk_rank"] <= 1.0
    assert len(body["top_features"]) > 0
    for f in body["top_features"]:
        assert f["kind"] in ("vital", "text")
    assert body["outcome"] == "admitted"
    assert body["outcome_detail"]


def test_known_well_case_is_not_flagged():
    # well-01: bank self-check confirmed not flagged (risk 0.094)
    r = client.post("/submit", json={"case_id": "well-01", "judge_call": "flag"})
    assert r.status_code == 200

    r = client.post("/reveal", json={"case_id": "well-01", "session_id": SESSION})
    assert r.status_code == 200
    body = r.json()

    assert body["model_flag"] is False
    assert body["outcome"] == "discharged"


def test_reveal_without_submit_is_rejected():
    r = client.post("/reveal", json={"case_id": "well-02", "session_id": SESSION})
    assert r.status_code == 400


def test_next_case_cycles_through_all_16_in_stable_order():
    seen = [client.post("/next-case").json()["case_id"] for _ in range(16)]
    assert seen == [c["case_id"] for c in CASE_BANK]


def test_reveal_logs_to_leaderboard_via_db_fallback():
    # force_fallback_db forces every /reveal in this module through the graceful-
    # degradation fallback path (constraint 1) — assert it actually recorded those
    # events and the leaderboard reads them back correctly.
    assert db.TIGER_DSN is None
    result = db.get_leaderboard()
    assert result["source"] == "fallback"
    assert result["n_events"] >= 2  # at least the two reveals above
    assert result["human_catch_rate"] is not None
    assert result["model_catch_rate"] is not None
    assert 0.0 <= result["human_catch_rate"] <= 1.0
    assert 0.0 <= result["model_catch_rate"] <= 1.0


def test_leaderboard_endpoint_never_raises_when_db_down():
    r = client.get("/leaderboard")
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "fallback"
    assert "timeseries" in body
