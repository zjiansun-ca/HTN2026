import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import db


def test_circuit_breaker_trips_and_stops_retrying(monkeypatch):
    # simulate a configured-but-unreachable DB (the filtered-port scenario) without
    # touching the network or the real TIGER_DSN
    monkeypatch.setattr(db, "TIGER_DSN", "postgres://fake:fake@example.invalid:5432/fake")
    monkeypatch.setattr(db, "_DB_DOWN", False)
    monkeypatch.setattr(db, "_FALLBACK_EVENTS", [])

    call_count = {"n": 0}

    def fake_tcp_reachable(timeout=db.CONNECT_TIMEOUT_S):
        call_count["n"] += 1
        return False

    monkeypatch.setattr(db, "_tcp_reachable", fake_tcp_reachable)

    assert db._DB_DOWN is False

    db.log_event("s1", "case-a", "under_triaged", True, True)
    assert db._DB_DOWN is True, "first failure must trip the breaker"
    assert call_count["n"] == 1, "exactly one connection attempt before tripping"

    # a second call must skip the connection attempt entirely — no retry
    db.log_event("s2", "case-b", "well", False, False)
    assert call_count["n"] == 1, "breaker must prevent any further connection attempts"

    # the leaderboard read must also respect the tripped breaker, not retry independently
    result = db.get_leaderboard()
    assert result["source"] == "fallback"
    assert result["n_events"] == 2
    assert call_count["n"] == 1
