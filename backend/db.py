"""Tiger Data (TimescaleDB) event log + leaderboard, with mandatory graceful degradation.

HARD CONSTRAINT: the networked DB can never be a single point of failure for the demo,
and degradation must be FAST, not just safe:
- A per-process circuit breaker (_DB_DOWN) trips on the first failure and stays tripped
  (reset only by restarting the process) — every call after that skips straight to the
  in-memory fallback with zero connection attempts.
- Before that trip, any single connection attempt is bounded to ~3s by a raw-socket
  precheck. psycopg's own connect_timeout did NOT reliably bound a reachable-host/
  filtered-port scenario in testing (~24s observed) — an OS-level socket connect with an
  explicit deadline is a primitive that's actually enforced regardless.
- DEMO_MODE (env var) skips the DB entirely in the write path and serves a one-time
  snapshot of the real board for reads — zero DB calls in the hot path at demo time.

Callers (backend/main.py) never see an exception from this module and never need to
check DB health themselves. Data logged here is entirely synthetic/judge-generated
(session_id, case_id, booleans) — never a MIMIC record. Safe for Tiger Cloud.
"""
import os
import socket
from datetime import datetime, timezone
from urllib.parse import urlsplit

import psycopg
from dotenv import load_dotenv

from backend import observability

load_dotenv()

TIGER_DSN = os.environ.get("TIGER_DSN")
DEMO_MODE = os.environ.get("DEMO_MODE", "").strip().lower() in ("1", "true", "yes")
CONNECT_TIMEOUT_S = 3  # a dead DB must fail fast, not hang a live-demo request

# in-memory fallback — list of dicts, same shape as a decision_events row
_FALLBACK_EVENTS: list[dict] = []

# circuit breaker: once True, every DB-touching function below skips straight to
# fallback with NO connection attempt, for the rest of this process's life
_DB_DOWN = False

# DEMO_MODE: one real snapshot of the live board, taken at startup, served forever after
_DEMO_SNAPSHOT: dict | None = None

# Each statement runs separately, on an autocommit connection: TimescaleDB's continuous
# aggregate DDL (CREATE MATERIALIZED VIEW ... WITH (timescaledb.continuous) and
# add_continuous_aggregate_policy) cannot run inside a multi-statement transaction block,
# and the policy call requires the aggregate to already exist — so table+hypertable,
# aggregate, and policy are three separate round trips, in that order.
SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS decision_events (
        ts             TIMESTAMPTZ NOT NULL DEFAULT now(),
        session_id     TEXT NOT NULL,
        case_id        TEXT NOT NULL,
        subpopulation  TEXT NOT NULL CHECK (subpopulation IN ('under_triaged', 'well')),
        judge_flagged  BOOLEAN NOT NULL,
        model_flagged  BOOLEAN NOT NULL
    );
    """,
    "SELECT create_hypertable('decision_events', 'ts', if_not_exists => TRUE);",
    """
    CREATE MATERIALIZED VIEW IF NOT EXISTS catch_rate_by_bucket
    WITH (timescaledb.continuous) AS
    SELECT
        time_bucket('1 hour', ts) AS bucket,
        count(*) FILTER (WHERE subpopulation = 'under_triaged') AS under_total,
        count(*) FILTER (WHERE subpopulation = 'under_triaged' AND judge_flagged) AS under_judge_caught,
        count(*) FILTER (WHERE subpopulation = 'under_triaged' AND model_flagged) AS under_model_caught
    FROM decision_events
    GROUP BY bucket
    WITH NO DATA;
    """,
    """
    SELECT add_continuous_aggregate_policy('catch_rate_by_bucket',
        start_offset => INTERVAL '1 day',
        end_offset => INTERVAL '1 minute',
        schedule_interval => INTERVAL '1 minute',
        if_not_exists => TRUE);
    """,
]


def _trip_breaker(reason: str) -> None:
    global _DB_DOWN
    if not _DB_DOWN:
        print(f"[db] circuit breaker tripped — every further call this process uses the "
              f"in-memory fallback, no more connection attempts ({reason})", flush=True)
        observability.log_event("circuit breaker tripped", level="warning", reason=reason)
    _DB_DOWN = True


def _db_available() -> bool:
    return bool(TIGER_DSN) and not _DB_DOWN


def _tcp_reachable(timeout: float = CONNECT_TIMEOUT_S) -> bool:
    """Hard, OS-enforced socket-level deadline that's actually honored (unlike psycopg's
    connect_timeout in the filtered-port case) — always returns within `timeout` seconds."""
    parts = urlsplit(TIGER_DSN)
    host, port = parts.hostname, (parts.port or 5432)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def _connect():
    if not TIGER_DSN:
        raise RuntimeError("TIGER_DSN not set")
    if not _tcp_reachable():
        raise RuntimeError("TCP precheck failed (host unreachable or port filtered)")
    return psycopg.connect(TIGER_DSN, connect_timeout=CONNECT_TIMEOUT_S, autocommit=True)


def init_schema() -> bool:
    """Idempotent. Returns True if the real schema was created/confirmed, False if we
    couldn't/shouldn't reach the DB (caller should treat that as expected, not fatal)."""
    if DEMO_MODE:
        print("[db] DEMO_MODE — skipping schema init, no DB calls in the hot path", flush=True)
        return False
    if not _db_available():
        return False
    try:
        with _connect() as conn:
            for stmt in SCHEMA_STATEMENTS:
                conn.execute(stmt)
        return True
    except Exception as e:
        _trip_breaker(str(e))
        return False


def load_demo_snapshot() -> None:
    """Called once at startup when DEMO_MODE is on: takes ONE real read of the live
    Tiger board so the demo shows genuine numbers with zero further DB calls. If even
    this one bootstrap read fails, DEMO_MODE just serves the empty fallback board
    instead of crashing."""
    global _DEMO_SNAPSHOT
    if not DEMO_MODE:
        return
    if not TIGER_DSN:
        print("[db] DEMO_MODE but no TIGER_DSN — nothing to snapshot", flush=True)
        return
    try:
        _DEMO_SNAPSHOT = _read_leaderboard_from_db()
        print(f"[db] DEMO_MODE snapshot loaded: {_DEMO_SNAPSHOT['n_events']} events "
              f"across {_DEMO_SNAPSHOT['n_sessions']} sessions", flush=True)
    except Exception as e:
        print(f"[db] DEMO_MODE snapshot fetch failed, will serve fallback board instead ({e})", flush=True)


def log_event(session_id: str, case_id: str, subpopulation: str, judge_flagged: bool,
              model_flagged: bool, ts: datetime | None = None) -> None:
    ts = ts or datetime.now(timezone.utc)

    if DEMO_MODE or not _db_available():
        observability.log_event("fallback engaged", case_id=case_id,
                                 reason="demo_mode" if DEMO_MODE else "db_unavailable")
        _FALLBACK_EVENTS.append({
            "ts": ts, "session_id": session_id, "case_id": case_id, "subpopulation": subpopulation,
            "judge_flagged": judge_flagged, "model_flagged": model_flagged,
        })
        return

    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO decision_events (ts, session_id, case_id, subpopulation, judge_flagged, model_flagged) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (ts, session_id, case_id, subpopulation, judge_flagged, model_flagged),
            )
        try:
            # best-effort — a failed manual refresh must not undo the successful insert above,
            # and must not itself trip the breaker (the policy will catch up regardless)
            with _connect() as conn:
                conn.execute("CALL refresh_continuous_aggregate('catch_rate_by_bucket', NULL, NULL)")
        except Exception as e:
            print(f"[db] manual cagg refresh failed (non-fatal, policy will catch up): {e}", flush=True)
        return
    except Exception as e:
        _trip_breaker(str(e))
        observability.log_event("fallback engaged", case_id=case_id, reason="write_failed")

    _FALLBACK_EVENTS.append({
        "ts": ts, "session_id": session_id, "case_id": case_id, "subpopulation": subpopulation,
        "judge_flagged": judge_flagged, "model_flagged": model_flagged,
    })


def _catch_rate(events, flag_key: str) -> float | None:
    under = [e for e in events if e["subpopulation"] == "under_triaged"]
    if not under:
        return None
    return sum(1 for e in under if e[flag_key]) / len(under)


def _bucket_fallback(events) -> list[dict]:
    buckets: dict[datetime, dict] = {}
    for e in events:
        if e["subpopulation"] != "under_triaged":
            continue
        b = e["ts"].replace(minute=0, second=0, microsecond=0)
        row = buckets.setdefault(b, {"bucket": b, "under_total": 0, "under_judge_caught": 0, "under_model_caught": 0})
        row["under_total"] += 1
        row["under_judge_caught"] += int(e["judge_flagged"])
        row["under_model_caught"] += int(e["model_flagged"])
    return [buckets[b] for b in sorted(buckets)]


def _read_leaderboard_from_db() -> dict:
    """Raises on any failure — callers decide whether that means 'fall back' or
    'give up on the snapshot'. Never called while DEMO_MODE writes are active."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT count(*) FILTER (WHERE subpopulation='under_triaged' AND judge_flagged)::float "
            "       / NULLIF(count(*) FILTER (WHERE subpopulation='under_triaged'), 0) AS human_catch_rate, "
            "count(*) FILTER (WHERE subpopulation='under_triaged' AND model_flagged)::float "
            "       / NULLIF(count(*) FILTER (WHERE subpopulation='under_triaged'), 0) AS model_catch_rate, "
            "count(DISTINCT session_id) AS n_sessions, "
            "count(*) AS n_events "
            "FROM decision_events"
        ).fetchone()
        timeseries = conn.execute(
            "SELECT bucket, under_total, under_judge_caught, under_model_caught "
            "FROM catch_rate_by_bucket ORDER BY bucket"
        ).fetchall()
    return {
        "human_catch_rate": row[0],
        "model_catch_rate": row[1],
        "n_sessions": row[2],
        "n_events": row[3],
        "source": "tiger",
        "timeseries": [
            {"bucket": b.isoformat(), "under_total": t, "under_judge_caught": jc, "under_model_caught": mc}
            for b, t, jc, mc in timeseries
        ],
    }


def _fallback_leaderboard() -> dict:
    events = _FALLBACK_EVENTS
    return {
        "human_catch_rate": _catch_rate(events, "judge_flagged"),
        "model_catch_rate": _catch_rate(events, "model_flagged"),
        "n_sessions": len({e["session_id"] for e in events}),
        "n_events": len(events),
        "source": "fallback",
        "timeseries": [
            {"bucket": row["bucket"].isoformat(), "under_total": row["under_total"],
             "under_judge_caught": row["under_judge_caught"], "under_model_caught": row["under_model_caught"]}
            for row in _bucket_fallback(events)
        ],
    }


def get_leaderboard() -> dict:
    if DEMO_MODE:
        return _DEMO_SNAPSHOT if _DEMO_SNAPSHOT is not None else _fallback_leaderboard()

    if _db_available():
        try:
            return _read_leaderboard_from_db()
        except Exception as e:
            _trip_breaker(str(e))

    return _fallback_leaderboard()
