import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import observability


def test_init_is_noop_without_dsn(monkeypatch):
    monkeypatch.setattr(observability, "SENTRY_DSN", None)
    monkeypatch.setattr(observability, "_ENABLED", False)

    result = observability.init_sentry()

    assert result is False
    assert observability._ENABLED is False


def test_log_event_never_raises_without_sentry(monkeypatch):
    monkeypatch.setattr(observability, "_ENABLED", False)
    # must not raise even though nothing is initialized
    observability.log_event("test message", level="info", foo="bar")


def test_start_span_is_usable_without_sentry(monkeypatch):
    monkeypatch.setattr(observability, "_ENABLED", False)
    with observability.start_span(op="test", description="test"):
        pass  # must not raise
