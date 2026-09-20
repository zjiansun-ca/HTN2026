"""Sentry: error monitoring + tracing + structured logs on the decision path.

Presentation/observability only — never touches the flag decision, the DB
fallback logic, or DEMO_MODE. Mirrors backend/db.py's philosophy: absent or
bad config must be a silent no-op, never a crash or a hang. If SENTRY_DSN is
unset, sentry_sdk itself makes every capture/logger call below a no-op with
no network activity, so call sites never need to check "is Sentry on?".
"""
import os

from dotenv import load_dotenv

load_dotenv()

SENTRY_DSN = os.environ.get("SENTRY_DSN", "").strip() or None

_ENABLED = False


def init_sentry() -> bool:
    """Idempotent. Returns True if Sentry was initialized, False otherwise
    (no DSN, or init itself failed) — either way the caller can proceed."""
    global _ENABLED
    if not SENTRY_DSN:
        print("[observability] no SENTRY_DSN — running without error monitoring/tracing", flush=True)
        return False
    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration

        sentry_sdk.init(
            dsn=SENTRY_DSN,
            traces_sample_rate=1.0,
            enable_logs=True,
            integrations=[StarletteIntegration(), FastApiIntegration()],
        )
        _ENABLED = True
        print("[observability] Sentry initialized (tracing + logs)", flush=True)
        return True
    except Exception as e:
        print(f"[observability] Sentry init failed, continuing without it: {e}", flush=True)
        return False


def log_event(message: str, level: str = "info", **fields) -> None:
    """Structured log on the decision path. A no-op (not an error) if Sentry
    isn't initialized — always safe to call unconditionally."""
    print(f"[decision] {message} {fields}", flush=True)
    if not _ENABLED:
        return
    try:
        from sentry_sdk import logger as sentry_logger

        emit = {"info": sentry_logger.info, "warning": sentry_logger.warning,
                "error": sentry_logger.error}.get(level, sentry_logger.info)
        emit(message, **fields)
    except Exception:
        pass  # observability must never break the decision path


def start_span(op: str, description: str):
    """Returns a Sentry span context manager, or a no-op context manager if
    Sentry isn't initialized — callers can always use it with `with`."""
    if _ENABLED:
        try:
            import sentry_sdk
            return sentry_sdk.start_span(op=op, description=description)
        except Exception:
            pass
    from contextlib import nullcontext
    return nullcontext()
