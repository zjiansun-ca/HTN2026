"""Seeds the leaderboard so it isn't empty at demo time. Requires TIGER_DSN — with no DB
configured, db.log_event() just appends to a fallback list scoped to THIS process and
nothing persists, so running this script without a real Tiger Data connection is a no-op
in practice (useful only to sanity-check the simulation logic itself).

Honest framing, not fabrication:
- model_flagged is the REAL, LOCKED model's real behavior on the real 16-case bank —
  computed live via backend.scoring._score_case, the exact same function /reveal uses.
  Never invented, never tuned to look good.
- judge_flagged is a SEEDED, CLEARLY-LABELED SIMULATION of a plausibly-imperfect human
  judge (catches most but not all under-triaged cases, occasionally over-flags a well
  case) — not real human performance data. Real judge data accumulates from actual
  playthroughs of the running app; this script exists only so the leaderboard has
  something to show before that happens.
"""
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from backend import db
from backend.main import CASE_BANK
from backend.scoring import _score_case

SEED_RNG = np.random.default_rng(7)

# plausibly-imperfect simulated judge: catches most under-triaged cases but not all,
# rarely over-flags an obviously-well case — NOT calibrated to match or beat the model
JUDGE_CATCH_PROB_UNDER = 0.75
JUDGE_FALSE_FLAG_PROB_WELL = 0.15


def simulate_judge_call(subpopulation: str) -> bool:
    if subpopulation == "under_triaged":
        return bool(SEED_RNG.random() < JUDGE_CATCH_PROB_UNDER)
    return bool(SEED_RNG.random() < JUDGE_FALSE_FLAG_PROB_WELL)


def seed(n_sessions: int = 8):
    print(f"seeding {n_sessions} simulated sessions x {len(CASE_BANK)} cases "
          f"({'live Tiger Data' if db.TIGER_DSN else 'NO TIGER_DSN — this will be a no-op'})")

    now = datetime.now(timezone.utc)
    for session_num in range(n_sessions):
        session_id = f"seed-{uuid.uuid4()}"
        # spread sessions over the past several hours so the hourly-bucketed chart has
        # more than one point to show
        session_ts = now - timedelta(hours=(n_sessions - session_num) * 2)

        for case in CASE_BANK:
            subpop = case["truth"]["subpopulation"]
            _, model_flagged, _ = _score_case(case)  # the real, locked model — never faked
            judge_flagged = simulate_judge_call(subpop)

            db.log_event(
                session_id=session_id,
                case_id=case["case_id"],
                subpopulation=subpop,
                judge_flagged=judge_flagged,
                model_flagged=model_flagged,
                ts=session_ts,
            )
        print(f"  session {session_num+1}/{n_sessions} ({session_id}) logged at {session_ts.isoformat()}")

    result = db.get_leaderboard()
    print(f"\nleaderboard after seeding (source={result['source']}):")
    print(f"  human catch rate: {result['human_catch_rate']}")
    print(f"  model catch rate: {result['model_catch_rate']}")
    print(f"  n_sessions={result['n_sessions']}  n_events={result['n_events']}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    seed(n)
