"""Converts a case's raw risk score into its percentile within the held-out
non-urgent (ESI 4-5) MIMIC population — presentation only. Never influences the
flag decision, which is already fixed by backend/scoring.py's locked P20
threshold before this is ever called.

Derived from artifacts/risk_percentiles.json (data_prep/compute_risk_percentiles.py),
an aggregate array of anonymous risk scores from the same held-out split
eval_undertriage.py used — no patient records, same governance category as
artifacts/eval_summary.json.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

with open(ROOT / "artifacts" / "risk_percentiles.json") as f:
    _data = json.load(f)
_SORTED_SCORES = np.array(_data["sorted_scores"])
N = _data["n"]


def percentile_for(risk_rank: float) -> float:
    """Returns this score's percentile (0-100) within the held-out non-urgent
    population — e.g. 93.2 means it scores higher than 93.2% of that
    population. Honest interpolation against the real distribution, never
    invented; a risk_rank outside the observed range clamps to 0 or 100."""
    idx = np.searchsorted(_SORTED_SCORES, risk_rank, side="right")
    return round(100.0 * idx / N, 1)
