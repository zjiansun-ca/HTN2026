import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.scoring import MEDIANS, VITALS, _score_case


def test_pain_name_collision_does_not_misattribute():
    # "pain" is both a numeric vital column and a common word in chief-complaint text.
    # Force both to contribute meaningfully: push the pain-intensity vital far from its
    # median, and make the chief complaint a single-word "pain" (maximizes that token's
    # tfidf weight). Both should surface in top_features as separate, correctly-kind-tagged
    # entries — never collapsed or misattributed to the wrong kind.
    case = {
        "vitals": {v: MEDIANS[v] for v in VITALS},
        "chief_complaint": "pain",
    }
    case["vitals"]["pain"] = 10

    _, _, top_features = _score_case(case)

    pain_entries = [f for f in top_features if f["feature"] == "pain"]
    assert len(pain_entries) == 2, f"expected both the vital and text 'pain' to surface, got {pain_entries}"

    kinds = {f["kind"] for f in pain_entries}
    assert kinds == {"vital", "text"}, f"expected one vital and one text entry, got kinds {kinds}"

    for f in top_features:
        assert f["kind"] in ("vital", "text")
