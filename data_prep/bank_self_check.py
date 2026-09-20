# THE GATE: runs the real standardized flagger over the finished 16-case bank and reports
# catches/misses/false-flags, classifying every well-case flag as DEFENSIBLE (a real vital
# crossing a clinical threshold is among the dominant positive contributors) or ARTIFACT
# (the positive drivers are text tokens or normal-value vitals). Asserts a non-degenerate
# spread. Never hand-tunes a case to change whether it flags — a degenerate spread may only
# be fixed by resampling the offending POOL from its own aggregate distribution again.
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.scoring import _score_case  # noqa: E402  (exact same scoring path as /reveal, no web framework needed)

# same clinical-threshold table the frontend uses for qualifiers — kept in sync by hand,
# both are small and reviewed together
THRESHOLDS = {
    "sbp": lambda v: v < 90,
    "heartrate": lambda v: v > 100,
    "o2sat": lambda v: v < 92,
    "resprate": lambda v: v > 20,
    "temperature": lambda v: v > 100.4 or v < 96.8,
}


def classify(case: dict, top_features: list) -> str:
    positives = [f for f in top_features[:3] if f["contribution"] > 0]
    for f in positives:
        if f["kind"] == "vital":
            check = THRESHOLDS.get(f["feature"])
            if check and check(case["vitals"][f["feature"]]):
                return "DEFENSIBLE"
    return "ARTIFACT"


def run(bank_path: str = "backend/cases.json"):  # canonical file — same one the backend serves
    with open(bank_path) as f:
        cases = json.load(f)

    under = [c for c in cases if c["truth"]["subpopulation"] == "under_triaged"]
    well = [c for c in cases if c["truth"]["subpopulation"] == "well"]

    print(f"=== Under-triaged pool (n={len(under)}) ===")
    caught, missed = [], []
    for c in under:
        risk, flag, top = _score_case(c)
        (caught if flag else missed).append(c["case_id"])
        status = "CAUGHT" if flag else "MISSED"
        print(f"  {c['case_id']:20s} risk={risk:.3f}  {status}")
    print(f"  -> {len(caught)}/{len(under)} caught, {len(missed)}/{len(under)} missed")

    print(f"\n=== Well pool (n={len(well)}) ===")
    flagged_rows = []
    n_flagged = 0
    for c in well:
        risk, flag, top = _score_case(c)
        if flag:
            n_flagged += 1
            cls = classify(c, top)
            flagged_rows.append((c["case_id"], risk, top, cls))
            print(f"  {c['case_id']:20s} risk={risk:.3f}  FLAGGED  [{cls}]")
            for f in top:
                print(f"      {f['kind']:5s} {f['feature']!r:20s} contribution={f['contribution']:+.3f}")
        else:
            print(f"  {c['case_id']:20s} risk={risk:.3f}  not flagged")
    print(f"  -> {n_flagged}/{len(well)} false-flagged "
          f"({sum(1 for r in flagged_rows if r[3]=='DEFENSIBLE')} defensible, "
          f"{sum(1 for r in flagged_rows if r[3]=='ARTIFACT')} artifact)")

    # non-degenerate spread: some caught AND some missed among under-triaged;
    # at least one false flag among well (this operating point is known to false-flag —
    # zero would itself be a sign the sample isn't representative)
    degenerate = not (1 <= len(caught) <= len(under) - 1) or n_flagged == 0
    return {
        "under_caught": caught,
        "under_missed": missed,
        "well_flagged": flagged_rows,
        "degenerate": degenerate,
    }


if __name__ == "__main__":
    result = run()
    if result["degenerate"]:
        print("\nDEGENERATE SPREAD — resample the offending pool (representativeness only, not to hit a target rate).")
        sys.exit(1)
    print("\nnon-degenerate spread confirmed.")
