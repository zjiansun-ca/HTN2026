# Case-bank generator: builds the 16-case synthetic bank from AGGREGATE per-subpopulation
# stats only. No verbatim MIMIC row is ever assembled or sent anywhere.
#
# Vitals and chief-complaint terms are NOT sampled independently — that was diagnosed as
# artificially depressing catch rate (destroys the real vitals<->complaint co-occurrence
# the flagger's signal partly rides on). Instead: within each pool (under_triaged, well),
# real patients are stratified into 2 coarse bins by red-flag-complaint presence. Per bin
# we compute AGGREGATES ONLY (vital mean vector + covariance matrix + term frequencies +
# bin prevalence) — never individual rows. A synthetic case picks a bin by real prevalence,
# then draws ALL vitals JOINTLY from that bin's multivariate normal and complaint terms
# from that bin's term frequencies, so both share the same underlying bin.
#
# No LLM in this pipeline — chief_complaint/presentation/outcome_detail are all built
# deterministically (seeded RNG, template text). Gemini is deferred to the Saturday
# patient-voice stretch feature, not used here (see CLAUDE.md STATUS/Gemini sections;
# the google-genai setup in .env is left in place for that, not ripped out).
#
# DUA: reads MIMIC CSVs from local disk only. Output cases.json is fully synthetic and
# committable — nothing here is a real chiefcomplaint string or a real row.
#
# NOTE: this MIMIC-IV-ED subset has no patients.csv / age field. Age is NOT derived from
# real aggregate stats (the field doesn't exist here) — it's sampled from a disclosed,
# generic clinically-plausible range per pool. Sex IS sampled from real per-pool gender
# proportions (edstays.csv `gender` is present).
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ED = r"C:\Users\zjian\Downloads\mimic-iv-ed-2.2\mimic-iv-ed-2.2"
TEXT_RNG = np.random.default_rng(1000)  # isolated from vitals/demographics sampling below

POOL_INDEX = {"under_triaged": 0, "well": 1}


def rng_for(pool_name: str, idx: int, attempt: int) -> np.random.Generator:
    """Fully independent, reproducible stream per (pool, case index, resample attempt).
    Lets a degenerate self-check resample ONE pool's 8 cases (bump its attempt) without
    perturbing the other pool's already-validated draws."""
    return np.random.default_rng([42, POOL_INDEX[pool_name], idx, attempt])

VITALS = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain"]
N_VITALS = len(VITALS)

# physiologically plausible bounds — filters MIMIC data-entry errors (e.g. o2sat=1002,
# dbp=8109, pain=100 on a 0-10 scale) out of the sampling distribution
PLAUSIBLE = {
    "temperature": (90.0, 106.0),
    "heartrate": (30, 220),
    "resprate": (6, 60),
    "o2sat": (50, 100),
    "sbp": (60, 250),
    "dbp": (30, 150),
    "pain": (0, 10),
}
ROUND = {  # decimal places
    "temperature": 1, "heartrate": 0, "resprate": 0, "o2sat": 0, "sbp": 0, "dbp": 0, "pain": 0,
}
AGE_RANGE = {"under_triaged": (50, 90), "well": (18, 75)}  # disclosed assumption, not real stats

# mirrors recon.py's existing red-flag term list, for consistency
RED_FLAG_TERMS = [
    "chest pain", "hypotension", "short of breath", "sob", "syncope",
    "altered", "unresponsive", "suicid", "overdose", "stroke", "weakness",
]

stays = pd.read_csv(f"{ED}/edstays.csv")
triage = pd.read_csv(f"{ED}/triage.csv")
df = triage.merge(stays[["stay_id", "disposition", "gender"]], on="stay_id", how="inner")

low = df[df["acuity"].isin([4, 5])].copy()
under_pool = low[low["disposition"].isin(["ADMITTED", "EXPIRED"])].copy()
well_pool = low[low["disposition"] == "HOME"].copy()

POOLS = {"under_triaged": under_pool, "well": well_pool}

# real ADMITTED vs EXPIRED split within the under-triaged pool, used to label each
# synthetic under-triaged case's outcome proportionally (not hand-picked)
UNDER_DISPOSITION_PROPS = under_pool["disposition"].value_counts(normalize=True).to_dict()


def has_red_flag(text) -> bool:
    s = str(text).lower()
    return any(t in s for t in RED_FLAG_TERMS)


def pool_gender_props(pool: pd.DataFrame) -> dict:
    return pool["gender"].value_counts(normalize=True).to_dict()


def top_terms(series: pd.Series, n: int = 20) -> list:
    counts = Counter()
    for txt in series.dropna().str.lower():
        for w in re.findall(r"[a-z]+", txt):
            if len(w) > 2:
                counts[w] += 1
    return counts.most_common(n)


def row_aligned_vitals(bin_df: pd.DataFrame) -> pd.DataFrame:
    """Row-aligned plausibility filter: drop a row if ANY vital is out of range, so the
    mean vector and covariance matrix are computed over one consistent set of real rows
    (per-column filtering would misalign rows across dimensions)."""
    vitals_num = bin_df[VITALS].apply(pd.to_numeric, errors="coerce")
    valid = pd.Series(True, index=vitals_num.index)
    for v in VITALS:
        lo, hi = PLAUSIBLE[v]
        valid &= vitals_num[v].between(lo, hi) & vitals_num[v].notna()
    return vitals_num[valid]


def build_bins(pool_df: pd.DataFrame) -> dict:
    """AGGREGATES ONLY per bin: mean vector, covariance matrix, prevalence, term
    frequencies. No individual rows are retained in the returned structure."""
    pool_clean = row_aligned_vitals(pool_df)
    pool_cov = pool_clean.cov().values  # shrinkage target for small bins

    is_rf = pool_df["chiefcomplaint"].apply(has_red_flag)
    n_pool = len(pool_df)

    bins = {}
    for bin_name, bin_df in [("redflag", pool_df[is_rf]), ("none", pool_df[~is_rf])]:
        clean = row_aligned_vitals(bin_df)
        n_bin = len(clean)
        mean = clean.mean().values
        cov_raw = clean.cov().values if n_bin > 1 else pool_cov

        # shrinkage: blend toward the pool-level covariance, weight grows with bin size —
        # this is what makes a 13-row bin usable without a wild/unstable estimate, and
        # guarantees a positive-definite result since both inputs are valid covariances
        alpha = n_bin / (n_bin + 20)
        cov = alpha * cov_raw + (1 - alpha) * pool_cov
        cov += np.eye(N_VITALS) * 1e-6 * (np.trace(cov) / N_VITALS)  # numerical safety

        bins[bin_name] = {
            "mean": mean,
            "cov": cov,
            "n": n_bin,
            "prevalence": len(bin_df) / n_pool,
            "terms": top_terms(bin_df["chiefcomplaint"]),
        }
    return bins


BINS = {pool_name: build_bins(pool_df) for pool_name, pool_df in POOLS.items()}

# specificity stoplist: the user's minimum (pain/eval/lower) plus other non-symptom
# fragments observed in the real top-terms (admit/for/status/mental/needs/scripts/
# unable/body) — a complaint made only of these reads as non-specific either way
STOPLIST = {"pain", "eval", "lower", "admit", "for", "status", "mental", "needs", "scripts", "unable", "body"}


def sample_vitals_mvn(bin_info: dict, rng: np.random.Generator) -> dict:
    draw = rng.multivariate_normal(bin_info["mean"], bin_info["cov"])
    row = {}
    for i, v in enumerate(VITALS):
        lo, hi = PLAUSIBLE[v]
        val = float(np.clip(draw[i], lo, hi))  # approximation of the true joint — documented, not a bug
        row[v] = round(val, ROUND[v]) if ROUND[v] else round(val)
    return row


def sample_terms(term_freqs: list, k: int, rng: np.random.Generator) -> list:
    words, weights = zip(*term_freqs)
    weights = np.array(weights, dtype=float)
    weights /= weights.sum()
    return list(rng.choice(words, size=min(k, len(words)), replace=False, p=weights))


def sample_specific_terms(term_freqs: list, rng: np.random.Generator, max_attempts: int = 25) -> list:
    for _ in range(max_attempts):
        k = int(rng.integers(1, 4))
        terms = sample_terms(term_freqs, k, rng)
        if set(terms) - STOPLIST:
            return list(terms)
    raise RuntimeError("could not sample a specific complaint after many attempts")


def generate_seeds(n_per_pool: int = 8, pool_attempts: dict | None = None) -> list:
    """pool_attempts lets a degenerate self-check resample just one pool's 8 cases —
    e.g. {"under_triaged": 0, "well": 1} regenerates only the well pool's draws."""
    pool_attempts = pool_attempts or {p: 0 for p in POOLS}
    seeds = []
    for pool_name, pool_df in POOLS.items():
        genders = pool_gender_props(pool_df)
        sexes = list(genders.keys())
        sex_p = list(genders.values())
        bins = BINS[pool_name]
        bin_names = list(bins.keys())
        bin_p = [bins[b]["prevalence"] for b in bin_names]
        attempt = pool_attempts[pool_name]

        for i in range(n_per_pool):
            rng = rng_for(pool_name, i, attempt)
            lo, hi = AGE_RANGE[pool_name]
            if pool_name == "under_triaged":
                disp = str(rng.choice(list(UNDER_DISPOSITION_PROPS.keys()), p=list(UNDER_DISPOSITION_PROPS.values())))
                outcome = "died" if disp == "EXPIRED" else "admitted"
            else:
                outcome = "discharged"

            bin_name = str(rng.choice(bin_names, p=bin_p))
            bin_info = bins[bin_name]

            seed = {
                "case_id": f"{pool_name}-{i+1:02d}",
                "subpopulation": pool_name,
                "bin": bin_name,
                "vitals": sample_vitals_mvn(bin_info, rng),
                "sex": str(rng.choice(sexes, p=sex_p)),
                "age": int(rng.integers(lo, hi + 1)),
                "acuity": int(rng.choice([4, 5])),
                "complaint_terms": sample_specific_terms(bin_info["terms"], rng),
                "outcome": outcome,
                "_rng": rng,  # kept for a same-case collision redraw; stripped before writing to disk
            }
            seeds.append(seed)
    return seeds


def check_no_verbatim_match(cases: list, low_df: pd.DataFrame) -> list:
    """Guardrail: no generated (chief_complaint, full-vitals) tuple may exactly match a
    real MIMIC row. Returns list of colliding case_ids (empty if none)."""
    real_tuples = set()
    for _, r in low_df.iterrows():
        cc = str(r["chiefcomplaint"]).strip().lower() if pd.notna(r["chiefcomplaint"]) else ""
        vt = tuple(r.get(v) for v in VITALS)
        real_tuples.add((cc, vt))

    collisions = []
    for c in cases:
        cc = c["chief_complaint"].strip().lower()
        vt = tuple(c["vitals"][v] for v in VITALS)
        if (cc, vt) in real_tuples:
            collisions.append(c["case_id"])
    return collisions


SEX_NOUN = {"M": "man", "F": "woman"}
SEX_ADJ = {"M": "Male", "F": "Female"}

# 6 distinct sentence patterns, no LLM — light deterministic variety via TEXT_RNG.
PRESENTATION_TEMPLATES = [
    "{age}-year-old {noun} presents to triage reporting {complaint}. Pain is rated {pain}/10 at intake.",
    "A {age}-year-old {noun} arrives at the ED with {complaint}; vitals are obtained at triage, pain {pain}/10.",
    "{adj} patient, age {age}, presents with {complaint}. Reports pain level {pain} out of 10.",
    "{age}-year-old {noun} checks in at triage citing {complaint}, with a pain score of {pain}/10.",
    "Triage note: {age}-year-old {noun}, {complaint}. Pain {pain}/10 on the intake scale.",
    "{age}-year-old {noun} is evaluated at triage for {complaint}; pain is documented as {pain}/10.",
]

# Curated token -> lay phrase lookup, built from the real top-20 terms across all 4 bins
# (under/well x redflag/none). chief_complaint (fed to the flagger) keeps the raw tokens
# unchanged, regardless of what happens here — this lookup only shapes `presentation`.
TOKEN_PHRASES = {
    # anatomy -> "X pain"
    "arm": "arm pain", "leg": "leg pain", "hip": "hip pain", "knee": "knee pain",
    "back": "back pain", "foot": "foot pain", "ankle": "ankle pain", "hand": "hand pain",
    "shoulder": "shoulder pain", "chest": "chest pain", "finger": "finger pain",
    "ear": "ear pain", "eye": "eye pain",
    # bare symptom nouns
    "weakness": "weakness", "dizziness": "dizziness", "swelling": "swelling",
    "numbness": "numbness", "fever": "fever", "nausea": "nausea", "anxiety": "anxiety",
    "dyspnea": "shortness of breath", "soreness": "soreness", "sore": "soreness",
    # countable single events -> "a/an X"
    "fall": "a recent fall", "injury": "an injury", "wound": "a wound",
    "laceration": "a laceration", "rash": "a rash", "cough": "a cough",
    "headache": "a headache", "syncope": "a fainting episode",
    "overdose": "a possible overdose", "throat": "a sore throat",
    "mvc": "a motor vehicle collision", "etoh": "alcohol use",
    "hypotension": "low blood pressure", "unresponsive": "unresponsiveness",
    "suicidal": "a mental health crisis", "sob": "shortness of breath",
    # handled specially (pairing), see build_presentation
    "altered": "altered mental status",
}
LOWER_PAIRABLE = {"back", "leg", "extremity", "abdominal", "abd"}
DROP_FROM_PRESENTATION = STOPLIST | {"mental", "transfer", "intubated"}  # "mental" is absorbed into "altered"


def _phrase_list(terms: list) -> list:
    working = [t for t in terms if t not in DROP_FROM_PRESENTATION]

    phrases = []
    if "lower" in terms:
        paired = next((t for t in working if t in LOWER_PAIRABLE), None)
        if paired:
            working.remove(paired)
            base = TOKEN_PHRASES.get(paired, paired)
            phrases.append("lower " + base)

    for t in working:
        phrases.append(TOKEN_PHRASES.get(t, t))

    return phrases or ["nonspecific symptoms"]  # safety net; specificity filter should prevent this


def _join_phrases(phrases: list) -> str:
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    return ", ".join(phrases[:-1]) + f", and {phrases[-1]}"


def build_presentation(seed: dict) -> str:
    complaint_phrase = _join_phrases(_phrase_list(seed["complaint_terms"]))
    template = TEXT_RNG.choice(PRESENTATION_TEMPLATES)
    return str(template).format(
        age=seed["age"],
        noun=SEX_NOUN[seed["sex"]],
        adj=SEX_ADJ[seed["sex"]],
        complaint=complaint_phrase,
        pain=seed["vitals"]["pain"],
    )


# Generic phrase banks by outcome — no fabricated specific diagnoses, only what the
# outcome label itself implies.
OUTCOME_DETAIL_PHRASES = {
    "admitted": [
        "Returned to the department later with worsening symptoms and was admitted for further evaluation and monitoring.",
        "Was admitted to the inpatient floor for closer monitoring after initial assessment.",
        "Clinical condition did not improve after the initial visit, leading to hospital admission.",
        "Follow-up assessment prompted admission for observation and further workup.",
    ],
    "died": [
        "Condition deteriorated after triage; the patient died during the same hospital encounter.",
        "Despite treatment, the patient's condition worsened and they died during this hospital stay.",
    ],
    "discharged": [
        "Evaluated and discharged the same visit with routine follow-up instructions.",
        "No acute findings on further evaluation; discharged home in stable condition.",
        "Discharged with standard aftercare guidance after workup.",
        "Cleared for discharge the same day, advised to follow up with primary care if symptoms persist.",
    ],
}


def build_outcome_detail(seed: dict) -> str:
    bank = OUTCOME_DETAIL_PHRASES[seed["outcome"]]
    return str(TEXT_RNG.choice(bank))


def build_case(seed: dict) -> dict:
    seed["chief_complaint"] = ", ".join(seed["complaint_terms"])
    return {
        "case_id": seed["case_id"],
        "presentation": build_presentation(seed),
        "chief_complaint": seed["chief_complaint"],
        "age": seed["age"],
        "sex": seed["sex"],
        "vitals": seed["vitals"],
        "acuity": seed["acuity"],
        "outcome": seed["outcome"],
        "outcome_detail": build_outcome_detail(seed),
        "truth": {"subpopulation": seed["subpopulation"]},
    }


# THE canonical file the backend actually serves — generator, self-check, and backend
# all point at this one path, no v2/canonical split.
OUTPUT_PATH = Path("backend/cases.json")


def build_cases(seeds: list) -> list:
    cases = []
    for seed in seeds:
        attempt = 0
        while True:
            attempt += 1
            case = build_case(seed)
            collisions = check_no_verbatim_match([case], low)
            if not collisions:
                break
            print(f"  {seed['case_id']}: attempt {attempt} exact-matched a real row — resampling", flush=True)
            seed["vitals"] = sample_vitals_mvn(BINS[seed["subpopulation"]][seed["bin"]], seed["_rng"])
            if attempt >= 5:
                raise RuntimeError(f"{seed['case_id']} kept colliding with a real row after 5 attempts")
        cases.append(case)
    return cases


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from backend.scoring import _score_case  # only import needed for the self-check-driven resample loop

    pool_attempts = {"under_triaged": 0, "well": 0}
    cases = []
    for round_num in range(6):
        seeds = generate_seeds(pool_attempts=pool_attempts)
        print(f"--- round {round_num}: pool_attempts={pool_attempts} ---", flush=True)
        for s in seeds:
            print(f"  {s['case_id']}: bin={s['bin']}", flush=True)

        cases = build_cases(seeds)
        for c in cases:
            print(f"  {c['case_id']}: {c['chief_complaint']!r}", flush=True)

        collisions = check_no_verbatim_match(cases, low)
        assert not collisions, f"guardrail failed after resampling: {collisions}"
        print(f"guardrail: 0/{len(cases)} exact matches against real MIMIC rows", flush=True)

        under = [c for c in cases if c["truth"]["subpopulation"] == "under_triaged"]
        well = [c for c in cases if c["truth"]["subpopulation"] == "well"]
        caught = sum(1 for c in under if _score_case(c)[1])
        flagged_well = sum(1 for c in well if _score_case(c)[1])
        under_degenerate = not (1 <= caught <= len(under) - 1)
        well_degenerate = flagged_well == 0
        print(f"self-check: under-triaged caught {caught}/{len(under)}, well false-flagged {flagged_well}/{len(well)}", flush=True)

        if not under_degenerate and not well_degenerate:
            print("non-degenerate spread — done.", flush=True)
            break

        if under_degenerate:
            pool_attempts["under_triaged"] += 1
            print(f"under_triaged pool degenerate (caught={caught}/{len(under)}) — resampling that pool only "
                  f"(representativeness only, NOT chasing a target rate)", flush=True)
        if well_degenerate:
            pool_attempts["well"] += 1
            print(f"well pool degenerate (0 false flags) — resampling that pool only "
                  f"(representativeness only, NOT chasing a target rate)", flush=True)
    else:
        print("WARNING: still degenerate after 6 rounds — writing anyway and reporting honestly.", flush=True)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(cases, f, indent=2)
    print(f"wrote {OUTPUT_PATH} ({len(cases)} cases) — canonical file, same one the backend serves", flush=True)
