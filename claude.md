# CLAUDE.md — Triage Second Reader (HTN 2026)

## What this is
An interactive triage **second reader**. A judge sits in the nurse's seat: for each
patient the nurse already triaged NON-URGENT, the judge decides "keep in waiting room"
or "pull back for a second look." Then the tool reveals the model's under-triage flag
(with a faithful reason), the real outcome, and a live human-vs-model leaderboard.
Decision support — NOT nurse replacement. Research prototype — NOT medical advice.

## ⚠️ NON-NEGOTIABLE — DATA GOVERNANCE (DUA)
Model trained on MIMIC-IV-ED under a PhysioNet credentialed DUA. Violating this risks
the user's credentialing.
- No raw MIMIC record EVER leaves the machine — not to Gemini, Tiger Data Cloud, or a repo.
- The case-bank generator draws profiles from AGGREGATE per-subpopulation stats, never
  verbatim rows. Gemini receives only synthetic structured seeds — never a real record.
- `.gitignore` MUST exclude `data_prep/*.csv*` and `artifacts/`. `cases.json` is fully
  synthetic and IS committable. Commit code that regenerates the model, never the model.
- If a task would send MIMIC anywhere external, STOP and flag it.

## STATUS — LOCKED (model + bank). No further iteration on either without explicit sign-off.
- `artifacts/flagger.joblib` = {model (LogReg), tfidf, vitals, medians, scaler}; `thresholds.json` = {p20}.
  Trained on the low-acuity (ESI 4–5) subpopulation, numeric vitals standardized (StandardScaler
  fit on train split only) — `/reveal`/`backend/scoring.py` applies the same scaler at serve time.
  **Final numbers**: AUC 0.786 (text+vitals) / 0.633 (vitals-only). p20 = 0.5858. Held-out catch
  rate 58.7% (84/143). Threshold is FIXED — the demo does not tune it.
- The model's prediction is ALWAYS computed live from the flagger — never stored in cases.json.
- tfidf/`vitals` name-collision (e.g. `"pain"` as both a vital column and a text token) is fixed:
  `top_features` entries carry `kind: "vital"|"text"`, computed in `backend/scoring.py` (no FastAPI
  dependency) and rendered by `kind` in the frontend — never by string-matching a name. Imported by
  both `backend/main.py` and `data_prep/bank_self_check.py`, so the self-check gate scores
  byte-identically to `/reveal` without a running server.
- `backend/cases.json` is THE canonical 16-case bank (8 under-triaged, 8 well) — the same file the
  generator, self-check, and backend all read; no stub/v2 split. Built by
  `data_prep/generate_case_bank.py`, no LLM. Vitals are sampled from per-bin multivariate-normal
  distributions (bins = red-flag-complaint presence within each pool), conditioning vitals on
  complaint type instead of sampling them independently — aggregates only (mean vector +
  covariance + term frequencies + prevalence per bin), never verbatim rows. Complaint terms pass a
  minimum-specificity filter (no bare "pain"/"eval"/"lower"). `presentation` comes from a curated
  token→phrase lookup + seeded sentence templates; `chief_complaint` stays the raw sampled tokens
  (unchanged register, feeds the flagger). Verbatim-match guardrail against real MIMIC rows: 0/16
  collisions.
  **Final self-check** (`data_prep/bank_self_check.py`): 4/8 under-triaged caught, 4/8 missed;
  3/8 well cases false-flag, all 3 classified ARTIFACT (none DEFENSIBLE) — driven by text tokens
  or non-crossing vitals, never a real threshold-crossing vital. Non-degenerate, reviewed, locked.
- **Known, accepted limitations (README material, NOT bugs to fix):**
  (a) the `'fall'`/`'pain fall'` text tokens carry a positive coefficient large enough that benign
  mechanical falls in the well pool can false-flag on text alone — a real, disclosed failure mode
  of a text-heavy linear model, kept on purpose rather than papered over.
  (b) per-bin multivariate-normal + clip-to-plausible-range is an approximation of the true joint
  vital distribution, not an exact resample of it — documented, not a defect.
  (c) some fitted coefficient signs are clinically counterintuitive (e.g. low SBP reading as
  protective in places) — a known class-imbalance/retrospective-data confound, not a scoring bug.
- **Gemini reconciliation (built, guarded):** `backend/explain.py` + `POST /explain` add an
  OPTIONAL plain-language paraphrase of the flag, kept strictly out of the decision/explanation
  path this section otherwise protects. The faithful reason (contribution bars, `backend/scoring.py`,
  TEMPLATE-rendered) remains the sole source of truth and is computed and shown with zero LLM
  involvement, exactly as below. Gemini is called separately, after the flag/score already exist,
  sees ONLY the already-computed top_features (name/kind/direction + shown vital values) — never
  chief-complaint text, presentation, outcome, or a real record — and is instructed to add no new
  facts. Any failure/timeout/quota/absent-key silently returns `null`; the bars alone remain the
  reveal either way. This does not reopen the "no LLM in the decision path" rule below — Gemini
  can only restate a decision already made, never make or explain-from-scratch one. The Saturday
  patient-voice stretch feature (voice/speech) is a separate, still-undone idea and still conflicts
  with OUT OF SCOPE below; reconcile that one separately if it's ever built.

## Architecture (all local at demo time)
Case (synthetic) ──► [Flagger] risk → flag (vs fixed p20) ──► faithful reason (top features)
Judge's binary call ──► compare to model + real outcome ──► log event ──► leaderboard
- Faithful reason = the model's real top contributing features (coef × value), rendered by
  TEMPLATE. No LLM in the flag/score decision, and none in the faithful reason itself — that stays
  template-rendered and is the reveal's source of truth. An optional Gemini paraphrase of that
  already-computed reason exists on top (`/explain`), strictly guarded; see STATUS above.
- The case bank is built entirely deterministically (seeded RNG + templates) — no LLM anywhere in
  bank generation.

## cases.json schema
Every case MUST have: `case_id, presentation, chief_complaint, vitals {temperature, heartrate,
resprate, o2sat, sbp, dbp, pain}, acuity, age, sex, outcome, outcome_detail, truth.subpopulation
("under_triaged"|"well")`. `age`/`sex`/`outcome_detail` are display-only — passed through by the
API verbatim, never fed to the flagger. `outcome` holds the literal disposition-style value
("admitted"/"died"/"discharged"); the binary classification lives in `truth.subpopulation`. All
fields are built deterministically by `data_prep/generate_case_bank.py` (no LLM) — a case missing
any of these renders as `undefined` in the frontend.

## Stack
- Backend: Python + FastAPI. Frontend: single-file HTML/JS (no build step).
- DB: Tiger Data (Postgres + TimescaleDB) via psycopg. OS: Windows/PowerShell.
- Secrets in `.env` (gitignored): GEMINI_API_KEY, TIGER_DSN.

## Gemini — IMPORTANT
- SDK: `google-genai` (`from google import genai`). Legacy `google-generativeai` is SUNSET.
- Structured output via Pydantic `response_schema`. Model: `gemini-2.5-flash` (or current flash).
- `chief_complaint` output MUST be terse, ED-note style ("weakness, dizziness") to match the
  register the flagger's TF-IDF was trained on. Verbose prose silently breaks the flag.

## Build order — walking skeleton FIRST, one milestone at a time, git commit each
1. FastAPI: /next-case, /submit (logs judge call), /reveal (LIVE flagger flag + template reason
   + outcome), /leaderboard (stub). Real flagger wired. 2-case hardcoded stub bank. Smoke test.
2. Single-file frontend: case → keep/flag → three-way reveal. End-to-end. (Milestone: works.)
3. `data_prep/eval_undertriage.py` (real AUC + p20 catch-rate on held-out MIMIC) AND the
   case-bank generator → real 16-case cases.json (8 under-triaged + 8 well, Gemini narratives).
   SELF-CHECK: run the flagger over the finished bank; assert a non-degenerate spread
   (some caught, ≥1 missed, ≥1 false flag). Resample if degenerate.
4. Distribution-match panel: synthetic-vs-real histograms (acuity, SBP, HR, SpO2, temp, admit
   rate) → a figure. The "fake but faithful" proof.
5. Tiger Data: hypertable of {session, case_id, judge_call, model_flag, outcome, ts} +
   continuous aggregate → /leaderboard (human vs model catch-rate) + leaderboard UI.
6. Polish, README (problem, method, real numbers, limitations, clinical-translation path),
   demo fallback (canned run if a service dies), rehearse.

## Conventions
- Keep it simple; ships in hours. Smoke test after each milestone; run it.
- Tests required: the template explanation (asserts it names only real top features), and the
  bank self-check.

## OUT OF SCOPE — do not build
- Voice / speech / live translation. GPTZero. Patient routing. Wait-time simulation.
- Any LLM in the flag decision itself, or generating the faithful reason from scratch (that stays
  template-rendered off the model's real coefficients). The one built exception — an optional
  Gemini paraphrase of an already-computed reason, guarded so it can never influence or replace it
  — is reconciled in STATUS above; do not widen it beyond that without new sign-off.
- Full CTAS 1–5 assignment (judge call is binary). Any "AI replaces the nurse" framing.
- Auth, accounts, deploy infra.