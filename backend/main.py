"""Kairos — Triage Decision Support. FastAPI app.

Decision support, NOT nurse replacement. Research prototype, NOT medical advice.
Faithful reason = the flagger's real top contributing features (coef x value),
rendered by template. No LLM in the explanation or decision path. The model's
prediction is always computed live here — never stored in cases.json.
"""
import json
from pathlib import Path
from typing import Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend import db, explain, observability
from backend.scoring import _score_case

ROOT = Path(__file__).resolve().parent.parent

with open(ROOT / "backend" / "cases.json") as f:
    CASE_BANK = json.load(f)
CASES_BY_ID = {c["case_id"]: c for c in CASE_BANK}

# in-memory demo state (single-process — fine for this walking-skeleton scope)
NEXT_INDEX = 0
SUBMISSIONS: dict[str, str] = {}

app = FastAPI(title="Kairos")
observability.init_sentry()  # no-op if SENTRY_DSN is unset — never blocks startup
db.init_schema()  # best-effort; db.py falls back to in-memory logging if this can't reach Tiger Data
db.load_demo_snapshot()  # only does anything if DEMO_MODE is set


@app.get("/")
def frontend():
    return FileResponse(ROOT / "frontend" / "index.html")


class SubmitRequest(BaseModel):
    case_id: str
    judge_call: Literal["keep", "flag"]


class RevealRequest(BaseModel):
    case_id: str
    session_id: str


class ExplainRequest(BaseModel):
    case_id: str


@app.post("/next-case")
def next_case():
    global NEXT_INDEX
    if not CASE_BANK:
        raise HTTPException(status_code=404, detail="case bank is empty")
    case = CASE_BANK[NEXT_INDEX % len(CASE_BANK)]
    NEXT_INDEX += 1
    return {
        "case_id": case["case_id"],
        "presentation": case["presentation"],
        "chief_complaint": case["chief_complaint"],
        "vitals": case["vitals"],
        "acuity": case["acuity"],
        "age": case["age"],
        "sex": case["sex"],
    }


@app.post("/submit")
def submit(req: SubmitRequest):
    if req.case_id not in CASES_BY_ID:
        raise HTTPException(status_code=404, detail=f"unknown case_id '{req.case_id}'")
    SUBMISSIONS[req.case_id] = req.judge_call
    return {"ok": True, "case_id": req.case_id, "judge_call": req.judge_call}


@app.post("/reveal")
def reveal(req: RevealRequest, background_tasks: BackgroundTasks):
    case = CASES_BY_ID.get(req.case_id)
    if case is None:
        raise HTTPException(status_code=404, detail=f"unknown case_id '{req.case_id}'")
    judge_call = SUBMISSIONS.get(req.case_id)
    if judge_call is None:
        raise HTTPException(status_code=400, detail="submit a judge call before revealing")

    with observability.start_span(op="flagger.score", description="score_case"):
        risk_rank, model_flag, top_features = _score_case(case)

    if model_flag:
        observability.log_event("flag fired", case_id=case["case_id"], risk_rank=round(risk_rank, 4))

    # fire-and-forget: runs after the response is sent, so a slow or down DB never
    # delays the judge's reveal, even on the first attempt before the breaker trips
    background_tasks.add_task(
        db.log_event,
        session_id=req.session_id,
        case_id=case["case_id"],
        subpopulation=case["truth"]["subpopulation"],
        judge_flagged=(judge_call == "flag"),
        model_flagged=model_flag,
    )

    return {
        "case_id": case["case_id"],
        "judge_call": judge_call,
        "model_flag": model_flag,
        "risk_rank": round(risk_rank, 4),
        "top_features": top_features,
        "outcome": case["outcome"],
        "outcome_detail": case["outcome_detail"],
    }


@app.post("/explain")
def explain_reveal(req: ExplainRequest):
    """Optional plain-language paraphrase of an already-computed flag. Never
    influences the flag/score (computed in /reveal, independently recomputed
    here from the same deterministic scorer). Always returns 200 — a disabled
    key, timeout, or quota error yields explanation: null, never an error."""
    case = CASES_BY_ID.get(req.case_id)
    if case is None:
        raise HTTPException(status_code=404, detail=f"unknown case_id '{req.case_id}'")

    _, _, top_features = _score_case(case)
    sentence = explain.explain_case(req.case_id, top_features, case["vitals"])
    return {"case_id": req.case_id, "explanation": sentence}


@app.get("/leaderboard")
def leaderboard():
    return db.get_leaderboard()
