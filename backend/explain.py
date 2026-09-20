"""Optional Gemini plain-language paraphrase of an ALREADY-COMPUTED reveal.

Guardrails (non-negotiable, per CLAUDE.md's DUA + no-LLM-in-decision-path rules):
- Gemini sees ONLY the model's own top_features (name, kind, direction) plus the
  vital values already shown to the judge on the case card — never chief-complaint
  text, never the narrative presentation, never the outcome, never a raw record.
  These are synthetic cases already (cases.json), but the exposure surface here is
  kept even narrower than that on purpose.
- The flag/score is computed elsewhere (backend/scoring.py) before this is ever
  called — this module can only rephrase a decision that already happened, never
  influence it.
- Any failure (no key, timeout, quota, network) returns None silently. Callers
  (the /explain endpoint) must treat None as "show the bars alone, no error UI."
"""
import concurrent.futures
import os

from dotenv import load_dotenv
from pydantic import BaseModel

from backend import observability

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip() or None
MODEL_NAME = "gemini-3.6-flash"  # gemini-2.5-flash was retired; API points new callers here
TIMEOUT_S = 10  # hard backstop — a stalled Gemini call must never hang a reveal
# (10s is the API's own enforced minimum for a manually-set deadline; anything
# shorter is rejected outright rather than honored, so this is as tight as it goes)

# per-case_id cache — both successes and failures, so replaying a case (or a
# transient quota error) never triggers a second Gemini call for that case
_CACHE: dict[str, str | None] = {}

_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="gemini-explain")


class _ExplanationSentence(BaseModel):
    sentence: str


def _direction(contribution: float) -> str:
    return "increases risk" if contribution >= 0 else "decreases risk"


def _build_prompt(top_features: list[dict], vitals: dict) -> str:
    lines = []
    for f in top_features:
        entry = f"- {f['feature']} ({f['kind']}): {_direction(f['contribution'])}"
        if f["kind"] == "vital" and f["feature"] in vitals:
            entry += f", value {vitals[f['feature']]}"
        lines.append(entry)
    factors = "\n".join(lines)
    return (
        "You are summarizing an already-computed clinical risk flag for a triage "
        "decision-support tool. Below is the COMPLETE list of contributing factors "
        "the model used, and nothing else about the patient.\n\n"
        f"{factors}\n\n"
        "Rephrase ONLY these contributing factors into one plain-language clinical "
        "sentence. Add NO new symptoms, diagnoses, or facts that are not in this "
        "list. Do not speculate about the patient's condition beyond what these "
        "factors state."
    )


def _call_gemini(prompt: str) -> str | None:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY, http_options=types.HttpOptions(timeout=TIMEOUT_S * 1000))
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=_ExplanationSentence,
        ),
    )
    parsed = response.parsed
    if parsed is None:
        return None
    sentence = parsed.sentence.strip()
    return sentence or None


def explain_case(case_id: str, top_features: list[dict], vitals: dict) -> str | None:
    if case_id in _CACHE:
        return _CACHE[case_id]

    if not GEMINI_API_KEY:
        _CACHE[case_id] = None
        return None

    prompt = _build_prompt(top_features, vitals)
    try:
        future = _executor.submit(_call_gemini, prompt)
        sentence = future.result(timeout=TIMEOUT_S)
        _CACHE[case_id] = sentence
        observability.log_event("explanation generated", case_id=case_id)
        return sentence
    except Exception as e:
        observability.log_event("explanation fallback", level="warning", case_id=case_id, reason=str(e))
        _CACHE[case_id] = None
        return None
