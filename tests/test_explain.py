import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import explain

TOP_FEATURES = [
    {"feature": "heartrate", "kind": "vital", "contribution": 0.8},
    {"feature": "fall", "kind": "text", "contribution": 0.5},
]
VITALS = {"heartrate": 110, "sbp": 100}


def test_no_api_key_never_calls_gemini(monkeypatch):
    monkeypatch.setattr(explain, "GEMINI_API_KEY", None)
    monkeypatch.setattr(explain, "_CACHE", {})

    def fail_if_called(prompt):
        raise AssertionError("must not call Gemini when no API key is configured")

    monkeypatch.setattr(explain, "_call_gemini", fail_if_called)

    result = explain.explain_case("case-a", TOP_FEATURES, VITALS)
    assert result is None


def test_success_is_cached_per_case_id(monkeypatch):
    monkeypatch.setattr(explain, "GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(explain, "_CACHE", {})

    call_count = {"n": 0}

    def fake_call(prompt):
        call_count["n"] += 1
        return "The elevated heart rate and a fall in the history both raise concern."

    monkeypatch.setattr(explain, "_call_gemini", fake_call)

    first = explain.explain_case("case-b", TOP_FEATURES, VITALS)
    second = explain.explain_case("case-b", TOP_FEATURES, VITALS)

    assert first == "The elevated heart rate and a fall in the history both raise concern."
    assert second == first
    assert call_count["n"] == 1, "replaying the same case_id must not re-call Gemini"


def test_failure_falls_back_to_none_and_is_cached(monkeypatch):
    monkeypatch.setattr(explain, "GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(explain, "_CACHE", {})

    call_count = {"n": 0}

    def failing_call(prompt):
        call_count["n"] += 1
        raise RuntimeError("simulated quota/timeout error")

    monkeypatch.setattr(explain, "_call_gemini", failing_call)

    first = explain.explain_case("case-c", TOP_FEATURES, VITALS)
    second = explain.explain_case("case-c", TOP_FEATURES, VITALS)

    assert first is None
    assert second is None
    assert call_count["n"] == 1, "a failed call must be cached too, not retried on replay"


def test_prompt_contains_only_feature_list_and_shown_vitals():
    prompt = explain._build_prompt(TOP_FEATURES, VITALS)
    assert "heartrate" in prompt
    assert "110" in prompt  # the shown vital value for the vital-kind feature
    assert "fall" in prompt
    # sbp is in vitals but not a top feature here — must not leak into the prompt
    assert "100" not in prompt
