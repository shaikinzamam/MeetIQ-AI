"""Run with:  pytest -q      (no API key needed: we use a fake LLM client)"""
import json

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.errors import AnalysisError, TranscriptError
from app.llm_client import LLMClient, MockClient
from app.service import MeetingAnalyzer, extract_json

LONG_TEXT = "Alice: Let's ship the report on Friday. Bob: I will send it to the client. " * 3


class FakeClient(LLMClient):
    """Returns pre-defined answers in order, and counts how many times it was called."""

    name = "fake"

    def __init__(self, answers):
        self.answers, self.calls = list(answers), 0

    def complete(self, system, user):
        self.calls += 1
        return self.answers.pop(0)


VALID = json.dumps({"title": "T", "summary": "S", "action_items": [{"task": "Do X", "owner": "Bob", "deadline": "Friday", "priority": "HIGH"}]})


def test_extract_json_handles_code_fences_and_chatter():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here it is: {"a": 1} Hope that helps.') == {"a": 1}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_valid_output_is_parsed_and_normalized():
    report = MeetingAnalyzer(FakeClient([VALID])).analyze(LONG_TEXT)
    assert report.analysis.action_items[0].priority == "high"  # "HIGH" normalized
    assert report.analysis.decisions == []                      # missing -> empty list
    assert report.metrics.accountability_score == 100.0


def test_retry_once_on_bad_output():
    client = FakeClient(["this is not json", VALID])
    report = MeetingAnalyzer(client).analyze(LONG_TEXT)
    assert client.calls == 2
    assert report.analysis.title == "T"


def test_fails_cleanly_after_two_bad_outputs():
    with pytest.raises(AnalysisError):
        MeetingAnalyzer(FakeClient(["bad", "still bad"])).analyze(LONG_TEXT)


def test_transcript_validation():
    analyzer = MeetingAnalyzer(FakeClient([VALID]))
    with pytest.raises(TranscriptError):
        analyzer.analyze("   ")
    with pytest.raises(TranscriptError):
        analyzer.analyze("x" * 100_000)


def test_unassigned_task_lowers_accountability():
    data = json.loads(VALID)
    data["action_items"].append({"task": "Update docs", "owner": None, "deadline": "tbd"})
    report = MeetingAnalyzer(FakeClient([json.dumps(data)])).analyze(LONG_TEXT)
    assert report.analysis.action_items[1].owner == "Unassigned"
    assert report.analysis.action_items[1].deadline is None
    assert report.metrics.accountability_score == 50.0


def test_api_endpoint_with_mock_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    from app import main

    main.get_analyzer.cache_clear()
    client = TestClient(main.app)
    assert client.get("/health").json() == {"status": "ok"}
    ok = client.post("/analyze", json={"transcript": LONG_TEXT})
    assert ok.status_code == 200 and "analysis" in ok.json()
    short = client.post("/analyze", json={"transcript": "hi"})
    assert short.status_code == 422
    main.get_analyzer.cache_clear()


def test_gemini_provider_wiring(monkeypatch):
    """No real key needed: we replace the Google client with a fake and check the call."""
    pytest.importorskip("google.genai")
    from google import genai

    seen = {}

    class FakeModels:
        def generate_content(self, model, contents, config):
            seen.update(model=model, config=config)
            return type("R", (), {"text": VALID})()

    class FakeGenaiClient:
        def __init__(self, api_key):
            self.models = FakeModels()

    monkeypatch.setattr(genai, "Client", FakeGenaiClient)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    report = MeetingAnalyzer().analyze(LONG_TEXT)
    assert report.model_used.startswith("gemini:")
    assert seen["config"].response_mime_type == "application/json"
    assert report.analysis.title == "T"


def test_short_transcript_saves_no_time_but_long_one_does():
    tiny = "alpha bravo charlie delta echo foxtrot"  # 6 words: no shorter than the report
    short = MeetingAnalyzer(FakeClient([VALID])).analyze(tiny + " " * 20 + "x")
    assert short.metrics.est_seconds_saved == 0
    long_text = "Alice: we should ship the report on Friday and Bob will send it. " * 400
    long = MeetingAnalyzer(FakeClient([VALID])).analyze(long_text)
    assert long.metrics.est_seconds_saved > 60


def test_api_errors_get_friendly_hints():
    from app.llm_client import _api_error

    assert "Wait a minute" in str(_api_error("Gemini", Exception("429 RESOURCE_EXHAUSTED")))
    assert ".env" in str(_api_error("Gemini", Exception("403 API key not valid")))
    assert "retired" in str(_api_error("Gemini", Exception("404 NOT_FOUND model")))


def test_transient_503_is_retried_then_succeeds(monkeypatch):
    from app import llm_client

    monkeypatch.setattr(llm_client.time, "sleep", lambda s: None)  # don't actually wait
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("503 UNAVAILABLE. This model is currently experiencing high demand.")
        return "ok"

    assert llm_client._call_with_retries("Gemini", flaky) == "ok"
    assert calls["n"] == 3


def test_persistent_503_gives_friendly_error_and_permanent_errors_are_not_retried(monkeypatch):
    from app import llm_client

    monkeypatch.setattr(llm_client.time, "sleep", lambda s: None)
    n = {"c": 0}

    def always_503():
        n["c"] += 1
        raise RuntimeError("503 UNAVAILABLE")

    with pytest.raises(AnalysisError, match="click Analyze again"):
        llm_client._call_with_retries("Gemini", always_503)
    assert n["c"] == 3

    n["c"] = 0

    def bad_key():
        n["c"] += 1
        raise RuntimeError("403 API key not valid")

    with pytest.raises(AnalysisError):
        llm_client._call_with_retries("Gemini", bad_key)
    assert n["c"] == 1  # no pointless retries for permanent errors