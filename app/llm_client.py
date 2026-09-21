"""Thin wrappers around LLM providers.

Every provider exposes the same method:  complete(system, user) -> str
The rest of the app never needs to know which vendor is being used, so switching
providers is a one-line change in .env.
"""
from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod

from app.config import Settings
from app.errors import AnalysisError, ConfigError

MAX_OUTPUT_TOKENS = 4000


RETRY_DELAYS = (2, 6)  # seconds to wait before the 2nd and 3rd attempt
_TRANSIENT_RE = re.compile(r"\b(500|502|503|504)\b|unavailable|overloaded|high demand|timed? ?out|connection", re.I)


def _is_transient(exc: Exception) -> bool:
    """Temporary server-side problems (like Gemini's 503 'high demand') that usually clear up on their own."""
    return bool(_TRANSIENT_RE.search(str(exc)))


def _call_with_retries(provider: str, fn):
    """Run fn(); retry a few times with a short wait if the failure looks temporary."""
    for attempt in range(len(RETRY_DELAYS) + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt < len(RETRY_DELAYS) and _is_transient(exc):
                time.sleep(RETRY_DELAYS[attempt])
                continue
            raise _api_error(provider, exc) from exc


def _api_error(provider: str, exc: Exception) -> AnalysisError:
    """Turn a raw SDK exception into a short message that tells the user what to do."""
    text = str(exc)
    low = text.lower()
    if _is_transient(exc):
        hint = "The AI service is temporarily busy or unreachable. MeetIQ already retried; wait a minute and click Analyze again."
    elif "429" in text or "resource_exhausted" in low or "rate limit" in low or "quota" in low:
        hint = "Rate limit or quota reached. Wait a minute and try again."
    elif "401" in text or "403" in text or "api key" in low or "permission" in low:
        hint = "Check that the API key in your .env file is correct and active."
    elif "404" in text or "not found" in low:
        hint = "The model name may be retired. Set a current model in .env (see GEMINI_MODEL / ANTHROPIC_MODEL / OPENAI_MODEL)."
    else:
        hint = ""
    return AnalysisError(f"{provider} request failed. {hint} Details: {text[:300]}".replace("  ", " "))


class LLMClient(ABC):
    name: str

    @abstractmethod
    def complete(self, system: str, user: str) -> str:
        """Send a prompt, return the model's raw text answer."""


class AnthropicClient(LLMClient):
    def __init__(self, api_key: str, model: str):
        from anthropic import Anthropic  # imported lazily: only needed for this provider

        self._client = Anthropic(api_key=api_key)
        self._model = model
        self.name = f"anthropic:{model}"

    def complete(self, system: str, user: str) -> str:
        resp = _call_with_retries(
            "Anthropic",
            lambda: self._client.messages.create(
                    model=self._model,
                    max_tokens=MAX_OUTPUT_TOKENS,
                    system=system,
                    messages=[{"role": "user", "content": user}],
            ),
        )
        return "".join(block.text for block in resp.content if block.type == "text")


class OpenAIClient(LLMClient):
    def __init__(self, api_key: str, model: str):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model
        self.name = f"openai:{model}"

    def complete(self, system: str, user: str) -> str:
        resp = _call_with_retries(
            "OpenAI",
            lambda: self._client.chat.completions.create(
                    model=self._model,
                    temperature=0.2,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
            ),
        )
        return resp.choices[0].message.content or ""


class GeminiClient(LLMClient):
    """Google Gemini via the `google-genai` SDK (API keys from Google AI Studio)."""

    def __init__(self, api_key: str, model: str):
        from google import genai  # imported lazily: only needed for this provider
        from google.genai import types

        self._client = genai.Client(api_key=api_key)
        self._types = types
        self._model = model
        self.name = f"gemini:{model}"

    def complete(self, system: str, user: str) -> str:
        config = self._types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.2,
            max_output_tokens=8192,  # generous: newer Gemini models may spend tokens "thinking"
            response_mime_type="application/json",  # ask Gemini for JSON-only output
        )
        resp = _call_with_retries(
            "Gemini",
            lambda: self._client.models.generate_content(model=self._model, contents=user, config=config),
        )
        return resp.text or ""


class MockClient(LLMClient):
    """Offline demo: ALWAYS returns the same canned analysis of data/sample_transcript.txt.

    Useful to try the UI/API without an API key. It does NOT read your transcript.
    """

    name = "mock:demo"

    def complete(self, system: str, user: str) -> str:
        return json.dumps(
            {
                "title": "[DEMO MODE] Website Redesign - Sprint Planning",
                "summary": (
                    "The team met to lock the scope of the new website's beta launch. "
                    "Checkout development is on track but blocked by missing payment "
                    "gateway credentials. Dark mode was postponed to keep the beta "
                    "achievable, and the team committed to a beta launch on October 15th "
                    "with a scope check on October 10th."
                ),
                "key_points": [
                    "Checkout API is about 70% complete",
                    "Desktop mockups approved; mobile mockups need one more round",
                    "Dark mode would add about a week of work plus design on every screen",
                    "QA has a single engineer and no test plan yet",
                    "API documentation is out of date",
                ],
                "decisions": [
                    {
                        "decision": "Postpone dark mode to the next release",
                        "context": "It needs at least a week of dev work and design on all screens.",
                    },
                    {
                        "decision": "Beta scope is checkout, product pages and new navigation only",
                        "context": None,
                    },
                    {
                        "decision": "Commit to October 15th for the beta launch",
                        "context": "Progress will be reviewed on October 10th.",
                    },
                ],
                "action_items": [
                    {"task": "Email the payment vendor to obtain sandbox credentials", "owner": "Priya", "deadline": "October 3rd", "priority": "high"},
                    {"task": "Finish the checkout API", "owner": "Arjun", "deadline": "October 8th", "priority": "high"},
                    {"task": "Deliver final mobile designs", "owner": "Meera", "deadline": "October 5th", "priority": "medium"},
                    {"task": "Draft the QA test plan", "owner": "Rahul", "deadline": "October 10th", "priority": "high"},
                    {"task": "Update the API documentation before the beta", "owner": "Unassigned", "deadline": None, "priority": "medium"},
                ],
                "risks_blockers": [
                    {"description": "Payment gateway sandbox credentials have not been received", "type": "blocker", "severity": "high", "mitigation": "Priya will chase the vendor by October 3rd"},
                    {"description": "Only one QA engineer; late credentials could squeeze regression testing", "type": "risk", "severity": "high", "mitigation": "Test plan due October 10th to check feasibility"},
                    {"description": "Outdated API docs may slow the mobile team", "type": "risk", "severity": "low", "mitigation": None},
                ],
            }
        )


def get_client(settings: Settings) -> LLMClient:
    """Factory: pick the provider named in LLM_PROVIDER."""
    try:
        return _build_client(settings)
    except ImportError as exc:
        raise ConfigError(
            f"A required package is missing ({exc.name}). Run: pip install -r requirements.txt"
        ) from exc


def _build_client(settings: Settings) -> LLMClient:
    if settings.provider == "mock":
        return MockClient()
    if settings.provider == "gemini":
        if not settings.gemini_api_key:
            raise ConfigError("GEMINI_API_KEY is missing. Add it to your .env file (or set LLM_PROVIDER=mock).")
        return GeminiClient(settings.gemini_api_key, settings.gemini_model)
    if settings.provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ConfigError("ANTHROPIC_API_KEY is missing. Add it to your .env file (or set LLM_PROVIDER=mock).")
        return AnthropicClient(settings.anthropic_api_key, settings.anthropic_model)
    if settings.provider == "openai":
        if not settings.openai_api_key:
            raise ConfigError("OPENAI_API_KEY is missing. Add it to your .env file (or set LLM_PROVIDER=mock).")
        return OpenAIClient(settings.openai_api_key, settings.openai_model)
    raise ConfigError(f"Unknown LLM_PROVIDER '{settings.provider}'. Use gemini, anthropic, openai, or mock.")