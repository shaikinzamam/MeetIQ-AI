"""Core pipeline:  transcript -> validate -> LLM -> parse JSON -> validate schema -> metrics.

Both the FastAPI backend and the Streamlit UI call `MeetingAnalyzer.analyze()`,
so the AI logic lives in exactly one place.
"""
from __future__ import annotations

import json
import re

from pydantic import ValidationError

from app.config import Settings, get_settings
from app.errors import AnalysisError, TranscriptError
from app.llm_client import LLMClient, get_client
from app.metrics import compute_metrics
from app.prompts import SYSTEM_PROMPT, build_repair_prompt, build_user_prompt
from app.schemas import MeetingAnalysis, MeetingReport

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def extract_json(text: str) -> dict:
    """Pull a JSON object out of an LLM answer, even if it added code fences or chatter."""
    cleaned = _FENCE_RE.sub("", text.strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in the model output.")
    return json.loads(cleaned[start : end + 1])


class MeetingAnalyzer:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.client = client or get_client(self.settings)

    # ---- public API -------------------------------------------------------
    def analyze(self, transcript: str) -> MeetingReport:
        transcript = self._validate_transcript(transcript)
        analysis = self._run_llm(transcript)
        return MeetingReport(
            analysis=analysis,
            metrics=compute_metrics(transcript, analysis),
            model_used=self.client.name,
        )

    # ---- internals --------------------------------------------------------
    def _validate_transcript(self, transcript: str) -> str:
        transcript = (transcript or "").strip()
        if len(transcript) < self.settings.min_transcript_chars:
            raise TranscriptError(
                f"Transcript is too short (minimum {self.settings.min_transcript_chars} characters)."
            )
        if len(transcript) > self.settings.max_transcript_chars:
            raise TranscriptError(
                f"Transcript is too long (maximum {self.settings.max_transcript_chars} characters). "
                "Split it into smaller parts."
            )
        return transcript

    def _parse(self, raw: str) -> MeetingAnalysis:
        return MeetingAnalysis.model_validate(extract_json(raw))

    def _run_llm(self, transcript: str) -> MeetingAnalysis:
        raw = self.client.complete(SYSTEM_PROMPT, build_user_prompt(transcript))
        try:
            return self._parse(raw)
        except (ValueError, ValidationError) as first_error:
            # One self-repair attempt: show the model its mistake and ask again.
            raw2 = self.client.complete(
                SYSTEM_PROMPT, build_repair_prompt(transcript, raw, str(first_error)[:500])
            )
            try:
                return self._parse(raw2)
            except (ValueError, ValidationError) as second_error:
                raise AnalysisError(
                    f"The model did not return valid structured output after a retry: {second_error}"
                ) from second_error
