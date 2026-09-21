"""Central configuration, read from environment variables (and an optional .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    provider: str
    anthropic_api_key: str | None
    anthropic_model: str
    openai_api_key: str | None
    openai_model: str
    gemini_api_key: str | None
    gemini_model: str
    min_transcript_chars: int
    max_transcript_chars: int


def get_settings() -> Settings:
    """Build settings fresh each call so tests can change env vars."""
    return Settings(
        provider=os.getenv("LLM_PROVIDER", "gemini").strip().lower(),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        gemini_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
        min_transcript_chars=int(os.getenv("MIN_TRANSCRIPT_CHARS", "50")),
        max_transcript_chars=int(os.getenv("MAX_TRANSCRIPT_CHARS", "60000")),
    )
