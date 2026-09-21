"""FastAPI backend.   Run:  uvicorn app.main:app --reload   ->  docs at /docs"""
from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.errors import AnalysisError, ConfigError, TranscriptError
from app.schemas import MeetingReport
from app.service import MeetingAnalyzer

app = FastAPI(
    title="MeetIQ API",
    version="1.0.0",
    description="Turn meeting transcripts into summaries, decisions, action items and risks.",
)


class AnalyzeRequest(BaseModel):
    transcript: str = Field(..., description="Plain-text meeting transcript.")


@lru_cache
def get_analyzer() -> MeetingAnalyzer:
    """Created once and reused (so we don't rebuild the API client on every request)."""
    try:
        return MeetingAnalyzer()
    except ConfigError as exc:  # e.g. missing API key -> clear message instead of a crash
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# A normal `def` (not `async def`) on purpose: the LLM SDK call is blocking, and FastAPI
# runs sync endpoints in a thread pool so one slow request doesn't freeze the server.
@app.post("/analyze", response_model=MeetingReport)
def analyze(req: AnalyzeRequest, analyzer: MeetingAnalyzer = Depends(get_analyzer)) -> MeetingReport:
    try:
        return analyzer.analyze(req.transcript)
    except TranscriptError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except AnalysisError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
