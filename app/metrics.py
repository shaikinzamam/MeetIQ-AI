"""Simple, transparent productivity metrics linked to SDG 8.

SDG 8 (Decent Work & Economic Growth) targets productivity and decent working
conditions. Two things a meeting tool can honestly measure:
  1. Time saved   -> people catch up from a short report instead of a long transcript.
  2. Accountability -> tasks with a clear owner + deadline avoid rework and follow-up meetings.

These are ESTIMATES (based on an average reading speed), not exact measurements.
"""
from __future__ import annotations

from app.schemas import EfficiencyMetrics, MeetingAnalysis

READING_WORDS_PER_MINUTE = 200  # typical adult silent-reading speed


def _word_count(text: str) -> int:
    return len(text.split())


def _report_words(a: MeetingAnalysis) -> int:
    parts = [a.title, a.summary, *a.key_points]
    parts += [d.decision for d in a.decisions]
    parts += [f"{t.task} {t.owner} {t.deadline or ''}" for t in a.action_items]
    parts += [r.description for r in a.risks_blockers]
    return sum(_word_count(p) for p in parts)


def compute_metrics(transcript: str, analysis: MeetingAnalysis) -> EfficiencyMetrics:
    transcript_words = _word_count(transcript)
    read_transcript = transcript_words / READING_WORDS_PER_MINUTE
    read_report = _report_words(analysis) / READING_WORDS_PER_MINUTE

    saved_minutes = max(0.0, read_transcript - read_report)

    items = analysis.action_items
    with_owner = sum(1 for t in items if t.owner != "Unassigned")
    with_deadline = sum(1 for t in items if t.deadline)
    complete = sum(1 for t in items if t.owner != "Unassigned" and t.deadline)
    score = round(100 * complete / len(items), 1) if items else None

    return EfficiencyMetrics(
        transcript_words=transcript_words,
        est_read_minutes_transcript=round(read_transcript, 1),
        est_read_minutes_report=round(read_report, 1),
        est_minutes_saved_per_reader=round(saved_minutes, 1),
        est_seconds_saved=round(saved_minutes * 60),
        action_items_total=len(items),
        action_items_with_owner=with_owner,
        action_items_with_deadline=with_deadline,
        accountability_score=score,
    )