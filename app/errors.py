"""Custom exceptions so the API/UI can show clear, friendly messages."""


class MeetIQError(Exception):
    """Base class for all MeetIQ errors."""


class ConfigError(MeetIQError):
    """Missing API key, unknown provider, etc."""


class TranscriptError(MeetIQError):
    """The input transcript is empty, too short, or too long."""


class AnalysisError(MeetIQError):
    """The LLM failed or returned output we could not turn into valid JSON."""
