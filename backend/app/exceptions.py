from __future__ import annotations

from dataclasses import dataclass

from app.schemas import AttemptLog


class LLMError(RuntimeError):
    pass


class LLMResponseValidationError(LLMError):
    pass


class LLMRetryExhausted(LLMError):
    def __init__(self, message: str, attempts: list[AttemptLog], retry_exhausted: bool = True):
        super().__init__(message)
        self.attempts = attempts
        self.retry_exhausted = retry_exhausted


@dataclass
class LLMHTTPError(LLMError):
    status_code: int
    message: str

    def __str__(self) -> str:
        return f"LLM HTTP {self.status_code}: {self.message}"
