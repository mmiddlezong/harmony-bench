"""Provider interface shared by every API backend."""

from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field

from ..config import ModelSpec


@dataclass
class Usage:
    """Normalized token usage. `output_tokens` is ALL billable output tokens, including
    hidden reasoning; `reasoning_tokens` is the reasoning subset (informational).
    `input_tokens` includes cached tokens; `cached_input_tokens` is the cached subset."""

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_input_tokens: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProviderResult:
    text: str | None
    usage: Usage = field(default_factory=Usage)
    stop_reason: str | None = None
    request_id: str | None = None
    refused: bool = False
    truncated: bool = False
    provider_cost_usd: float | None = None  # cost reported by the API itself, if any
    served_model: str | None = None
    raw_usage: dict | None = None  # the provider's usage block, verbatim, for auditing


@dataclass
class Request:
    """One model call: a prompt, optionally with an image, optionally constrained to a
    JSON schema (benchmark answers are free text; only the judge uses a schema)."""

    prompt: str
    image: bytes | None = None
    media_type: str = "image/png"
    schema: dict | None = None
    schema_name: str = "answer"


class ProviderError(Exception):
    """Non-retryable failure in building or sending a request (bad config, auth, 400)."""


class Provider(ABC):
    def __init__(self, spec: ModelSpec):
        self.spec = spec

    @abstractmethod
    def build_request(self, request: Request) -> dict:
        """Return the provider-specific request kwargs (pure; no network)."""

    @abstractmethod
    async def complete(self, request: Request) -> ProviderResult:
        """Send one request and return the normalized result."""

    async def aclose(self) -> None:  # pragma: no cover - optional hook
        return None


def billable_output(output_tokens: int, reasoning_tokens: int, input_tokens: int, total_tokens: int | None) -> int:
    """Billable output tokens, robust to both OpenAI-style conventions: some servers include
    reasoning tokens in the output count (OpenAI), others report them separately and only
    include them in total_tokens (e.g. xAI)."""
    if total_tokens:
        return max(output_tokens, total_tokens - input_tokens)
    return output_tokens


def dump_usage(usage) -> dict | None:
    if usage is None:
        return None
    try:
        return usage.model_dump(exclude_none=True)
    except AttributeError:
        return dict(usage) if isinstance(usage, dict) else None


def b64(image: bytes) -> str:
    return base64.standard_b64encode(image).decode("ascii")


def data_url(image: bytes, media_type: str) -> str:
    return f"data:{media_type};base64,{b64(image)}"
