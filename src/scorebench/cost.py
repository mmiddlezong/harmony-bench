"""Cost accounting (actual, from reported usage) and cost estimation (before a run)."""

from __future__ import annotations

from dataclasses import dataclass

from .config import ModelSpec
from .providers.base import Usage


def usage_cost(usage: Usage, spec: ModelSpec) -> float:
    """USD cost of one request from its normalized usage and the model's list prices."""
    p = spec.pricing
    cached = min(usage.cached_input_tokens, usage.input_tokens)
    uncached = usage.input_tokens - cached
    cached_rate = p.cached_input if p.cached_input is not None else p.input
    return (uncached * p.input + cached * cached_rate + usage.output_tokens * p.output) / 1e6


@dataclass
class CostEstimate:
    model_id: str
    n_requests: int
    input_tokens_per_req: int
    output_tokens_per_req: int
    low_usd: float
    expected_usd: float
    high_usd: float

    @property
    def per_request_usd(self) -> float:
        return self.expected_usd / self.n_requests if self.n_requests else 0.0


# Hidden reasoning length is the dominant uncertainty; the range brackets it.
REASONING_LOW, REASONING_HIGH = 0.5, 2.5


def estimate_cost(spec: ModelSpec, n_requests: int, condition: str = "image") -> CostEstimate:
    """condition: "image" (prompt + score image), "musicxml" (prompt + MusicXML text), or
    "text" (prompt only, e.g. a judge call)."""
    e = spec.estimate
    p = spec.pricing
    extra = {"image": e.image_tokens, "musicxml": e.musicxml_tokens, "text": 0}[condition]
    inp = e.text_tokens + extra

    def total(reasoning_mult: float) -> float:
        out = e.output_tokens + e.reasoning_tokens * reasoning_mult
        return n_requests * (inp * p.input + out * p.output) / 1e6

    return CostEstimate(
        model_id=spec.id,
        n_requests=n_requests,
        input_tokens_per_req=inp,
        output_tokens_per_req=e.output_tokens + e.reasoning_tokens,
        low_usd=total(REASONING_LOW),
        expected_usd=total(1.0),
        high_usd=total(REASONING_HIGH),
    )
