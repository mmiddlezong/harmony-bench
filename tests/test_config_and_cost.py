import pytest

from scorebench.config import load_registry
from scorebench.cost import estimate_cost, usage_cost
from scorebench.providers import Request, make_provider
from scorebench.providers.base import Usage

KNOWN_PROVIDERS = {"anthropic", "openai", "google", "xai", "openrouter", "openai_chat"}


def test_registry_is_valid():
    reg = load_registry()
    assert reg.pricing_checked
    for m in reg.models:
        assert m.provider in KNOWN_PROVIDERS, m.id
        assert m.pricing.input > 0 and m.pricing.output > 0, f"{m.id} has no pricing"
        assert m.estimate.image_tokens > 0, f"{m.id} has no image token estimate"
        assert m.groups, f"{m.id} is in no group"


def test_judge_is_configured():
    reg = load_registry()
    j = reg.judge_spec()
    assert j.id == "gpt-6-luna" and j.params == {"reasoning_effort": "high"}
    assert j.estimate.text_tokens == 700 and j.estimate.image_tokens == reg.get("gpt-6-luna").estimate.image_tokens


def test_every_model_builds_requests():
    """Offline dry run: every configured model can build image and text-only requests."""
    for m in load_registry().models:
        p = make_provider(m)
        assert isinstance(p.build_request(Request(prompt="x", image=b"\x89PNG fake")), dict), m.id
        assert isinstance(p.build_request(Request(prompt="x")), dict), m.id


def test_group_selection():
    reg = load_registry()
    everything = reg.select(["all"])
    assert len(everything) == len([m for m in reg.models if m.enabled])
    assert {m.id for m in reg.select(["anthropic"])} >= {"claude-fable-5-1", "claude-haiku-4-5"}
    first = everything[0].id
    assert [m.id for m in reg.select([f"{first},{first}"])] == [first]
    with pytest.raises(KeyError):
        reg.select(["no-such-model"])


def test_usage_cost(spec_factory):
    spec = spec_factory()  # $2 in, $10 out, $0.2 cached
    u = Usage(input_tokens=1_000_000, output_tokens=100_000, cached_input_tokens=500_000)
    assert usage_cost(u, spec) == pytest.approx(0.5 * 2 + 0.5 * 0.2 + 0.1 * 10)


def test_estimate_depends_on_condition(spec_factory):
    spec = spec_factory()
    img, xml, text = (estimate_cost(spec, 10, c) for c in ("image", "musicxml", "text"))
    e = spec.estimate
    assert img.input_tokens_per_req == e.text_tokens + e.image_tokens
    assert xml.input_tokens_per_req == e.text_tokens + e.musicxml_tokens
    assert text.input_tokens_per_req == e.text_tokens
    assert img.low_usd < img.expected_usd < img.high_usd
