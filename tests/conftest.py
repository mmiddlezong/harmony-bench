from __future__ import annotations

import json

import pytest

from harmonybench.config import Estimate, ModelSpec, Pricing
from harmonybench.dataset import Item

JUDGMENT = {"final_answer": "F# minor", "verdict": "correct", "explanation": "Matches the key."}


@pytest.fixture
def good_json() -> str:
    return json.dumps(JUDGMENT)


def make_spec(**overrides) -> ModelSpec:
    base = dict(
        id="test-model",
        display_name="Test Model",
        lab="Test",
        provider="anthropic",
        model="test-model-api-id",
        pricing=Pricing(input=2.0, output=10.0, cached_input=0.2),
        estimate=Estimate(image_tokens=400, reasoning_tokens=1000),
        api_key_env=["HARMONYBENCH_TEST_KEY"],
        timeout_s=10,
    )
    base.update(overrides)
    return ModelSpec(**base)


@pytest.fixture
def spec_factory():
    return make_spec


def make_item(index: int, root: str = "C", quality: str = "major", key_signature: int = 0) -> Item:
    return Item(
        index=index,
        item_id=f"item-{index:03d}",
        subset="test",
        image=f"images/item-{index:03d}.png",
        musicxml=f"musicxml/item-{index:03d}.musicxml",
        image_sha256="",
        musicxml_sha256="",
        label={"root": root, "quality": quality, "canonical": f"{root} {quality}"},
        meta={"key_signature": key_signature, "voicing": {"S": "E5", "A": "G4", "T": "C4", "B": "C3"}},
    )


@pytest.fixture
def item_factory():
    return make_item


@pytest.fixture(autouse=True)
def _test_key(monkeypatch):
    monkeypatch.setenv("HARMONYBENCH_TEST_KEY", "sk-test")
