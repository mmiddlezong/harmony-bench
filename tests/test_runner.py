import json

import pytest

from harmonybench import runner
from harmonybench.dataset import Item
from harmonybench.providers.base import Provider, ProviderError, ProviderResult, Usage


class FakeProvider(Provider):
    """Scripted provider: behavior[item_index] -> 'ok' | 'error' | 'fatal' | 'refuse' | 'truncate' | 'empty'."""

    calls: list[tuple[int, bool]] = []

    def __init__(self, spec, behavior=None, default="ok"):
        super().__init__(spec)
        self.behavior = behavior or {}
        self.default = default

    def build_request(self, request):
        return {}

    async def complete(self, request):
        # The fake items encode their index in the image bytes / MusicXML text.
        idx = int(request.image.decode()) if request.image is not None else int(request.prompt.rsplit(" ", 1)[1])
        FakeProvider.calls.append((idx, request.image is not None))
        mode = self.behavior.get(idx, self.default)
        usage = Usage(input_tokens=1000, output_tokens=1000)
        if mode == "error":
            raise RuntimeError("503 Service Unavailable")
        if mode == "fatal":
            raise ProviderError("BadRequestError: invalid model")
        if mode == "refuse":
            return ProviderResult(text="", usage=usage, stop_reason="refusal", refused=True)
        if mode == "truncate":
            return ProviderResult(text="The bass note is", usage=usage, stop_reason="max_tokens", truncated=True)
        if mode == "empty":
            return ProviderResult(text="  ", usage=usage, stop_reason="end_turn")
        return ProviderResult(text="This is a C major triad.", usage=usage, stop_reason="end_turn")


@pytest.fixture
def items(item_factory, monkeypatch):
    monkeypatch.setattr(Item, "load_image", lambda self: str(self.index).encode())
    monkeypatch.setattr(Item, "load_musicxml", lambda self: f"<xml> {self.index}")
    monkeypatch.setattr(runner, "manifest_hash", lambda subset: "manifest-v1")
    monkeypatch.setattr(runner, "items_hash", lambda ids, subset: "items-v1")
    return [item_factory(i) for i in range(6)]


@pytest.fixture
def use_fake(monkeypatch):
    FakeProvider.calls = []

    def install(**kw):
        FakeProvider.calls = []
        monkeypatch.setattr(runner, "make_provider", lambda spec: FakeProvider(spec, **kw))

    return install


async def test_statuses_and_resume(tmp_path, spec_factory, items, use_fake):
    spec = spec_factory()
    use_fake(behavior={1: "error", 2: "refuse", 3: "truncate", 4: "empty"})
    s = await runner.run_model(spec, items, "triads_root", conditions=("image",), concurrency=2, results_dir=tmp_path)
    assert s.statuses == {"ok": 2, "api_error": 1, "refusal": 1, "truncated": 1, "empty": 1}
    # cost: 1000 in * $2/M + 1000 out * $10/M = $0.012 per completed request (5 of them)
    assert s.cost_usd == pytest.approx(5 * 0.012)

    use_fake()  # everything succeeds now
    s2 = await runner.run_model(spec, items, "triads_root", conditions=("image",), concurrency=2, results_dir=tmp_path)
    assert s2.skipped == 5 and s2.attempted == 1  # only the api_error item is retried
    assert FakeProvider.calls == [(1, True)]

    recs = runner.read_records(runner.predictions_path(spec.id, "triads_root", tmp_path))
    assert len(recs) == 7
    assert recs[-1]["raw_text"] == "This is a C major triad." and recs[-1]["condition"] == "image"
    meta = json.loads((runner.run_dir(spec.id, "triads_root", tmp_path) / "meta.json").read_text())
    assert meta["prompt_version"] == "v1" and meta["request_config"]["model"] == spec.model


async def test_both_conditions(tmp_path, spec_factory, items, use_fake):
    use_fake()
    s = await runner.run_model(spec_factory(), items[:3], "triads_root", results_dir=tmp_path)
    assert s.planned == 6 and s.attempted == 6
    assert sorted(FakeProvider.calls) == [(0, False), (0, True), (1, False), (1, True), (2, False), (2, True)]


async def test_budget_cap_stops_early(tmp_path, spec_factory, items, use_fake):
    spec = spec_factory()  # estimate ≈ $0.0135/image request
    use_fake()
    s = await runner.run_model(
        spec, items, "triads_root", conditions=("image",), concurrency=1, max_cost=0.03, results_dir=tmp_path
    )
    assert s.aborted and "budget" in s.aborted
    assert s.attempted == 2
    assert s.cost_usd <= 0.03


async def test_consecutive_fatal_errors_abort(tmp_path, spec_factory, items, use_fake):
    use_fake(default="fatal")
    s = await runner.run_model(spec_factory(), items, "triads_root", concurrency=1, results_dir=tmp_path)
    assert s.aborted and "non-retryable" in s.aborted
    assert s.attempted == runner.MAX_CONSECUTIVE_FATAL


async def test_config_change_requires_fresh(tmp_path, spec_factory, items, use_fake):
    use_fake()
    await runner.run_model(spec_factory(), items[:2], "triads_root", results_dir=tmp_path)
    changed = spec_factory(params={"effort": "low"})
    with pytest.raises(runner.RunConfigMismatch):
        await runner.run_model(changed, items[:2], "triads_root", results_dir=tmp_path)
    s = await runner.run_model(changed, items[:2], "triads_root", fresh=True, results_dir=tmp_path)
    assert s.attempted == 4
    backups = list(runner.run_dir(changed.id, "triads_root", tmp_path).glob("predictions.*.bak.jsonl"))
    assert len(backups) == 1


async def test_subset_growth_keeps_results_but_edits_do_not(tmp_path, spec_factory, items, use_fake, monkeypatch):
    spec = spec_factory()
    use_fake()
    await runner.run_model(spec, items[:3], "triads_root", conditions=("image",), results_dir=tmp_path)

    monkeypatch.setattr(runner, "manifest_hash", lambda subset: "grown-manifest")
    s = await runner.run_model(spec, items, "triads_root", conditions=("image",), results_dir=tmp_path)
    assert s.skipped == 3 and s.attempted == 3

    monkeypatch.setattr(runner, "manifest_hash", lambda subset: "edited-manifest")
    monkeypatch.setattr(runner, "items_hash", lambda ids, subset: "different")
    with pytest.raises(runner.RunConfigMismatch):
        await runner.run_model(spec, items, "triads_root", conditions=("image",), results_dir=tmp_path)


def test_read_records_tolerates_torn_line(tmp_path):
    p = tmp_path / "p.jsonl"
    p.write_text('{"item_id": "a", "status": "ok"}\n{"item_id": "b", "sta')
    assert [r["item_id"] for r in runner.read_records(p)] == ["a"]


def test_items_hash_skips_removed_items(tmp_path, monkeypatch):
    from harmonybench import dataset

    manifest = tmp_path / "labels.jsonl"
    manifest.write_text('{"id": "a", "label": 1}\n{"id": "b", "label": 2}\n')
    monkeypatch.setattr(dataset, "items_path", lambda subset: manifest)
    # "c" was answered but has since been removed from the subset: it no longer counts.
    assert dataset.items_hash({"a", "b", "c"}, "x") == dataset.items_hash({"a", "b"}, "x")
    assert dataset.items_hash({"a"}, "x") != dataset.items_hash({"a", "b"}, "x")
