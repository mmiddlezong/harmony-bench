"""End to end without the network: fake model answers -> fake judge -> scores."""

import asyncio
import json

import pytest

from scorebench import judge, pipeline, runner
from scorebench.dataset import Item
from scorebench.metrics import collect_outcomes, compare_models, score_model
from scorebench.providers.base import Provider, ProviderResult, Usage
from scorebench.tasks import TRIADS_ROOT

# Image-condition answers per item (every item's key is C major); MusicXML answers are all right.
ANSWERS = {
    0: "The notes are C, E, G with C in the bass, so this is C major.",
    1: "This spells B# major.",
    2: "C minor.",
    3: None,  # refusal
    4: "I can't make out the notes.",
    5: "Looks like A minor.",
}
# What the fake judge says about each answer. Item 5's verdict contradicts its own
# extracted answer, so the parser cross-check must flag it.
JUDGE = {
    ANSWERS[0]: ("C major", "correct"),
    ANSWERS[1]: ("B# major", "enharmonic"),
    ANSWERS[2]: ("C minor", "incorrect"),
    ANSWERS[4]: ("", "no_answer"),
    ANSWERS[5]: ("A minor", "correct"),
    "C major": ("C major", "correct"),
}


class FakeModel(Provider):
    def build_request(self, request):
        return {}

    async def complete(self, request):
        usage = Usage(input_tokens=500, output_tokens=100)
        if request.image is None:
            return ProviderResult(text="C major", usage=usage)
        text = ANSWERS[int(request.image.decode())]
        if text is None:
            return ProviderResult(text="", usage=usage, refused=True)
        return ProviderResult(text=text, usage=usage)


class FakeJudge(Provider):
    prompts: list[str] = []

    def build_request(self, request):
        return {}

    async def complete(self, request):
        assert request.image is None and request.schema == TRIADS_ROOT.judge_schema()
        FakeJudge.prompts.append(request.prompt)
        response = request.prompt.split("<response>\n", 1)[1].split("\n</response>", 1)[0]
        final, verdict = JUDGE[response]
        body = {"final_answer": final, "verdict": verdict, "explanation": "test"}
        return ProviderResult(text=json.dumps(body), usage=Usage(input_tokens=700, output_tokens=50))


@pytest.fixture
def items(item_factory, monkeypatch):
    monkeypatch.setattr(Item, "load_image", lambda self: str(self.index).encode())
    monkeypatch.setattr(Item, "load_musicxml", lambda self: "<xml/>")
    monkeypatch.setattr(runner, "manifest_hash", lambda subset: "m")
    monkeypatch.setattr(runner, "items_hash", lambda ids, subset: "h")
    return [item_factory(i, key_signature=(i % 2) * 2) for i in range(6)]


async def _run_and_judge(tmp_path, spec_factory, items, monkeypatch):
    spec = spec_factory()
    jspec = spec_factory(id="judge", provider="openai")
    monkeypatch.setattr(runner, "make_provider", lambda s: FakeModel(s))
    monkeypatch.setattr(judge, "make_provider", lambda s: FakeJudge(s))
    FakeJudge.prompts = []
    await runner.run_model(spec, items, "triads_root", results_dir=tmp_path)
    summary = await judge.judge_model(jspec, spec.id, items, "triads_root", results_dir=tmp_path)
    fp = judge.judge_fingerprint(jspec, TRIADS_ROOT)
    recs = runner.read_records(runner.predictions_path(spec.id, "triads_root", tmp_path))
    judgments = judge.current_judgments(spec.id, "triads_root", fp, tmp_path)
    return spec, jspec, summary, recs, judgments


async def test_judge_grades_and_flags(tmp_path, spec_factory, items, monkeypatch):
    spec, jspec, summary, recs, judgments = await _run_and_judge(tmp_path, spec_factory, items, monkeypatch)
    # 12 requests, 1 refusal -> 11 answers to judge
    assert summary.attempted == 11 and summary.statuses == {"ok": 11}
    assert summary.disagreements == 1
    assert "Answer key: C major" in FakeJudge.prompts[0]
    flagged = [j for j in judgments.values() if j["agrees_with_parser"] is False]
    assert [(j["item_id"], j["final_answer"], j["parser_verdict"]) for j in flagged] == [
        ("item-005", "A minor", "incorrect")
    ]

    # Judging again is a no-op; a new judge setup re-judges everything.
    again = await judge.judge_model(jspec, spec.id, items, "triads_root", results_dir=tmp_path)
    assert again.attempted == 0
    changed = spec_factory(id="judge", provider="openai", params={"reasoning_effort": "low"})
    redo = await judge.judge_model(changed, spec.id, items, "triads_root", results_dir=tmp_path)
    assert redo.attempted == 11


async def test_scores(tmp_path, spec_factory, items, monkeypatch):
    spec, _, _, recs, judgments = await _run_and_judge(tmp_path, spec_factory, items, monkeypatch)
    s = score_model(spec.id, items, recs, judgments)
    img, xml = s.conditions["image"], s.conditions["musicxml"]
    assert s.coverage == 1 and s.unjudged == 0 and s.judge_disagreements == 1
    assert img["accuracy"] == pytest.approx(2 / 6)  # items 0 and 5 (judge says correct)
    assert img["lenient_accuracy"] == pytest.approx(3 / 6)
    assert img["failure_rate"] == pytest.approx(2 / 6)  # refusal + no_answer
    assert img["breakdowns"]["Accidentals"] == pytest.approx(1 / 3)  # items 0, 2, 4
    assert img["breakdowns"]["Key sig."] == pytest.approx(1 / 3)  # items 1, 3, 5
    assert xml["accuracy"] == 1.0
    assert s.reading_gap["gap"] == pytest.approx(1 - 2 / 6)


async def test_unjudged_answers_are_missing_not_wrong(tmp_path, spec_factory, items, monkeypatch):
    spec, _, _, recs, _ = await _run_and_judge(tmp_path, spec_factory, items, monkeypatch)
    s = score_model(spec.id, items, recs, judgments={})
    assert s.unjudged == 11
    assert s.conditions["image"]["n_items"] == 1  # only the refusal is graded (as wrong)
    assert s.conditions["image"]["accuracy"] == 0


async def test_compare(tmp_path, spec_factory, items, monkeypatch):
    spec, _, _, recs, judgments = await _run_and_judge(tmp_path, spec_factory, items, monkeypatch)
    outcomes, _ = collect_outcomes(items, recs, judgments)
    c = compare_models(items, "a", outcomes, "b", outcomes, "image")
    assert c.n_common == 6 and c.diff == 0


def test_bad_judgment_is_rejected():
    with pytest.raises(ValueError):
        judge._parse_judgment(
            '{"final_answer": "C major", "verdict": "probably", "explanation": ""}', TRIADS_ROOT.verdicts
        )


async def test_pipeline_runs_models_together_and_judges_as_answers_arrive(tmp_path, spec_factory, items, monkeypatch):
    """Model B can't answer until the judge has graded one of model A's answers, so this
    only finishes if models run concurrently and judging overlaps the runs."""
    first_judgment = asyncio.Event()

    class WaitsForJudge(FakeModel):
        async def complete(self, request):
            await first_judgment.wait()
            return await super().complete(request)

    a, b = spec_factory(id="model-a"), spec_factory(id="model-b")
    jspec = spec_factory(id="judge", provider="openai")
    monkeypatch.setattr(runner, "make_provider", lambda s: WaitsForJudge(s) if s.id == "model-b" else FakeModel(s))
    monkeypatch.setattr(judge, "make_provider", lambda s: FakeJudge(s))
    runs, judged = await asyncio.wait_for(
        pipeline.run_and_judge(
            [b, a],
            items,
            "triads_root",
            judge_spec=jspec,
            results_dir=tmp_path,
            on_judged=lambda mid, out, spent: first_judgment.set(),
        ),
        timeout=5,
    )
    for mid in ("model-a", "model-b"):
        assert runs[mid].attempted == 12
        assert judged[mid].attempted == 11 and judged[mid].statuses == {"ok": 11}
    # Nothing left over for `scorebench judge`.
    assert await judge.judge_model(jspec, "model-b", items, "triads_root", results_dir=tmp_path) == judge.JudgeSummary(
        model_id="model-b"
    )


async def test_pipeline_judges_old_backlog_and_survives_a_config_mismatch(tmp_path, spec_factory, items, monkeypatch):
    monkeypatch.setattr(runner, "make_provider", lambda s: FakeModel(s))
    monkeypatch.setattr(judge, "make_provider", lambda s: FakeJudge(s))
    a, b = spec_factory(id="model-a"), spec_factory(id="model-b")
    await runner.run_model(a, items, "triads_root", results_dir=tmp_path)  # answered, never judged
    await runner.run_model(b, items[:2], "triads_root", results_dir=tmp_path)
    b_changed = spec_factory(id="model-b", params={"effort": "low"})

    runs, judged = await pipeline.run_and_judge(
        [a, b_changed],
        items,
        "triads_root",
        judge_spec=spec_factory(id="judge", provider="openai"),
        results_dir=tmp_path,
    )
    assert runs["model-a"].attempted == 0 and judged["model-a"].attempted == 11
    assert isinstance(runs["model-b"], runner.RunConfigMismatch) and "model-b" not in judged


async def test_repeats_add_samples_and_count_mixed_items(tmp_path, spec_factory, items, monkeypatch):
    """A second pass (--repeats 2) adds a sample per item without touching the first; items
    answered right once and wrong once count as mixed and score 0.5."""
    spec, jspec, _, recs, judgments = await _run_and_judge(tmp_path, spec_factory, items, monkeypatch)

    # Second pass: now every image answer is right ("C major"), so items 1-5 become mixed.
    class RightNow(FakeModel):
        async def complete(self, request):
            return ProviderResult(text="C major", usage=Usage(input_tokens=500, output_tokens=100))

    monkeypatch.setattr(runner, "make_provider", lambda s: RightNow(s))
    s2 = await runner.run_model(spec, items, "triads_root", repeats=2, results_dir=tmp_path)
    assert s2.skipped == 12 and s2.attempted == 12  # only the new samples are asked
    j2 = await judge.judge_model(jspec, spec.id, items, "triads_root", results_dir=tmp_path)
    assert j2.attempted == 12  # only the new answers are judged

    recs = runner.read_records(runner.predictions_path(spec.id, "triads_root", tmp_path))
    judgments = judge.current_judgments(spec.id, "triads_root", judge.judge_fingerprint(jspec, TRIADS_ROOT), tmp_path)
    s = score_model(spec.id, items, recs, judgments)
    img = s.conditions["image"]
    assert img["samples_per_item"] == 2
    # first pass: items 0 and 5 right; second pass: all right -> 0 and 5 score 1, the rest 0.5
    assert img["accuracy"] == pytest.approx((1 + 0.5 * 4 + 1) / 6)
    assert img["mixed_items"] == 4
    assert s.conditions["musicxml"]["mixed_items"] == 0  # right both times
