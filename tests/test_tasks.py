import pytest

from harmonybench.dataset import Item, load_items
from harmonybench.judge import response_hash
from harmonybench.metrics import score_model
from harmonybench.tasks import TASKS, TRIADS_ROOT, WRONG_NOTE, get_task


def wrong_note_item(index=0, measure=29) -> Item:
    return Item(
        index=index,
        item_id=f"wrong_note-{index:03d}",
        subset="wrong_note",
        image="images/x.png",
        image_sha256="",
        label={"measure": measure},
        meta={"source": "test"},
    )


def test_wrong_note_prompt():
    item = wrong_note_item()
    assert WRONG_NOTE.build_prompt(item, "image") == (
        "This is an image of part of a score. One note has been modified from the original. "
        "Which measure is it in? Answer with its bar number."
    )


def test_wrong_note_judge_sees_only_the_response_and_the_key():
    item = wrong_note_item()
    j = WRONG_NOTE.build_judge_prompt(item, "It's in bar 29.")
    assert "Correct answer: measure 29" in j and "It's in bar 29." in j
    assert WRONG_NOTE.judge_schema()["required"] == ["verdict"]


def test_wrong_note_scoring():
    items = [wrong_note_item(i, measure=29) for i in range(2)]
    records, judgments = [], {}
    for i, (it, verdict) in enumerate(zip(items, ["correct", "incorrect"], strict=True)):
        records.append({"item_id": it.item_id, "condition": "image", "sample": 0, "status": "ok", "raw_text": str(i)})
        judgments[(it.item_id, "image", 0, response_hash(str(i)))] = {
            "verdict": verdict,
            "response_hash": response_hash(str(i)),
        }
    s = score_model("m", items, records, judgments)
    assert list(s.conditions) == ["image"] and s.coverage == 1
    assert s.accuracy == 0.5
    assert s.primary["lenient_accuracy"] is None and s.reading_gap == {}


def test_triad_prompts_are_unchanged():
    """Existing triads_root results are keyed to this fingerprint; bump the version to change it."""
    assert TRIADS_ROOT.version == "v1" and TRIADS_ROOT.prompt_hash() == "08c9d6df19d58ce6"


def test_registry():
    assert set(TASKS) == {"triads_root", "wrong_note"}
    with pytest.raises(KeyError):
        get_task("nope")


def test_wrong_note_manifest_loads_if_present():
    try:
        items = load_items("wrong_note")
    except FileNotFoundError:
        pytest.skip("data/wrong_note is private and not in the repo")
    assert all(it.label["measure"] for it in items)
    assert all(it.meta.get("source") for it in items)
