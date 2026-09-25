import pytest

from harmonybench.chords import all_triads, root_position_voicings
from harmonybench.parsing import grade, parse_chord


@pytest.mark.parametrize(
    "text,expected",
    [
        ("ANSWER: F# minor", ("F#", "minor")),
        ("ANSWER: F-sharp minor", ("F#", "minor")),
        ("ANSWER: Gb major", ("Gb", "major")),
        ("ANSWER: B-flat major", ("Bb", "major")),
        ("ANSWER: E♭ major", ("Eb", "major")),
        ("ANSWER: Bbm", ("Bb", "minor")),
        ("ANSWER: bb", ("Bb", "minor")),
        ("ANSWER: C", ("C", "major")),
        ("ANSWER: Dmin", ("D", "minor")),
        ("ANSWER: B major (B–D#–F#)", ("B", "major")),
        ("Looks like C major at first.\nANSWER: C minor", ("C", "minor")),
        ("no idea", None),
    ],
)
def test_parse(text, expected):
    assert parse_chord(text) == expected


@pytest.mark.parametrize("triad", all_triads(), ids=lambda t: t.name)
def test_canonical_answer_is_correct(triad):
    label = {"root": triad.root, "quality": triad.quality}
    assert grade(f"ANSWER: {triad.name}", label)["correct"]


def test_enharmonic_answer_gets_partial_credit_only():
    r = grade("ANSWER: F# major", {"root": "Gb", "quality": "major"})
    assert not r["correct"] and r["enharmonic_correct"]


@pytest.mark.parametrize("triad", all_triads(), ids=lambda t: t.name)
def test_every_triad_has_voicings(triad):
    voicings = root_position_voicings(triad)
    assert voicings
    for v in voicings:
        assert v["B"].name == triad.tones()[0]
        assert {p.name for p in v.values()} == set(triad.tones())
