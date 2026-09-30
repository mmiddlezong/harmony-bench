"""Parse free-text chord answers and grade them against labels."""

import re

from music21 import pitch

_ACCIDENTAL = {
    "": "",
    "#": "#",
    "♯": "#",
    "sharp": "#",
    "-sharp": "#",
    " sharp": "#",
    "b": "-",
    "♭": "-",
    "flat": "-",
    "-flat": "-",
    " flat": "-",
}

_MAJOR = {"major", "maj", "M", "Δ"}
_MINOR = {"minor", "min", "m", "-"}

_CHORD_RE = re.compile(
    r"(?<![A-Za-z])([A-Ga-g])"
    r"(#|♯|b|♭|-sharp|-flat| sharp| flat|sharp|flat)?"
    r"\s*(major|minor|maj|min|M|m|-|Δ)?"
    r"(?![A-Za-z])",
)


def parse_chord(text: str) -> tuple[str, str] | None:
    """Return (root, quality) from a response, e.g. ("F#", "minor"), using
    the last line that starts with "ANSWER:" if present, else the whole text.
    Root spelling is kept (Gb stays Gb). A bare lowercase root with no quality
    ("f#") is read as minor; a bare uppercase root as major."""
    answer_lines = [ln for ln in text.splitlines() if ln.strip().upper().startswith("ANSWER:")]
    target = answer_lines[-1].split(":", 1)[1] if answer_lines else text
    matches = list(_CHORD_RE.finditer(target.strip()))
    if not matches:
        return None
    m = matches[0] if answer_lines else matches[-1]
    letter, acc, qual = m.group(1), (m.group(2) or "").lower(), m.group(3)
    if qual in _MAJOR:
        quality = "major"
    elif qual in _MINOR:
        quality = "minor"
    else:
        quality = "minor" if letter.islower() else "major"
    root = pitch.Pitch(letter.upper() + _ACCIDENTAL[acc]).name.replace("-", "b")
    return root, quality


def grade(response: str, label: dict) -> dict:
    parsed = parse_chord(response)
    if parsed is None:
        return {"parsed": None, "correct": False, "enharmonic_correct": False}
    root, quality = parsed
    same_pc = pitch.Pitch(root.replace("b", "-")).pitchClass == pitch.Pitch(label["root"].replace("b", "-")).pitchClass
    return {
        "parsed": f"{root} {quality}",
        "correct": root == label["root"] and quality == label["quality"],
        "enharmonic_correct": same_pc and quality == label["quality"],
    }
