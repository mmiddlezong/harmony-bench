"""The benchmark prompts.

Answers are free text: models may reason out loud and phrase the chord however they like,
and a judge model (see judge.py) grades the answer against the answer key.

Each item is asked under two conditions, to separate misreading the score from bad
analysis:
  image     the rendered score (PNG)
  musicxml  the same score as MusicXML text, no image

The prompts are versioned: results are stored under results/<subset>/<PROMPT_VERSION>/
and a run refuses to resume if the prompt text has changed. Bump PROMPT_VERSION whenever
the wording changes so results from different prompts are never mixed.
"""

from __future__ import annotations

import hashlib
import json

PROMPT_VERSION = "v1"
CONDITIONS = ("image", "musicxml")

_QUESTION = "What chord is this? Name its root, including any sharp or flat, and its quality (major or minor)."

PROMPTS = {
    "image": (
        "This image shows a single chord written in four-part harmony (soprano, alto, tenor, bass) "
        f"on a grand staff. {_QUESTION}"
    ),
    "musicxml": (
        "Below is a MusicXML file containing a single chord written in four-part harmony "
        f"(soprano, alto, tenor, bass) on a grand staff. {_QUESTION}\n\n{{musicxml}}"
    ),
}


def build_prompt(condition: str, musicxml: str | None = None) -> str:
    if condition == "musicxml":
        if musicxml is None:
            raise ValueError("the musicxml condition needs the item's MusicXML")
        return PROMPTS["musicxml"].format(musicxml=musicxml)
    return PROMPTS[condition]


def prompt_hash() -> str:
    """Fingerprint of everything that defines the task as seen by the model."""
    blob = json.dumps(PROMPTS, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
