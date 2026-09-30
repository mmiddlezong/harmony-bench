"""Chord spelling and SATB voicing for root-position triads."""

import itertools
import random
from dataclasses import dataclass

from music21 import key, pitch

# Roots of the 15 major and 15 minor key names, so every triad is spelled
# without double sharps or flats.
MAJOR_ROOTS = ["Cb", "Gb", "Db", "Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "C#"]
MINOR_ROOTS = ["Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "C#", "G#", "D#", "A#"]

THIRD = {"major": "M3", "minor": "m3"}

# Voice ranges as MIDI numbers (Aldwell & Schachter-style textbook ranges).
RANGES = {
    "B": (40, 60),  # E2-C4
    "T": (48, 67),  # C3-G4
    "A": (55, 74),  # G3-D5
    "S": (60, 79),  # C4-G5
}


@dataclass(frozen=True)
class Triad:
    root: str
    quality: str  # "major" | "minor"

    @property
    def name(self) -> str:
        return f"{self.root} {self.quality}"

    def tones(self) -> tuple[str, str, str]:
        r = pitch.Pitch(self.root)
        return (r.name, r.transpose(THIRD[self.quality]).name, r.transpose("P5").name)


def all_triads() -> list[Triad]:
    return [Triad(r, "major") for r in MAJOR_ROOTS] + [Triad(r, "minor") for r in MINOR_ROOTS]


def pitches_in_range(name: str, lo: int, hi: int) -> list[pitch.Pitch]:
    out = []
    for octave in range(1, 7):
        p = pitch.Pitch(f"{name}{octave}")
        if lo <= p.midi <= hi:
            out.append(p)
    return out


def root_position_voicings(triad: Triad) -> list[dict[str, pitch.Pitch]]:
    """Every textbook-legal root-position SATB voicing: bass on the root, root
    doubled, complete triad, no crossing or unisons, S-A and A-T within an
    octave, T-B within a twelfth."""
    root, third, fifth = triad.tones()
    voicings = []
    for b in pitches_in_range(root, *RANGES["B"]):
        # Upper voices hold the remaining {root, third, fifth} in some order.
        for upper in set(itertools.permutations([root, third, fifth])):
            t_name, a_name, s_name = upper
            for t, a, s in itertools.product(
                pitches_in_range(t_name, *RANGES["T"]),
                pitches_in_range(a_name, *RANGES["A"]),
                pitches_in_range(s_name, *RANGES["S"]),
            ):
                if not (b.midi < t.midi < a.midi < s.midi):
                    continue
                if s.midi - a.midi > 12 or a.midi - t.midi > 12 or t.midi - b.midi > 19:
                    continue
                voicings.append({"S": s, "A": a, "T": t, "B": b})
    voicings.sort(key=lambda v: [v[x].midi for x in "SATB"])
    return voicings


def diatonic_key_signatures(triad: Triad) -> list[int]:
    """Key signatures (as number of sharps, negative for flats) in which every
    chord tone is diatonic, so the score needs no accidentals."""
    tones = set(triad.tones())
    out = []
    for sharps in range(-7, 8):
        scale = {p.name for p in key.KeySignature(sharps).getScale("major").pitches}
        if tones <= scale:
            out.append(sharps)
    return out


def choose_voicing(triad: Triad, rng: random.Random) -> dict[str, pitch.Pitch]:
    return rng.choice(root_position_voicings(triad))
