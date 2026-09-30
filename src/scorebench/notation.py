"""Build a one-chord SATB grand-staff score and export it as MusicXML."""

import re

from music21 import clef, key, layout, metadata, meter, note, stream
from music21.musicxml.m21ToXml import GeneralObjectExporter


def build_score(voicing: dict, key_sharps: int) -> stream.Score:
    score = stream.Score()
    score.metadata = metadata.Metadata(title="", composer="")

    staves = []
    for part_id, clef_obj, (upper, lower) in [
        ("P1", clef.TrebleClef(), ("S", "A")),
        ("P2", clef.BassClef(), ("T", "B")),
    ]:
        part = stream.PartStaff(id=part_id)
        measure = stream.Measure(number=1)
        measure.append([clef_obj, key.KeySignature(key_sharps), meter.TimeSignature("4/4")])
        for voice_id, name in [("1", upper), ("2", lower)]:
            v = stream.Voice(id=voice_id)
            n = note.Note(voicing[name], type="whole")
            v.append(n)
            measure.insert(0, v)
        measure.rightBarline = "final"
        part.append(measure)
        score.insert(0, part)
        staves.append(part)

    score.insert(0, layout.StaffGroup(staves, symbol="brace", barTogether=True))
    return score


def to_musicxml(score: stream.Score) -> str:
    xml = GeneralObjectExporter(score).parse().decode("utf-8")
    # Drop the export date and music21's random part ids so rebuilding produces
    # byte-identical files.
    xml = re.sub(r"\s*<encoding-date>[^<]*</encoding-date>", "", xml)
    ids: dict[str, str] = {}
    return re.sub(r'id="(P[0-9a-f]{32})"', lambda m: f'id="{ids.setdefault(m[1], f"P{len(ids) + 1}")}"', xml)
