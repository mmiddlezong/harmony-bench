"""Generate the root-position triad subset: images, MusicXML, and labels.

Deterministic: the same seed always produces byte-identical files, so rebuilding never
invalidates stored results.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import random
from pathlib import Path

from music21 import chord, converter, key

from .chords import all_triads, choose_voicing, diatonic_key_signatures
from .notation import build_score, to_musicxml
from .render import musicxml_to_svg, svg_to_png

SUBSET = "triads_root"


def verify(xml_path: Path, label: dict, key_sharps: int) -> None:
    """Re-read the exported file and check it independently of the generator."""
    score = converter.parse(xml_path)
    notes = list(score.recurse().notes)
    assert len(notes) == 4, f"{xml_path}: expected 4 notes, got {len(notes)}"
    c = chord.Chord([n.pitch for n in notes])
    root = c.root().name.replace("-", "b")
    assert root == label["root"], f"{xml_path}: root {root} != {label['root']}"
    assert c.quality == label["quality"], f"{xml_path}: quality {c.quality} != {label['quality']}"
    assert c.bass().name == c.root().name, f"{xml_path}: not in root position"
    sigs = {ks.sharps for ks in score.recurse().getElementsByClass(key.KeySignature)}
    assert sigs == {key_sharps}, f"{xml_path}: key signatures {sigs} != {key_sharps}"


def generate(out: Path, per_chord: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    specs = []
    for triad in all_triads():
        for i in range(per_chord):
            # Alternate between no key signature (chord spelled with accidentals)
            # and a key signature in which the chord is diatonic.
            if i % 2 == 0:
                key_sharps = 0
            else:
                options = [k for k in diatonic_key_signatures(triad) if k != 0]
                key_sharps = rng.choice(options)
            specs.append((triad, key_sharps))
    rng.shuffle(specs)

    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "musicxml").mkdir(parents=True, exist_ok=True)

    items = []
    for n, (triad, key_sharps) in enumerate(specs, start=1):
        item_id = f"{SUBSET}-{n:03d}"
        voicing = choose_voicing(triad, rng)
        xml = to_musicxml(build_score(voicing, key_sharps))
        xml_path = out / "musicxml" / f"{item_id}.musicxml"
        png_path = out / "images" / f"{item_id}.png"
        xml_path.write_text(xml)
        png_path.write_bytes(svg_to_png(musicxml_to_svg(xml)))

        label = {"root": triad.root, "quality": triad.quality, "canonical": triad.name}
        verify(xml_path, label, key_sharps)
        items.append(
            {
                "id": item_id,
                "subset": SUBSET,
                "image": str(png_path.relative_to(out)),
                "musicxml": str(xml_path.relative_to(out)),
                "image_sha256": hashlib.sha256(png_path.read_bytes()).hexdigest(),
                "musicxml_sha256": hashlib.sha256(xml_path.read_bytes()).hexdigest(),
                "label": label,
                "meta": {
                    "voicing": {v: p.nameWithOctave.replace("-", "b") for v, p in voicing.items()},
                    "key_signature": key_sharps,
                    "has_accidentals": key_sharps == 0 and any(p.accidental for p in voicing.values()),
                },
            }
        )

    with open(out / "items.jsonl", "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
    write_review_page(out, items)
    return items


def write_review_page(out: Path, items: list[dict]) -> None:
    """A self-contained contact sheet (PNGs embedded) for hand-checking every
    item against its label."""
    cards = []
    for it in items:
        v = it["meta"]["voicing"]
        png = base64.b64encode((out / it["image"]).read_bytes()).decode("ascii")
        cards.append(
            f'<figure><img src="data:image/png;base64,{png}" alt="{html.escape(it["id"])}">'
            f"<figcaption><b>{html.escape(it['id'])}</b> &middot; {html.escape(it['label']['canonical'])}<br>"
            f"S {v['S']} &middot; A {v['A']} &middot; T {v['T']} &middot; B {v['B']} "
            f"&middot; key sig {it['meta']['key_signature']:+d}</figcaption></figure>"
        )
    (out / "review.html").write_text(
        "<!doctype html><meta charset=utf-8><title>triads_root review</title>"
        "<style>body{font:14px system-ui;margin:16px;background:#fff}"
        "main{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}"
        "figure{margin:0;border:1px solid #ddd;padding:8px}img{width:100%}</style>"
        f"<h1>triads_root: {len(items)} items</h1><main>{''.join(cards)}</main>"
    )
