"""Access to a subset (data/<subset>/<manifest>.jsonl), its images, and its MusicXML if any."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .paths import DATA_DIR

DEFAULT_SUBSET = "triads_root"


def subset_dir(subset: str) -> Path:
    return DATA_DIR / subset


def items_path(subset: str) -> Path:
    from .tasks import get_task

    return subset_dir(subset) / get_task(subset).manifest


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class Item:
    index: int
    item_id: str
    subset: str
    image: str
    image_sha256: str
    label: dict
    meta: dict
    musicxml: str | None = None
    musicxml_sha256: str | None = None

    def _read(self, rel: str, sha: str) -> bytes:
        path = subset_dir(self.subset) / rel
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}.")
        data = path.read_bytes()
        if _sha256(data) != sha:
            raise ValueError(f"Checksum mismatch for {path}; the file changed after its manifest was written.")
        return data

    def load_image(self) -> bytes:
        return self._read(self.image, self.image_sha256)

    def load_musicxml(self) -> str:
        if self.musicxml is None:
            raise ValueError(f"{self.item_id} has no MusicXML")
        return self._read(self.musicxml, self.musicxml_sha256).decode("utf-8")


def load_items(subset: str = DEFAULT_SUBSET, limit: int | None = None) -> list[Item]:
    """Load a subset in file order. Generated subsets are shuffled at build time, so any
    prefix is a random sample."""
    path = items_path(subset)
    if not path.exists():
        raise FileNotFoundError(f"No items at {path}.")
    items = []
    with path.open() as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            items.append(
                Item(
                    index=i,
                    item_id=row["id"],
                    subset=row["subset"],
                    image=row["image"],
                    image_sha256=row["image_sha256"],
                    label=row["label"],
                    meta=row.get("meta", {}),
                    musicxml=row.get("musicxml"),
                    musicxml_sha256=row.get("musicxml_sha256"),
                )
            )
    return items[:limit] if limit else items


def manifest_hash(subset: str = DEFAULT_SUBSET) -> str:
    return _sha256(items_path(subset).read_bytes())[:16]


def items_hash(item_ids, subset: str = DEFAULT_SUBSET) -> str:
    """Fingerprint of specific items (labels and file checksums), ignoring their position.
    Lets results survive the subset changing: a run stays valid as long as every item it has
    answers for is unchanged. Items since removed from the subset are skipped (their answers
    are simply no longer scored)."""
    rows = {}
    with items_path(subset).open() as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows[row["id"]] = row
    h = hashlib.sha256()
    for item_id in sorted(set(item_ids) & set(rows)):
        h.update(json.dumps(rows[item_id], sort_keys=True).encode())
    return h.hexdigest()[:16]
