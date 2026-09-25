"""Leaderboard generation (Markdown + JSON) from stored predictions and judgments."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

from .config import Registry
from .dataset import Item
from .judge import current_judgments, judgments_path
from .metrics import ModelScore, score_model
from .paths import RESULTS_DIR
from .prompt import PROMPT_VERSION
from .runner import predictions_path, read_records


def results_base(subset: str, results_dir: Path = RESULTS_DIR) -> Path:
    return results_dir / subset / PROMPT_VERSION


def discover_models(subset: str, results_dir: Path = RESULTS_DIR) -> list[str]:
    base = results_base(subset, results_dir)
    if not base.exists():
        return []
    return sorted(p.name for p in base.iterdir() if (p / "predictions.jsonl").exists())


def score_all(
    items: list[Item], model_ids: list[str], subset: str, judge_fingerprint: str, results_dir: Path = RESULTS_DIR
) -> list[ModelScore]:
    scores = []
    for mid in model_ids:
        recs = read_records(predictions_path(mid, subset, results_dir))
        if recs:
            judgments = current_judgments(mid, subset, judge_fingerprint, results_dir)
            judge_recs = read_records(judgments_path(mid, subset, results_dir))
            scores.append(score_model(mid, items, recs, judgments, judge_recs))
    scores.sort(key=lambda s: -(s.image_accuracy if s.image_accuracy is not None else -1))
    return scores


def _pct(x: float | None, digits: int = 0) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{digits}f}%"


def _pp(x: float | None) -> str:
    """Signed percentage points."""
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:+.0f} pts"


def _name(registry: Registry | None, mid: str) -> tuple[str, str]:
    if registry is not None:
        try:
            spec = registry.get(mid)
            return spec.display_name, spec.lab
        except KeyError:
            pass
    return mid, ""


def leaderboard_markdown(scores: list[ModelScore], registry: Registry | None, subset: str) -> str:
    lines = [
        f"# HarmonyBench leaderboard: {subset} (prompt {PROMPT_VERSION})",
        "",
        f"_Generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}. "
        "Ranked by accuracy on score images. 95% CIs from 10,000 item-level bootstrap resamples._",
        "",
        "| # | Model | Image accuracy (95% CI) | Enharmonic | Key sig. | Accidentals | MusicXML accuracy "
        "| Reading gap | Fail | Judge flags | Cost | n |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rank = 0
    for s in scores:
        img = s.conditions.get("image")
        if not img:
            continue
        rank += 1
        name, _ = _name(registry, s.model_id)
        lo, hi = img["accuracy_ci95"]
        xml = s.conditions.get("musicxml", {})
        n = f"{img['n_items']}/{s.n_total}" + ("" if s.coverage == 1 else " ⚠")
        lines.append(
            f"| {rank} | **{name}** | {_pct(img['accuracy'])} ({_pct(lo)}–{_pct(hi)}) "
            f"| {_pct(img['enharmonic_accuracy'])} | {_pct(img['accuracy_with_key_signature'])} "
            f"| {_pct(img['accuracy_with_accidentals'])} | {_pct(xml.get('accuracy'))} "
            f"| {_pp(s.reading_gap.get('gap'))} | {_pct(img['failure_rate'])} | {s.judge_disagreements} "
            f"| ${s.usage.get('projected_cost_full_run_usd', 0):.2f} | {n} |"
        )
    lines += [
        "",
        "**Columns.** *Image accuracy*: share of score images where the model named the chord exactly "
        "(root spelled as written, and the right quality), as graded by the judge model. "
        "*Enharmonic*: also counts a root that sounds right but is spelled differently (F# for Gb). "
        "*Key sig.* / *Accidentals*: image accuracy on items written with a key signature / with no key "
        "signature and accidentals on the notes. *MusicXML accuracy*: the same items given as MusicXML "
        "text instead of an image. *Reading gap*: MusicXML accuracy minus image accuracy on the same items; "
        "a large gap means the model knows the harmony but misreads the image. *Fail*: empty answers, "
        "refusals, truncations and answers that name no chord (all scored wrong). *Judge flags*: judge "
        "verdicts that disagree with the rule-based parser, worth checking by hand "
        "(`harmonybench disagreements`). *Cost*: API spend to run every item once in both conditions, at "
        "list prices. ⚠ = incomplete run.",
        "",
        "## Usage",
        "",
        "| Model | Mean output tokens | Median latency | Spent so far | Judge cost |",
        "|---|---:|---:|---:|---:|",
    ]
    for s in scores:
        name, _ = _name(registry, s.model_id)
        u = s.usage
        tokens = f"{u['mean_output_tokens']:.0f}" if u.get("mean_output_tokens") else "–"
        if u.get("mean_reasoning_tokens"):
            tokens += f" ({u['mean_reasoning_tokens']:.0f} reasoning)"
        latency = f"{u['median_latency_s']:.1f}s" if u.get("median_latency_s") else "–"
        lines.append(
            f"| {name} | {tokens} | {latency} | ${u.get('total_cost_usd', 0):.2f} | ${u.get('judge_cost_usd', 0):.3f} |"
        )
    return "\n".join(lines) + "\n"


def write_leaderboard(
    scores: list[ModelScore], registry: Registry | None, subset: str, results_dir: Path = RESULTS_DIR
) -> tuple[Path, Path]:
    base = results_base(subset, results_dir)
    base.mkdir(parents=True, exist_ok=True)
    md_path = base / "leaderboard.md"
    json_path = base / "leaderboard.json"
    md_path.write_text(leaderboard_markdown(scores, registry, subset))
    json_path.write_text(
        json.dumps(
            {
                "subset": subset,
                "prompt_version": PROMPT_VERSION,
                "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "models": [s.to_dict() for s in scores],
            },
            indent=2,
        )
        + "\n"
    )
    return md_path, json_path


README_START = "<!-- LEADERBOARD:START -->"
README_END = "<!-- LEADERBOARD:END -->"


def readme_block(scores: list[ModelScore], registry: Registry | None, n_items: int) -> str:
    """Leaderboard table for the top of the README: complete runs only."""
    done = [s for s in scores if s.conditions.get("image") and s.coverage >= 1]
    lines = [
        "| Rank | Model | Named the chord correctly | Same chord, given as text | Cost |",
        "|:---:|---|---:|---:|---:|",
    ]
    for rank, s in enumerate(done, 1):
        name, lab = _name(registry, s.model_id)
        img = s.conditions["image"]
        lo, hi = img["accuracy_ci95"]
        xml = s.conditions.get("musicxml", {}).get("accuracy")
        lines.append(
            f"| {rank} | **{name}**{f' ({lab})' if lab else ''} | **{_pct(img['accuracy'])}** ({_pct(lo)}–{_pct(hi)}) "
            f"| {_pct(xml)} | ${s.usage.get('projected_cost_full_run_usd', 0):.2f} |"
        )
    lines += [
        "",
        f"Each model saw {n_items} chords, one image at a time, and was asked to name each one. The range in "
        "parentheses is a 95% confidence interval: when two models' ranges overlap, the difference between them "
        'may be luck. "Same chord, given as text" is the score when the model got the same chords as MusicXML '
        'instead of a picture. "Cost" is the API bill for one run over every chord in both forms. '
        f"Updated {datetime.now(UTC).strftime('%B %-d, %Y')}.",
    ]
    return "\n".join(lines)


def update_readme(readme: Path, scores: list[ModelScore], registry: Registry | None, n_items: int) -> bool:
    """Replace the marked leaderboard block in the README. Returns True if it changed."""
    if not readme.exists():
        return False
    text = readme.read_text()
    if README_START not in text or README_END not in text:
        return False
    head, rest = text.split(README_START, 1)
    _, tail = rest.split(README_END, 1)
    new = f"{head}{README_START}\n{readme_block(scores, registry, n_items)}\n{README_END}{tail}"
    if new != text:
        readme.write_text(new)
        return True
    return False
