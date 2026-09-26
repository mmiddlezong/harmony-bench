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
from .runner import predictions_path, read_records
from .tasks import Task, get_task

CONDITION_NAMES = {"image": "Image", "musicxml": "MusicXML"}


def results_base(subset: str, results_dir: Path = RESULTS_DIR) -> Path:
    return results_dir / subset / get_task(subset).version


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
    scores.sort(key=lambda s: -(s.accuracy if s.accuracy is not None else -1))
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


def _columns(task: Task) -> list[tuple[str, callable]]:
    """(header, cell function) for every leaderboard column after the model name."""
    first, others = task.conditions[0], task.conditions[1:]

    def acc(s):
        p = s.primary
        lo, hi = p["accuracy_ci95"]
        return f"{_pct(p['accuracy'])} ({_pct(lo)}–{_pct(hi)})"

    cols = [(f"{CONDITION_NAMES.get(first, first)} accuracy (95% CI)", acc)]
    if task.lenient:
        cols.append((task.lenient_name, lambda s: _pct(s.primary.get("lenient_accuracy"))))
    for name in task.breakdowns:
        cols.append((name, lambda s, name=name: _pct(s.primary["breakdowns"].get(name))))
    for c in others:
        cols.append(
            (f"{CONDITION_NAMES.get(c, c)} accuracy", lambda s, c=c: _pct(s.conditions.get(c, {}).get("accuracy")))
        )
    if "musicxml" in others:
        cols.append(("Reading gap", lambda s: _pp(s.reading_gap.get("gap"))))
    cols += [
        ("Fail", lambda s: _pct(s.primary["failure_rate"])),
        ("Judge flags", lambda s: str(s.judge_disagreements)),
        ("Cost", lambda s: f"${s.usage.get('projected_cost_full_run_usd', 0):.2f}"),
        ("n", lambda s: f"{s.primary['n_items']}/{s.n_total}" + ("" if s.coverage == 1 else " ⚠")),
    ]
    return cols


def leaderboard_markdown(scores: list[ModelScore], registry: Registry | None, subset: str) -> str:
    task = get_task(subset)
    cols = _columns(task)
    ranked = [s for s in scores if s.primary]
    lines = [
        f"# HarmonyBench leaderboard: {subset} (prompt {task.version})",
        "",
        f"_Generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}. Ranked by "
        f"{CONDITION_NAMES.get(task.conditions[0], task.conditions[0]).lower()} accuracy. 95% CIs from 10,000 "
        "item-level bootstrap resamples." + "_",
        "",
        "| # | Model | " + " | ".join(h for h, _ in cols) + " |",
        "|---:|---|" + "|".join("---:" if i else "---" for i in range(len(cols))) + "|",
    ]
    for rank, s in enumerate(ranked, 1):
        name, _ = _name(registry, s.model_id)
        lines.append(f"| {rank} | **{name}** | " + " | ".join(f(s) for _, f in cols) + " |")
    lines += [
        "",
        "**Columns.** *Accuracy*: share of items the judge model graded correct. "
        + (f"*{task.lenient_name}*: also counts near misses ({', '.join(task.lenient)}). " if task.lenient else "")
        + (
            "*Reading gap*: MusicXML accuracy minus image accuracy on the same items; a large gap means the "
            "model knows the harmony but misreads the image. "
            if "musicxml" in task.conditions
            else ""
        )
        + "*Fail*: empty answers, refusals, truncations and answers that name nothing (all scored wrong). "
        "*Judge flags*: judge verdicts that disagree with the rule-based parser, worth checking by hand "
        "(`harmonybench disagreements`). *Cost*: API spend to run every item once in every condition, at "
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
                "prompt_version": get_task(subset).version,
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
    """Leaderboard table for the top of the README (triads_root): complete runs only."""
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
