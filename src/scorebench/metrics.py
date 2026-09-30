"""Scoring: per-model accuracy, bootstrap confidence intervals, and paired comparisons.

Headline metric
---------------
Accuracy on the task's first condition (the score image): the share of items the judge
grades "correct". Secondary, where the task defines them:
  * lenient accuracy: also credits near misses (e.g. an enharmonically spelled chord)
  * accuracy on the other condition (e.g. the same items as MusicXML text)
  * reading gap: MusicXML accuracy minus image accuracy, over items answered in both.
    A large gap means the model understands the harmony but misreads the score.
  * breakdowns: accuracy on subsets of items

Conventions
-----------
* The unit of analysis is the item. With --repeats > 1, per-item correctness is averaged
  over samples first, then over items (so every item carries equal weight). `mixed_items`
  counts items a model got right on some samples and wrong on others.
* A model failure (empty answer, refusal, truncation, or a judge verdict of no_answer)
  counts as wrong. This keeps denominators equal across models.
* Infrastructure errors (status api_error) and answers the judge has not graded yet are
  NOT counted against the model; those items are simply missing, and `coverage` shows how
  complete the run is. Re-run `scorebench run` / `scorebench judge` to fill them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dataset import Item
from .judge import response_hash
from .runner import FINAL_STATUSES
from .tasks import Task, get_task

BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 0


def final_records(records: list[dict]) -> dict[tuple[str, str, int], dict]:
    """Latest model-outcome record per (item, condition, sample); falls back to the latest record."""
    best: dict[tuple[str, str, int], dict] = {}
    for r in records:
        key = (r["item_id"], r["condition"], r.get("sample", 0))
        prev = best.get(key)
        if prev is None or r.get("status") in FINAL_STATUSES or prev.get("status") not in FINAL_STATUSES:
            best[key] = r
    return best


@dataclass
class Outcome:
    """One graded sample. verdict: correct | enharmonic | incorrect | no_answer | failed."""

    item: Item
    condition: str
    sample: int
    verdict: str
    record: dict
    judgment: dict | None = None


def collect_outcomes(items: list[Item], records: list[dict], judgments: dict) -> tuple[list[Outcome], int]:
    """Graded samples, plus the number of answered samples still waiting for the judge."""
    by_id = {it.item_id: it for it in items}
    outcomes, unjudged = [], 0
    for (item_id, condition, sample), rec in final_records(records).items():
        item = by_id.get(item_id)
        status = rec.get("status")
        if item is None or status not in FINAL_STATUSES:
            continue
        if status != "ok":
            outcomes.append(Outcome(item, condition, sample, "failed", rec))
            continue
        j = judgments.get((item_id, condition, sample, response_hash(rec.get("raw_text", ""))))
        if j is None:
            unjudged += 1
            continue
        outcomes.append(Outcome(item, condition, sample, j["verdict"], rec, j))
    outcomes.sort(key=lambda o: (o.item.index, o.condition, o.sample))
    return outcomes, unjudged


def per_item(outcomes: list[Outcome], condition: str, credit=("correct",)) -> dict[str, float]:
    """item_id -> share of samples credited (averaged over samples)."""
    hits: dict[str, list[float]] = {}
    for o in outcomes:
        if o.condition == condition:
            hits.setdefault(o.item.item_id, []).append(1.0 if o.verdict in credit else 0.0)
    return {k: float(np.mean(v)) for k, v in hits.items()}


def _bootstrap_ci(values: np.ndarray, n: int = BOOTSTRAP_SAMPLES) -> tuple[float, float]:
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    lo, hi = np.percentile(values[idx].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def _condition_stats(outcomes: list[Outcome], items: list[Item], condition: str, task: Task) -> dict:
    exact = per_item(outcomes, condition)
    if not exact:
        return {}
    ids = [it.item_id for it in items if it.item_id in exact]
    acc = np.array([exact[i] for i in ids])
    lenient = per_item(outcomes, condition, credit=("correct", *task.lenient))
    mine = [o for o in outcomes if o.condition == condition]
    by_id = {it.item_id: it for it in items}

    def acc_where(pred) -> float | None:
        vals = [exact[i] for i in ids if pred(by_id[i])]
        return float(np.mean(vals)) if vals else None

    samples = {}
    for o in mine:
        samples[o.item.item_id] = samples.get(o.item.item_id, 0) + 1
    return {
        "n_items": len(ids),
        "samples_per_item": float(np.mean([samples[i] for i in ids])),
        # items graded more than once with mixed results (right on some runs, wrong on others)
        "mixed_items": sum(1 for i in ids if samples[i] > 1 and 0 < exact[i] < 1),
        "accuracy": float(acc.mean()),
        "accuracy_ci95": _bootstrap_ci(acc),
        "lenient_accuracy": float(np.mean([lenient[i] for i in ids])) if task.lenient else None,
        "failure_rate": float(np.mean([o.verdict in ("failed", "no_answer") for o in mine])),
        "breakdowns": {name: acc_where(pred) for name, pred in task.breakdowns.items()},
    }


@dataclass
class ModelScore:
    model_id: str
    n_total: int
    coverage: float  # share of (item, condition) pairs that are graded
    status_counts: dict[str, int]
    unjudged: int
    judge_disagreements: int
    conditions: dict[str, dict] = field(default_factory=dict)
    reading_gap: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)

    @property
    def primary(self) -> dict:
        """Stats for the headline condition (the task's first: the score image)."""
        return next(iter(self.conditions.values()), {})

    @property
    def accuracy(self) -> float | None:
        return self.primary.get("accuracy")

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "n_total": self.n_total,
            "coverage": self.coverage,
            "status_counts": self.status_counts,
            "unjudged": self.unjudged,
            "judge_disagreements": self.judge_disagreements,
            "conditions": self.conditions,
            "reading_gap": self.reading_gap,
            "usage": self.usage,
        }


def score_model(
    model_id: str,
    items: list[Item],
    records: list[dict],
    judgments: dict,
    judge_records: list[dict] | None = None,
) -> ModelScore:
    task = get_task(items[0].subset) if items else None
    conditions = task.conditions if task else ()
    outcomes, unjudged = collect_outcomes(items, records, judgments)
    finals = final_records(records)
    counts: dict[str, int] = {}
    for rec in finals.values():
        counts[rec.get("status", "unknown")] = counts.get(rec.get("status", "unknown"), 0) + 1
    graded_pairs = {(o.item.item_id, o.condition) for o in outcomes}
    used = {(o.item.item_id, o.condition, o.sample, o.judgment["response_hash"]) for o in outcomes if o.judgment}

    score = ModelScore(
        model_id=model_id,
        n_total=len(items),
        coverage=len(graded_pairs) / (len(items) * len(conditions)) if items else 0.0,
        status_counts=counts,
        unjudged=unjudged,
        judge_disagreements=sum(1 for k, j in judgments.items() if k in used and j.get("agrees_with_parser") is False),
    )
    for c in conditions:
        stats = _condition_stats(outcomes, items, c, task)
        if stats:
            score.conditions[c] = stats

    img, xml = per_item(outcomes, "image"), per_item(outcomes, "musicxml")
    common = [it.item_id for it in items if it.item_id in img and it.item_id in xml]
    if common:
        diffs = np.array([xml[i] - img[i] for i in common])
        score.reading_gap = {"n_items": len(common), "gap": float(diffs.mean()), "gap_ci95": _bootstrap_ci(diffs)}

    recs = list(finals.values())
    spent = sum(r.get("cost_usd", 0.0) or 0.0 for r in records)  # includes superseded retries
    judge_spent = sum(j.get("cost_usd", 0.0) or 0.0 for j in judge_records or [])
    per_cond_cost = {
        c: [r.get("cost_usd", 0.0) or 0.0 for r in recs if r["condition"] == c and r.get("status") in FINAL_STATUSES]
        for c in conditions
    }
    lat = [r["latency_s"] for r in recs if r.get("status") in FINAL_STATUSES and "latency_s" in r]
    outs = [r["usage"]["output_tokens"] for r in recs if r.get("usage")]
    reas = [r["usage"]["reasoning_tokens"] for r in recs if r.get("usage")]
    score.usage = {
        "total_cost_usd": float(spent),
        "judge_cost_usd": float(judge_spent),
        "projected_cost_full_run_usd": float(sum(np.mean(v) * len(items) for v in per_cond_cost.values() if v)),
        "mean_output_tokens": float(np.mean(outs)) if outs else 0.0,
        "mean_reasoning_tokens": float(np.mean(reas)) if reas else 0.0,
        "median_latency_s": float(np.median(lat)) if lat else 0.0,
    }
    return score


@dataclass
class Comparison:
    model_a: str
    model_b: str
    condition: str
    n_common: int
    acc_a: float
    acc_b: float
    diff: float  # acc_a - acc_b (positive = A better)
    diff_ci95: tuple[float, float]
    p_value: float


def compare_models(
    items: list[Item], id_a: str, outcomes_a: list[Outcome], id_b: str, outcomes_b: list[Outcome], condition: str
) -> Comparison:
    """Paired bootstrap on per-item correctness, over items both models were graded on."""
    a_map, b_map = per_item(outcomes_a, condition), per_item(outcomes_b, condition)
    common = [it.item_id for it in items if it.item_id in a_map and it.item_id in b_map]
    nan = float("nan")
    if len(common) < 2:
        return Comparison(id_a, id_b, condition, len(common), nan, nan, nan, (nan, nan), nan)
    a = np.array([a_map[i] for i in common])
    b = np.array([b_map[i] for i in common])
    diffs = a - b
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(diffs), size=(BOOTSTRAP_SAMPLES, len(diffs)))
    boot = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return Comparison(
        id_a,
        id_b,
        condition,
        len(common),
        float(a.mean()),
        float(b.mean()),
        float(diffs.mean()),
        (float(lo), float(hi)),
        float(min(1.0, p)),
    )
