"""Grade free-text answers with a cheap judge model (configs/models.yaml `judge:`).

The judge sees the answer key and the model's full response, and returns the answer the
model finally committed to plus a verdict. The rubric comes from the task (tasks.py).
Judgments live next to the predictions in
results/<subset>/<task_version>/<model_id>/judgments.jsonl, so the judge can be changed
and re-run without asking the benchmarked models again.

Cross-check: the judge's extracted answer is also graded by the task's deterministic
parser. When the two verdicts disagree the judgment is flagged
(`agrees_with_parser: false`) for a human to look at; `harmonybench disagreements` lists them.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .concurrency import AdaptiveLimiter
from .config import ModelSpec
from .cost import usage_cost
from .dataset import Item
from .paths import RESULTS_DIR
from .providers import ProviderError, Request, make_provider
from .runner import _now, predictions_path, read_records, run_dir
from .tasks import Task, get_task


def judge_fingerprint(spec: ModelSpec, task: Task) -> str:
    """Identifies the judge setup; judgments made under a different one are re-done."""
    blob = json.dumps(
        {
            "version": task.judge_version,
            "prompt": task.judge_prompt,
            "schema": task.judge_schema(),
            "request": spec.model_dump(include={"provider", "model", "params", "max_output_tokens"}),
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def response_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def judgments_path(model_id: str, subset: str, results_dir: Path = RESULTS_DIR) -> Path:
    return run_dir(model_id, subset, results_dir) / "judgments.jsonl"


def current_judgments(model_id: str, subset: str, fingerprint: str, results_dir: Path = RESULTS_DIR) -> dict:
    """Latest usable judgment per (item, condition, sample, response hash) under this judge."""
    out = {}
    for j in read_records(judgments_path(model_id, subset, results_dir)):
        if j.get("status") == "ok" and j.get("judge_fingerprint") == fingerprint:
            out[(j["item_id"], j["condition"], j.get("sample", 0), j["response_hash"])] = j
    return out


def pending(model_id: str, subset: str, fingerprint: str, results_dir: Path = RESULTS_DIR) -> list[dict]:
    """Answered predictions that have no current judgment yet."""
    from .metrics import final_records

    done = current_judgments(model_id, subset, fingerprint, results_dir)
    todo = []
    for rec in final_records(read_records(predictions_path(model_id, subset, results_dir))).values():
        if rec.get("status") != "ok":
            continue
        key = (rec["item_id"], rec["condition"], rec.get("sample", 0), response_hash(rec.get("raw_text", "")))
        if key not in done:
            todo.append(rec)
    return todo


@dataclass
class JudgeSummary:
    model_id: str
    attempted: int = 0
    statuses: dict[str, int] = field(default_factory=dict)
    disagreements: int = 0
    cost_usd: float = 0.0


def _parse_judgment(text: str | None, verdicts: tuple[str, ...]) -> dict:
    obj = json.loads(text or "")
    if not isinstance(obj, dict) or obj.get("verdict") not in verdicts:
        raise ValueError(f"bad judgment: {text!r}"[:500])
    return {
        "final_answer": str(obj.get("final_answer", ""))[:200],
        "verdict": obj["verdict"],
        "explanation": str(obj.get("explanation", ""))[:1000],
    }


class Judge:
    """One judge client and concurrency limit, shared by every model being graded, so
    answers can be judged the moment they arrive (see pipeline.py)."""

    def __init__(
        self, spec: ModelSpec, items: list[Item], subset: str, *, concurrency: int = 16, results_dir=RESULTS_DIR
    ):
        self.spec = spec
        self.subset = subset
        self.results_dir = results_dir
        self.task = get_task(subset)
        self.fingerprint = judge_fingerprint(spec, self.task)
        self.schema = self.task.judge_schema()
        self.by_id = {it.item_id: it for it in items}
        self.provider = make_provider(spec)
        self.limiter = AdaptiveLimiter(concurrency)
        self.summaries: dict[str, JudgeSummary] = {}
        self.cost_usd = 0.0
        self._lock = asyncio.Lock()

    def summary(self, model_id: str) -> JudgeSummary:
        return self.summaries.setdefault(model_id, JudgeSummary(model_id=model_id))

    def pending(self, model_id: str) -> list[dict]:
        return [
            r for r in pending(model_id, self.subset, self.fingerprint, self.results_dir) if r["item_id"] in self.by_id
        ]

    async def grade(self, model_id: str, rec: dict) -> dict:
        """Judge one answered prediction, append the judgment to disk, and return it."""
        task = self.task
        item = self.by_id[rec["item_id"]]
        summary = self.summary(model_id)
        text = rec.get("raw_text", "")
        out = {
            "item_id": rec["item_id"],
            "condition": rec["condition"],
            "sample": rec.get("sample", 0),
            "response_hash": response_hash(text),
            "judge_fingerprint": self.fingerprint,
            "judge_model": self.spec.id,
            "answer_key": task.answer(item),
        }
        request = Request(prompt=task.build_judge_prompt(item, text), schema=self.schema, schema_name="judgment")
        async with self.limiter:
            t0 = time.perf_counter()

            async def send():
                nonlocal t0
                t0 = time.perf_counter()
                return await self.provider.complete(request)

            try:
                result = await self.limiter.call(send)
                cost = usage_cost(result.usage, self.spec)
                out.update({"usage": result.usage.to_dict(), "cost_usd": round(cost, 6)})
                summary.cost_usd += cost
                self.cost_usd += cost
                try:
                    out.update(_parse_judgment(result.text, task.verdicts))
                    pv = task.parser_verdict(out["final_answer"], item) if task.parser_verdict else None
                    out["parser_verdict"] = pv
                    out["agrees_with_parser"] = None if pv is None else pv == out["verdict"]
                    out["status"] = "ok"
                    if out["agrees_with_parser"] is False:
                        summary.disagreements += 1
                except (ValueError, json.JSONDecodeError) as e:
                    out.update(
                        {"status": "parse_error", "error": str(e)[:1000], "raw_text": (result.text or "")[:2000]}
                    )
            except ProviderError as e:
                out.update({"status": "api_error", "fatal": True, "error": str(e)[:2000]})
            except Exception as e:  # network/rate-limit/server errors after SDK retries
                out.update({"status": "api_error", "error": f"{type(e).__name__}: {e}"[:2000]})
            out["latency_s"] = round(time.perf_counter() - t0, 3)
        out["timestamp"] = _now()
        summary.attempted += 1
        summary.statuses[out["status"]] = summary.statuses.get(out["status"], 0) + 1
        async with self._lock:
            with judgments_path(model_id, self.subset, self.results_dir).open("a") as f:
                f.write(json.dumps(out) + "\n")
        return out

    async def aclose(self) -> None:
        await self.provider.aclose()


async def judge_models(
    judge_spec: ModelSpec,
    model_ids: list[str],
    items: list[Item],
    subset: str,
    *,
    concurrency: int = 16,
    results_dir: Path = RESULTS_DIR,
    on_start=None,
    on_record=None,
) -> dict[str, JudgeSummary]:
    """Judge every answered prediction of these models that lacks a current judgment,
    all models at once through one shared judge."""
    fingerprint = judge_fingerprint(judge_spec, get_task(subset))
    ids = {it.item_id for it in items}
    todo = [
        (mid, r) for mid in model_ids for r in pending(mid, subset, fingerprint, results_dir) if r["item_id"] in ids
    ]
    if on_start:
        on_start(len(todo))
    if not todo:
        return {mid: JudgeSummary(model_id=mid) for mid in model_ids}

    judge = Judge(judge_spec, items, subset, concurrency=concurrency, results_dir=results_dir)

    async def one(mid: str, rec: dict) -> None:
        out = await judge.grade(mid, rec)
        if on_record:
            on_record(mid, out, judge.cost_usd)

    try:
        await asyncio.gather(*(one(mid, r) for mid, r in todo))
    finally:
        await judge.aclose()
    return {mid: judge.summary(mid) for mid in model_ids}


async def judge_model(
    judge_spec: ModelSpec,
    model_id: str,
    items: list[Item],
    subset: str,
    *,
    concurrency: int = 16,
    results_dir: Path = RESULTS_DIR,
    on_start=None,
    on_record=None,
) -> JudgeSummary:
    """Judge every answered prediction of `model_id` that lacks a current judgment."""
    summaries = await judge_models(
        judge_spec,
        [model_id],
        items,
        subset,
        concurrency=concurrency,
        results_dir=results_dir,
        on_start=on_start,
        on_record=on_record and (lambda _mid, out, spent: on_record(out, spent)),
    )
    return summaries[model_id]
