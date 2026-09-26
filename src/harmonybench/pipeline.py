"""Run many models at once and judge their answers as they arrive.

Every model runs concurrently (each with its own adaptive concurrency limit, see
concurrency.py), and each answer is handed to one shared judge the moment it is written,
so judging overlaps the runs instead of waiting for the slowest model to finish. Answers
left ungraded by an earlier invocation are judged too. Results on disk are exactly what
run_model / judge_model write on their own, so everything stays resumable.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from .config import ModelSpec
from .dataset import Item
from .judge import Judge, JudgeSummary
from .paths import RESULTS_DIR
from .runner import RunConfigMismatch, RunSummary, run_model


async def run_and_judge(
    specs: list[ModelSpec],
    items: list[Item],
    subset: str,
    *,
    judge_spec: ModelSpec | None = None,
    conditions: tuple[str, ...] | None = None,
    repeats: int = 1,
    concurrency: int = 8,
    judge_concurrency: int = 16,
    parallel_models: int | None = None,
    max_cost: float | None = None,
    fresh: bool = False,
    results_dir: Path = RESULTS_DIR,
    on_start=None,
    on_record=None,
    on_judge_queued=None,
    on_judged=None,
) -> tuple[dict[str, RunSummary | RunConfigMismatch], dict[str, JudgeSummary]]:
    """Returns ({model_id: run summary, or the mismatch that stopped it}, {model_id: judge summary}).

    Callbacks: on_start(model_id, n_todo), on_record(model_id, record, spent),
    on_judge_queued(model_id), on_judged(model_id, judgment, judge_spent)."""
    judge = None
    if judge_spec is not None:
        judge = Judge(judge_spec, items, subset, concurrency=judge_concurrency, results_dir=results_dir)
    judge_tasks: list[asyncio.Task] = []
    gate = asyncio.Semaphore(parallel_models or max(1, len(specs)))
    runs: dict[str, RunSummary | RunConfigMismatch] = {}

    def grade(model_id: str, rec: dict) -> None:
        async def go() -> None:
            out = await judge.grade(model_id, rec)
            if on_judged:
                on_judged(model_id, out, judge.cost_usd)

        if on_judge_queued:
            on_judge_queued(model_id)
        judge_tasks.append(asyncio.create_task(go()))

    async def one_model(spec: ModelSpec) -> None:
        def started(n_todo: int) -> None:
            # Called after the results dir is checked (and archived under --fresh) but
            # before any new answer is written, so this picks up exactly the old backlog.
            if judge:
                for rec in judge.pending(spec.id):
                    grade(spec.id, rec)
            if on_start:
                on_start(spec.id, n_todo)

        def recorded(rec: dict, spent: float) -> None:
            if on_record:
                on_record(spec.id, rec, spent)
            if judge and rec["status"] == "ok" and rec["item_id"] in judge.by_id:
                grade(spec.id, rec)

        async with gate:
            try:
                runs[spec.id] = await run_model(
                    spec,
                    items,
                    subset,
                    conditions=conditions,
                    repeats=repeats,
                    concurrency=concurrency,
                    max_cost=max_cost,
                    fresh=fresh,
                    results_dir=results_dir,
                    on_start=started,
                    on_record=recorded,
                )
            except RunConfigMismatch as e:
                runs[spec.id] = e

    try:
        await asyncio.gather(*(one_model(s) for s in specs))
        await asyncio.gather(*judge_tasks)
    finally:
        for t in judge_tasks:
            t.cancel()
        if judge:
            await judge.aclose()
    judged = {mid: judge.summary(mid) for mid, r in runs.items() if isinstance(r, RunSummary)} if judge else {}
    return {s.id: runs[s.id] for s in specs if s.id in runs}, judged
