"""Command-line interface: `uv run harmonybench --help`."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from .config import load_registry
from .cost import estimate_cost
from .dataset import DEFAULT_SUBSET, load_items
from .paths import ROOT
from .tasks import TASKS, get_task

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="HarmonyBench: benchmark AI models at reading harmony from music score images.",
)
console = Console()
SEED_ENV = "HARMONYBENCH_SEED"

ModelsArg = Annotated[
    list[str] | None, typer.Argument(help="Model ids and/or group names (see `harmonybench models`). Default: all.")
]
SubsetOpt = Annotated[str, typer.Option("--subset", "-s", help=f"Which subset under data/ ({', '.join(TASKS)}).")]
LimitOpt = Annotated[
    int | None, typer.Option("--limit", "-n", help="Use only the first N items (items are stored shuffled).")
]
ConditionOpt = Annotated[
    list[str] | None,
    typer.Option("--condition", help="image and/or musicxml (repeatable). Default: all of the task's conditions."),
]


@app.callback()
def _main() -> None:
    load_dotenv(ROOT / ".env")


def _conditions(conditions: list[str] | None, subset: str) -> tuple[str, ...]:
    allowed = get_task(subset).conditions
    chosen = tuple(conditions or allowed)
    bad = [c for c in chosen if c not in allowed]
    if bad:
        raise typer.BadParameter(f"condition(s) {bad} not in {subset}; choose from {list(allowed)}")
    return chosen


def _progress() -> Progress:
    return Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("{task.fields[spent]}"),
        console=console,
    )


@app.command()
def build(
    per_chord: Annotated[int, typer.Option(help="Items per chord (alternating no key signature / key signature).")] = 2,
    seed: Annotated[
        int | None,
        typer.Option(help=f"Random seed; the same seed rebuilds identical files. Default: ${SEED_ENV} from .env."),
    ] = None,
) -> None:
    """Generate the root-position triad subset (images, MusicXML, labels, review page).

    The real test set's seed is private (in .env), so the public code alone can't rebuild it."""
    from .build import SUBSET, generate
    from .dataset import subset_dir

    if seed is None:
        raw = os.environ.get(SEED_ENV, "").strip()
        if not raw:
            console.print(
                f"[red]No seed: set {SEED_ENV} in .env (the private seed of the real test set) "
                "or pass --seed for a throwaway build.[/]"
            )
            raise typer.Exit(1)
        try:
            seed = int(raw)
        except ValueError:
            console.print(f"[red]{SEED_ENV} must be an integer.[/]")
            raise typer.Exit(1) from None
    out = subset_dir(SUBSET)
    items = generate(out, per_chord, seed)
    console.print(f"[green]✓[/] wrote {len(items)} verified items to {out.relative_to(ROOT)}")
    console.print(f"[dim]Check them by eye: {(out / 'review.html').relative_to(ROOT)}[/]")


@app.command()
def models(group: Annotated[str | None, typer.Option(help="Only show this group.")] = None) -> None:
    """List configured models, their prices, and whether an API key is set."""
    reg = load_registry()
    table = Table(title=f"Model registry (prices checked {reg.pricing_checked}, USD per 1M tokens)")
    for col in ("id", "lab", "provider", "API model", "in $", "out $", "groups", "key"):
        table.add_column(col)
    for m in reg.models:
        if group and group not in m.groups:
            continue
        key = "[green]✓[/]" if m.has_credentials() else f"[red]✗[/] {'/'.join(m.key_envs())}"
        table.add_row(
            m.id + ("" if m.enabled else " (disabled)"),
            m.lab,
            m.provider,
            m.model,
            f"{m.pricing.input:g}",
            f"{m.pricing.output:g}",
            ",".join(m.groups),
            key,
        )
    console.print(table)
    console.print("Groups: " + ", ".join(f"{g} ({len(ids)})" for g, ids in reg.groups().items()))
    if reg.judge:
        j = reg.judge_spec()
        console.print(f"Judge: {j.id} {j.params} {'[green]✓[/]' if j.has_credentials() else '[red]✗ no key[/]'}")


def _estimate_table(specs, counts: dict[str, dict[str, int]], judge_spec=None) -> tuple[Table, float, float, float]:
    """counts: {model_id: {condition: n_requests}}. The judge makes one call per answer."""
    table = Table(title="Estimated cost")
    for col in ("model", "requests", "in tok/req", "out tok/req", "low", "expected", "high"):
        table.add_column(col, justify="left" if col == "model" else "right")
    lo = ex = hi = 0.0
    for spec in specs:
        ests = [estimate_cost(spec, n, c) for c, n in counts[spec.id].items()]
        n = sum(e.n_requests for e in ests)
        l_, e_, h_ = (sum(getattr(e, f) for e in ests) for f in ("low_usd", "expected_usd", "high_usd"))
        lo, ex, hi = lo + l_, ex + e_, hi + h_
        ins = "/".join(str(e.input_tokens_per_req) for e in ests)
        table.add_row(
            spec.id,
            str(n),
            ins,
            str(ests[0].output_tokens_per_req if ests else 0),
            f"${l_:.2f}",
            f"${e_:.2f}",
            f"${h_:.2f}",
        )
    if judge_spec is not None:
        n_judge = sum(sum(c.values()) for c in counts.values())
        est = estimate_cost(judge_spec, n_judge, "text")
        lo, ex, hi = lo + est.low_usd, ex + est.expected_usd, hi + est.high_usd
        table.add_row(
            f"[dim]judge ({judge_spec.id})[/]",
            str(n_judge),
            str(est.input_tokens_per_req),
            str(est.output_tokens_per_req),
            f"${est.low_usd:.2f}",
            f"${est.expected_usd:.2f}",
            f"${est.high_usd:.2f}",
        )
    table.add_section()
    table.add_row("[bold]TOTAL[/]", "", "", "", f"${lo:.2f}", f"[bold]${ex:.2f}[/]", f"${hi:.2f}")
    return table, lo, ex, hi


@app.command()
def estimate(
    model_names: ModelsArg = None,
    subset: SubsetOpt = DEFAULT_SUBSET,
    limit: LimitOpt = None,
    condition: ConditionOpt = None,
    repeats: Annotated[int, typer.Option(help="Samples per item and condition.")] = 1,
) -> None:
    """Estimate API cost before running (no API calls are made)."""
    reg = load_registry()
    specs = reg.select(model_names)
    n = len(load_items(subset, limit)) * repeats
    counts = {s.id: {c: n for c in _conditions(condition, subset)} for s in specs}
    table, *_ = _estimate_table(specs, counts, reg.judge_spec() if reg.judge else None)
    console.print(table)
    console.print(
        "[dim]in tok/req is per condition (image/musicxml). Range reflects uncertainty in hidden reasoning "
        "tokens (0.5×–2.5× the assumed amount). Actual cost is computed from API-reported usage.[/]"
    )


def _redact(obj, prompts: set[str]):
    """Shorten image payloads and the prompt so a request can be printed readably."""
    if isinstance(obj, dict):
        return {k: _redact(v, prompts) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact(v, prompts) for v in obj]
    if isinstance(obj, bytes):
        return f"<{len(obj):,} image bytes>"
    if isinstance(obj, str):
        if obj in prompts:
            return f"<prompt, {len(obj)} chars>"
        if len(obj) > 300:
            return f"{obj[:40]}…<{len(obj):,} chars>"
    return obj


def _judge_spec(reg):
    """The configured judge, or None (with a warning) if there is none or it has no key."""
    if reg.judge is None:
        console.print("[yellow]No judge configured in configs/models.yaml; answers stay ungraded.[/]")
        return None
    jspec = reg.judge_spec()
    if not jspec.has_credentials():
        console.print(f"[yellow]Judge {jspec.id} has no API key ({' or '.join(jspec.key_envs())}); skipping.[/]")
        return None
    return jspec


def _report_judging(summaries) -> bool:
    """Print one line per judged model. Returns False if any judgment failed."""
    ok = True
    for mid, s in summaries.items():
        if s.attempted:
            statuses = ", ".join(f"{k}: {v}" for k, v in sorted(s.statuses.items()))
            flag = f"; [yellow]{s.disagreements} flagged[/]" if s.disagreements else ""
            console.print(f"judge → {mid}: {s.attempted} graded ({statuses}){flag}; spent ${s.cost_usd:.4f}")
        if s.statuses.get("api_error") or s.statuses.get("parse_error"):
            ok = False
    if not ok:
        console.print("[yellow]Some judgments failed; run `harmonybench judge` to retry them.[/]")
    return ok


def _judge(reg, model_ids: list[str], items, subset: str, concurrency: int) -> bool:
    """Judge every ungraded answer for these models. Returns False if anything failed."""
    from .judge import judge_models

    jspec = _judge_spec(reg)
    if jspec is None:
        return False
    with _progress() as prog:
        task = prog.add_task(f"judging ({jspec.id})", total=None, spent="")
        summaries = asyncio.run(
            judge_models(
                jspec,
                model_ids,
                items,
                subset,
                concurrency=concurrency,
                on_start=lambda n: prog.update(task, total=n),
                on_record=lambda mid, r, spent: prog.update(task, advance=1, spent=f"${spent:.4f}"),
            )
        )
    return _report_judging(summaries)


@app.command()
def run(
    model_names: ModelsArg = None,
    subset: SubsetOpt = DEFAULT_SUBSET,
    limit: LimitOpt = None,
    condition: ConditionOpt = None,
    repeats: Annotated[int, typer.Option(help="Samples per item and condition (default 1).")] = 1,
    concurrency: Annotated[
        int,
        typer.Option(
            "--concurrency", "-c", help="Max parallel requests per model (lowered automatically on rate limits)."
        ),
    ] = 16,
    parallel_models: Annotated[
        int | None, typer.Option("--parallel-models", "-p", help="Models to run at once. Default: all of them.")
    ] = None,
    judge_concurrency: Annotated[int, typer.Option(help="Max parallel judge requests.")] = 32,
    max_cost: Annotated[float | None, typer.Option(help="Per-model spend cap (USD).")] = None,
    fresh: Annotated[bool, typer.Option(help="Archive existing results for these models and start over.")] = False,
    judge: Annotated[bool, typer.Option(help="Grade the answers with the judge model as they come in.")] = True,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the cost confirmation prompt.")] = False,
    dry_run: Annotated[bool, typer.Option(help="Build one request per model and print it; send nothing.")] = False,
) -> None:
    """Run models on the benchmark (all at once) and judge their answers as they come in.
    Resumable: re-running only fills in missing answers and judgments."""
    from .pipeline import run_and_judge
    from .providers import make_provider
    from .runner import RunConfigMismatch, build_request, remaining

    reg = load_registry()
    specs = reg.select(model_names)
    items = load_items(subset, limit)
    conditions = _conditions(condition, subset)

    if dry_run:
        prompts: set[str] = set()
        for spec in specs:
            for c in conditions:
                req = build_request(items[0], c)
                prompts.add(req.prompt)
                body = make_provider(spec).build_request(req)
                console.rule(f"{spec.id} ({spec.provider}) · {c}")
                console.print_json(json.dumps(_redact(body, prompts)), indent=1)
        console.print("[dim]Dry run: nothing was sent.[/]")
        return

    missing = [s for s in specs if not s.has_credentials()]
    for s in missing:
        console.print(f"[yellow]skipping {s.id}: set {' or '.join(s.key_envs())}[/]")
    specs = [s for s in specs if s.has_credentials()]
    if not specs:
        raise typer.Exit(1)

    if not yes:
        counts = {}
        for spec in specs:
            todo = remaining(spec.id, subset, items, conditions, repeats)
            counts[spec.id] = {c: sum(1 for _, tc, _ in todo if tc == c) for c in conditions}
        jspec = reg.judge_spec() if (judge and reg.judge) else None
        table, lo, ex, hi = _estimate_table(specs, counts, jspec)
        console.print(table)
        console.print("[dim]Only requests without an answer yet are counted; finished ones are skipped.[/]")
        if not typer.confirm(f"Proceed? (expected ≈ ${ex:.2f}, range ${lo:.2f}–${hi:.2f})"):
            raise typer.Exit(0)

    jspec = _judge_spec(reg) if judge else None
    any_failed = judge and jspec is None
    with _progress() as prog:
        rows = {s.id: prog.add_task(s.id, total=None, spent="") for s in specs}
        jrow = prog.add_task(f"[dim]judge ({jspec.id})[/]", total=0, spent="") if jspec else None

        def on_record(mid: str, rec: dict, spent: float) -> None:
            prog.update(rows[mid], advance=1, spent=f"${spent:.3f} · last: {rec['status']}")

        queued = 0

        def on_judge_queued(mid: str) -> None:
            nonlocal queued
            queued += 1
            prog.update(jrow, total=queued)

        runs, judged = asyncio.run(
            run_and_judge(
                specs,
                items,
                subset,
                judge_spec=jspec,
                conditions=conditions,
                repeats=repeats,
                concurrency=concurrency,
                judge_concurrency=judge_concurrency,
                parallel_models=parallel_models,
                max_cost=max_cost,
                fresh=fresh,
                on_start=lambda mid, n: prog.update(rows[mid], total=n),
                on_record=on_record,
                on_judge_queued=on_judge_queued,
                on_judged=lambda mid, out, spent: prog.update(jrow, advance=1, spent=f"${spent:.4f}"),
            )
        )

    for spec in specs:
        summary = runs[spec.id]
        if isinstance(summary, RunConfigMismatch):
            any_failed = True
            console.print(f"[red]{summary}[/]")
            continue
        statuses = ", ".join(f"{k}: {v}" for k, v in sorted(summary.statuses.items())) or "nothing to do"
        console.print(
            f"{spec.id}: {summary.skipped} already done, {summary.attempted} attempted "
            f"({statuses}); spent ${summary.cost_usd:.4f}"
        )
        if summary.aborted:
            any_failed = True
            console.print(f"[red]  aborted: {summary.aborted}[/]")
        if summary.statuses.get("api_error"):
            any_failed = True
            console.print("[yellow]  some requests hit API errors; re-run the same command to retry them.[/]")
    if judged:
        any_failed |= not _report_judging(judged)
    console.print(f"\nNext: [bold]uv run harmonybench score --subset {subset}[/]")
    if any_failed:
        raise typer.Exit(2)


@app.command("judge")
def judge_cmd(
    model_names: ModelsArg = None,
    subset: SubsetOpt = DEFAULT_SUBSET,
    concurrency: Annotated[int, typer.Option("--concurrency", "-c", help="Max parallel judge requests.")] = 32,
) -> None:
    """Grade answers that have no judgment yet (or were judged under an older judge setup)."""
    from .report import discover_models

    reg = load_registry()
    available = discover_models(subset)
    if model_names:
        wanted = {s.id for s in reg.select(model_names)}
        available = [m for m in available if m in wanted]
    if not _judge(reg, available, load_items(subset), subset, concurrency):
        raise typer.Exit(2)


def _scores(subset: str, model_names):
    from .judge import judge_fingerprint
    from .report import discover_models, score_all

    reg = load_registry()
    available = discover_models(subset)
    if model_names:
        wanted = {s.id for s in reg.select(model_names)}
        available = [m for m in available if m in wanted]
    if not available:
        console.print(
            f"[yellow]No results under results/{subset}/{get_task(subset).version}/. Run `harmonybench run` first.[/]"
        )
        raise typer.Exit(1)
    items = load_items(subset)
    return reg, items, score_all(items, available, subset, judge_fingerprint(reg.judge_spec(), get_task(subset)))


@app.command()
def score(
    model_names: ModelsArg = None,
    subset: SubsetOpt = DEFAULT_SUBSET,
    write: Annotated[bool, typer.Option(help="Write leaderboard.md / leaderboard.json.")] = True,
) -> None:
    """Score judged answers and write the leaderboard."""
    from .report import _columns

    reg, items, scores = _scores(subset, model_names)
    task = get_task(subset)
    cols = _columns(task)
    table = Table(title=f"HarmonyBench {subset} (prompt {task.version}) — ranked by {task.conditions[0]} accuracy")
    table.add_column("model")
    for header, _ in cols:
        table.add_column(header, justify="right")
    unjudged = 0
    for s in scores:
        unjudged += s.unjudged
        if s.primary:
            table.add_row(s.model_id, *(f(s) for _, f in cols))
    console.print(table)
    if unjudged:
        console.print(f"[yellow]{unjudged} answers are not judged yet and are left out; run `harmonybench judge`.[/]")
    if any(s.judge_disagreements for s in scores):
        console.print(
            "[dim]flags: judge verdicts the rule-based parser disagrees with; see `harmonybench disagreements`.[/]"
        )
    if write:
        from .report import update_readme, write_leaderboard

        md, js = write_leaderboard(scores, reg, subset)
        console.print(f"[green]✓[/] wrote {md.relative_to(ROOT)} and {js.relative_to(ROOT)}")
        if task.readme and not model_names and update_readme(ROOT / "README.md", scores, reg, len(items)):
            console.print("[green]✓[/] updated the README leaderboard (complete runs only)")


@app.command()
def compare(
    model_a: str,
    model_b: str,
    subset: SubsetOpt = DEFAULT_SUBSET,
    condition: Annotated[str | None, typer.Option(help="image or musicxml. Default: the task's first.")] = None,
) -> None:
    """Paired bootstrap test: is model A's accuracy different from model B's?"""
    from .judge import current_judgments, judge_fingerprint
    from .metrics import collect_outcomes, compare_models
    from .runner import predictions_path, read_records

    reg = load_registry()
    fp = judge_fingerprint(reg.judge_spec(), get_task(subset))
    items = load_items(subset)
    condition = condition or get_task(subset).conditions[0]

    def outcomes(mid):
        return collect_outcomes(items, read_records(predictions_path(mid, subset)), current_judgments(mid, subset, fp))[
            0
        ]

    c = compare_models(items, model_a, outcomes(model_a), model_b, outcomes(model_b), condition)
    if c.n_common < 2:
        console.print("[yellow]Not enough items graded for both models.[/]")
        raise typer.Exit(1)
    better = model_a if c.diff > 0 else model_b
    console.print(
        f"{c.n_common} common items ({condition}) | accuracy {model_a}: {100 * c.acc_a:.1f}%  "
        f"{model_b}: {100 * c.acc_b:.1f}%"
    )
    console.print(
        f"difference (A−B): {100 * c.diff:+.1f} pts, 95% CI [{100 * c.diff_ci95[0]:+.1f}, "
        f"{100 * c.diff_ci95[1]:+.1f}], p = {c.p_value:.3f}"
    )
    verdict = "significant" if c.p_value < 0.05 else "not significant"
    console.print(f"→ {better} is better; difference is [bold]{verdict}[/] at α = 0.05")


def _item_context(item) -> str:
    """A short description of the item for error analysis."""
    m = item.meta
    if "voicing" in m:
        v = m["voicing"]
        return " ".join(f"{k} {v[k]}" for k in "SATB" if k in v) + f"; key signature {m.get('key_signature', 0):+d}"
    if "source" in m:
        return m["source"]
    return ""


def _print_outcome(o, show_text: bool) -> None:
    console.rule(f"{o.item.item_id} · {o.condition} · [bold]{o.verdict}[/]")
    console.print(f"[bold]key[/] {get_task(o.item.subset).answer(o.item)}   ({_item_context(o.item)})")
    if o.judgment:
        j = o.judgment
        console.print(f"[bold]model said[/] {j.get('final_answer') or '—'}   [dim]judge: {j.get('explanation', '')}[/]")
        if j.get("agrees_with_parser") is False:
            console.print(f"[yellow]parser would say {j.get('parser_verdict')}[/]")
    else:
        console.print(f"[red]{o.record.get('status')}[/] {o.record.get('error', '')}")
    if show_text:
        console.print(f"[dim]{(o.record.get('raw_text') or '').strip()}[/]")
    console.print(f"[dim]image: data/{o.item.subset}/{o.item.image}[/]")


def _outcomes_for(model_id: str, subset: str):
    from .judge import current_judgments, judge_fingerprint
    from .metrics import collect_outcomes
    from .runner import predictions_path, read_records

    fp = judge_fingerprint(load_registry().judge_spec(), get_task(subset))
    items = load_items(subset)
    return collect_outcomes(
        items, read_records(predictions_path(model_id, subset)), current_judgments(model_id, subset, fp)
    )[0]


@app.command()
def errors(
    model_id: str,
    subset: SubsetOpt = DEFAULT_SUBSET,
    condition: Annotated[str | None, typer.Option(help="image or musicxml. Default: the task's first.")] = None,
    n: Annotated[int, typer.Option("-n", help="How many to show.")] = 20,
    full: Annotated[bool, typer.Option(help="Also print the model's whole response.")] = False,
) -> None:
    """Show the items a model got wrong (for error analysis)."""
    condition = condition or get_task(subset).conditions[0]
    wrong = [o for o in _outcomes_for(model_id, subset) if o.condition == condition and o.verdict != "correct"]
    console.print(f"{len(wrong)} wrong answers for {model_id} ({condition})")
    for o in wrong[:n]:
        _print_outcome(o, full)


@app.command()
def disagreements(
    model_names: ModelsArg = None,
    subset: SubsetOpt = DEFAULT_SUBSET,
) -> None:
    """Show judge verdicts that the rule-based parser disagrees with, to check by hand."""
    from .report import discover_models

    reg = load_registry()
    ids = discover_models(subset)
    if model_names:
        wanted = {s.id for s in reg.select(model_names)}
        ids = [m for m in ids if m in wanted]
    total = 0
    for mid in ids:
        flagged = [
            o for o in _outcomes_for(mid, subset) if o.judgment and o.judgment.get("agrees_with_parser") is False
        ]
        if flagged:
            console.print(f"\n[bold]{mid}[/]: {len(flagged)} flagged")
        for o in flagged:
            _print_outcome(o, True)
        total += len(flagged)
    console.print(f"\n{total} flagged judgments")


@app.command()
def prompt(subset: SubsetOpt = DEFAULT_SUBSET) -> None:
    """Print the prompt templates for a subset ({fields} are filled in per item)."""
    task = get_task(subset)
    for c in task.conditions:
        console.print(f"[bold]{subset} · prompt {task.version} · {c}[/]\n", highlight=False)
        print(task.prompts[c].replace("{musicxml}", "<the item's MusicXML file>"))
        print()


if __name__ == "__main__":
    app()
