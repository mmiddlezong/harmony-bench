"""Command-line interface: `uv run harmonybench --help`."""

from __future__ import annotations

import asyncio
import json
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
from .prompt import CONDITIONS, PROMPT_VERSION, PROMPTS

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="HarmonyBench: benchmark AI models at reading harmony from music score images.",
)
console = Console()

ModelsArg = Annotated[
    list[str] | None, typer.Argument(help="Model ids and/or group names (see `harmonybench models`). Default: all.")
]
SubsetOpt = Annotated[str, typer.Option("--subset", "-s", help="Which generated subset under data/.")]
LimitOpt = Annotated[
    int | None, typer.Option("--limit", "-n", help="Use only the first N items (items are stored shuffled).")
]
ConditionOpt = Annotated[
    list[str] | None,
    typer.Option("--condition", help="image and/or musicxml (repeatable). Default: both."),
]


@app.callback()
def _main() -> None:
    load_dotenv(ROOT / ".env")


def _conditions(conditions: list[str] | None) -> tuple[str, ...]:
    chosen = tuple(conditions or CONDITIONS)
    bad = [c for c in chosen if c not in CONDITIONS]
    if bad:
        raise typer.BadParameter(f"unknown condition(s) {bad}; choose from {list(CONDITIONS)}")
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
    seed: Annotated[int, typer.Option(help="Random seed; the same seed rebuilds identical files.")] = 0,
) -> None:
    """Generate the root-position triad subset (images, MusicXML, labels, review page)."""
    from .build import SUBSET, generate
    from .dataset import subset_dir

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
    counts = {s.id: {c: n for c in _conditions(condition)} for s in specs}
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
            return f"<prompt {PROMPT_VERSION}, {len(obj)} chars>"
        if len(obj) > 300:
            return f"{obj[:40]}…<{len(obj):,} chars>"
    return obj


def _judge(reg, model_ids: list[str], items, subset: str, concurrency: int) -> bool:
    """Judge every ungraded answer for these models. Returns False if anything failed."""
    from .judge import judge_model

    if reg.judge is None:
        console.print("[yellow]No judge configured in configs/models.yaml; answers stay ungraded.[/]")
        return False
    jspec = reg.judge_spec()
    if not jspec.has_credentials():
        console.print(f"[yellow]Judge {jspec.id} has no API key ({' or '.join(jspec.key_envs())}); skipping.[/]")
        return False
    ok = True
    for mid in model_ids:
        with _progress() as prog:
            task = prog.add_task(f"judging {mid}", total=None, spent="")
            s = asyncio.run(
                judge_model(
                    jspec,
                    mid,
                    items,
                    subset,
                    concurrency=concurrency,
                    on_start=lambda n, _t=task: prog.update(_t, total=n),
                    on_record=lambda r, spent, _t=task: prog.update(_t, advance=1, spent=f"${spent:.4f}"),
                )
            )
        if s.attempted:
            statuses = ", ".join(f"{k}: {v}" for k, v in sorted(s.statuses.items()))
            flag = f"; [yellow]{s.disagreements} flagged[/]" if s.disagreements else ""
            console.print(f"judge → {mid}: {s.attempted} graded ({statuses}){flag}; spent ${s.cost_usd:.4f}")
        if s.statuses.get("api_error") or s.statuses.get("parse_error"):
            ok = False
            console.print("[yellow]Some judgments failed; run `harmonybench judge` to retry them.[/]")
    return ok


@app.command()
def run(
    model_names: ModelsArg = None,
    subset: SubsetOpt = DEFAULT_SUBSET,
    limit: LimitOpt = None,
    condition: ConditionOpt = None,
    repeats: Annotated[int, typer.Option(help="Samples per item and condition (default 1).")] = 1,
    concurrency: Annotated[int, typer.Option("--concurrency", "-c", help="Parallel requests per model.")] = 8,
    max_cost: Annotated[float | None, typer.Option(help="Per-model spend cap (USD).")] = None,
    fresh: Annotated[bool, typer.Option(help="Archive existing results for these models and start over.")] = False,
    judge: Annotated[bool, typer.Option(help="Grade the answers with the judge model afterwards.")] = True,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the cost confirmation prompt.")] = False,
    dry_run: Annotated[bool, typer.Option(help="Build one request per model and print it; send nothing.")] = False,
) -> None:
    """Run models on the benchmark, then judge their answers. Resumable: re-running only
    fills in missing answers and judgments."""
    from .providers import make_provider
    from .runner import RunConfigMismatch, build_request, remaining, run_model

    reg = load_registry()
    specs = reg.select(model_names)
    items = load_items(subset, limit)
    conditions = _conditions(condition)

    if dry_run:
        prompts = set(PROMPTS.values())
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

    any_failed = False
    ran = []
    for spec in specs:
        console.rule(f"[bold]{spec.display_name}[/] ({spec.id})")
        with _progress() as prog:
            task = prog.add_task(spec.id, total=None, spent="")

            def on_start(n_todo: int, _task=task) -> None:
                prog.update(_task, total=n_todo)

            def on_record(rec: dict, spent: float, _task=task) -> None:
                prog.update(_task, advance=1, spent=f"${spent:.3f} · last: {rec['status']}")

            try:
                summary = asyncio.run(
                    run_model(
                        spec,
                        items,
                        subset,
                        conditions=conditions,
                        repeats=repeats,
                        concurrency=concurrency,
                        max_cost=max_cost,
                        fresh=fresh,
                        on_start=on_start,
                        on_record=on_record,
                    )
                )
            except RunConfigMismatch as e:
                any_failed = True
                console.print(f"[red]{e}[/]")
                continue
        ran.append(spec.id)
        statuses = ", ".join(f"{k}: {v}" for k, v in sorted(summary.statuses.items())) or "nothing to do"
        console.print(
            f"{spec.id}: {summary.skipped} already done, {summary.attempted} attempted "
            f"({statuses}); spent ${summary.cost_usd:.4f}"
        )
        if summary.aborted:
            any_failed = True
            console.print(f"[red]aborted: {summary.aborted}[/]")
        if summary.statuses.get("api_error"):
            any_failed = True
            console.print("[yellow]Some requests hit API errors; re-run the same command to retry them.[/]")

    if judge and ran:
        console.rule("Judging answers")
        any_failed |= not _judge(reg, ran, items, subset, concurrency=16)
    console.print(f"\nNext: [bold]uv run harmonybench score --subset {subset}[/]")
    if any_failed:
        raise typer.Exit(2)


@app.command("judge")
def judge_cmd(
    model_names: ModelsArg = None,
    subset: SubsetOpt = DEFAULT_SUBSET,
    concurrency: Annotated[int, typer.Option("--concurrency", "-c", help="Parallel judge requests.")] = 16,
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
        console.print(f"[yellow]No results under results/{subset}/{PROMPT_VERSION}/. Run `harmonybench run` first.[/]")
        raise typer.Exit(1)
    items = load_items(subset)
    return reg, items, score_all(items, available, subset, judge_fingerprint(reg.judge_spec()))


@app.command()
def score(
    model_names: ModelsArg = None,
    subset: SubsetOpt = DEFAULT_SUBSET,
    write: Annotated[bool, typer.Option(help="Write leaderboard.md / leaderboard.json.")] = True,
) -> None:
    """Score judged answers and write the leaderboard."""
    reg, items, scores = _scores(subset, model_names)
    table = Table(title=f"HarmonyBench {subset} (prompt {PROMPT_VERSION}) — ranked by image accuracy")
    for col in ("model", "image acc", "95% CI", "enharm.", "musicxml acc", "gap", "fail", "flags", "cost", "n"):
        table.add_column(col, justify="left" if col == "model" else "right")
    unjudged = 0
    for s in scores:
        unjudged += s.unjudged
        img = s.conditions.get("image")
        if not img:
            continue
        lo, hi = img["accuracy_ci95"]
        xml = s.conditions.get("musicxml", {}).get("accuracy")
        gap = s.reading_gap.get("gap")
        table.add_row(
            s.model_id,
            f"{100 * img['accuracy']:.0f}%",
            f"{100 * lo:.0f}–{100 * hi:.0f}%",
            f"{100 * img['enharmonic_accuracy']:.0f}%",
            "–" if xml is None else f"{100 * xml:.0f}%",
            "–" if gap is None else f"{100 * gap:+.0f}",
            f"{100 * img['failure_rate']:.0f}%",
            str(s.judge_disagreements),
            f"${s.usage.get('projected_cost_full_run_usd', 0):.2f}",
            f"{img['n_items']}/{s.n_total}",
        )
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
        if not model_names and update_readme(ROOT / "README.md", scores, reg, len(items)):
            console.print("[green]✓[/] updated the README leaderboard (complete runs only)")


@app.command()
def compare(
    model_a: str,
    model_b: str,
    subset: SubsetOpt = DEFAULT_SUBSET,
    condition: Annotated[str, typer.Option(help="image or musicxml.")] = "image",
) -> None:
    """Paired bootstrap test: is model A's accuracy different from model B's?"""
    from .judge import current_judgments, judge_fingerprint
    from .metrics import collect_outcomes, compare_models
    from .runner import predictions_path, read_records

    reg = load_registry()
    fp = judge_fingerprint(reg.judge_spec())
    items = load_items(subset)

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


def _print_outcome(o, show_text: bool) -> None:
    v = o.item.meta.get("voicing", {})
    voicing = " ".join(f"{k} {v[k]}" for k in "SATB" if k in v)
    console.rule(f"{o.item.item_id} · {o.condition} · [bold]{o.verdict}[/]")
    console.print(f"[bold]key[/] {o.item.answer}   ({voicing}; key signature {o.item.meta.get('key_signature', 0):+d})")
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

    fp = judge_fingerprint(load_registry().judge_spec())
    items = load_items(subset)
    return collect_outcomes(
        items, read_records(predictions_path(model_id, subset)), current_judgments(model_id, subset, fp)
    )[0]


@app.command()
def errors(
    model_id: str,
    subset: SubsetOpt = DEFAULT_SUBSET,
    condition: Annotated[str, typer.Option(help="image or musicxml.")] = "image",
    n: Annotated[int, typer.Option("-n", help="How many to show.")] = 20,
    full: Annotated[bool, typer.Option(help="Also print the model's whole response.")] = False,
) -> None:
    """Show the items a model got wrong (for error analysis)."""
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
def prompt() -> None:
    """Print the exact prompts sent with every item."""
    for c in CONDITIONS:
        console.print(f"[bold]Prompt {PROMPT_VERSION} · {c}[/]\n")
        print(PROMPTS[c].replace("{musicxml}", "<the item's MusicXML file>"))
        print()


if __name__ == "__main__":
    app()
