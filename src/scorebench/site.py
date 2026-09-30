"""The results website for the wrong_note subset: `scorebench site`.

The page is built from private data (the excerpts and the stored answers), so it is built
locally and the output committed, like CalorieBench's site. Two versions:

  public   site/index.html, deployed to GitHub Pages: ranking, answer patterns and every
           answer as a bar number, but no excerpt images and no model responses
  private  results/wrong_note/<version>/report.html (gitignored): also every excerpt with
           its image and each model's full answers

Look: engraved sheet music rather than a generic report: a title page, the answers laid out
as a score (models as staves braced by lab, excerpts as bars, a final double barline),
excerpts numbered with boxed rehearsal marks, and editorial notes.
"""

from __future__ import annotations

import base64
import hashlib
import html
import re
import statistics as st
from datetime import date
from pathlib import Path

from .config import load_registry
from .dataset import Item, load_items
from .judge import current_judgments, judge_fingerprint
from .metrics import collect_outcomes, score_model
from .paths import ROOT
from .report import discover_models
from .runner import predictions_path, read_records
from .tasks import WRONG_NOTE

e = html.escape
STYLESHEET = Path(__file__).with_name("site.css")
# An example excerpt for the "Try one" section. It is public on purpose and is not one of the test items.
SAMPLE_IMAGE = ROOT / "site" / "sample.png"
SAMPLE_ANSWER = 2
AUTHOR = "Michael Middlezong"
AUTHOR_URL = "https://github.com/mmiddlezong"
REPO_URL = "https://github.com/mmiddlezong/score-bench"
FAVICON = ROOT / "site" / "favicon.svg"


def first_line(text: str) -> str:
    return next((ln.replace("**", "").strip() for ln in text.splitlines() if ln.strip()), "")


def stated_bar(text: str) -> int | None:
    """The bar an answer commits to: an explicit "Answer: ... N" line wins, then a bar number
    on the first line, then the last "measure/bar N" mentioned anywhere."""
    t = text.replace("**", "")
    answers = re.findall(r"answer\W*(?:(?:measure|bar)\s*)?(\d+)", t, re.I)
    if answers:
        return int(answers[-1])
    first = first_line(text)
    m = re.match(r"^\W*(\d+)\W*$", first) or re.search(r"(?:measure|bar)\s*(\d+)", first, re.I)
    if m:
        return int(m.group(1))
    mentions = re.findall(r"(?:measure|bar)\s*(\d+)", t, re.I)
    return int(mentions[-1]) if mentions else None


def _check_public(page: str, items: list[Item]) -> None:
    """Refuse to produce a public page that contains any private test material."""
    leaks = []
    if "data:image" in page:
        leaks.append("an embedded image")
    if 'class="said"' in page or 'class="answers"' in page:
        leaks.append("model responses")
    for it in items:
        src = it.meta.get("source")
        if src and src in page:
            leaks.append(f"the arrangement name {src!r}")
    if leaks:
        raise ValueError("public page would include " + ", ".join(sorted(set(leaks))))


def render(subset: str = "wrong_note", public: bool = True) -> str:
    """The results page as one self-contained HTML string."""
    MODELS = discover_models(subset)
    reg = load_registry()
    items = load_items(subset)
    fp = judge_fingerprint(reg.judge_spec(), WRONG_NOTE)
    # ---------------------------------------------------------------- data
    models = {}
    for mid in MODELS:
        spec = reg.get(mid)
        recs = read_records(predictions_path(mid, subset))
        judg = current_judgments(mid, subset, fp)
        score = score_model(mid, items, recs, judg)
        outcomes, _ = collect_outcomes(items, recs, judg)
        by_item: dict[str, list] = {}
        for o in outcomes:  # collect_outcomes sorts by item, then sample
            by_item.setdefault(o.item.item_id, []).append(o)
        recs_ok = [o.record for o in outcomes]
        passes = max(len(v) for v in by_item.values())
        models[mid] = {
            "name": spec.display_name,
            "lab": spec.lab,
            "reasoning": spec.reasoning,
            "score": score,
            "outcomes": outcomes,
            "by_item": by_item,
            "right": sum(o.verdict == "correct" for o in outcomes),
            "answers": len(outcomes),
            "acc": score.primary["accuracy"],
            "mixed": score.primary.get("mixed_items", 0),
            "cost": sum(r.get("cost_usd", 0) for r in recs_ok) / passes,  # one pass over every excerpt
            "reasoning_med": st.median(r["usage"]["reasoning_tokens"] for r in recs_ok),
            "latency_med": st.median(r["latency_s"] for r in recs_ok),
            "visible_med": st.median(r["usage"]["output_tokens"] - r["usage"]["reasoning_tokens"] for r in recs_ok),
        }
    order = sorted(MODELS, key=lambda m: (-models[m]["acc"], models[m]["cost"]))
    short = {m: models[m]["name"].replace("GPT-6 ", "").replace("Claude ", "").split(" ")[0] for m in MODELS}
    SAMPLES = max(len(v) for m in MODELS for v in models[m]["by_item"].values())
    n = len(items)
    TIMES = {1: "once", 2: "twice"}.get(SAMPLES, f"{SAMPLES} times")

    def state(k: int, total: int) -> str:
        return "ok" if k == total else "no" if k == 0 else "mix"

    per_item = {}
    for it in items:
        rows = []
        for mid in order:
            runs = []
            for o in models[mid]["by_item"][it.item_id]:
                text = o.record.get("raw_text", "")
                runs.append(
                    {
                        "verdict": o.verdict,
                        "text": text,
                        "bar": stated_bar(text),
                        "sample": o.sample,
                        "reasoning": o.record["usage"]["reasoning_tokens"],
                        "latency": o.record["latency_s"],
                    }
                )
            k = sum(r["verdict"] == "correct" for r in runs)
            rows.append({"mid": mid, "runs": runs, "k": k, "state": state(k, len(runs))})
        per_item[it.item_id] = rows

    right_answers = {it.item_id: sum(row["k"] for row in per_item[it.item_id]) for it in items}
    answers_per_item = {it.item_id: sum(len(row["runs"]) for row in per_item[it.item_id]) for it in items}

    # ---------------------------------------------------------------- rendering
    # Deliberately plain: one column, the system font, ordinary tables and links.

    def pct(x):
        return f"{100 * x:.0f}%"

    def clean(text: str) -> str:
        """Model responses without markdown noise (bold markers, heading hashes)."""
        return re.sub(r"(?m)^#+\s*", "", text.replace("**", "")).strip()

    def bars(runs) -> str:
        return " / ".join("?" if r["bar"] is None else str(r["bar"]) for r in runs)

    # ranking
    rank_rows = ""
    for i, m in enumerate(order, 1):
        d = models[m]
        lo, hi = d["score"].primary["accuracy_ci95"]
        rank_rows += (
            f"<tr><td>{i}</td><td>{e(d['name'])}</td><td class=num>{pct(d['acc'])}</td>"
            f"<td class=num>{pct(lo)}–{pct(hi)}</td></tr>\n"
        )

    # table of answers: a row per model, a column per excerpt
    ex_ids = [it.item_id for it in items]
    head_cells = "".join(
        f"<th>{iid[-3:]}</th>" if public else f'<th><a href="#{iid}">{iid[-3:]}</a></th>' for iid in ex_ids
    )
    key_cells = "".join(f"<td><b>{it.label['measure']}</b></td>" for it in items)
    model_rows = ""
    for m in order:
        cells = ""
        for iid in ex_ids:
            row = next(r for r in per_item[iid] if r["mid"] == m)
            cells += f'<td class="{row["state"]}">{bars(row["runs"])}</td>'
        model_rows += f"<tr><th>{e(short[m])}</th>{cells}</tr>\n"
    total_cells = "".join(f"<td>{right_answers[iid]}</td>" for iid in ex_ids)

    # private copy only: every excerpt with each model's full answers
    item_secs = ""
    for it in [] if public else items:  # the public page never touches the excerpt images
        img = base64.b64encode(it.load_image()).decode()
        k, tot = right_answers[it.item_id], answers_per_item[it.item_id]
        rows = ""
        for row in per_item[it.item_id]:
            said = "".join(
                f"<p><b>Try {r['sample'] + 1}</b> ({r['reasoning']:,} reasoning tokens, {r['latency']:.0f}s)</p>"
                f"<pre>{e(clean(r['text']))}</pre>"
                for r in row["runs"]
            )
            rows += (
                f'<details class="{row["state"]}"><summary>{e(models[row["mid"]]["name"])}: {bars(row["runs"])}</summary>'
                f'<div class="said">{said}</div></details>\n'
            )
        item_secs += (
            f'<h3 id="{it.item_id}">{it.item_id[-3:]} ({e(it.meta.get("source", ""))})</h3>\n'
            f"<p>The changed note is in bar {it.label['measure']}. {k} of {tot} answers found it.</p>\n"
            f'<p><img src="data:image/png;base64,{img}" alt="Excerpt {it.item_id[-3:]}"></p>\n'
            f'<div class="answers">{rows}</div>\n'
        )
    excerpts_html = "" if public else f"<h2>Excerpts</h2>\n{item_secs}"

    # the tab icon is a file next to the public page; the private copy lives elsewhere, so it carries it inline
    favicon = (
        "favicon.svg"
        if public or not FAVICON.exists()
        else "data:image/svg+xml;base64," + base64.b64encode(FAVICON.read_bytes()).decode()
    )

    try_html = ""
    if SAMPLE_IMAGE.exists():
        sample = SAMPLE_IMAGE.read_bytes()
        if hashlib.sha256(sample).hexdigest() in {it.image_sha256 for it in items}:
            raise ValueError(f"{SAMPLE_IMAGE} is one of the test excerpts; the example must not be")
        src = "sample.png" if public else "data:image/png;base64," + base64.b64encode(sample).decode()
        try_html = f"""<h2 id="try">Try one</h2>
<p>This is a sample and not included in the benchmark. Here's the prompt:</p>
<blockquote>{e(WRONG_NOTE.prompts["image"])}</blockquote>
<p><img src="{src}" alt="A sample excerpt: the opening bars of a choir arrangement"></p>
<details><summary>Show the answer</summary><p>Bar {SAMPLE_ANSWER}.</p></details>
"""

    judge_prompt = (
        WRONG_NOTE.judge_prompt.replace("{answer}", "N")
        .replace("{response}", "…the model's full response…")
        .replace("{{", "{")  # the template escapes literal braces for str.format
        .replace("}}", "}")
    )
    css = STYLESHEET.read_text()

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ScoreBench</title>
<link rel="icon" type="image/svg+xml" href="{favicon}">
<style>
{css}</style>
</head>
<body>

<h1>Can AI find the <mark>wrong note</mark>?</h1>
<p>I took {n} excerpts from various musical arrangements and compositions, modified one note, and asked
frontier AI models which measure the changed note is in.</p>

{try_html}
<h2 id="ranking">Ranking</h2>
<table>
<tr><th>#</th><th>Model</th><th>Accuracy</th><th>95% CI</th></tr>
{rank_rows}</table>
<p>Each excerpt was given {TIMES} to each model separately to reduce variance.</p>

<h2 id="answers">Table of answers</h2>
<div class="wide"><table class="grid">
<tr><th></th>{head_cells}</tr>
<tr><th>Correct bar</th>{key_cells}</tr>
{model_rows}<tr><th># correct</th>{total_cells}</tr>
</table></div>

{excerpts_html}
<h2 id="methodology">Methodology</h2>
<p>Each model got the excerpt image and {"the prompt above" if try_html else "this prompt: " + e(WRONG_NOTE.prompts["image"])},
with no system prompt and no tools. All models ran at their
<code>high</code> reasoning setting. Each excerpt was asked {TIMES}, in separate requests.</p>
<p>For grading, GPT-6 Luna read each model's full response next to the right bar number, without the image, and replied correct
or incorrect:</p>
<blockquote class="pre">{e(judge_prompt)}</blockquote>

<hr>
<p class="small">Made by <a href="{AUTHOR_URL}">{AUTHOR}</a>. The code is <a href="{REPO_URL}">on GitHub</a>.
Last updated {date.today():%B %-d, %Y}.{"" if public else " This private copy includes the test excerpts."}</p>

</body>
</html>
"""

    if public:
        _check_public(page, items)
    return page
