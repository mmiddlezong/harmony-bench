"""The results website for the wrong_note subset: `harmonybench site`.

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
    pip_tag = "span" if public else "a"

    def link(iid: str) -> str:
        return "" if public else f' href="#{iid}"'

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

    def pct(x):
        return f"{100 * x:.0f}%"

    def clean(text: str) -> str:
        """Model responses without markdown noise (bold markers, heading hashes)."""
        return re.sub(r"(?m)^#+\s*", "", text.replace("**", "")).strip()

    MARK = {"ok": "✓", "no": "✗"}
    ex_ids = [it.item_id for it in items]

    def bars_of(row) -> str:
        return " / ".join("–" if r["bar"] is None else str(r["bar"]) for r in row["runs"])

    # ranking: a dot plot on one shared 0-100% axis, with each model's answer pattern beside it
    rank_rows = ""
    for i, m in enumerate(order, 1):
        d = models[m]
        lo, hi = d["score"].primary["accuracy_ci95"]
        pattern = "".join(
            f'<{pip_tag} class="pip {row["state"]}"{link(iid)} title="{iid[-3:]}: {bars_of(row)} (answer {it.label["measure"]})"></{pip_tag}>'
            for it, iid in zip(items, ex_ids, strict=True)
            for row in per_item[iid]
            if row["mid"] == m
        )
        rank_rows += f"""<li>
      <span class="rn">{i}</span>
      <span class="who"><b>{e(d["name"])}</b></span>
      <span class="track"><span class="ci" style="left:{100 * lo:.1f}%;width:{100 * (hi - lo):.1f}%"></span><span class="dot" style="left:{100 * d["acc"]:.1f}%"><span>{pct(d["acc"])}</span></span></span>
      <span class="pattern">{pattern}</span>
    </li>"""
    pattern_w = 13 * n + 3 * (n - 1)  # n squares of 13px with 3px gaps
    css = f":root {{ --pattern-w: {pattern_w}px; }}\n" + STYLESHEET.read_text()
    axis = "".join(f'<span style="left:{t}%">{t}%</span>' for t in (0, 25, 50, 75, 100))

    # every answer, as a score: one staff per model, grouped by lab like instrument families
    labs: dict[str, list[str]] = {}
    for m in order:
        labs.setdefault(models[m]["lab"], []).append(m)
    sys_rows = ""
    bar_head = "".join(
        f"<th>{iid[-3:]}</th>" if public else f'<th><a href="#{iid}">{iid[-3:]}</a></th>' for iid in ex_ids
    )
    key_row = "".join(f'<td class="k">{it.label["measure"]}</td>' for it in items)
    for lab, mids in labs.items():
        for j, m in enumerate(mids):
            cells = ""
            for iid in ex_ids:
                row = next(r for r in per_item[iid] if r["mid"] == m)
                label = "<i>/</i>".join("–" if r["bar"] is None else str(r["bar"]) for r in row["runs"])
                cells += f'<td class="c {row["state"]}">{label}</td>'
            brace = f'<td class="brace" rowspan="{len(mids)}"><span>{e(lab)}</span></td>' if j == 0 else ""
            sys_rows += (
                f'<tr class="{"first" if j == 0 else ""}">{brace}<th class="staff">{e(short[m])}</th>{cells}</tr>'
            )
    tot_row = "".join(f'<td class="t">{right_answers[iid]}</td>' for iid in ex_ids)

    # excerpts, each opened by a boxed rehearsal mark
    item_secs = ""
    for it in [] if public else items:  # the public page never touches the excerpt images
        img = base64.b64encode(it.load_image()).decode()
        k, tot = right_answers[it.item_id], answers_per_item[it.item_id]
        rows = ""
        for row in per_item[it.item_id]:
            runs = "".join(
                f'<span class="r {"ok" if r["verdict"] == "correct" else "no"}">'
                f"{MARK['ok' if r['verdict'] == 'correct' else 'no']}&thinsp;{'–' if r['bar'] is None else r['bar']}</span>"
                for r in row["runs"]
            )
            bodies = "".join(
                f"<div><h5>Run {r['sample'] + 1}<small>{r['reasoning']:,} reasoning tokens, {r['latency']:.0f}s</small></h5>"
                f"<p>{e(clean(r['text']))}</p></div>"
                for r in row["runs"]
            )
            rows += (
                f'<details class="{row["state"]}"><summary><span class="m">{e(models[row["mid"]]["name"])}</span>{runs}</summary>'
                f'<div class="said">{bodies}</div></details>'
            )
        item_secs += f"""
    <article class="ex" id="{it.item_id}">
      <div class="exhead"><span class="mark">{it.item_id[-3:]}</span>
        <div><div class="src">{e(it.meta.get("source", ""))}</div>
        <div class="ans">The changed note is in bar <b>{it.label["measure"]}</b>. {k} of {tot} answers found it.</div></div></div>
      <div class="plate"><img src="data:image/png;base64,{img}" alt="Excerpt {it.item_id[-3:]}" loading="lazy"></div>
      <div class="answers"><div class="anskey"><span>Model</span><span>1st and 2nd try</span></div>{rows}</div>
    </article>"""

    try_html = ""
    if SAMPLE_IMAGE.exists():
        sample = SAMPLE_IMAGE.read_bytes()
        if hashlib.sha256(sample).hexdigest() in {it.image_sha256 for it in items}:
            raise ValueError(f"{SAMPLE_IMAGE} is one of the test excerpts; the example must not be")
        src = "sample.png" if public else "data:image/png;base64," + base64.b64encode(sample).decode()
        try_html = f"""<h2 id="try">Try one</h2>
    <div class="try">
      <div class="plate"><img src="{src}" alt="Example excerpt: the opening bars of a choir arrangement"></div>
      <div class="trytext">
        <p>This is an example, not one of the {n} in the test. Every excerpt looks like this: part of a real score with
        one note changed. The models get the image and this question, and nothing else:</p>
        <blockquote>{e(WRONG_NOTE.prompts["image"])}</blockquote>
        <details class="reveal"><summary>Show the answer</summary><p>Bar {SAMPLE_ANSWER}.</p></details>
      </div>
    </div>"""

    excerpts_html = (
        ""
        if public
        else '<h2 id="excerpts">The excerpts</h2><p class="fine">Click a model to read what it said.</p>' + item_secs
    )
    judge_prompt = (
        WRONG_NOTE.judge_prompt.replace("{answer}", "N")
        .replace("{response}", "…the model's full response…")
        .replace("{{", "{")  # the template escapes literal braces for str.format
        .replace("}}", "}")
    )

    page = f"""<!doctype html>
    <html lang="en">
    <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Can AI find the wrong note? – HarmonyBench</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400;1,6..72,500&family=Barlow+Semi+Condensed:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
{css}</style>
    </head>
    <body>
    <div class="page">

    <header class="title">
      <h1>Can AI find the wrong note?</h1>
      <p class="deck">I took {n} excerpts from various musical arrangements and compositions, modified one note, and asked
      frontier AI models which measure the changed note is in.</p>
    </header>

    {try_html}

    <h2 id="ranking">Ranking</h2>
    <div class="axis"><span></span><span></span><span class="scale">{axis}</span><span></span></div>
    <ol class="ranking">{rank_rows}</ol>
    <p class="fine">Gray bars around the accuracy are 95% confidence intervals. Each excerpt was given twice to each model
    separately to reduce variance. The squares are the {n} excerpts in order: green if both answers were right, gold if one
    was, hollow if neither was.{"" if public else " Click one to jump to the excerpt."}</p>

    <h2 id="score">Table of answers</h2>
    <div class="scroll"><table class="system">
      <thead><tr><th></th><th></th>{bar_head}</tr></thead>
      <tbody>
        <tr class="key"><td></td><th class="rowlab">Answer</th>{key_row}</tr>
        {sys_rows}
        <tr class="tot"><td></td><th class="rowlab"># correct</th>{tot_row}</tr>
      </tbody>
    </table></div>
    <div class="legend"><span><i class="sw ok"></i>Right both times</span><span><i class="sw mix"></i>Right once</span><span><i class="sw no"></i>Wrong both times</span></div>

    {excerpts_html}

    <h2 id="methodology">Methodology</h2>
    <div class="notes">
      <h3>The question</h3>
      <p>Each model got the excerpt image and this text, with no system prompt and no tools:</p>
      <blockquote>{e(WRONG_NOTE.prompts["image"])}</blockquote>
      <p>All models ran at their <code>high</code> reasoning setting. Each excerpt was asked {TIMES}, in separate requests.</p>
      <h3>The grading</h3>
      <p>GPT-6 Luna read each model's full response next to the right bar number, without the image, and replied correct
      or incorrect:</p>
      <blockquote>{e(judge_prompt)}</blockquote>
    </div>
    <p class="colophon">HarmonyBench, updated {date.today():%B %-d, %Y}. {"The test excerpts are unpublished, so they aren't shown here." if public else "Private copy: includes the unpublished test excerpts."}</p>
    </div>
    </body>
    </html>"""

    if public:
        _check_public(page, items)
    return page
