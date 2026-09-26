# HarmonyBench leaderboard: triads_root (prompt v1)

_Generated 2026-09-26 03:17 UTC. Ranked by image accuracy. 95% CIs from 10,000 item-level bootstrap resamples._

| # | Model | Image accuracy (95% CI) | Enharmonic | Key sig. | Accidentals | MusicXML accuracy | Reading gap | Fail | Judge flags | Cost | n |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **GPT-6 Sol** | 95% (88%–100%) | 95% | 97% | 93% | 100% | +5 pts | 0% | 0 | $0.38 | 60/60 |

**Columns.** *Accuracy*: share of items the judge model graded correct. *Enharmonic*: also counts near misses (enharmonic). *Reading gap*: MusicXML accuracy minus image accuracy on the same items; a large gap means the model knows the harmony but misreads the image. *Fail*: empty answers, refusals, truncations and answers that name nothing (all scored wrong). *Judge flags*: judge verdicts that disagree with the rule-based parser, worth checking by hand (`harmonybench disagreements`). *Cost*: API spend to run every item once in every condition, at list prices. ⚠ = incomplete run.

## Usage

| Model | Mean output tokens | Median latency | Spent so far | Judge cost |
|---|---:|---:|---:|---:|
| GPT-6 Sol | 190 (179 reasoning) | 3.8s | $0.38 | $0.007 |
