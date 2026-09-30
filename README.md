# HarmonyBench

Can AI find the wrong note? I took 15 excerpts from various musical arrangements and compositions, modified one
note, and asked frontier AI models which measure the changed note is in.

Results: https://mmiddlezong.github.io/harmony-bench/

## How it works

Each model gets the excerpt image and this prompt, nothing else:

> This is an image of part of a score. One note has been modified from the original. Which measure is it in? Answer
> with its bar number.

Every model runs at its `high` reasoning setting, and each excerpt is asked twice. GPT-6 Luna grades each answer
against the correct bar without seeing the image.

The excerpts aren't in this repo so they stay out of training data.

## Running it

You need [uv](https://docs.astral.sh/uv/), API keys in `.env` (see `.env.example`), and your own excerpts in
`data/wrong_note/`.

```bash
uv sync
uv run harmonybench run gpt-6-sol claude-opus-5-5 -s wrong_note --repeats 2
uv run harmonybench score -s wrong_note
uv run harmonybench site
```

`harmonybench models` lists the configured models. To add one, add it to `configs/models.yaml`.

There's also a generated warm-up set of root-position triads (`harmonybench build`, subset `triads_root`), which I
used to test the pipeline.
