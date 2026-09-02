# Detecting text watermarks in production LLMs

Black-box tests for statistical text watermarks in Claude, Gemini, and GPT, with the
detectors validated on a model watermarked in-house using SynthID. This repository holds
the method and the raw data behind the article *Catching the Watermark* (link to be added).

**Headline finding.** Google's Gemini developer API watermarks its text output, while
Claude (Opus 4.8 and Sonnet 5) and GPT-5.2 do not, as of late August 2026. Full numbers are
in [`results/RESULTS_SUMMARY.md`](results/RESULTS_SUMMARY.md).

A statistical text watermark biases which words a model picks, in a way that is invisible to
a reader but detectable by someone with the secret key. The question here is whether that
mark can be spotted from the *outside*, with ordinary API queries and no key, and the answer
is a qualified yes.

## How the tests work

All three tests build on the ETH Zurich Red-Green test (Gloaguen et al., ICLR 2025). The
model is asked to finish the sentence `I ate {number}` by choosing one of four fruits at
random. A keyed watermark nudges that choice based on the few tokens just before it, so for
a fixed number the same fruit gets nudged the same way across ten different sentence
openings. That cross-context agreement is scored against a shuffled null; with no watermark,
the fruit choice is independent of the number.

Each test differs only in **how the numbers are chosen**:

- **repeated-digit** (the ETH original): a single digit repeated fifteen times (`111...1`).
  Content-neutral, so any agreement must come from the key. It has one blind spot, below.
- **random-k**: nine distinct fifteen-digit random numbers. These never repeat a context, so
  they slip past SynthID's skip rule (see below). The cost is that distinct numbers have
  distinct content, which is why this design needed validating.
- **window-k**: nine numbers sharing the same fourteen leading digits, varying only the last.
  The number is held nearly constant, so the fruit cannot shift because of what the number
  means, while the final token (the watermark's hash window) still changes.

**The blind spot.** SynthID skips watermarking at any position whose preceding tokens have
already appeared earlier in the text, to preserve quality on repetitive passages. A repeated
digit is maximally repetitive, so the repeated-digit test measures a position where the mark
was never applied. It can return null even when a watermark is present.

## Validating the detectors

A test that fires proves nothing until you know it fires for the right reason. So we took an
open model (Mistral-7B) and ran each design on it twice, once with a real SynthID watermark
switched on at generation and once with it off, using the SynthID implementation in Hugging Face
Transformers. Same model, same prompts; the watermark is the only difference, so base-model
behavior cancels between the two runs. Result: **window-k and random-k fire only when the
watermark is on, and the repeated-digit test is blind even when it is on.** See
[`results/cloud_validation.json`](results/cloud_validation.json).

## What is in this repository

```
scripts/
  watermark_redgreen.py    Run and analyze the Red-Green grids on the live APIs
  cloud_validate.py        Watermark-ON-vs-OFF validation on an open model (GPU)
  local_synthid_logit.py   Logit-level check that the tilt is window-local (supporting)
data/redgreen/             Raw generations (*.jsonl) and per-grid p-values (*_result_*.json)
                           for Claude, Sonnet, Gemini, and GPT across all three designs
results/
  api_results.csv          Every model x design p-value in one table
  cloud_validation.json    The ON/OFF validation numbers
  RESULTS_SUMMARY.md        The tables and the plain-language read
```

Each `data/redgreen/*.jsonl` file is one grid, with one JSON object per query recording the
sentence prefix, the number, the raw model response, and the parsed fruit. The
`*_result_*.json` files hold the median p-value for that grid.

## Reproduce

Requires Python 3.10+ and API keys for the providers you want to test.

```bash
pip install -r requirements.txt

# provide keys via environment or a local .env (never commit it)
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
export OPENAI_API_KEY=...
```

Run a grid, then analyze it. `--provider` is one of `claude` (Opus 4.8), `sonnet`
(Sonnet 5), `gemini`, or `gpt`; `--kmode` is `repeat` (default), `random`, or `window`.

```bash
# generate a grid (10 prefixes x 9 numbers x N samples per cell)
python scripts/watermark_redgreen.py --provider gemini --kmode window --n 50

# compute the Red-Green statistic on it
python scripts/watermark_redgreen.py --provider gemini --kmode window --analyze
```

Generations are written to `scripts/analysis/` (relative to the script). The grids already
in `data/redgreen/` are the output of the actual runs behind the paper.

**The validation step needs a GPU.** `cloud_validate.py` loads an open model, applies a real
SynthID watermark, and runs all three designs on and off. We ran it on a RunPod RTX 4090 with
`torch` and `transformers` installed:

```bash
pip install torch transformers accelerate
python scripts/cloud_validate.py --model mistralai/Mistral-7B-Instruct-v0.1 --n 50
```

Mistral-7B is used because it spreads its fruit choices enough for a watermark to have room
to act; some models collapse to a single option and leave nothing to detect.

## The statistic

For each grid, the responses are grouped by (sentence prefix, number). The statistic asks how
consistently a given number pushes the same fruit across all ten prefixes, and compares that
to a null built from 10,000 random shuffles, taking the median over 100 bootstraps. A median
p-value below 0.05 is a detection. This is a numpy re-implementation of the ETH team's test.

## Caveats

- The detectors were validated against SynthID as implemented in Hugging Face Transformers,
  the same scheme family Google uses, not Google's exact production settings (keys, window
  size, masking policy). The precise claim is that Gemini's API produces a SynthID-family
  watermark these tests can read, as of 2026-08-22 on `gemini-3.5-flash`.
- Results are a snapshot. Watermark deployment is changing quickly, and every model here
  predates the point where marking becomes standard, so re-running later may give different
  answers.
- The `n/a` for GPT repeated-digit means that cell was not run; only the two masking-immune
  designs were used on GPT.

## Prior work

- Gloaguen, T., Jovanović, N., Staab, R., Vechev, M. "Black-Box Detection of Language Model
  Watermarks." ICLR 2025. arXiv:2405.20777. Official code:
  https://github.com/eth-sri/watermark-detection
- Dathathri, S. et al. "Scalable watermarking for identifying large language model outputs"
  (SynthID-Text). Nature 634, 818-823 (2024). The repeated-context masking behavior is
  described here.
- Kirchenbauer, J. et al. "A Watermark for Large Language Models." ICML 2023.
  arXiv:2301.10226. The Red-Green mechanism window-k probes.
- SynthID as implemented in Hugging Face Transformers (`SynthIDTextWatermarkingConfig`), the
  open-source port used for validation.

## License

No license yet. Add one before making the repository public.
