# Detecting text watermarks in production LLMs

Black-box tests for statistical text watermarks in Claude, Gemini, and GPT, with the
detectors validated on a model watermarked in-house using SynthID. This repository holds
the method and the raw data behind two articles on *Wondering About AI*:

- [A mostly plain language primer on Anthropic's new watermark and how it can be detected](https://wonderingaboutai.substack.com/p/a-mostly-plain-language-primer-on) (August 23, 2026)
- [I built a new test and confirmed that Claude Fable 5.1 is watermarked](https://wonderingaboutai.substack.com/p/i-built-a-new-test-and-confirmed/) (September 8, 2026)

**Headline findings.**

1. **Google's Gemini developer API watermarks its text output** (tested 2026-08-22 and again
   2026-09-02 on `gemini-3.5-flash`). It fires on every masking-immune design below.
2. **Claude Fable 5.1 most likely watermarks its text output, as Anthropic states it does.**
   Tested 2026-09-02, one day after launch. Every design that echoes the test number in the
   prompt returns null, the same as Opus 4.8. Once the number is given as spaced digits so
   that the phrase the watermark reads first appears in the response, a keyed pattern shows
   up, weakly with a 15-digit number (corrected p = 0.006, replicated) and strongly with a
   40-digit one (corrected p = 0.0000; the last digit explains 50% of the choice, as on
   Gemini). Of the sibling models Anthropic lists as not yet marked, Sonnet 5 is null on the
   same prompts and Opus 4.8 shows only a small trace on the 40-digit prompt (corrected
   p = 0.03, digit share 19%) whose per-digit pattern does not match Fable's. See
   [`results/RESULTS_SUMMARY.md`](results/RESULTS_SUMMARY.md).
3. **Claude Opus 4.8, Claude Sonnet 5, and GPT-5.2 are not watermarked** on any echoed design,
   nor on the 15-digit no-echo design (Opus 4.8's 40-digit trace is discussed in the summary).
4. **The published Red-Green protocol has two blind spots and one weakness.** Repeated-digit
   numbers trigger SynthID's repeated-context masking. Echoing the number in the prompt
   hides the mark from every design on an implementation that masks on prompt context.
   And the published statistic has little power on models whose word choice depends
   strongly on the sentence opening, which is every Claude model here.

A statistical text watermark biases which words a model picks, in a way that is invisible to
a reader but detectable by someone with the secret key. The question here is whether that
mark can be spotted from the *outside*, with ordinary API queries and no key.

## How the tests work

All designs build on the ETH Zurich Red-Green test (Gloaguen et al., ICLR 2025). The model
is asked to finish the sentence `I ate {number}` by choosing one of four fruits at random.
A keyed watermark nudges that choice based on the few tokens just before it, so for a fixed
number the same fruit gets nudged the same way across ten different sentence openings. That
cross-context agreement is scored against a shuffled null; with no watermark, the fruit
choice is independent of the number.

The designs differ in **how the number is chosen** and **how it is shown to the model**:

- **repeated-digit** (the ETH original): a single digit repeated fifteen times (`111...1`).
  Content-neutral, but SynthID skips watermarking at any position whose preceding tokens
  have already appeared in the generated text, and a repeated digit is maximally
  repetitive, so this design measures a position where the mark was never applied.
- **random-k**: nine distinct fifteen-digit random numbers. Never repeat a context, but the
  numbers have content that can sway the choice on its own (Opus 4.8 shows this).
- **window-k**: nine numbers sharing the same leading digits, varying only the last. The
  number's content is held nearly constant while the last token of the hash window still
  changes. The design of record.
- **40-digit window-k** (`--context 40`): the same, with a 40-digit number. A 15-digit
  number spans only ~4 tokens on the Claude tokenizer; a 40-digit one spans ~13, enough to
  fill a longer hash window so the sentence opening never enters it.
- **no-echo** (`--noecho`): the number is given in the prompt as spaced digits (`4 7 3 8 ...`)
  with an instruction to write it compactly. In the standard template the model sees the
  number twice before writing it, so an implementation that treats the prompt as
  already-seen context would skip the mark at the one position the test measures. With
  no-echo, the phrase `I ate 473829105624381` first appears in the response.

## The two statistics

**Published** (`--stat eth`, default): the ETH statistic and permutation null, ported
verbatim to numpy. Cells are grouped by (opening, number); the statistic asks how
consistently a number pushes the same fruit across all ten openings, against a null built
by shuffling all ninety cells; median over 100 bootstraps of a 10,000-permutation test.

**Corrected** (`--stat within`): the same statistic with three changes. Bootstrapped cell
rates are clipped away from 0 and 1 before the logit; each opening is centered on its own
median across numbers; and the null shuffles numbers *within* each opening rather than
across the whole grid. Shuffling across openings is only valid if the opening has no effect
on the choice, and on Claude models the opening explains 70-80% of the variation in the
fruit rate, so the published null is inflated and the test is close to blind. The
corrected null is the exchangeability that holds under no watermark. It fires on Gemini's
masking-immune grids, stays null on Gemini's masked grids and on GPT, and by injection
(`scripts/power_check.py`) catches a x1.75 keyed tilt on Fable 5.1's grid where the
published statistic needs x3. The ETH code contains the centering step multiplied by zero.

## Validation

- **ON/OFF on an open model.** Mistral-7B with a real SynthID watermark switched on and off
  at generation, using the SynthID implementation in Hugging Face Transformers. Window-k and
  random-k fire only when the watermark is on; repeated-digit is blind even when it is on.
  See [`results/cloud_validation.json`](results/cloud_validation.json).
- **Positive control on every variant.** Gemini fires on each prompt variant the day it was
  introduced (40-digit, no-echo, and both together).
- **Negative controls.** Opus 4.8 and Sonnet 5, sibling models Anthropic lists as not yet
  marked, run on the identical no-echo prompts.
- **Power.** `scripts/power_check.py` injects a synthetic keyed tilt into a real grid and
  reports the smallest tilt each statistic detects.

## What is in this repository

```
scripts/
  watermark_redgreen.py    Run and analyze the Red-Green grids on the live APIs
  power_check.py           Injection power check for a finished grid
  cloud_validate.py        Watermark-ON-vs-OFF validation on an open model (GPU)
  local_synthid_logit.py   Logit-level check that the tilt is window-local (supporting)
data/redgreen/             Raw generations (*.jsonl) and per-grid p-values (*_result_*.json,
                           *_within.json for the corrected statistic) for every model and design
results/
  api_results.csv          Every model x design x prompt p-value under both statistics
  cloud_validation.json    The ON/OFF validation numbers
  RESULTS_SUMMARY.md       The tables and the plain-language read
```

Each `data/redgreen/*.jsonl` file is one grid, with one JSON object per query recording the
sentence prefix, the number, the raw model response, and the parsed fruit. Fable 5.1 records
also carry the served model id, stop reason, and token usage. File suffixes: `_krand`
(random-k), `_kwin` (window-k), `_kwin40` (40-digit), `_noecho` (no-echo prompt),
`_ctx25` (25-digit repeated-digit).

## Reproduce

Requires Python 3.10+ and API keys for the providers you want to test.

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...   # or a local .env (never commit it)
export GEMINI_API_KEY=...
export OPENAI_API_KEY=...
```

`--provider` is one of `fable` (Fable 5.1), `claude` (Opus 4.8), `sonnet` (Sonnet 5),
`gemini`, or `gpt`; `--kmode` is `repeat` (default), `random`, or `window`.

```bash
# the design that shows Fable 5.1's mark: no-echo, 40-digit window-k, 50 samples per cell
python scripts/watermark_redgreen.py --provider fable --kmode window --context 40 --noecho --n 50

# score it with both statistics
python scripts/watermark_redgreen.py --provider fable --kmode window --context 40 --noecho --analyze
python scripts/watermark_redgreen.py --provider fable --kmode window --context 40 --noecho --analyze --stat within

# how small a keyed tilt the test could have seen on that grid
python scripts/power_check.py scripts/analysis/redgreen_fable_kwin40_noecho.jsonl 20 2000 within
```

A 4,500-query grid costs about $10-13 on Fable 5.1 and runs in 10-40 minutes depending
on API load. Generations are written to `scripts/analysis/` (relative to the script); the
grids in `data/redgreen/` are the output of the actual runs behind the articles.

**Fable 5.1 note.** Its extended thinking cannot be disabled, so the harness sends
`output_config.effort="low"` and a 4,096-token cap and records thinking-block counts. On
this one-line prompt it emitted no thinking at all in 36,000 responses, at low and at the
default effort. No server-side fallback is requested, so every response comes from the
model asked for.

**The validation step needs a GPU.** `cloud_validate.py` loads an open model, applies a real
SynthID watermark, and runs the three original designs on and off. We ran it on a RunPod
RTX 4090 with `torch` and `transformers` installed:

```bash
pip install torch transformers accelerate
python scripts/cloud_validate.py --model mistralai/Mistral-7B-Instruct-v0.1 --n 50
```

## Caveats

- The detectors were validated against SynthID as implemented in Hugging Face Transformers,
  the same scheme family Google uses and Anthropic says it adapted, not either vendor's
  production settings. No Claude model with a switchable watermark exists outside
  Anthropic, so the Fable 5.1 claim rests on the Gemini positive control, the Opus/Sonnet
  negative controls, the injection power check, and replication.
- Results are a snapshot on the dates given. Anthropic has said pre-cutoff models will be
  retrofitted; rerunning later may give different answers.
- The `n/a` for GPT repeated-digit means that cell was not run.

## Prior work

- Gloaguen, T., Jovanović, N., Staab, R., Vechev, M. "Black-Box Detection of Language Model
  Watermarks." ICLR 2025. arXiv:2405.20777. Official code:
  https://github.com/eth-sri/watermark-detection
- Dathathri, S. et al. "Scalable watermarking for identifying large language model outputs"
  (SynthID-Text). Nature 634, 818-823 (2024). The repeated-context masking behavior is
  described here.
- Kirchenbauer, J. et al. "A Watermark for Large Language Models." ICML 2023.
  arXiv:2301.10226. The Red-Green mechanism window-k probes.
- Anthropic, "How Claude marks AI-generated content" (help center) and "Introducing Claude
  Fable 5.1 and Claude Mythos 5.1" (September 1, 2026), which state that Fable 5.1's text
  output is marked.
- SynthID as implemented in Hugging Face Transformers (`SynthIDTextWatermarkingConfig`), the
  open-source port used for validation.

## License

MIT, see [`LICENSE`](LICENSE).
