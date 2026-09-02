# Results summary

All values are median p-values from the Red-Green statistic over 100 bootstraps of a
10,000-permutation test. **Below 0.05 is a detection.** Two statistics are reported: the
published ETH statistic ("published") and the prefix-stratified variant added on
2026-09-02 ("corrected", `--stat within`; see the README). The full table with every run is
in [`api_results.csv`](api_results.csv).

## 1. The standard prompt (number echoed in the prompt), all designs

| Model | repeated-digit | random-k | window-k | window-k, 40-digit |
|---|---|---|---|---|
| Claude Fable 5.1 (2026-09-02) | 1.00 / 0.69 | 1.00 / 0.21 | 1.00 / 0.22 | 1.00 / 0.85 |
| Claude Opus 4.8 (2026-08-22) | 1.00 / 0.18 | 1.00 / 0.0003* | 1.00 / 0.70 | |
| Claude Sonnet 5 (2026-08-23) | 0.99 / 0.73 | 0.98 / 0.84 | 0.99 / 0.92 | |
| Gemini 3.5 Flash (2026-08-22, 09-02) | 0.58 / 0.75 | **0.005** / **0.0006** | **0.0006** / **0.0001** | **0.0001** / **0.0003** |
| GPT-5.2 (2026-08-22) | n/a | 0.70 / 0.66 | 0.72 / 0.89 | |

Each cell is published / corrected. \*Opus 4.8's random-k grid fires under the corrected
statistic while its window-k grid does not: distinct random numbers have content that
sways Opus's choice consistently across openings. That is the confound window-k removes,
so window-k is the column of record.

## 2. The no-echo prompt (number given as spaced digits; the response is the first place the phrase appears)

| Model | window-k, 15-digit | window-k, 40-digit | share of variance explained by the last digit (40-digit) |
|---|---|---|---|
| Claude Fable 5.1 | 0.16 / **0.006** (n=100; 0.005 at n=50) | 0.08 / **0.0000** | 50% |
| Claude Opus 4.8 (control) | 1.00 / 0.82 | 0.14 / 0.03 | 19% |
| Claude Sonnet 5 (control) | 0.63 / 0.47 | 0.99 / 0.84 | 7% |
| Gemini 3.5 Flash (positive control) | **0.0000** / **0.0000** | **0.0000** / **0.0000** | 48% |

On the 40-digit no-echo grid the published statistic gives 0.08 for Fable 5.1 only because
more than half its cells have all 50 samples on one fruit, which the published code turns
into infinite logits; clipping those rates alone (no change to the null) gives 0.0000.
Opus 4.8's 0.03 is a small effect concentrated on one digit (largest per-digit swing 29
points vs Fable's 71); its per-digit pattern is uncorrelated with Fable's (r = -0.39), so it
is not a shared content effect of the spaced-digit prompt. Whether it is Opus's known number-
content sensitivity or an early trace of the retrofit cannot be told from outside.

## 3. Detector validation (Mistral-7B, SynthID switched ON vs OFF, 2026-08-22)

| Design | Watermark ON | Watermark OFF | Verdict |
|---|---|---|---|
| repeated-digit (published) | 0.96 | 1.00 | blind, the mark is skipped under masking |
| random-k | **0.00** | 0.65 | valid detector |
| window-k | **0.00** | 0.65 | valid detector |

## 4. Power of the two statistics (semi-synthetic injection, `scripts/power_check.py`)

The smallest keyed tilt (one fruit's odds multiplied by the factor, for a given number,
across all ten openings) at which the median p drops below 0.05, on each model's real
echoed window-k grid:

| Grid | published statistic | corrected statistic |
|---|---|---|
| Claude Fable 5.1 | x3 | x1.75 (a 50/50 choice pushed to 64/36) |
| Claude Opus 4.8 | not at x4 | x3 |
| Claude Sonnet 5 | never | x3 |
| Gemini 3.5 Flash | fires with no injection (already marked) | same |

## Conclusions

- **Gemini's developer API watermarks its text output** (Aug 22; every variant since).
- **Claude Fable 5.1 most likely watermarks its text output, as Anthropic states.** The mark
  is invisible to every design that echoes the number in the prompt, and appears once the
  number is not echoed, weakly with a 15-digit number and strongly with a 40-digit one.
  Of the sibling models Anthropic lists as not yet marked, Sonnet 5 is null on the same
  prompts and Opus 4.8 shows only the small, differently-patterned trace noted above. The parsimonious reading is that Anthropic's masking rule treats the
  prompt as already-seen context, and that its hash window is longer than the four tokens
  a 15-digit number spans.
- **Claude Opus 4.8 and Sonnet 5 are not watermarked** on any echoed design or the 15-digit
  no-echo design, with a corrected statistic that would catch a x3 tilt. Sonnet 5 is also
  null on the 40-digit no-echo design; Opus 4.8 shows a weak trace there (see above).
- **Two blind spots in the published protocol:** repeated-context masking hides the mark
  from the repeated-digit design, and echoing the number in the prompt hides it from every
  design on an implementation that masks on prompt context. **One weakness in the published
  statistic:** it has little power on models whose word choice depends strongly on the
  sentence opening (all Claude models here).

Caveats: the detectors were validated against SynthID in Hugging Face Transformers, not
Google's or Anthropic's production settings. No Claude model with a switchable watermark
exists outside Anthropic, so the Fable 5.1 claim rests on the Gemini positive control, the
Opus/Sonnet negative controls, the injection power check, and replication, not on an ON/OFF
contrast. Snapshot dates as given; results can change with model updates.
