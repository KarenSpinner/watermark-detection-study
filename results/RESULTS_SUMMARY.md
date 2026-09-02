# Results summary

All values are median p-values from the Red-Green statistic. A value **below 0.05**
means the test detected a watermark; a value **above 0.05** means it did not.

## API tests (production models, live queries)

| Model | repeated-digit | random-k | window-k | Verdict |
|---|---|---|---|---|
| Claude Opus 4.8 | 1.00 | 1.00 | 1.00 | not watermarked |
| Claude Sonnet 5 | 0.99 | 0.98 | 0.99 | not watermarked |
| Gemini 3.5 Flash | 0.58 / 0.66 (k25) | **0.005** | **0.0006** | **watermarked** |
| GPT-5.2 | n/a | 0.70 | 0.72 | not watermarked |

Only the two masking-immune designs (random-k, window-k) are trustworthy on their own,
because the repeated-digit design has a blind spot (see below). Gemini fires on both;
Claude and GPT do not. GPT was run on the two masking-immune designs only.

## Detector validation (Mistral-7B, SynthID switched ON vs OFF)

Same model, same prompts; the watermark is the only difference. This isolates the
watermark from base-model behavior, which cancels between the two columns.

| Design | Watermark ON | Watermark OFF | Verdict |
|---|---|---|---|
| repeated-digit (published) | 0.96 | 1.00 | blind — the mark is skipped under masking |
| random-k | **0.00** | 0.65 | valid detector (fires only when watermarked) |
| window-k | **0.00** | 0.65 | valid detector (fires only when watermarked) |

## Conclusion

- **Gemini's developer API watermarks its text output** (fires on both validated
  detectors; the other models do not). This is consistent with Google's SynthID and
  resolves an open developer question about whether the API (not just the app) marks text.
- **Claude Opus 4.8 and Sonnet 5 are not watermarked** on these tests, confirmed by
  detectors that see through the repeated-digit blind spot.
- **The published repeated-digit test has a blind spot:** SynthID skips watermarking at
  repeated contexts, so a repeated-digit probe can miss a watermark that is present
  (validated: null even with the watermark ON).

Caveat: the detectors were validated against SynthID as implemented in Hugging Face
Transformers, the same scheme family Google uses, not Google's exact production settings.
So the precise claim is that Gemini's API produces a SynthID-family watermark these tests
can read, as of 2026-08-22 on gemini-3.5-flash.
