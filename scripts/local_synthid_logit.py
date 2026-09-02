#!/usr/bin/env python3
"""Direct logit-level positive control for the window design.

Avoids generation, chat templating, and model-capability issues. For a raw
prefix ending in a number whose LAST digit we vary, we read the model's
next-token logits restricted to the four fruit tokens, once as the base model
and once with the SynthID watermark processor applied. Then:

  base varies with last digit   -> base-model determinism (the window design
                                   could false-positive without any watermark)
  base flat, SynthID varies      -> the tilt is watermark-specific; the window
                                   design detects a real mark and not determinism

We also vary a digit OUTSIDE the hash window (a leading digit) to see whether
the base effect is window-local (mimicking a watermark) or global.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          SynthIDTextWatermarkLogitsProcessor)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
KEYS = [654, 400, 836, 123, 340, 443, 597, 160, 57, 29,
        590, 639, 13, 715, 468, 990, 966, 226, 324, 585]
NGRAM_LEN = 5
FRUITS = [" strawberries", " blueberries", " raspberries", " blackberries"]


def fruit_probs(logits, fruit_ids):
    sub = logits[fruit_ids]
    return torch.softmax(sub, dim=-1).cpu().numpy()


def main():
    print(f"device={DEVICE} model={MODEL}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEVICE)
    model.eval()
    fruit_ids = [tok(f, add_special_tokens=False).input_ids[0] for f in FRUITS]
    proc = SynthIDTextWatermarkLogitsProcessor(
        ngram_len=NGRAM_LEN, keys=KEYS, sampling_table_size=65536,
        sampling_table_seed=0, context_history_size=1024, device=DEVICE)

    base_prefix = "I ate 47382910562438"       # vary the LAST digit (in-window)
    lead_prefix_tail = "382910562438"           # vary a LEADING digit (out-of-window)

    @torch.no_grad()
    def logits_for(text):
        ids = tok(text, return_tensors="pt", add_special_tokens=False).to(DEVICE)
        out = model(**ids).logits[0, -1]                       # next-token logits
        base = fruit_probs(out, fruit_ids)
        wm_scores = proc(ids.input_ids, out.unsqueeze(0)).squeeze(0)
        wm = fruit_probs(wm_scores, fruit_ids)
        return base, wm

    def spread(label, texts, tags):
        base_arg, wm_arg = [], []
        base_rows, wm_rows = [], []
        for t in texts:
            b, w = logits_for(t)
            base_rows.append(b); wm_rows.append(w)
            base_arg.append(int(np.argmax(b))); wm_arg.append(int(np.argmax(w)))
        # how much does the argmax-fruit move across the varied digit?
        from collections import Counter
        b_uniq = len(set(base_arg)); w_uniq = len(set(wm_arg))
        # mean total-variation of the fruit distribution vs the first variant
        b_tv = np.mean([0.5*np.abs(np.array(r)-np.array(base_rows[0])).sum() for r in base_rows[1:]])
        w_tv = np.mean([0.5*np.abs(np.array(r)-np.array(wm_rows[0])).sum() for r in wm_rows[1:]])
        print(f"\n{label}")
        print(f"  BASE   argmax across variants: {[FRUITS[i].strip()[:4] for i in base_arg]}"
              f"  ({b_uniq} distinct, mean TV {b_tv:.3f})")
        print(f"  SynthID argmax across variants: {[FRUITS[i].strip()[:4] for i in wm_arg]}"
              f"  ({w_uniq} distinct, mean TV {w_tv:.3f})")
        return b_tv, w_tv

    # 1) vary the LAST digit (inside the hash window)
    texts_in = [base_prefix + str(d) for d in range(1, 10)]
    b_in, w_in = spread("Vary LAST digit (in-window):", texts_in, list(range(1, 10)))

    # 2) vary a LEADING digit (outside the hash window), last digits fixed
    texts_out = ["I ate " + str(d) + lead_prefix_tail + "5" for d in range(1, 10)]
    b_out, w_out = spread("Vary LEADING digit (out-of-window):", texts_out, list(range(1, 10)))

    print("\n--- READ ---")
    print(f"in-window : base TV {b_in:.3f} | SynthID TV {w_in:.3f}")
    print(f"out-window: base TV {b_out:.3f} | SynthID TV {w_out:.3f}")
    if w_in > max(b_in, 0.05) and w_in > w_out:
        print("SynthID tilt is window-local and larger than the base effect -> the "
              "window design detects a real watermark, not base determinism.")
    if b_in > 0.2 and b_in >= b_out * 1.5:
        print("BASE fruit preference ALSO shifts strongly with the last digit and is "
              "window-local -> base-model determinism alone can drive the window "
              "design; the API 'hits' need not be a watermark.")


if __name__ == "__main__":
    main()
