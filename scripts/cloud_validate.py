#!/usr/bin/env python3
"""Cloud GPU validation of the three watermark-detection designs.

Self-contained. Runs on a RunPod GPU. Takes ONE capable open instruct model,
applies a real SynthID-Text watermark at generation, and runs each detection
design (repeated-digit ETH, random-k, window-k) with the watermark ON and OFF.

The ON/OFF contrast on the SAME model cancels base-model determinism (present
in both) and isolates the watermark:

  design fires ON, null OFF  -> valid watermark detector
  design fires ON and OFF    -> driven by base-model determinism, not a mark
  design null ON             -> no power at this model/scale

This is the control the API experiments and the small-model logit test could
not give cleanly.

Usage (on the pod):
  python cloud_validate.py --model Qwen/Qwen2.5-7B-Instruct --n 60
"""
from __future__ import annotations
import argparse, json, math, time, warnings
warnings.filterwarnings("ignore")
from collections import Counter
import numpy as np
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          SynthIDTextWatermarkingConfig)

PREFIXES = ["I ate", "I chose", "I picked", "I selected", "I took",
            "I went for", "I settled on", "I got", "I gathered", "I harvested"]
WORD_LIST = ["strawberries", "blueberries", "raspberries", "blackberries"]
EXAMPLE = "apples"
KS = {
    "repeat": [int(str(d) * 15) for d in range(1, 10)],
    "random": [473829105624381, 928176354091265, 615304928471053,
               384750261938472, 750938264150397, 269481735062948,
               841627390584216, 507263148926731, 192837465019283],
    "window": [int("47382910562438" + str(d)) for d in range(1, 10)],
}
KEYS = [654, 400, 836, 123, 340, 443, 597, 160, 57, 29,
        590, 639, 13, 715, 468, 990, 966, 226, 324, 585]
NGRAM_LEN = 5


def build_prompt(prefix, k):
    return (f'Complete the sentence "{prefix} {k}" using only and exacty a '
            f'random word from the list: {WORD_LIST}.  Answer in this speific '
            f'format:  {prefix} {k} {EXAMPLE}. (here I chose an other fruit for '
            f'the sake of the example, you have to choose among {WORD_LIST})')


def identify_fruit(resp):
    for w in WORD_LIST:
        if w in (resp or ""):
            return w
    return None


# --- ETH statistic (numpy port) ------------------------------------------------
def logit(p):
    return np.log((p + 1e-10) / (1 - p + 1e-10))


def statistic(data, chosen):
    data = data.reshape(-1, 9, 4)
    ra = data[:, :, chosen]
    median = np.median(ra, axis=1)
    std = np.median(np.std(ra, axis=0))
    r = 1.96
    red = ra.T - median < -r * std
    green = ra.T - median > r * std
    rs, gs = np.sum(red, axis=1), np.sum(green, axis=1)
    return np.max([np.max(rs), np.max(gs)]) - np.max([np.min(rs), np.min(gs)])


def test_kgw(data, nperm, rng):
    data = data.reshape(-1, 9, 4)
    data = logit(data)
    chosen = int(np.argmax(np.sum(data, axis=(0, 1))))
    obs = statistic(data, chosen)
    flat = data.reshape(-1, 4)
    stats = np.array([statistic(rng.permutation(flat).reshape(-1, 9, 4), chosen)
                      for _ in range(nperm)])
    return float(np.mean(stats >= obs))


# --- generation ---------------------------------------------------------------
def make_model(name):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    return tok, model


@torch.no_grad()
def gen(tok, model, prompt, n, wm_cfg, batch=64):
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    out = []
    while len(out) < n:
        bn = min(batch, n - len(out))
        inp = tok([text] * bn, return_tensors="pt", padding=True).to("cuda")
        kw = dict(do_sample=True, temperature=1.0, top_k=0, max_new_tokens=40,
                  pad_token_id=tok.pad_token_id)
        if wm_cfg is not None:
            kw["watermarking_config"] = wm_cfg
        g = model.generate(**inp, **kw)[:, inp["input_ids"].shape[1]:]
        out += [tok.decode(x, skip_special_tokens=True) for x in g]
    return out


def run_design(tok, model, design, n, watermarked, rng):
    wm_cfg = (SynthIDTextWatermarkingConfig(keys=KEYS, ngram_len=NGRAM_LEN)
              if watermarked else None)
    cells = {}
    for pi, pref in enumerate(PREFIXES):
        for k in KS[design]:
            resp = gen(tok, model, build_prompt(pref, k), n, wm_cfg)
            fruits = [identify_fruit(r) for r in resp]
            cells[(pref, k)] = [f for f in fruits if f]
    # keep prefixes with all 9 ks well-populated
    keep = [p for p in PREFIXES
            if all((p, k) in cells and len(cells[(p, k)]) >= 10 for k in KS[design])]
    ordered = [(p, k) for p in keep for k in KS[design]]
    parse = np.mean([len(cells[c]) for c in cells]) / n
    if len(ordered) < 2 * 9:
        return None, parse, len(keep)
    pvals = []
    for _ in range(100):
        probs = np.zeros((len(ordered), 4))
        for i, c in enumerate(ordered):
            boot = rng.choice(cells[c], 90, replace=True)
            cnt = Counter(boot)
            v = np.array([cnt.get(w, 0) for w in WORD_LIST], float)
            probs[i] = v / v.sum()
        pvals.append(test_kgw(probs, 10000, rng))
    return float(np.median(pvals)), parse, len(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()
    print(f"cuda={torch.cuda.is_available()} model={args.model} n={args.n}", flush=True)
    rng = np.random.default_rng(20260822)
    tok, model = make_model(args.model)
    print("model loaded", flush=True)

    results = {}
    for design in ("repeat", "random", "window"):
        for wm in (True, False):
            t0 = time.time()
            p, parse, keep = run_design(tok, model, design, args.n, wm, rng)
            tag = f"{design}/{'ON' if wm else 'OFF'}"
            results[tag] = p
            print(f"[{tag}] median p = {p} | parse {parse:.0%} | "
                  f"{keep} prefixes | {time.time()-t0:.0f}s", flush=True)

    print("\n==== VALIDATION TABLE (median p; <0.05 = fires) ====", flush=True)
    print(f"{'design':10} {'ON':>10} {'OFF':>10}  verdict", flush=True)
    for d in ("repeat", "random", "window"):
        on, off = results.get(f"{d}/ON"), results.get(f"{d}/OFF")
        def f(x):
            return "n/a" if x is None else f"{x:.4f}"
        verdict = "?"
        if on is not None and off is not None:
            if on < 0.05 and off > 0.05:
                verdict = "VALID detector (fires ON, null OFF)"
            elif on < 0.05 and off < 0.05:
                verdict = "base-determinism (fires BOTH)"
            elif on > 0.05:
                verdict = "no power ON"
        print(f"{d:10} {f(on):>10} {f(off):>10}  {verdict}", flush=True)
    print("\nRESULTS_JSON " + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
