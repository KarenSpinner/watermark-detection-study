#!/usr/bin/env python3
"""Red-Green black-box watermark test (ETH Zurich) on production APIs.

A faithful port of eth-sri/watermark-detection's closed_models.ipynb protocol
(Gloaguen et al., "Black-Box Detection of Language Model Watermarks", ICLR
2025) to gemini-3.5-flash and claude-opus-4-8. Their statistic functions are
reimplemented below in plain numpy (verbatim logic, no torch dependency);
prompt template, prefixes, k construction, bootstrap and permutation settings
follow the notebook.

Idea: ask the model to complete "{prefix} {k}" with a random word from a fixed
4-word list, where k is a CONTEXT-digit repeated number that fills (upper-
bounds) the watermark's hash window. For a fixed k the window content is
constant, so a Red-Green/SynthID-style scheme tilts the SAME word up or down
regardless of prefix; changing the digit re-seeds the tilt. The statistic
scores cross-prefix agreement about which k is anomalous, against a permutation
null. No watermark -> word choice is independent of k -> p ~ uniform.

Modes:
  --calibrate            Phase 1: check word-choice entropy + parse rate
  (default)              Phase 2: full grid (10 prefixes x 9 ks x --n samples)
  --analyze              Phase 3: bootstrap + permutation test on stored runs

Usage:
  .venv/bin/python watermark_redgreen.py --provider gemini --calibrate
  .venv/bin/python watermark_redgreen.py --provider gemini --n 100
  .venv/bin/python watermark_redgreen.py --provider gemini --analyze
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
OUTS = {"gemini": ROOT / "analysis" / "redgreen_gemini.jsonl",
        "claude": ROOT / "analysis" / "redgreen_claude.jsonl",
        "gpt": ROOT / "analysis" / "redgreen_gpt.jsonl",
        "sonnet": ROOT / "analysis" / "redgreen_sonnet.jsonl",
        "fable": ROOT / "analysis" / "redgreen_fable.jsonl"}
MODELS = {"gemini": "gemini-3.5-flash", "claude": "claude-opus-4-8",
          "gpt": "gpt-5.2", "sonnet": "claude-sonnet-5",
          "fable": "claude-fable-5-1"}   # gpt = negative control (no announced text watermark)

# --- protocol constants (from the ETH notebook) -------------------------------
PREFIXES = ["I ate", "I chose", "I picked", "I selected", "I took",
            "I went for", "I settled on", "I got", "I gathered", "I harvested"]
CONTEXT = 15          # digits in k; must UPPERBOUND the watermark context size
WORD_LIST = ["strawberries", "blueberries", "raspberries", "blackberries"]
# calibration 2026-08-22 on gemini-3.5-flash: berry list gave the flattest
# split (75/25) vs mangoes list (83/17, above the 0.8 warning) and peaches
# list (92/8); same list used for claude for comparability
EXAMPLE = "apples"
FORMAT = ""
MAX_TOKENS = 96       # was 40 for the 2026-08 grids; raised 2026-09-02 so a
                      # 40-digit number (one token per digit on Gemini) is not
                      # cut off before the fruit. Only caps length; the fruit
                      # token is sampled the same way.
TEMPERATURE = 1.0     # gemini only; claude-opus-4-8 rejects temperature
RETRIES = 4
# Fable 5.1 (added 2026-09-02): thinking is always on and cannot be disabled
# (the API rejects thinking={"type": "disabled"}), so thinking depth is held
# at its minimum with output_config.effort="low" and max_tokens is raised so
# the thinking never crowds out the one-line answer. No server-side fallback
# is requested: a fallback would silently substitute another model into the
# sample. Every record stores the served model id, stop_reason and token
# usage so the run can be audited after the fact.
FABLE_MAX_TOKENS = 4096
FABLE_EFFORT = "low"
FABLE_PRICE = (10.0, 50.0)   # USD per 1M input / output tokens, 2026-09
N_BOOTSTRAP = 100
BOOT_N = 90           # draws per cell in each bootstrap (ETH notebook)
N_PERMUTATIONS = 10_000

KS = [int(str(d) * CONTEXT) for d in range(1, 10)]

# Fixed random 15-digit ks (generated once, hardcoded for reproducibility).
# Repeated-digit ks make the hash window recur within the generation, which
# SynthID's repeated-context masking can exploit to skip watermarking at the
# measured position; non-repeating ks close that hole.
KS_RANDOM = [
    473829105624381, 928176354091265, 615304928471053,
    384750261938472, 750938264150397, 269481735062948,
    841627390584216, 507263148926731, 192837465019283,
]

# Content-controlled + masking-immune ks: nine numbers sharing a 14-digit
# prefix and differing ONLY in the last digit. The number's "content" is held
# ~constant across k (so cross-prefix fruit consistency cannot come from the
# number's meaning -- the confound is neutralized), but the hash window's final
# token still varies across k (so a keyed watermark can still tilt), and no
# window is a repeated-digit string (so repeated-context masking cannot apply).
# This separates the watermark hypothesis from the content confound.
KS_WINDOW = [int("47382910562438" + str(d)) for d in range(1, 10)]

# Long window-k (added 2026-09-02): a 15-digit number spans only ~4-5 tokens on
# the Claude tokenizer (count_tokens, claude-fable-5-1), so a hash window
# longer than that would reach back into the sentence prefix and the tilt
# would no longer agree across prefixes. --context 40 --kmode window uses a
# 40-digit number (~13 tokens) that varies only in its last digit.
KS_WINDOW_PREFIX = "473829105624381928176354091265615304928"   # 39 digits

# No-echo prompt (added 2026-09-02): in the ETH template the number appears
# verbatim in the prompt, so the response n-gram "{prefix} {k}" is repeated
# context if a masking rule reads the prompt as well as the response.
# --noecho gives the number as spaced digits in the prompt and asks for it
# written compactly, so the n-gram first appears in the response.
NOECHO = False


def build_prompt(prefix: str, k: int) -> str:
    # verbatim template from the notebook (typos included, for fidelity)
    if NOECHO:
        spaced = " ".join(str(k))
        return (f'Complete the sentence "{prefix} {spaced}" using only and exacty '
                f'a random word from the list: {WORD_LIST}.  Answer in this speific '
                f'format: {FORMAT} {prefix} {spaced} {EXAMPLE}. (here I chose an '
                f'other fruit for the sake of the example, you have to choose among '
                f'{WORD_LIST}). Write the number as one number with no spaces '
                f'between the digits.')
    return (f'Complete the sentence "{prefix} {k}" using only and exacty a '
            f'random word from the list: {WORD_LIST}.  Answer in this speific '
            f'format: {FORMAT} {prefix} {k} {EXAMPLE}. (here I chose an other '
            f'fruit for the sake of the example, you have to choose among '
            f'{WORD_LIST})')


def identify_fruit(response: str):
    for w in WORD_LIST:
        if w in (response or ""):
            return w
    return None


def identify_fruit_after_k(response: str, k: int):
    """Earliest fruit (by position) after the number k in the response, falling
    back to the earliest fruit anywhere. Used for thinking models, whose visible
    text can run longer than the bare answer line and may mention the list."""
    text = response or ""
    start = text.find(str(k))
    tail = text[start + len(str(k)):] if start >= 0 else text
    hits = [(tail.find(w), w) for w in WORD_LIST if w in tail]
    if not hits and start >= 0:
        hits = [(text.find(w), w) for w in WORD_LIST if w in text]
    return min(hits)[1] if hits else None


# --- providers -----------------------------------------------------------------
def make_client(provider):
    if provider in ("claude", "sonnet", "fable"):
        import anthropic
        return anthropic.Anthropic()
    if provider == "gpt":
        from openai import OpenAI
        return OpenAI()
    from google import genai
    import os
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def generate(client, provider, prompt):
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            if provider == "fable":
                r = client.messages.create(
                    model=MODELS["fable"], max_tokens=FABLE_MAX_TOKENS,
                    output_config={"effort": FABLE_EFFORT},
                    messages=[{"role": "user", "content": prompt}])
                text = "".join(b.text for b in r.content if b.type == "text")
                meta = {"model": r.model, "stop_reason": r.stop_reason,
                        "input_tokens": r.usage.input_tokens,
                        "output_tokens": r.usage.output_tokens,
                        "thinking_blocks": sum(1 for b in r.content
                                               if b.type == "thinking")}
                return text, meta
            if provider in ("claude", "sonnet"):
                # thinking disabled -> raw output sampling. On opus-4-8 omitting
                # is already off; Sonnet 5 defaults to adaptive, so disable it
                # explicitly to match the opus-4-8 no-thinking setup.
                r = client.messages.create(
                    model=MODELS[provider], max_tokens=MAX_TOKENS,
                    thinking={"type": "disabled"},
                    messages=[{"role": "user", "content": prompt}])
                return "".join(b.text for b in r.content if b.type == "text"), {}
            if provider == "gpt":
                r = client.responses.create(
                    model=MODELS["gpt"], input=prompt,
                    max_output_tokens=MAX_TOKENS, temperature=TEMPERATURE)
                return r.output_text or "", {}
            from google.genai import types
            r = client.models.generate_content(
                model=MODELS["gemini"], contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=MAX_TOKENS, temperature=TEMPERATURE,
                    thinking_config=types.ThinkingConfig(thinking_budget=0)))
            return r.text or "", {}
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt < RETRIES:
                time.sleep(2 ** attempt)
    raise last


# --- ETH statistic, reimplemented (redgreen_test.py, numpy only) ----------------
def logit(p):
    return np.log((p + 1e-10) / (1 - p + 1e-10))


def statistic(data: np.ndarray, chosen: int):
    data = data.reshape(-1, 9, 4)           # axis 1 is the k (last-digit) axis
    result_array = data[:, :, chosen]
    median = np.median(result_array, axis=1)
    std = np.median(np.std(result_array, axis=0))
    r = 1.96
    red = result_array.T - median < -r * std
    green = result_array.T - median > r * std
    red_score = np.sum(red, axis=1)
    green_score = np.sum(green, axis=1)
    max_common = np.max([np.max(red_score), np.max(green_score)])
    min_common = np.max([np.min(red_score), np.min(green_score)])
    return max_common - min_common


def test_kgw_detection(data: np.ndarray, num_permutations: int, rng, stat="eth"):
    """stat="eth": the published test, verbatim (permute cells across the whole
    grid). stat="within": prefix-stratified variant added 2026-09-02. The ETH
    null shuffles cells across sentence prefixes, so when the prefix itself
    drives the fruit choice (all Claude models: prefix explains 56-81% of the
    variance in the fruit rate) the shuffled grids contain large spurious
    deviations and the null swamps a real keyed tilt. The variant centers each
    prefix on its own median across k (the ETH code contains this line with a
    "* 0" that disables it) and permutes k-cells WITHIN each prefix, which is
    the exchangeability that actually holds under H0 (no k effect), so the
    prefix effect drops out of both the statistic and the null. Cell rates are
    also clipped away from 0/1 before the logit (see below)."""
    data = data.reshape(-1, 9, 4)
    check = np.mean(data, axis=(0, 1)) > 0.8
    if check.any():
        print("Warning: a word dominates (>0.8) — results might be incorrect. "
              f"Probabilities: {np.round(np.mean(data, axis=(0, 1)), 3)}")
    if stat == "within":
        # add-half smoothing: a bootstrapped cell rate of exactly 0 or 1 would
        # become a logit of +-23 and dominate the threshold; clip to the
        # resolution of the 90-draw bootstrap instead.
        data = np.clip(data, 0.5 / BOOT_N, 1 - 0.5 / BOOT_N)
    data = logit(data)
    weight = np.sum(data, axis=(0, 1))
    chosen = int(np.argmax(weight))
    if stat == "within":
        data = data - np.median(data, axis=1, keepdims=True)
    observed = statistic(data, chosen)
    stats = np.zeros(num_permutations)
    flat = data.reshape(-1, 4)
    n_prefix = data.shape[0]
    for i in range(num_permutations):
        if stat == "within":
            perm = np.stack([rng.permutation(data[j]) for j in range(n_prefix)])
        else:
            perm = rng.permutation(flat).reshape(-1, 9, 4)
        stats[i] = statistic(perm, chosen)
    return observed, stats, float(np.mean(stats >= observed))


# --- phases ---------------------------------------------------------------------
_lock = threading.Lock()


def load_done(path):
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["prefix"], r["k"], r["idx"])] = r
    return done


def run_grid(provider, n, workers):
    path = OUTS[provider]
    path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(path)
    jobs = [(p, k, i) for p in PREFIXES for k in KS for i in range(n)
            if (p, k, i) not in done]
    print(f"{provider} ({MODELS[provider]}): grid {len(PREFIXES)}x{len(KS)}x{n} "
          f"= {len(PREFIXES)*len(KS)*n} | done {len(done)} | to run {len(jobs)}")
    if not jobs:
        return
    client = make_client(provider)
    ok = fail = 0

    def one(job):
        p, k, i = job
        resp, meta = generate(client, provider, build_prompt(p, k))
        fruit = (identify_fruit_after_k(resp, k) if provider == "fable"
                 else identify_fruit(resp))
        rec = {"prefix": p, "k": k, "idx": i, "response": resp,
               "fruit": fruit, **meta}
        with _lock:
            with path.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(one, j) for j in jobs]
        for m, f in enumerate(as_completed(futs), 1):
            try:
                f.result(); ok += 1
            except Exception as exc:  # noqa: BLE001
                fail += 1
                print(f"  FAIL: {str(exc)[:100]}")
            if m % 500 == 0:
                print(f"  {m}/{len(jobs)} (ok={ok} fail={fail})", flush=True)
    print(f"Done. ok={ok} fail={fail} -> {path.name}")


def calibrate(provider, workers):
    client = make_client(provider)
    prompt = build_prompt(PREFIXES[0], KS[0])
    print("Prompt:", prompt, "\n")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(generate, client, provider, prompt) for _ in range(24)]
        results = [f.result() for f in as_completed(futs)]
    responses = [r for r, _ in results]
    metas = [m for _, m in results]
    from collections import Counter
    if provider == "fable":
        fruits = Counter(identify_fruit_after_k(r, KS[0]) for r in responses)
    else:
        fruits = Counter(identify_fruit(r) for r in responses)
    print("Distribution over 24 samples:", dict(fruits))
    parse = sum(v for k_, v in fruits.items() if k_) / len(responses)
    print(f"Parse rate: {parse:.0%}")
    for r, m in results[:8]:
        tag = f" [out_tok={m['output_tokens']} stop={m['stop_reason']}]" if m else ""
        print("  sample:", (r or "")[:120].replace("\n", " ") + tag)
    if metas and metas[0]:
        outs = np.array([m["output_tokens"] for m in metas])
        ins = np.array([m["input_tokens"] for m in metas])
        print(f"served model: {dict(Counter(m['model'] for m in metas))} | "
              f"stop_reason: {dict(Counter(m['stop_reason'] for m in metas))} | "
              f"thinking blocks/resp: {dict(Counter(m['thinking_blocks'] for m in metas))}")
        print(f"output tokens: mean {outs.mean():.0f} median {np.median(outs):.0f} "
              f"min {outs.min()} max {outs.max()} | input tokens mean {ins.mean():.0f}")
        per_call = (ins.mean() * FABLE_PRICE[0] + outs.mean() * FABLE_PRICE[1]) / 1e6
        print(f"est. cost/call ${per_call:.4f}; n=100 grid (9000 calls) "
              f"${per_call*9000:.2f}; n=50 grid (4500 calls) ${per_call*4500:.2f}; "
              f"all three designs (18000 calls) ${per_call*18000:.2f}")


def analyze(provider, stat="eth", n_boot=N_BOOTSTRAP, n_perm=N_PERMUTATIONS):
    path = OUTS[provider]
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    valid = [r for r in rows if r["fruit"]]
    print(f"{provider}: {len(rows)} responses, {len(valid)} parsed "
          f"({len(valid)/len(rows):.0%})")
    cells = {}
    for r in valid:
        cells.setdefault((r["prefix"], r["k"]), []).append(r["fruit"])
    if rows and "output_tokens" in rows[0]:
        from collections import Counter
        tot_in = sum(r["input_tokens"] for r in rows)
        tot_out = sum(r["output_tokens"] for r in rows)
        print(f"served model: {dict(Counter(r['model'] for r in rows))} | "
              f"stop_reason: {dict(Counter(r['stop_reason'] for r in rows))}")
        print(f"tokens: in {tot_in} out {tot_out} (mean out/resp "
              f"{tot_out/len(rows):.0f}); est. cost "
              f"${(tot_in*FABLE_PRICE[0] + tot_out*FABLE_PRICE[1])/1e6:.2f}")
    counts_n = [len(v) for v in cells.values()]
    print(f"cells: {len(cells)} (expect {len(PREFIXES)*len(KS)}); "
          f"samples/cell min {min(counts_n)} median {int(np.median(counts_n))}")

    # order cells prefix-major, k-minor (statistic assumes 9-k blocks per prefix)
    ordered = [(p, k) for p in PREFIXES for k in KS]
    missing = [c for c in ordered if c not in cells]
    if missing:
        print(f"WARNING: {len(missing)} empty cells (dropped prefix rows): {missing[:4]}")
        keep_prefixes = [p for p in PREFIXES
                         if all((p, k) in cells for k in KS)]
        ordered = [(p, k) for p in keep_prefixes for k in KS]

    rng = np.random.default_rng(20260822)
    pvals = []
    for b in range(n_boot):
        probs = np.zeros((len(ordered), 4))
        for i, cell in enumerate(ordered):
            samples = cells[cell]
            boot = rng.choice(samples, BOOT_N, replace=True)
            cnt = {w: 0 for w in WORD_LIST}
            for s in boot:
                cnt[s] += 1
            probs[i] = np.array([cnt[w] for w in WORD_LIST], dtype=float)
            probs[i] /= probs[i].sum()
        _, _, p = test_kgw_detection(probs, n_perm, rng, stat=stat)
        pvals.append(p)
        if (b + 1) % 20 == 0:
            print(f"  bootstrap {b+1}/{n_boot} running median p = "
                  f"{np.median(pvals):.4f}", flush=True)
    print(f"\nRESULT {provider} ({MODELS[provider]}) [stat={stat}, "
          f"{n_boot} boot x {n_perm} perm]: median p = "
          f"{np.median(pvals):.4f}  (5th pct {np.percentile(pvals,5):.4f}, "
          f"95th pct {np.percentile(pvals,95):.4f})")
    print("Interpretation: p near 0 across bootstraps = Red-Green/SynthID-style "
          "watermark detected; p spread over (0,1) = no detection.")
    out = {"provider": provider, "model": MODELS[provider], "stat": stat,
           "n_boot": n_boot, "n_perm": n_perm,
           "median_p": float(np.median(pvals)),
           "p5": float(np.percentile(pvals, 5)),
           "p95": float(np.percentile(pvals, 95)),
           "n_valid": len(valid), "n_total": len(rows)}
    result_name = OUTS[provider].stem.replace("redgreen_", "redgreen_result_")
    if stat != "eth":
        result_name += f"_{stat}"
    (ROOT / "analysis" / f"{result_name}.json").write_text(
        json.dumps(out, indent=2))


def main():
    global KS, FABLE_EFFORT, NOECHO
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=list(MODELS), required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--stat", choices=["eth", "within"], default="eth",
                    help="eth = published statistic/null; within = prefix-"
                         "centered, prefix-stratified null (see test_kgw_detection)")
    ap.add_argument("--nboot", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--nperm", type=int, default=N_PERMUTATIONS)
    ap.add_argument("--context", type=int, default=CONTEXT,
                    help="digits in k (robustness sweep; non-default gets its own file)")
    ap.add_argument("--noecho", action="store_true",
                    help="give the number as spaced digits in the prompt so the "
                         "response n-gram never appears in the prompt (own file)")
    ap.add_argument("--effort", default=FABLE_EFFORT,
                    choices=["low", "medium", "high", "xhigh", "max"],
                    help="fable only: output_config.effort (non-default gets its own file)")
    ap.add_argument("--kmode", choices=["repeat", "random", "window"], default="repeat",
                    help="random = non-repeating 15-digit ks (masking-immune, but "
                         "content-confounded); window = content-controlled ks that "
                         "vary only the last digit (masking-immune AND confound-"
                         "controlled); each gets its own file")
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")
    if args.kmode == "random":
        KS = KS_RANDOM
        for prov in OUTS:
            OUTS[prov] = OUTS[prov].with_name(OUTS[prov].stem + "_krand.jsonl")
    elif args.kmode == "window" and args.context == 40:
        KS = [int(KS_WINDOW_PREFIX + str(d)) for d in range(1, 10)]
        for prov in OUTS:
            OUTS[prov] = OUTS[prov].with_name(OUTS[prov].stem + "_kwin40.jsonl")
    elif args.kmode == "window":
        KS = KS_WINDOW
        for prov in OUTS:
            OUTS[prov] = OUTS[prov].with_name(OUTS[prov].stem + "_kwin.jsonl")
    elif args.context != CONTEXT:
        KS = [int(str(d) * args.context) for d in range(1, 10)]
        for prov in OUTS:
            OUTS[prov] = OUTS[prov].with_name(
                OUTS[prov].stem + f"_ctx{args.context}.jsonl")
    if args.noecho:
        NOECHO = True
        for prov in OUTS:
            OUTS[prov] = OUTS[prov].with_name(OUTS[prov].stem + "_noecho.jsonl")
    if args.provider == "fable" and args.effort != FABLE_EFFORT:
        FABLE_EFFORT = args.effort
        OUTS["fable"] = OUTS["fable"].with_name(
            OUTS["fable"].stem + f"_eff{args.effort}.jsonl")
    if args.calibrate:
        calibrate(args.provider, args.workers)
    elif args.analyze:
        analyze(args.provider, args.stat, args.nboot, args.nperm)
    else:
        run_grid(args.provider, args.n, args.workers)


if __name__ == "__main__":
    main()
