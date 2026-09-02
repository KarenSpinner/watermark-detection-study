#!/usr/bin/env python3
"""Semi-synthetic power check for one finished Red-Green grid.

Takes the observed per-cell fruit distributions of a real grid (so base-model
skew and prefix/number sensitivity are exactly what the live model produced),
injects a synthetic keyed tilt of the kind a Red-Green/SynthID watermark would
add (for each number k, one fruit's probability is multiplied by (1+tilt)
across all ten prefixes), resamples the grid at the original per-cell sample
size, and runs the ETH statistic. The smallest tilt at which the median p
drops below 0.05 is the effect size the live test could have missed.

Usage: python scripts/power_check.py data/redgreen/redgreen_fable_kwin.jsonl [n_boot] [n_perm] [eth|within]
"""
import json, sys
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import watermark_redgreen as wr

path = sys.argv[1]
n_boot = int(sys.argv[2]) if len(sys.argv) > 2 else 20
n_perm = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
stat = sys.argv[4] if len(sys.argv) > 4 else "eth"
rows = [json.loads(l) for l in open(path) if l.strip()]
cells = {}
for r in rows:
    if r["fruit"]:
        cells.setdefault((r["prefix"], r["k"]), []).append(wr.WORD_LIST.index(r["fruit"]))
ks = sorted({k for _, k in cells})
prefixes = [p for p in wr.PREFIXES if all((p, k) in cells for k in ks)]
n_cell = int(np.median([len(v) for v in cells.values()]))
base = {c: np.bincount(v, minlength=4) / len(v) for c, v in cells.items()}
marg = np.mean([base[(p, k)] for p in prefixes for k in ks], axis=0)
print(f"{path}: {len(prefixes)} prefixes x {len(ks)} ks, ~{n_cell}/cell, "
      f"marginal {np.round(marg, 3)}")

def median_p(tilt, seed):
    rng = np.random.default_rng(seed)
    inplay = [i for i in range(4) if marg[i] > 0.05]      # fruits the model actually picks
    green = {k: int(rng.choice(inplay)) for k in ks}       # keyed fruit per k
    pv = []
    for _ in range(n_boot):
        pr = np.zeros((len(prefixes) * len(ks), 4)); i = 0
        for p in prefixes:
            for k in ks:
                probs = base[(p, k)].copy()
                if tilt:
                    probs[green[k]] *= (1 + tilt)
                probs = probs / probs.sum()
                draws = rng.choice(4, n_cell, p=probs)
                boot = rng.choice(draws, 90, replace=True)
                cnt = np.bincount(boot, minlength=4).astype(float)
                pr[i] = cnt / cnt.sum(); i += 1
        _, _, p_ = wr.test_kgw_detection(pr, n_perm, rng, stat=stat)
        pv.append(p_)
    return float(np.median(pv))

print(f"stat={stat}  tilt   median_p (median over 3 keys)")
for tilt in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
    p = np.median([median_p(tilt, s) for s in range(3)])
    print(f"{tilt:4.2f}   {p:.3f}", flush=True)
