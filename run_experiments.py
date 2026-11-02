"""Runs every experiment and writes CSVs into results/.  Usage: python run_experiments.py [--quick]"""
import sys, time, json
import numpy as np, pandas as pd
from core import *
from sim import *

QUICK = "--quick" in sys.argv
SEEDS = range(3) if QUICK else range(10)
PRM = ACOParams(n_ants=20, n_iter=8, extra_stages=6)
t0 = time.time()

# ---------------------------------------------------------------- 1. main result (paper Fig. 7 analogue)
rows = []
for kind in CLUSTER_KINDS:
    spec = preset(kind)
    for mix in MIXES:
        for seed in SEEDS:
            r = run_fill(spec, mix, seed, "RAMPA", params=PRM)
            rows.append(dict(cluster=kind, mix=mix, seed=seed, method="RAMPA", pattern=-1, count=r.count))
            for m in ("NCAR", "PE"):
                for pat in (0, 1, 2):
                    r = run_fill(spec, mix, seed, m, pattern=pat, params=PRM)
                    rows.append(dict(cluster=kind, mix=mix, seed=seed, method=m, pattern=pat, count=r.count))
    print(f"[{time.time()-t0:6.0f}s] fill done for {kind}", flush=True)
pd.DataFrame(rows).to_csv("results/fill.csv", index=False)

# ---------------------------------------------------------------- 2. stage-count distribution (Fig. 8 analogue) + fragmentation
base = preset("Base")
stage_rows, frag_rows = [], []
for seed in SEEDS:
    for m, pat in (("RAMPA", 0), ("NCAR", 0), ("PE", 0)):
        r = run_fill(base, "Same demand", seed, m, pattern=pat, params=PRM)
        for rec in r.records:
            stage_rows.append(dict(method=m, seed=seed, svc=rec["svc"], Np=rec["Np"]))
            frag_rows.append(dict(method=m, seed=seed, step=rec["step"], util=rec["util"], full_free=rec["full_free"]))
pd.DataFrame(stage_rows).to_csv("results/stages.csv", index=False)
pd.DataFrame(frag_rows).to_csv("results/fragmentation.csv", index=False)
print(f"[{time.time()-t0:6.0f}s] stages + fragmentation done", flush=True)

# ---------------------------------------------------------------- 3. EXTENSION A: profiling-error sensitivity + safety margin
noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
margins = [0.0, 0.05, 0.10, 0.20, 0.30]
nrows, mrows = [], []
for seed in SEEDS:
    runs = {"PE": run_fill(base, "Same demand", seed, "PE", params=PRM),
            "NCAR": run_fill(base, "Same demand", seed, "NCAR", params=PRM)}
    for mg in margins:
        runs[f"RAMPA m={int(mg*100)}%"] = run_fill(base, "Same demand", seed, "RAMPA", margin=mg, params=PRM)
    for name, r in runs.items():
        mrows.append(dict(seed=seed, variant=name, admitted=r.count))
        for nz in noise_levels:
            nrows.append(dict(seed=seed, variant=name, noise=nz, violation=violation_rate(r.allocs, nz, trials=40, seed=seed)))
pd.DataFrame(nrows).to_csv("results/noise.csv", index=False)
pd.DataFrame(mrows).to_csv("results/margin.csv", index=False)
print(f"[{time.time()-t0:6.0f}s] noise/margin done", flush=True)

# ---------------------------------------------------------------- 4. EXTENSION B: dynamic arrivals / departures
loads = [10, 20, 30, 40, 50]
drows = []
for load in loads:
    for seed in (range(2) if QUICK else range(5)):
        for m in METHODS:
            d = run_dynamic(base, "Same demand", seed, m, load, n_arrivals=300 if QUICK else 500, params=PRM)
            drows.append(d | dict(seed=seed))
    print(f"[{time.time()-t0:6.0f}s] dynamic load {load} done", flush=True)
pd.DataFrame(drows).to_csv("results/dynamic.csv", index=False)
print("ALL DONE in %.0fs" % (time.time() - t0))
