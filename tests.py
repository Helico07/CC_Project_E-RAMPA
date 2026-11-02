"""Structural checks: aggregated Eq.(5) == brute-force per-GPU Eq.(5); DP respects constraints."""
import itertools, random, json, subprocess, sys
from collections import deque
import numpy as np
from core import *
from energy import alloc_series

def brute_force_link_costs(cl: Cluster):
    # explicit GPU-level graph: GPU nodes ('g',r,k) attached to switch ('s',r)
    adj = {}
    def add(u, v): adj.setdefault(u, set()).add(v); adj.setdefault(v, set()).add(u)
    gpus = []
    for r in range(cl.R):
        for k in range(cl.free[r]):
            gpus.append(("g", r, k)); add(("g", r, k), ("s", r))
    for (u, v) in cl.links: add(("s", u), ("s", v))
    C = {g: (cl.rack_flops[g[1]] / 1e12) * (cl.rack_mem[g[1]] / 1e9) * cl.free[g[1]] for g in gpus}
    def bfs(s):
        d = {s: 0}; sg = {s: 1}; q = deque([s])
        while q:
            u = q.popleft()
            for v in sorted(adj[u]):
                if v not in d: d[v] = d[u] + 1; sg[v] = 0; q.append(v)
                if d[v] == d[u] + 1: sg[v] += sg[u]
        return d, sg
    info = {g: bfs(g) for g in gpus}
    links = {frozenset(e) for u in adj for e in ((u, v) for v in adj[u])}
    out = {l: 0.0 for l in links}
    for s, t in itertools.combinations(gpus, 2):
        ds, ss = info[s]; dt, st = info[t]
        H, N = ds[t], ss[t]
        for l in links:
            u, v = tuple(l); thru = 0
            for a, b in ((u, v), (v, u)):
                if ds[a] + 1 + dt[b] == H: thru += ss[a] * st[b]
            if thru: out[l] += (thru / N) * (C[s] + C[t]) / H
    return out

def test_eq5():
    spec = ClusterSpec("tiny", 3, 3, 3, tuple((r % 4 + r // 3) % 4 for r in range(9)))
    cl = Cluster(spec)
    rng = random.Random(1)
    for r in rng.sample(range(9), 4):                      # make the state non-uniform
        cl.free[r] -= rng.randint(1, 2); [cl.free_ids[r].pop() for _ in range(spec.gpus_per_rack - cl.free[r])]
    Cg, acc, linkc = cl.costs()
    bf = brute_force_link_costs(cl)
    for li, (u, v) in enumerate(cl.links):
        assert np.isclose(linkc[li], bf[frozenset((("s", u), ("s", v)))], rtol=1e-9), (u, v)
    for r in range(9):
        if cl.free[r] > 0:
            assert np.isclose(acc[r], bf[frozenset((("g", r, 0), ("s", r)))], rtol=1e-9), r
    print("Eq.(5) aggregated == brute force  OK")

def test_plans():
    cl = Cluster(preset("Base"))
    for i, s in enumerate(SERVICES):
        n0 = n_min(s.model, s.sla, cl.bw)
        assert n0 is not None, s.name
        for Np in range(n0, min(s.model.n, n0 + 2) + 1):
            for t in range(4):
                p = plan_partition(s.model, s.sla, cl.bw, (t,) * Np, (0,) * (Np - 1))
                if p:
                    assert 1 / p.Tu >= s.sla.min_tps - 1e-9 and p.Tr <= s.sla.max_latency + 1e-9
                    assert len(p.cuts) == Np - 1 and list(p.cuts) == sorted(set(p.cuts))
        print(f"{s.name:28s} n_min={n0}")
    print("plan sanity OK")

def test_bookkeeping():
    cl = Cluster(preset("Base")); rng = random.Random(0); prm = ACOParams()
    allocs = []
    for i in arrival_stream("Same demand", 0, 12):
        a = allocate(cl, i, "RAMPA", rng, prm)
        if a: allocs.append(a)
    for a in allocs: cl.release(a)
    assert cl.free == [cl.gpr] * cl.R and cl.cap == [cl.spec.circuits] * cl.L
    print("alloc/release bookkeeping OK")

# ==============================================================================================
# NEW TESTS (T1 – T4): e-RAMPA additions
# ==============================================================================================

_BASELINE_DIR = r"C:\Users\hardi\AppData\Local\Temp\rampa_baseline"

def _run_fill_json(workdir, spec_kind, mix, seed, method, pattern, lam):
    """Run run_fill in a subprocess from workdir, return JSON with count+records."""
    script = f"""
import sys, json
sys.path.insert(0, {repr(workdir)})
from core import preset
from sim import run_fill
spec = preset({repr(spec_kind)})
kw = dict(lam={lam}) if {int(lam != 0 or workdir != repr(_BASELINE_DIR))} else dict()
try:
    r = run_fill(spec, {repr(mix)}, {seed}, {repr(method)}, pattern={pattern}, **kw)
except TypeError:
    r = run_fill(spec, {repr(mix)}, {seed}, {repr(method)}, pattern={pattern})
out = dict(count=r.count,
           records=[dict(racks=rec['racks'], cost=rec['cost']) for rec in r.records])
print(json.dumps(out))
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return json.loads(result.stdout)


def test_lam0_regression():
    """T1: lam=0 new code == baseline code for all 16 (cluster,mix) x seeds 0-2, RAMPA and NCAR p=0."""
    import os
    new_dir = os.path.dirname(os.path.abspath(sys.modules[__name__].__file__)) if hasattr(sys.modules.get(__name__), '__file__') else os.getcwd()
    failures = []
    for kind in CLUSTER_KINDS:
        for mix in list(MIXES.keys()):
            for seed in range(3):
                for method, pat in [("RAMPA", 0), ("NCAR", 0)]:
                    base = _run_fill_json(_BASELINE_DIR, kind, mix, seed, method, pat, lam=0.0)
                    new  = _run_fill_json(new_dir,       kind, mix, seed, method, pat, lam=0.0)
                    if base["count"] != new["count"] or base["records"] != new["records"]:
                        failures.append((kind, mix, seed, method))
    if failures:
        raise AssertionError(f"T1 FAILED — mismatches: {failures}")
    print("T1 lam=0 regression  OK  (16 envs x 3 seeds x 2 methods)")


def _brute_force_link_costs_param(cl, Cg_rack):
    """Parameterised brute-force link cost using a given per-rack Cg array (for T2)."""
    adj = {}
    def add(u, v): adj.setdefault(u, set()).add(v); adj.setdefault(v, set()).add(u)
    gpus = []
    for r in range(cl.R):
        for k in range(cl.free[r]):
            gpus.append(("g", r, k)); add(("g", r, k), ("s", r))
    for (u, v) in cl.links: add(("s", u), ("s", v))
    C = {g: Cg_rack[g[1]] for g in gpus}   # use passed-in per-rack Cg (already per-rack total)

    def bfs(s):
        d = {s: 0}; sg = {s: 1}; q = deque([s])
        while q:
            u = q.popleft()
            for v in sorted(adj[u]):
                if v not in d: d[v] = d[u] + 1; sg[v] = 0; q.append(v)
                if d[v] == d[u] + 1: sg[v] += sg[u]
        return d, sg

    info = {g: bfs(g) for g in gpus}
    links = {frozenset(e) for u in adj for e in ((u, v) for v in adj[u])}
    out = {l: 0.0 for l in links}
    for s, t in itertools.combinations(gpus, 2):
        ds, ss = info[s]; dt, st = info[t]
        H, N = ds[t], ss[t]
        for l in links:
            u, v = tuple(l); thru = 0
            for a, b in ((u, v), (v, u)):
                if ds[a] + 1 + dt[b] == H: thru += ss[a] * st[b]
            if thru: out[l] += (thru / N) * (C[s] + C[t]) / H
    return out


def test_penalty_factor():
    """T2: penalty factor correctness and penalize_links toggle."""
    spec = ClusterSpec("tiny", 3, 3, 3, tuple((r % 4 + r // 3) % 4 for r in range(9)))
    rng = random.Random(1)

    # Base state: non-uniform free counts (same as test_eq5 setup)
    cl0 = Cluster(spec)
    for r in rng.sample(range(9), 4):
        cl0.free[r] -= rng.randint(1, 2)
        [cl0.free_ids[r].pop() for _ in range(spec.gpus_per_rack - cl0.free[r])]

    # Clone state into lam=1 cluster
    cl1 = Cluster(spec)
    cl1.free = list(cl0.free)
    cl1.free_ids = [list(ids) for ids in cl0.free_ids]
    cl1.lam = 1.0
    cl1.penalize_links = False

    Cg0, acc0, lc0 = cl0.costs()   # lam=0 (default)
    Cg1, acc1, lc1 = cl1.costs()   # lam=1, penalize_links=False

    # Check gpu_cost_pen == gpu_cost0 * (1 + TDP/TDP_MAX)
    expected_pen = np.array([1.0 + TDP_W[GPU_TYPES[t].short] / TDP_MAX for t in cl1.types])
    assert np.allclose(Cg1, Cg0 * expected_pen), "gpu_cost penalty wrong"

    # With penalize_links=False, access and link costs must equal lam=0 values
    assert np.array_equal(acc0, acc1), "access_cost should be unchanged (penalize_links=False)"
    assert np.array_equal(lc0, lc1),  "link_cost should be unchanged (penalize_links=False)"

    # penalize_links=True: brute-force should agree with the penalised Cg
    cl1p = Cluster(spec)
    cl1p.free = list(cl0.free)
    cl1p.free_ids = [list(ids) for ids in cl0.free_ids]
    cl1p.lam = 1.0
    cl1p.penalize_links = True
    Cg1p, acc1p, lc1p = cl1p.costs()

    # Build per-rack Cg for brute-force (penalised, one entry per rack)
    Cg_rack_pen = Cg1p   # already accounts for free_counts in aggregation
    bf = _brute_force_link_costs_param(cl1p, Cg_rack_pen)
    for li, (u, v) in enumerate(cl1p.links):
        assert np.isclose(lc1p[li], bf[frozenset((("s", u), ("s", v)))], rtol=1e-9), (u, v)
    for r in range(9):
        if cl1p.free[r] > 0:
            assert np.isclose(acc1p[r], bf[frozenset((("g", r, 0), ("s", r)))], rtol=1e-9), r

    print("T2 penalty factor / penalize_links  OK")


def test_energy_accounting():
    """T3: alloc_series gpus_in_use and alloc_tdp match ground truth from snapshots."""
    from sim import run_fill
    spec = preset("Base")
    r = run_fill(spec, "Same demand", 0, "RAMPA", snapshots=True)
    series = alloc_series(r.records, spec)

    for i, (rec, row) in enumerate(zip(r.records, series)):
        # ground truth from snapshot: free[r] GPUs still free => gpr - free[r] in use
        gt_gpus = sum(spec.gpus_per_rack - f for f in rec["free"])
        gt_tdp  = sum(
            (spec.gpus_per_rack - rec["free"][rack]) * TDP_W[GPU_TYPES[spec.rack_types[rack]].short]
            for rack in range(len(spec.rack_types))
        )
        assert row["gpus_in_use"] == gt_gpus, \
            f"step {i+1}: gpus_in_use {row['gpus_in_use']} != {gt_gpus}"
        assert abs(row["alloc_tdp"] - gt_tdp) < 1e-9, \
            f"step {i+1}: alloc_tdp {row['alloc_tdp']} != {gt_tdp}"

    print("T3 energy accounting  OK")


def test_pe_ignores_lam():
    """T4: run_fill with method=PE and lam=1 produces identical results to lam=0."""
    from sim import run_fill
    spec = preset("Base")
    r0 = run_fill(spec, "Same demand", 0, "PE", lam=0.0)
    r1 = run_fill(spec, "Same demand", 0, "PE", lam=1.0)
    assert r0.count == r1.count, f"PE count differs: {r0.count} vs {r1.count}"
    assert r0.records == r1.records, "PE records differ with lam=1"
    print("T4 PE ignores lam  OK")



    test_eq5(); test_plans(); test_bookkeeping()
    test_lam0_regression(); test_penalty_factor(); test_energy_accounting()
    test_pe_ignores_lam()
    print("ALL TESTS PASSED")
