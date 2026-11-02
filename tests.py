"""Structural checks: aggregated Eq.(5) == brute-force per-GPU Eq.(5); DP respects constraints."""
import itertools, random
from collections import deque
import numpy as np
from core import *

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

if __name__ == "__main__":
    test_eq5(); test_plans(); test_bookkeeping()
