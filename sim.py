"""sim.py -- experiment drivers: fill-until-failure (paper metric), dynamic arrivals/departures, profiling-noise test."""
from __future__ import annotations
import heapq, random
from dataclasses import dataclass, field
import numpy as np
from core import *

@dataclass
class FillResult:
    method: str
    count: int
    records: list = field(default_factory=list)   # per admitted service: step, util, full_free, svc, Np, (snapshot)
    allocs: list = field(default_factory=list)
    failed_svc: int | None = None

def run_fill(spec, mix, seed, method, pattern=0, margin=0.0, params=None, max_services=2000, snapshots=False):
    """Paper metric: keep deploying requests until the first one cannot be allocated."""
    params = params or ACOParams()
    cl = Cluster(spec)
    rng = random.Random(seed * 101 + METHODS.index(method) * 7 + pattern)
    res = FillResult(method, 0)
    for step, i in enumerate(arrival_stream(mix, seed, max_services)):
        a = allocate(cl, i, method, rng, params, pattern=pattern, margin=margin)
        if a is None:
            res.failed_svc = i
            break
        res.allocs.append(a)
        rec = dict(step=step + 1, util=cl.util(), full_free=cl.full_free_racks(), svc=i, Np=len(a.racks),
                   racks=list(a.racks), cost=a.cost)
        if snapshots:
            rec["free"] = list(cl.free)
            rec["cap_left"] = list(cl.cap)
        res.records.append(rec)
    res.count = len(res.allocs)
    return res

def run_dynamic(spec, mix, seed, method, load, n_arrivals=400, pattern=0, params=None, warmup=0.25):
    """Poisson arrivals (rate 1), exponential holding time (mean = load) => ~`load` concurrent services offered."""
    params = params or ACOParams()
    cl = Cluster(spec)
    rng = random.Random(seed * 101 + METHODS.index(method) * 7 + pattern)
    trng = random.Random(seed * 977 + int(load))
    t, heap, uid = 0.0, [], 0
    blocked = total = 0
    utils = []
    for k, i in enumerate(arrival_stream(mix, seed, n_arrivals)):
        t += trng.expovariate(1.0)
        hold = trng.expovariate(1.0 / load)
        while heap and heap[0][0] <= t:
            cl.release(heapq.heappop(heap)[2])
        a = allocate(cl, i, method, rng, params, pattern=pattern, t=t)
        if a is not None:
            uid += 1
            heapq.heappush(heap, (t + hold, uid, a))
        if k >= warmup * n_arrivals:
            total += 1
            blocked += a is None
            utils.append(cl.util())
    return dict(method=method, load=load, blocking=blocked / total, util=float(np.mean(utils)))

def violation_rate(allocs, noise, trials=200, seed=0):
    """Fraction of deployed services whose SLA breaks when real stage times deviate from the profile by U(-noise,+noise)."""
    if not allocs:
        return float("nan")
    rng = np.random.default_rng(seed)
    bad = 0
    for _ in range(trials):
        for a in allocs:
            sla = SERVICES[a.svc].sla
            Te = np.array(a.plan.Te) * (1 + rng.uniform(-noise, noise, len(a.plan.Te)))
            Tc = np.array(a.plan.Tc)
            Tu = max(Te.max(), Tc.max() if len(Tc) else 0.0)
            Tr = len(Te) * Tu + Tc.sum()
            bad += (1 / Tu < sla.min_tps) or (Tr > sla.max_latency)
    return bad / (trials * len(allocs))
