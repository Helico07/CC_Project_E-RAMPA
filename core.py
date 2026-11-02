"""
core.py -- RAMPA simulation engine (re-implementation of IEEE TCC 2025, Ikoma/Ohsita/Murata).

What is implemented
  * Heterogeneous GPU cluster: racks of homogeneous GPUs, 2-D torus of optical switches,
    limited optical circuits per torus link (paper: 4 fibres per switch pair).
  * Eq.(4) GPU cost, Eq.(5) link cost (computed exactly, but aggregated per rack -- see tests.py
    for the brute-force check), Eq.(6) total allocation cost.
  * Eq.(1)-(3) pipeline-cycle performance model + memory constraint.
  * ACO search (GPU selection by pheromone * cost heuristic, pheromone evaporation/deposit,
    cost-bound pruning, performance-check rejection).
  * Three methods:  RAMPA (cost objective, #stages is a decision variable)
                    NCAR  (same cost objective, #stages fixed)
                    PE    (PipeEdge-style: minimise bottleneck stage time, #stages fixed)

Simplifications (all disclosed in the README)
  * Operation graph = chain of "blocks" (groups of layers), forward edges only.
  * GPUs in a rack are identical, so ants choose *racks*; pheromone is per rack.
  * Paths: cost-aware shortest feasible route (hops first, link cost as tie-break) instead of the
    paper's link-level ant walk.  Circuit capacity is still enforced.
  * Partition search per candidate GPU list: DP minimising (bottleneck, sum of comm) lexicographically.
"""
from __future__ import annotations
import heapq
import itertools
import math
import random
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

# ----------------------------------------------------------------------------------------------
# Hardware (paper Table III)
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class GPUType:
    name: str
    short: str
    flops: float
    mem: float
    mem_bw: float

GPU_TYPES = [
    GPUType("GPU_A (L4)", "A", 30.3e12, 24e9, 300e9),
    GPUType("GPU_B (V100)", "B", 14.0e12, 32e9, 900e9),
    GPUType("GPU_C (A30)", "C", 10.3e12, 24e9, 933e9),
    GPUType("GPU_D (T4)", "D", 8.1e12, 16e9, 320e9),
]

# e-RAMPA energy parameters (W).  V100 PCIe = 250 W (= TDP_MAX => factor 1 at lam=1).
TDP_W: dict[str, float] = {"A": 72.0, "B": 250.0, "C": 165.0, "D": 70.0}
TDP_MAX: float = 250.0

PROP_GPU_SW = 0.05e-6
PROP_SW_SW = 0.1e-6
SWITCH_DELAY = 30e-9


def edge_latency(hops: int) -> float:
    """Return propagation and switching delay for a path with the given hop count."""
    gpu_switch_delay = 2 * PROP_GPU_SW
    link_delay = hops * PROP_SW_SW
    switch_delay = (hops + 1) * SWITCH_DELAY
    return gpu_switch_delay + link_delay + switch_delay

# ----------------------------------------------------------------------------------------------
# Services (paper Table V; per-block profiles are synthetic but sized after the real models)
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SLA:
    min_tps: float
    max_latency: float


@dataclass(frozen=True)
class ModelProfile:
    name: str
    flops: tuple
    weights: tuple
    out_bytes: tuple
    in_bytes: float
    cumF: tuple = field(init=False, repr=False, compare=False)
    cumW: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        assert len(self.flops) == len(self.weights) == len(self.out_bytes) >= 1
        object.__setattr__(
            self, "cumF", tuple(itertools.accumulate((0.0,) + self.flops))
        )
        object.__setattr__(
            self, "cumW", tuple(itertools.accumulate((0.0,) + self.weights))
        )

    @property
    def n(self) -> int:
        return len(self.flops)


@dataclass(frozen=True)
class Service:
    name: str
    category: str
    model: ModelProfile
    sla: SLA


def _chain(name, n, total_flops, total_weight_bytes, out_bytes, in_bytes):
    return ModelProfile(
        name,
        (total_flops / n,) * n,
        (total_weight_bytes / n,) * n,
        (out_bytes,) * n,
        in_bytes,
    )


SERVICES = [
    Service(
        "S1 YOLOS (video detect)",
        "HT",
        _chain("YOLOS", 12, 0.54e12, 0.60e9, 8e6, 8e6),
        SLA(60, 0.1),
    ),
    Service(
        "S2 VideoMAE (video cls)",
        "HT",
        _chain("VideoMAE", 12, 0.60e12, 0.42e9, 5e6, 16e6),
        SLA(30, 0.2),
    ),
    Service(
        "S3 ViT (image cls)",
        "LD",
        _chain("ViT", 12, 0.80e12, 1.44e9, 4e6, 2e6),
        SLA(30, 0.05),
    ),
    Service(
        "S4 Gemma2-2b (textgen)",
        "LD",
        _chain("Gemma2-2b", 12, 1.30e12, 6.24e9, 1.2e6, 0.1e6),
        SLA(5, 0.8),
    ),
    Service(
        "S5 Gemma2-27b (chat)",
        "HM",
        _chain("Gemma2-27b", 12, 13.5e12, 64.8e9, 2.3e6, 0.05e6),
        SLA(0.1, 10),
    ),
    Service(
        "S6 Qwen2.5-32B (chat)",
        "HM",
        _chain("Qwen2.5-32B", 16, 16e12, 76.8e9, 2.6e6, 0.05e6),
        SLA(0.25, 5),
    ),
]
CATEGORY_IDX = {"HT": [0, 1], "LD": [2, 3], "HM": [4, 5]}
# paper Table IV
MIXES = {
    "Same demand": (1 / 3, 1 / 3, 1 / 3),
    "High-throughput heavy": (0.8, 0.1, 0.1),
    "Low-delay heavy": (0.1, 0.8, 0.1),
    "Huge-model heavy": (0.1, 0.1, 0.8),
}


def arrival_stream(mix: str, seed: int, n: int) -> list[int]:
    """Service indices (into SERVICES) of the next n deployment requests."""
    p = MIXES[mix]
    assert abs(sum(p) - 1) < 1e-9
    rng = random.Random(10_007 * seed + 13)
    services = []
    for _ in range(n):
        cat = rng.choices(["HT", "LD", "HM"], weights=p)[0]
        services.append(rng.choice(CATEGORY_IDX[cat]))
    return services

# ----------------------------------------------------------------------------------------------
# Partition search for a fixed ordered list of GPUs (performance model, Eq. 1-3 + memory constraint)
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Plan:
    cuts: tuple
    Tu: float
    Tr: float
    Te: tuple
    Tc: tuple


_DP_CACHE: dict = {}


def plan_partition(
    model: ModelProfile,
    sla: SLA,
    bw: float,
    tidx: tuple,
    hops: tuple,
):
    """Find the best stage cuts for a GPU sequence and its inter-stage hop counts.

    Return ``None`` if no partition meets the memory, throughput, and latency limits.
    """
    cache_key = (model.name, sla.min_tps, sla.max_latency, bw, tidx, hops)
    if cache_key in _DP_CACHE:
        return _DP_CACHE[cache_key]
    if len(_DP_CACHE) > 400_000:
        _DP_CACHE.clear()
    plan = _plan(model, sla, bw, tidx, hops)
    _DP_CACHE[cache_key] = plan
    return plan


def _plan(model, sla, bw, tidx, hops):
    block_count = model.n
    stage_count = len(tidx)
    assert len(hops) == stage_count - 1
    if stage_count > block_count:
        return None
    cycle_limit = 1.0 / sla.min_tps
    cumulative_flops = model.cumF
    cumulative_weights = model.cumW
    output_bytes = model.out_bytes
    previous = {0: (0.0, 0.0)}
    parents = []
    for stage_index in range(stage_count):
        gpu = GPU_TYPES[tidx[stage_index]]
        is_last_stage = stage_index == stage_count - 1
        current, parent = {}, {}
        for first_block, (previous_bottleneck, previous_comm) in previous.items():
            if stage_index > 0:
                comm_delay = (
                    edge_latency(hops[stage_index - 1])
                    + output_bytes[first_block - 1] / bw
                )
                if comm_delay > cycle_limit:
                    continue
            else:
                comm_delay = 0.0

            input_bytes = (
                model.in_bytes
                if first_block == 0
                else output_bytes[first_block - 1]
            )
            first_end = block_count if is_last_stage else first_block + 1
            last_end = (
                block_count
                if is_last_stage
                else block_count - (stage_count - stage_index - 1)
            )
            for end_block in range(first_end, last_end + 1):
                weight_bytes = (
                    cumulative_weights[end_block]
                    - cumulative_weights[first_block]
                )
                if weight_bytes + input_bytes > gpu.mem:
                    break
                execution_time = (
                    (cumulative_flops[end_block] - cumulative_flops[first_block])
                    / gpu.flops
                    + weight_bytes / gpu.mem_bw
                )
                if execution_time > cycle_limit:
                    break
                bottleneck = max(
                    previous_bottleneck,
                    execution_time,
                    comm_delay,
                )
                total_comm = previous_comm + comm_delay
                old = current.get(end_block)
                if old is None or (bottleneck, total_comm) < old:
                    current[end_block] = (bottleneck, total_comm)
                    parent[end_block] = first_block
        if not current:
            return None
        parents.append(parent)
        previous = current
    if block_count not in previous:
        return None

    cuts = []
    end_block = block_count
    for stage_index in range(stage_count - 1, -1, -1):
        first_block = parents[stage_index][end_block]
        if stage_index > 0:
            cuts.append(first_block)
        end_block = first_block

    cuts = tuple(reversed(cuts))
    bounds = (0,) + cuts + (block_count,)
    stage_times = tuple(
        (
            cumulative_flops[bounds[i + 1]] - cumulative_flops[bounds[i]]
        )
        / GPU_TYPES[tidx[i]].flops
        + (
            cumulative_weights[bounds[i + 1]] - cumulative_weights[bounds[i]]
        )
        / GPU_TYPES[tidx[i]].mem_bw
        for i in range(stage_count)
    )
    comm_times = tuple(
        edge_latency(hops[i]) + output_bytes[bounds[i + 1] - 1] / bw
        for i in range(stage_count - 1)
    )
    cycle_time = max(stage_times + comm_times)
    execution_latency = stage_count * cycle_time + sum(comm_times)
    if (
        1.0 / cycle_time < sla.min_tps - 1e-12
        or execution_latency > sla.max_latency
    ):
        return None
    return Plan(cuts, cycle_time, execution_latency, stage_times, comm_times)

_NMIN: dict = {}


def n_min(model: ModelProfile, sla: SLA, bw: float):
    """Return the fewest stages whose GPU sequence can satisfy the SLA."""
    cache_key = (model.name, sla.min_tps, sla.max_latency, bw)
    if cache_key in _NMIN:
        return _NMIN[cache_key]

    minimum_stage_count = None
    for stage_count in range(1, model.n + 1):
        if stage_count <= 3:
            gpu_sequences = itertools.product(
                range(len(GPU_TYPES)),
                repeat=stage_count,
            )
        else:
            gpu_sequences = (
                (gpu_type,) * stage_count
                for gpu_type in range(len(GPU_TYPES))
            )
        if any(
            plan_partition(
                model,
                sla,
                bw,
                gpu_sequence,
                (0,) * (stage_count - 1),
            )
            for gpu_sequence in gpu_sequences
        ):
            minimum_stage_count = stage_count
            break
    _NMIN[cache_key] = minimum_stage_count
    return minimum_stage_count

# ----------------------------------------------------------------------------------------------
# Cluster
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ClusterSpec:
    name: str
    rows: int
    cols: int
    gpus_per_rack: int
    rack_types: tuple
    link_gbps: float = 10.0
    circuits: int = 4

def preset(kind: str, gpus_per_rack: int = 6) -> ClusterSpec:
    """Scaled-down versions of the paper's four evaluation clusters (Fig. 6)."""
    if kind == "Base":
        rack_types = tuple((rack % 4 + rack // 4) % 4 for rack in range(16))
        return ClusterSpec("Base", 4, 4, gpus_per_rack, rack_types)
    if kind == "High-bandwidth":
        rack_types = tuple((rack % 4 + rack // 4) % 4 for rack in range(16))
        return ClusterSpec(
            "High-bandwidth",
            4,
            4,
            gpus_per_rack,
            rack_types,
            100.0,
        )
    if kind == "High-performance":
        return ClusterSpec("High-performance", 2, 4, gpus_per_rack, (0, 1, 0, 1, 1, 0, 1, 0))
    if kind == "Large":
        rack_types = tuple((rack % 4 + rack // 6) % 4 for rack in range(36))
        return ClusterSpec("Large", 6, 6, gpus_per_rack, rack_types)
    raise ValueError(kind)


CLUSTER_KINDS = ["Base", "High-bandwidth", "High-performance", "Large"]

@dataclass
class Allocation:
    svc: int
    racks: list
    gpu_ids: list
    edge_links: list          # per edge: list of torus link ids
    plan: Plan
    cost: float
    t_start: float = 0.0

class Cluster:
    def __init__(self, spec: ClusterSpec):
        self.spec = spec
        rack_count = spec.rows * spec.cols
        assert len(spec.rack_types) == rack_count
        self.R = rack_count
        self.gpr = spec.gpus_per_rack
        self.types = list(spec.rack_types)
        self.bw = spec.link_gbps * 1e9 / 8
        neighbors = [set() for _ in range(rack_count)]
        for i in range(spec.rows):
            for j in range(spec.cols):
                rack = i * spec.cols + j
                right_neighbor = i * spec.cols + (j + 1) % spec.cols
                lower_neighbor = ((i + 1) % spec.rows) * spec.cols + j
                for neighbor in (right_neighbor, lower_neighbor):
                    if rack != neighbor:
                        neighbors[rack].add(neighbor)
                        neighbors[neighbor].add(rack)
        self.nbrs = [sorted(rack_neighbors) for rack_neighbors in neighbors]
        self.links = sorted(
            {
                (min(rack, neighbor), max(rack, neighbor))
                for rack in range(rack_count)
                for neighbor in neighbors[rack]
            }
        )
        self.lid = {l: i for i, l in enumerate(self.links)}
        self.L = len(self.links)
        # shortest-path structure between racks (switch level)
        self.dist = np.zeros((rack_count, rack_count), dtype=int)
        sigma = np.zeros((rack_count, rack_count), dtype=np.int64)
        for source in range(rack_count):
            distances = {source: 0}
            path_counts = {source: 1}
            queue = [source]
            for rack in queue:
                for neighbor in self.nbrs[rack]:
                    if neighbor not in distances:
                        distances[neighbor] = distances[rack] + 1
                        path_counts[neighbor] = 0
                        queue.append(neighbor)
                    if distances[neighbor] == distances[rack] + 1:
                        path_counts[neighbor] += path_counts[rack]
            for target in range(rack_count):
                self.dist[source, target] = distances[target]
                sigma[source, target] = path_counts[target]

        self.pairs = [
            (a, b)
            for a in range(rack_count)
            for b in range(a + 1, rack_count)
        ]
        self.pa = np.array([pair[0] for pair in self.pairs])
        self.pb = np.array([pair[1] for pair in self.pairs])
        # F[l, p] = fraction of shortest paths of rack pair p that use torus link l   (the N^r(l)/N^r term of Eq.5)
        self.F = np.zeros((self.L, len(self.pairs)))
        for pair_index, (source, target) in enumerate(self.pairs):
            for link_index, (first_rack, second_rack) in enumerate(self.links):
                paths_through_link = 0
                for first, second in (
                    (first_rack, second_rack),
                    (second_rack, first_rack),
                ):
                    if (
                        self.dist[source, first]
                        + 1
                        + self.dist[second, target]
                        == self.dist[source, target]
                    ):
                        paths_through_link += (
                            sigma[source, first] * sigma[second, target]
                        )
                if paths_through_link:
                    self.F[link_index, pair_index] = (
                        paths_through_link / sigma[source, target]
                    )
        self.rack_flops = np.array([GPU_TYPES[t].flops for t in self.types])
        self.rack_mem = np.array([GPU_TYPES[t].mem for t in self.types])
        self.rack_tdp = np.array([TDP_W[GPU_TYPES[t].short] for t in self.types])
        self.lam: float = 0.0
        self.penalize_links: bool = False
        self.reset()

    def reset(self):
        self.free = [self.gpr] * self.R
        self.free_ids = [list(range(self.gpr)) for _ in range(self.R)]
        self.cap = [self.spec.circuits] * self.L

    # ---- state metrics ------------------------------------------------------------------------
    @property
    def total_gpus(self):
        return self.R * self.gpr

    def util(self):
        return 1.0 - sum(self.free) / self.total_gpus

    def full_free_racks(self):
        return sum(free_count == self.gpr for free_count in self.free)

    def free_by_type(self):
        free_by_type = Counter()
        for rack in range(self.R):
            gpu_type = GPU_TYPES[self.types[rack]].short
            free_by_type[gpu_type] += self.free[rack]
        return dict(free_by_type)

    # ---- cost model: Eq.(4), Eq.(5) ----------------------------------------------------------
    def costs(self):
        free_counts = np.array(self.free, dtype=float)
        gpu_cost0 = (
            (self.rack_flops / 1e12)
            * (self.rack_mem / 1e9)
            * free_counts
        )
        # e-RAMPA penalty: multiply Cg by (1 + lam * TDP / TDP_MAX).
        # With lam=0 penal == 1.0 everywhere => gpu_cost_pen == gpu_cost0 (bit-identical).
        penal = 1.0 + self.lam * self.rack_tdp / TDP_MAX
        gpu_cost_pen = gpu_cost0 * penal
        # Link/access cost: use penalised cost only when penalize_links=True.
        gpu_for_links = gpu_cost_pen if self.penalize_links else gpu_cost0
        gpu_hops = self.dist + 2
        partner_counts = free_counts[None, :] - np.eye(self.R)
        access_cost = (
            ((gpu_for_links[:, None] + gpu_for_links[None, :]) / gpu_hops)
            * partner_counts
        ).sum(axis=1)
        pair_weights = (
            free_counts[self.pa]
            * free_counts[self.pb]
            * (gpu_for_links[self.pa] + gpu_for_links[self.pb])
            / (self.dist[self.pa, self.pb] + 2)
        )
        link_cost = self.F @ pair_weights
        return gpu_cost_pen, access_cost, link_cost

    def route_table(self, linkc=None):
        """All-pairs cheapest feasible routes: fewest hops first, then lowest link cost; skips full links."""
        route_table = [[None] * self.R for _ in range(self.R)]
        for source in range(self.R):
            best_route = {source: (0, 0.0)}
            previous_hop = {}
            queue = [(0, 0.0, source)]
            while queue:
                hop_count, route_cost, rack = heapq.heappop(queue)
                if best_route.get(rack) != (hop_count, route_cost):
                    continue
                for neighbor in self.nbrs[rack]:
                    link_id = self.lid[(min(rack, neighbor), max(rack, neighbor))]
                    if self.cap[link_id] <= 0:
                        continue
                    next_route = (
                        hop_count + 1,
                        route_cost + (linkc[link_id] if linkc is not None else 0.0),
                    )
                    if (
                        neighbor not in best_route
                        or next_route < best_route[neighbor]
                    ):
                        best_route[neighbor] = next_route
                        previous_hop[neighbor] = (rack, link_id)
                        heapq.heappush(
                            queue,
                            (next_route[0], next_route[1], neighbor),
                        )

            for target in best_route:
                path = []
                rack = target
                while rack != source:
                    previous_rack, link_id = previous_hop[rack]
                    path.append(link_id)
                    rack = previous_rack
                route_table[source][target] = path[::-1]
        return route_table

    def context(self, objective: str):
        gpu_cost, access_cost, link_cost = self.costs()
        if objective == "cost":
            heuristic = np.where(
                np.array(self.free) > 0,
                1.0 / np.maximum(gpu_cost, 1e-9),
                0.0,
            )
            route_costs = link_cost
        else:
            heuristic = np.array(self.rack_flops) / 1e12
            route_costs = None

        return {
            "Cg": gpu_cost,
            "acc": access_cost,
            "linkc": link_cost,
            "eta": heuristic,
            "route": self.route_table(route_costs),
        }

    # ---- allocation bookkeeping ---------------------------------------------------------------
    def commit(
        self,
        svc: int,
        racks: list,
        edge_links: list,
        plan: Plan,
        cost: float,
        t=0.0,
    ) -> Allocation:
        gpu_ids = []
        for rack in racks:
            assert self.free[rack] > 0
            gpu_ids.append((rack, self.free_ids[rack].pop()))
            self.free[rack] -= 1
        for path in edge_links:
            for link_id in path:
                assert self.cap[link_id] > 0
                self.cap[link_id] -= 1
        return Allocation(svc, list(racks), gpu_ids, edge_links, plan, cost, t)

    def release(self, a: Allocation):
        for rack, gpu_id in a.gpu_ids:
            self.free_ids[rack].append(gpu_id)
            self.free[rack] += 1
        for path in a.edge_links:
            for link_id in path:
                self.cap[link_id] += 1
        assert all(0 <= free_count <= self.gpr for free_count in self.free)
        assert all(
            0 <= capacity <= self.spec.circuits for capacity in self.cap
        )

# ----------------------------------------------------------------------------------------------
# ACO search and the three allocation methods
# ----------------------------------------------------------------------------------------------
@dataclass
class ACOParams:
    n_ants: int = 14
    n_iter: int = 6
    alpha: float = 1.0
    beta: float = 2.0
    rho: float = 0.7
    phi: float = 1.0
    extra_stages: int = 5


@dataclass
class Candidate:
    racks: list
    plan: Plan
    cost: float
    score: float
    edge_links: list


def _evaluate(cl: Cluster, ctx, model, sla, racks, objective, bound):
    gpu_cost = ctx["Cg"]
    access_cost = ctx["acc"]
    link_cost = ctx["linkc"]
    routes = ctx["route"]
    total_cost = float(sum(gpu_cost[rack] for rack in racks))
    edge_links = []
    hop_counts = []

    for source, target in zip(racks, racks[1:]):
        path = [] if source == target else routes[source][target]
        if path is None:
            return None
        total_cost += float(
            access_cost[source]
            + access_cost[target]
            + sum(link_cost[link_id] for link_id in path)
        )
        edge_links.append(path)
        hop_counts.append(len(path))

    if objective == "cost" and total_cost >= bound:
        return None

    link_usage = Counter(
        link_id for path in edge_links for link_id in path
    )
    if any(
        usage > cl.cap[link_id]
        for link_id, usage in link_usage.items()
    ):
        return None

    gpu_types = tuple(cl.types[rack] for rack in racks)
    plan = plan_partition(model, sla, cl.bw, gpu_types, tuple(hop_counts))
    if plan is None:
        return None

    score = total_cost if objective == "cost" else plan.Tu
    if score >= bound:
        return None
    return Candidate(list(racks), plan, total_cost, score, edge_links)


def viable_types(cl, model, sla, Np):
    """Feasibility pre-filter: GPU types on which an Np-stage pipeline can meet the SLA (all stages in one rack).
    Without it the cost heuristic (which loves cheap GPUs) rarely samples the few GPUs that can host strict services."""
    feasible_types = {
        gpu_type
        for gpu_type in range(len(GPU_TYPES))
        if plan_partition(
            model,
            sla,
            cl.bw,
            (gpu_type,) * Np,
            (0,) * (Np - 1),
        )
    }
    return feasible_types or set(range(len(GPU_TYPES)))


def aco_search(cl, ctx, model, sla, Np, objective, rng, bound, p: ACOParams):
    pheromone = [1.0] * cl.R
    feasible_types = viable_types(cl, model, sla, Np)
    heuristic = [
        float(value) ** p.beta
        if cl.types[rack] in feasible_types
        else 0.0
        for rack, value in enumerate(ctx["eta"])
    ]
    best_candidate = None
    reference_score = None

    for _ in range(p.n_iter):
        iteration_best = None
        for _ in range(p.n_ants):
            available_gpus = list(cl.free)
            racks = []
            for _ in range(Np):
                weights = [
                    (pheromone[rack] ** p.alpha)
                    * heuristic[rack]
                    * available_gpus[rack]
                    for rack in range(cl.R)
                ]
                if sum(weights) <= 0:
                    break
                rack = rng.choices(range(cl.R), weights=weights)[0]
                available_gpus[rack] -= 1
                racks.append(rack)
            if len(racks) < Np:
                continue
            current_bound = (
                bound
                if best_candidate is None
                else min(bound, best_candidate.score)
            )
            candidate = _evaluate(
                cl,
                ctx,
                model,
                sla,
                racks,
                objective,
                current_bound,
            )
            if candidate and (
                iteration_best is None
                or candidate.score < iteration_best.score
            ):
                iteration_best = candidate

        pheromone = [value * p.rho for value in pheromone]
        if iteration_best is None:
            continue

        if reference_score is None:
            reference_score = iteration_best.score
        deposit = p.phi * reference_score / iteration_best.score
        for rack in set(iteration_best.racks):
            pheromone[rack] += deposit

        if (
            best_candidate is None
            or iteration_best.score < best_candidate.score
        ):
            best_candidate = iteration_best

    return best_candidate


METHODS = ["RAMPA", "NCAR", "PE"]


def allocate(
    cl: Cluster,
    svc_idx: int,
    method: str,
    rng: random.Random,
    params: ACOParams,
    pattern: int = 0,
    margin: float = 0.0,
    t: float = 0.0,
):
    """Try to deploy service `svc_idx`.  pattern = extra stages over n_min for the fixed-stage baselines.
    margin = planning safety margin (plan against gamma*(1+m), delta*(1-m))."""
    assert method in METHODS
    svc = SERVICES[svc_idx]
    model = svc.model
    plan_sla = SLA(
        svc.sla.min_tps * (1 + margin),
        svc.sla.max_latency * (1 - margin),
    )
    objective = "perf" if method == "PE" else "cost"
    ctx = cl.context(objective)

    if method == "RAMPA":
        minimum_stages = n_min(model, plan_sla, cl.bw)
        if minimum_stages is None:
            return None
        stage_counts = range(
            minimum_stages,
            min(model.n, minimum_stages + params.extra_stages) + 1,
        )
    else:
        minimum_stages = n_min(model, svc.sla, cl.bw)
        if minimum_stages is None:
            return None
        stage_counts = [min(model.n, minimum_stages + pattern)]

    best_candidate = None
    best_score = math.inf
    for stage_count in stage_counts:
        candidate = aco_search(
            cl,
            ctx,
            model,
            plan_sla,
            stage_count,
            objective,
            rng,
            best_score,
            params,
        )
        if candidate and candidate.score < best_score:
            best_candidate = candidate
            best_score = candidate.score

    if best_candidate is None:
        return None
    return cl.commit(
        svc_idx,
        best_candidate.racks,
        best_candidate.edge_links,
        best_candidate.plan,
        best_candidate.cost,
        t,
    )
