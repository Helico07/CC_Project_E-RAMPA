"""diagnose_part_a.py -- Diagnostic script for Part A with multiprocessing."""
import sys, math, time, os
import numpy as np
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from core import GPU_TYPES, TDP_W, TDP_MAX, Cluster, preset, CLUSTER_KINDS, MIXES, SERVICES, ACOParams, _evaluate
from sim import run_fill

def _worker_a1(task):
    kind, mix, seed, lam = task
    spec = preset(kind)
    r = run_fill(spec, mix, seed, "RAMPA", lam=lam, penalize_links=False)
    rack_seq = [rec["racks"] for rec in r.records]
    return (kind, mix, seed, lam, r.count, rack_seq)

def run_a1():
    print("=" * 80, flush=True)
    print("A1. Comparing lam in {0, 1, 2, 5, 10} across 16 envs x seeds 0..4", flush=True)
    print("=" * 80, flush=True)
    
    envs = [(k, m) for k in CLUSTER_KINDS for m in MIXES]
    seeds = list(range(5))
    lams = [0.0, 1.0, 2.0, 5.0, 10.0]
    
    tasks = [(k, m, s, l) for (k, m) in envs for s in seeds for l in lams]
    print(f"Total A1 tasks: {len(tasks)} across {os.cpu_count()} CPU cores", flush=True)
    
    t0 = time.time()
    data = {}
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        for kind, mix, seed, lam, count, rack_seq in pool.map(_worker_a1, tasks):
            data[(kind, mix, seed, lam)] = (count, rack_seq)
    
    print(f"A1 simulation finished in {time.time()-t0:.1f}s", flush=True)
    
    # Comparisons: (lam_a, lam_b)
    for lam_a, lam_b in [(1.0, 5.0), (1.0, 0.0), (1.0, 2.0), (5.0, 10.0)]:
        diff_racks_count = 0
        diff_admitted_count = 0
        first_diff_indices = []
        
        for kind, mix in envs:
            for seed in seeds:
                cnt_a, seq_a = data[(kind, mix, seed, lam_a)]
                cnt_b, seq_b = data[(kind, mix, seed, lam_b)]
                
                if cnt_a != cnt_b:
                    diff_admitted_count += 1
                
                # Check rack sequence
                min_len = min(len(seq_a), len(seq_b))
                diff_idx = None
                for idx in range(min_len):
                    if seq_a[idx] != seq_b[idx]:
                        diff_idx = idx + 1 # 1-based request index
                        break
                if diff_idx is None and len(seq_a) != len(seq_b):
                    diff_idx = min_len + 1
                
                if diff_idx is not None:
                    diff_racks_count += 1
                    first_diff_indices.append(diff_idx)
        
        total_runs = len(envs) * len(seeds)
        print(f"\n--- Comparison lam={lam_a} vs lam={lam_b} (Total {total_runs} env-seed pairs) ---", flush=True)
        print(f"Environments/seeds where rack sequence differs: {diff_racks_count} / {total_runs} ({diff_racks_count/total_runs*100:.1f}%)", flush=True)
        print(f"Environments/seeds where admitted services differ: {diff_admitted_count} / {total_runs} ({diff_admitted_count/total_runs*100:.1f}%)", flush=True)
        if first_diff_indices:
            print(f"First request index where racks differ: min={min(first_diff_indices)}, median={int(np.median(first_diff_indices))}, max={max(first_diff_indices)}, mean={np.mean(first_diff_indices):.1f}", flush=True)
        else:
            print("Racks never differ!", flush=True)

def run_a2():
    print("\n" + "=" * 80, flush=True)
    print("A2. Cg_pen per GPU type for a fully free rack (Base cluster, 6 GPUs/rack)", flush=True)
    print("=" * 80, flush=True)
    
    spec = preset("Base", gpus_per_rack=6)
    cl = Cluster(spec)
    
    test_lams = [0.0, 0.5, 1.0, 1.1703, 2.0, 5.0, 10.0]
    
    header = f"{'GPU Type':<16} {'TDP(W)':<8} " + " ".join([f"lam={l:<7}" for l in [0, 0.5, 1, 1.17, 2, 5, 10]])
    print(header, flush=True)
    print("-" * len(header), flush=True)
    
    racks = [0, 1, 2, 3] # types 0, 1, 2, 3
    
    res_table = {r: [] for r in racks}
    for lam in test_lams:
        cl.lam = lam
        cl.penalize_links = False
        gpu_cost_pen, _, _ = cl.costs()
        for r in racks:
            res_table[r].append(gpu_cost_pen[r])
            
    for r in racks:
        gt = GPU_TYPES[cl.types[r]]
        tdp = TDP_W[gt.short]
        costs_str = " ".join([f"{c:>11.2f}" for c in res_table[r]])
        print(f"{gt.name:<16} {tdp:<8.1f} {costs_str}", flush=True)
        
    print("\nOrdering (cheapest to most expensive):", flush=True)
    for idx, lam in enumerate(test_lams):
        order = sorted(racks, key=lambda r: res_table[r][idx])
        order_names = [GPU_TYPES[cl.types[r]].short for r in order]
        print(f"  lam={lam:<6.2f}: {' < '.join(order_names)} (costs: {', '.join([f'{GPU_TYPES[cl.types[r]].short}={res_table[r][idx]:.1f}' for r in order])})", flush=True)

def run_a3():
    print("\n" + "=" * 80, flush=True)
    print("A3. Decompose chosen placement's total_cost for 5 sample requests (Base, Same demand, seed 0)", flush=True)
    print("=" * 80, flush=True)
    
    spec = preset("Base")
    mix = "Same demand"
    seed = 0
    
    for lam in [1.0, 5.0]:
        print(f"\n--- lam = {lam} (penalize_links=False) ---", flush=True)
        r = run_fill(spec, mix, seed, "RAMPA", lam=lam, penalize_links=False)
        print(f"{'Step':<5} {'Service':<22} {'Stages':<6} {'Racks':<16} {'GPU Cost':>10} {'Access Cost':>12} {'Link Cost':>10} {'Total Cost':>10} {'GPU Share%':>12}", flush=True)
        print("-" * 105, flush=True)
        
        cl_replay = Cluster(spec)
        cl_replay.lam = lam
        cl_replay.penalize_links = False
        
        for step, a in enumerate(r.allocs[:5]):
            ctx = cl_replay.context("cost")
            gpu_cost = ctx["Cg"]
            access_cost = ctx["acc"]
            link_cost = ctx["linkc"]
            routes = ctx["route"]
            racks = a.racks
            
            c_gpu = float(sum(gpu_cost[rack] for rack in racks))
            c_acc = 0.0
            c_link = 0.0
            for src, dst in zip(racks, racks[1:]):
                path = [] if src == dst else routes[src][dst]
                c_acc += float(access_cost[src] + access_cost[dst])
                c_link += float(sum(link_cost[lid] for lid in path))
            c_tot = c_gpu + c_acc + c_link
            
            gpu_share = (c_gpu / c_tot) * 100 if c_tot > 0 else 0
            s_name = SERVICES[a.svc].name
            racks_str = str(racks)
            print(f"{step+1:<5} {s_name:<22} {len(racks):<6} {racks_str:<16} {c_gpu:>10.2f} {c_acc:>12.2f} {c_link:>10.2f} {c_tot:>10.2f} {gpu_share:>11.1f}%", flush=True)
            
            cl_replay.commit(a.svc, a.racks, a.edge_links, a.plan, a.cost)

if __name__ == "__main__":
    run_a2()
    run_a3()
    run_a1()
