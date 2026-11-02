"""run_energy.py -- Runs the e-RAMPA experiment grid and saves to results/energy/."""
import sys, os, time, math
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from core import CLUSTER_KINDS, MIXES, preset, ACOParams
from sim import run_fill
from energy import alloc_series

PRM = ACOParams(n_ants=20, n_iter=8, extra_stages=6)
SEEDS = list(range(10))

def get_task_list():
    tasks = []
    # Grid: 4 clusters x 4 mixes x 10 seeds = 160
    for kind in CLUSTER_KINDS:
        for mix in MIXES:
            for seed in SEEDS:
                # 1. RAMPA, penalize_links=False, lam in {0, 0.25, 0.5, 1, 2, 5, 10}
                for lam in [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]:
                    tasks.append((kind, mix, seed, "RAMPA", -1, lam, 0))
                # 2. RAMPA, penalize_links=True, lam in {1, 5}
                for lam in [1.0, 5.0]:
                    tasks.append((kind, mix, seed, "RAMPA", -1, lam, 1))
                # 3. NCAR pattern 0 at lam=0
                tasks.append((kind, mix, seed, "NCAR", 0, 0.0, 0))
                # 4. PE pattern 0 at lam=0
                tasks.append((kind, mix, seed, "PE", 0, 0.0, 0))
    return tasks

def _run_single_task(task):
    kind, mix, seed, method, pattern, lam, links = task
    penalize_links = bool(links)
    pat = max(0, pattern) # for NCAR/PE
    
    spec = preset(kind)
    t0 = time.time()
    r = run_fill(spec, mix, seed, method, pattern=pat, params=PRM,
                 snapshots=False, lam=lam, penalize_links=penalize_links)
    runtime_s = round(time.time() - t0, 3)
    
    # Run series
    series = alloc_series(r.records, spec)
    
    # Run row
    run_row = {
        "cluster": kind,
        "mix": mix,
        "seed": seed,
        "method": method,
        "pattern": pattern,
        "lam": lam,
        "links": links,
        "count": r.count,
        "runtime_s": runtime_s
    }
    
    # Step rows
    step_rows = []
    for s in series:
        step_rows.append({
            "cluster": kind,
            "mix": mix,
            "seed": seed,
            "method": method,
            "pattern": pattern,
            "lam": lam,
            "links": links,
            "step": s["step"],
            "gpus_in_use": s["gpus_in_use"],
            "alloc_tdp": s["alloc_tdp"],
            "n_A": s["n_A"],
            "n_B": s["n_B"],
            "n_C": s["n_C"],
            "n_D": s["n_D"]
        })
        
    return run_row, step_rows

def main():
    print("=" * 80, flush=True)
    print("run_energy.py: e-RAMPA Experiment Runner", flush=True)
    print("=" * 80, flush=True)
    
    # 1. Benchmark Base and Large
    print("Benchmarking single runs...", flush=True)
    t0 = time.time()
    r_base = run_fill(preset("Base"), "Same demand", 0, "RAMPA", params=PRM)
    t_base = time.time() - t0
    
    t0 = time.time()
    r_large = run_fill(preset("Large"), "Same demand", 0, "RAMPA", params=PRM)
    t_large = time.time() - t0
    
    print(f"  Base run:  {t_base:.2f}s (count={r_base.count})", flush=True)
    print(f"  Large run: {t_large:.2f}s (count={r_large.count})", flush=True)
    
    tasks = get_task_list()
    total_tasks = len(tasks)
    large_tasks = sum(1 for t in tasks if t[0] == "Large")
    small_tasks = total_tasks - large_tasks
    est_total_seq = small_tasks * t_base + large_tasks * t_large
    
    num_cpus = os.cpu_count() or 1
    est_total_par = est_total_seq / num_cpus
    print(f"Total tasks in grid: {total_tasks} (16 envs x 10 seeds x 11 configurations)", flush=True)
    print(f"Estimated sequential runtime: {est_total_seq:.0f}s ({est_total_seq/60:.1f} minutes)", flush=True)
    print(f"Estimated parallel runtime ({num_cpus} workers): {est_total_par:.0f}s ({est_total_par/60:.1f} minutes)", flush=True)
    
    if est_total_seq > 40 * 60:
        print(f"Sequential runtime exceeds 40 minutes -> using multiprocessing across {num_cpus} cores.", flush=True)
    
    os.makedirs("results/energy", exist_ok=True)
    
    start_time = time.time()
    run_records = []
    step_records = []
    
    print(f"\nStarting execution of {total_tasks} tasks...", flush=True)
    last_print = time.time()
    completed = 0
    
    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        for run_row, step_rows in executor.map(_run_single_task, tasks):
            run_records.append(run_row)
            step_records.extend(step_rows)
            completed += 1
            if completed % 100 == 0 or completed == total_tasks or time.time() - last_print > 10:
                elapsed = time.time() - start_time
                pct = completed / total_tasks * 100
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total_tasks - completed) / rate if rate > 0 else 0
                print(f"[{elapsed:6.1f}s] Progress: {completed}/{total_tasks} ({pct:.1f}%) | ETA: {eta:.0f}s", flush=True)
                last_print = time.time()
                
    total_elapsed = time.time() - start_time
    print(f"\nAll tasks finished in {total_elapsed:.1f}s ({total_elapsed/60:.2f} minutes).", flush=True)
    
    # 2. Sort and Save CSVs
    print("Sorting and saving CSVs...", flush=True)
    df_runs = pd.DataFrame(run_records)
    sort_cols_runs = ["cluster", "mix", "seed", "method", "pattern", "lam", "links"]
    df_runs = df_runs.sort_values(by=sort_cols_runs).reset_index(drop=True)
    df_runs.to_csv("results/energy/energy_runs.csv", index=False)
    print(f"Saved results/energy/energy_runs.csv ({len(df_runs)} rows)", flush=True)
    
    df_steps = pd.DataFrame(step_records)
    sort_cols_steps = ["cluster", "mix", "seed", "method", "pattern", "lam", "links", "step"]
    df_steps = df_steps.sort_values(by=sort_cols_steps).reset_index(drop=True)
    df_steps.to_csv("results/energy/energy_steps.csv", index=False)
    print(f"Saved results/energy/energy_steps.csv ({len(df_steps)} rows)", flush=True)
    
    # 3. Verification against results/fill.csv
    print("\nVerifying lam=0 RAMPA against baseline results/fill.csv...", flush=True)
    fill_path = "results/fill.csv"
    if os.path.exists(fill_path):
        df_fill = pd.read_csv(fill_path)
        base_rampa = df_fill[(df_fill["method"] == "RAMPA") & (df_fill["pattern"] == -1)]
        new_rampa = df_runs[(df_runs["method"] == "RAMPA") & (df_runs["lam"] == 0.0) & (df_runs["links"] == 0)]
        
        merged = pd.merge(base_rampa, new_rampa, on=["cluster", "mix", "seed"], suffixes=("_base", "_new"))
        matches = (merged["count_base"] == merged["count_new"]).sum()
        total_checks = len(merged)
        print(f"Verification: {matches} / {total_checks} match exact counts in baseline results/fill.csv")
        if matches == total_checks:
            print(">>> EXACT MATCH: All lam=0 RAMPA counts are 100% bit-identical to baseline results/fill.csv! <<<", flush=True)
        else:
            mismatches = merged[merged["count_base"] != merged["count_new"]]
            print(f"WARNING: Found {len(mismatches)} mismatches:\n{mismatches[['cluster', 'mix', 'seed', 'count_base', 'count_new']]}")
    else:
        print(f"Warning: {fill_path} not found for verification.")

if __name__ == "__main__":
    main()
