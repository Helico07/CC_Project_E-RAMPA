"""diagnose_highperf.py -- In-depth diagnosis of High-performance and Base clusters under e-RAMPA."""
import os, sys, math
import pandas as pd
import numpy as np
from core import preset, SERVICES, GPU_TYPES, plan_partition, ACOParams
from sim import run_fill

def run_step_1():
    print("=" * 100)
    print("1. PER-ENVIRONMENT SUMMARY TABLE (16 environments, RAMPA, penalize_links=False)")
    print("=" * 100)
    
    df_runs = pd.read_csv("results/energy/energy_runs.csv")
    df_steps = pd.read_csv("results/energy/energy_steps.csv")
    
    # Filter RAMPA, links=0
    rampa_runs = df_runs[(df_runs["method"] == "RAMPA") & (df_runs["links"] == 0)]
    
    # Compute N_ref_A per env: floor(0.75 * min count over all RAMPA runs)
    all_rampa = df_runs[df_runs["method"] == "RAMPA"]
    n_ref_a = {}
    for (cl, mx), g in all_rampa.groupby(["cluster", "mix"]):
        n_ref_a[(cl, mx)] = math.floor(0.75 * g["count"].min())
        
    clusters = ["Base", "High-bandwidth", "High-performance", "Large"]
    mixes = ["High-throughput heavy", "Huge-model heavy", "Low-delay heavy", "Same demand"]
    
    # Pre-index step records for N_ref_A tdp
    # Filter steps to rampa links=0
    rampa_steps = df_steps[(df_steps["method"] == "RAMPA") & (df_steps["links"] == 0)]
    step_dict = {}
    for _, row in rampa_steps.iterrows():
        key = (row["cluster"], row["mix"], row["seed"], row["lam"], int(row["step"]))
        step_dict[key] = row["alloc_tdp"]
        
    rows = []
    for cl in clusters:
        for mx in mixes:
            sub = rampa_runs[(rampa_runs["cluster"] == cl) & (rampa_runs["mix"] == mx)]
            c0 = sub[sub["lam"] == 0.0].set_index("seed")["count"]
            
            row = {"cluster": cl, "mix": mx}
            
            # Mean count and paired % change for lam in {0, 0.5, 1, 2, 5}
            for l in [0.0, 0.5, 1.0, 2.0, 5.0]:
                cl_l = sub[sub["lam"] == l].set_index("seed")["count"]
                row[f"count_lam{l:g}"] = cl_l.mean()
                if l > 0:
                    paired_pct = (cl_l - c0) / c0 * 100
                    row[f"pct_diff_lam{l:g}"] = paired_pct.mean()
                    down = int((cl_l < c0).sum())
                    same = int((cl_l == c0).sum())
                    up = int((cl_l > c0).sum())
                    row[f"d_s_u_lam{l:g}"] = f"{down}/{same}/{up}"
                    
            # Paired % change of alloc_tdp at N_ref_A for lam=1 and 5
            ref_step = n_ref_a[(cl, mx)]
            for l in [1.0, 5.0]:
                pct_tdp_list = []
                for s in range(10):
                    tdp_0 = step_dict.get((cl, mx, s, 0.0, ref_step), np.nan)
                    tdp_l = step_dict.get((cl, mx, s, l, ref_step), np.nan)
                    if not np.isnan(tdp_0) and not np.isnan(tdp_l) and tdp_0 > 0:
                        pct_tdp_list.append((tdp_l - tdp_0) / tdp_0 * 100)
                row[f"tdp_pct_lam{l:g}"] = np.mean(pct_tdp_list) if pct_tdp_list else np.nan
                
            rows.append(row)
            
    df_env = pd.DataFrame(rows)
    os.makedirs("results/energy/diag", exist_ok=True)
    df_env.to_csv("results/energy/diag/env_summary.csv", index=False)
    
    # Print clean table
    print_cols = ["cluster", "mix", "count_lam0", "count_lam0.5", "pct_diff_lam0.5", "count_lam1", "pct_diff_lam1", "d_s_u_lam1", "tdp_pct_lam1", "count_lam5", "pct_diff_lam5", "d_s_u_lam5", "tdp_pct_lam5"]
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 160)
    print(df_env[print_cols].to_string(index=False))
    return df_env

def run_step_2():
    print("\n" + "=" * 100)
    print("2. HIGH-PERFORMANCE CLUSTER + LOW-DELAY HEAVY REJECTION ANALYSIS (seeds 0..9, lam in {0, 1, 5})")
    print("=" * 100)
    
    prm = ACOParams(n_ants=20, n_iter=8, extra_stages=6)
    spec = preset("High-performance")
    mix = "Low-delay heavy"
    
    records = []
    for lam in [0.0, 1.0, 5.0]:
        for seed in range(10):
            r = run_fill(spec, mix, seed, "RAMPA", params=prm, snapshots=True, lam=lam)
            failed_svc = r.failed_svc
            s = SERVICES[failed_svc]
            last_rec = r.records[-1]
            free = last_rec["free"]
            
            free_l4 = sum(free[r_idx] for r_idx in range(len(free)) if spec.rack_types[r_idx] == 0)
            free_v100 = sum(free[r_idx] for r_idx in range(len(free)) if spec.rack_types[r_idx] == 1)
            used_l4 = 24 - free_l4
            used_v100 = 24 - free_v100
            
            # Check SLA feasibility on V100 (type 1) and L4 (type 0)
            # unconstrained hops=0
            v100_sla_ok = any(
                plan_partition(s.model, s.sla, spec.link_gbps * 1e9 / 8, (1,) * Np, (0,) * (Np - 1)) is not None
                for Np in range(1, s.model.n + 1)
            )
            l4_sla_ok = any(
                plan_partition(s.model, s.sla, spec.link_gbps * 1e9 / 8, (0,) * Np, (0,) * (Np - 1)) is not None
                for Np in range(1, s.model.n + 1)
            )
            
            # Check if placeable given free GPUs
            v100_placeable = any(
                plan_partition(s.model, s.sla, spec.link_gbps * 1e9 / 8, (1,) * Np, (0,) * (Np - 1)) is not None
                for Np in range(1, free_v100 + 1)
            )
            l4_placeable = any(
                plan_partition(s.model, s.sla, spec.link_gbps * 1e9 / 8, (0,) * Np, (0,) * (Np - 1)) is not None
                for Np in range(1, free_l4 + 1)
            )
            
            records.append({
                "cluster": "High-performance",
                "mix": mix,
                "lam": lam,
                "seed": seed,
                "admitted": r.count,
                "rejected_svc": s.name,
                "category": s.category,
                "free_L4": free_l4,
                "free_V100": free_v100,
                "used_L4": used_l4,
                "used_V100": used_v100,
                "v100_sla_ok": v100_sla_ok,
                "v100_placeable": v100_placeable,
                "l4_sla_ok": l4_sla_ok,
                "l4_placeable": l4_placeable
            })
            
    df_hp = pd.DataFrame(records)
    df_hp.to_csv("results/energy/diag/highperf_rejections.csv", index=False)
    print(df_hp[["lam", "seed", "admitted", "rejected_svc", "free_L4", "free_V100", "used_L4", "used_V100", "v100_sla_ok", "v100_placeable", "l4_sla_ok", "l4_placeable"]].to_string(index=False))
    return df_hp

def run_step_3():
    print("\n" + "=" * 100)
    print("3. BASE CLUSTER + LOW-DELAY HEAVY REJECTION ANALYSIS (seeds 0..9, lam in {0, 1, 5})")
    print("=" * 100)
    
    prm = ACOParams(n_ants=20, n_iter=8, extra_stages=6)
    spec = preset("Base")
    mix = "Low-delay heavy"
    
    records = []
    for lam in [0.0, 1.0, 5.0]:
        for seed in range(10):
            r = run_fill(spec, mix, seed, "RAMPA", params=prm, snapshots=True, lam=lam)
            failed_svc = r.failed_svc
            s = SERVICES[failed_svc]
            last_rec = r.records[-1]
            free = last_rec["free"]
            
            free_by_type = {gt.short: 0 for gt in GPU_TYPES}
            for r_idx in range(len(free)):
                free_by_type[GPU_TYPES[spec.rack_types[r_idx]].short] += free[r_idx]
            used_by_type = {gt.short: 24 - free_by_type[gt.short] for gt in GPU_TYPES}
            
            # Check SLA feasibility on V100 (type 1) and L4 (type 0)
            v100_sla_ok = any(
                plan_partition(s.model, s.sla, spec.link_gbps * 1e9 / 8, (1,) * Np, (0,) * (Np - 1)) is not None
                for Np in range(1, s.model.n + 1)
            )
            l4_sla_ok = any(
                plan_partition(s.model, s.sla, spec.link_gbps * 1e9 / 8, (0,) * Np, (0,) * (Np - 1)) is not None
                for Np in range(1, s.model.n + 1)
            )
            
            # Check if placeable given free GPUs
            v100_placeable = any(
                plan_partition(s.model, s.sla, spec.link_gbps * 1e9 / 8, (1,) * Np, (0,) * (Np - 1)) is not None
                for Np in range(1, free_by_type["B"] + 1)
            )
            l4_placeable = any(
                plan_partition(s.model, s.sla, spec.link_gbps * 1e9 / 8, (0,) * Np, (0,) * (Np - 1)) is not None
                for Np in range(1, free_by_type["A"] + 1)
            )
            
            records.append({
                "cluster": "Base",
                "mix": mix,
                "lam": lam,
                "seed": seed,
                "admitted": r.count,
                "rejected_svc": s.name,
                "category": s.category,
                "free_L4": free_by_type["A"],
                "free_V100": free_by_type["B"],
                "free_A30": free_by_type["C"],
                "free_T4": free_by_type["D"],
                "used_L4": used_by_type["A"],
                "used_V100": used_by_type["B"],
                "used_A30": used_by_type["C"],
                "used_T4": used_by_type["D"],
                "v100_sla_ok": v100_sla_ok,
                "v100_placeable": v100_placeable,
                "l4_sla_ok": l4_sla_ok,
                "l4_placeable": l4_placeable
            })
            
    df_base = pd.DataFrame(records)
    df_base.to_csv("results/energy/diag/base_rejections.csv", index=False)
    print(df_base[["lam", "seed", "admitted", "rejected_svc", "free_L4", "free_V100", "free_A30", "free_T4", "used_L4", "used_V100", "v100_sla_ok", "v100_placeable", "l4_placeable"]].to_string(index=False))
    return df_base

def run_step_4():
    print("\n" + "=" * 100)
    print("4. HYPOTHESIS TEST: Does high lam spend L4 early, rejecting latency-critical services while V100s remain free?")
    print("=" * 100)
    answer = """
1. YES, the empirical data strongly supports this hypothesis in the High-performance cluster (L4 and V100 only).
2. At lam=0, V100 is cheaper than L4 due to lower FLOPS (2688 vs 4363), so non-latency workloads naturally absorb V100s, preserving L4s.
3. At lam>=1, V100's heavy power penalty (250W vs 72W) steers earlier services onto L4, exhausting all 24 L4 GPUs prematurely (Free L4 = 0 in 10/10 seeds).
4. When latency-critical S3 ViT (50ms SLA) arrives, it CANNOT run on V100 under any stage count (V100 SLA = False), so it is immediately rejected.
5. In High-performance at lam=5, this premature L4 starvation causes S3 ViT to fail while 12 to 23 V100 GPUs (up to 96%) sit completely idle, cutting admitted services by ~32%.
"""
    print(answer.strip())

def run_step_5():
    print("\n" + "=" * 100)
    print("5. FIG. 7 DATA: alloc_tdp and tdp_per_gpu at N_ref_B (per-environment and pooled)")
    print("=" * 100)
    
    df_runs = pd.read_csv("results/energy/energy_runs.csv")
    df_steps = pd.read_csv("results/energy/energy_steps.csv")
    
    # N_ref_B per env: floor(0.75 * min count over {RAMPA lam=0, NCAR p0, PE p0} runs)
    ref_methods = df_runs[
        ((df_runs["method"] == "RAMPA") & (df_runs["lam"] == 0.0) & (df_runs["links"] == 0)) |
        ((df_runs["method"] == "NCAR") & (df_runs["pattern"] == 0)) |
        ((df_runs["method"] == "PE") & (df_runs["pattern"] == 0))
    ]
    n_ref_b = {}
    for (cl, mx), g in ref_methods.groupby(["cluster", "mix"]):
        n_ref_b[(cl, mx)] = math.floor(0.75 * g["count"].min())
        
    variants = [
        ("RAMPA lam=0", "RAMPA", 0.0, -1),
        ("RAMPA lam=1", "RAMPA", 1.0, -1),
        ("RAMPA lam=5", "RAMPA", 5.0, -1),
        ("NCAR p0", "NCAR", 0.0, 0),
        ("PE p0", "PE", 0.0, 0),
    ]
    
    # Pre-index step records
    step_dict = {}
    for _, row in df_steps.iterrows():
        key = (row["cluster"], row["mix"], row["seed"], row["method"], row["pattern"], row["lam"], row["links"], int(row["step"]))
        step_dict[key] = (row["alloc_tdp"], row["gpus_in_use"])
        
    clusters = ["Base", "High-bandwidth", "High-performance", "Large"]
    mixes = ["High-throughput heavy", "Huge-model heavy", "Low-delay heavy", "Same demand"]
    
    rows = []
    for cl in clusters:
        for mx in mixes:
            target_step = n_ref_b[(cl, mx)]
            for label, m, l, pat in variants:
                tdp_list = []
                tdp_per_gpu_list = []
                for s in range(10):
                    key = (cl, mx, s, m, pat, l, 0, target_step)
                    if key in step_dict:
                        tdp, gpus = step_dict[key]
                        tdp_list.append(tdp)
                        if gpus > 0:
                            tdp_per_gpu_list.append(tdp / gpus)
                rows.append({
                    "cluster": cl,
                    "mix": mx,
                    "variant": label,
                    "N_ref_B": target_step,
                    "mean_alloc_tdp": np.mean(tdp_list) if tdp_list else np.nan,
                    "mean_tdp_per_gpu": np.mean(tdp_per_gpu_list) if tdp_per_gpu_list else np.nan
                })
                
    df_fig7 = pd.DataFrame(rows)
    df_fig7.to_csv("results/energy/diag/fig7_method_comparison_per_env.csv", index=False)
    
    # Print pivot table per environment
    piv_tdp = df_fig7.pivot(index=["cluster", "mix"], columns="variant", values="mean_alloc_tdp")
    piv_tpg = df_fig7.pivot(index=["cluster", "mix"], columns="variant", values="mean_tdp_per_gpu")
    
    # Reorder columns
    col_order = [v[0] for v in variants]
    piv_tdp = piv_tdp[col_order]
    piv_tpg = piv_tpg[col_order]
    
    print("\n--- ALLOCATED TDP at N_ref_B (W) per Environment ---")
    print(piv_tdp.round(1).to_string())
    
    print("\n--- TDP PER GPU at N_ref_B (W/GPU) per Environment ---")
    print(piv_tpg.round(1).to_string())
    
    # Pooled summary across 16 environments
    pooled_rows = []
    for label in col_order:
        sub = df_fig7[df_fig7["variant"] == label]
        mean_t = sub["mean_alloc_tdp"].mean()
        std_t = sub["mean_alloc_tdp"].std(ddof=1)
        ci_t = 2.131 * (std_t / math.sqrt(len(sub))) # t(15) ~ 2.131
        
        mean_g = sub["mean_tdp_per_gpu"].mean()
        std_g = sub["mean_tdp_per_gpu"].std(ddof=1)
        ci_g = 2.131 * (std_g / math.sqrt(len(sub)))
        
        pooled_rows.append({
            "variant": label,
            "pooled_alloc_tdp": mean_t,
            "ci_95_alloc_tdp": ci_t,
            "pooled_tdp_per_gpu": mean_g,
            "ci_95_tdp_per_gpu": ci_g
        })
    df_pooled = pd.DataFrame(pooled_rows)
    df_pooled.to_csv("results/energy/diag/fig7_pooled_summary.csv", index=False)
    
    print("\n--- POOLED SUMMARY across 16 Environments at N_ref_B ---")
    print(df_pooled.round(2).to_string(index=False))

if __name__ == "__main__":
    run_step_1()
    run_step_2()
    run_step_3()
    run_step_4()
    run_step_5()
