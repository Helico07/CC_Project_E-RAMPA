"""energy_dash.py -- Helper functions, data processing, and visualization for the Energy (e-RAMPA) Streamlit tab."""
import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core import preset, GPU_TYPES, TDP_W, CLUSTER_KINDS
import viz

# Color palette aligned with viz.py
C_RAMPA = "#E4572E"
C_NCAR = "#2E86AB"
C_PE = "#8D99AE"
C_LAM = "#F3A712"
TYPE_COL = {"A": "#E4572E", "B": "#F3A712", "C": "#2E86AB", "D": "#6C757D"}

def compute_reference_points(df_runs):
    """Compute N_ref_A (for RAMPA explorer) and N_ref_B (for method comparison)."""
    rampa_runs = df_runs[df_runs["method"] == "RAMPA"]
    n_ref_a = {}
    for (cl, mx), g in rampa_runs.groupby(["cluster", "mix"]):
        n_ref_a[(cl, mx)] = math.floor(0.75 * g["count"].min())
        
    ref_methods = df_runs[
        ((df_runs["method"] == "RAMPA") & (df_runs["lam"] == 0.0) & (df_runs["links"] == 0)) |
        ((df_runs["method"] == "NCAR") & (df_runs["pattern"] == 0)) |
        ((df_runs["method"] == "PE") & (df_runs["pattern"] == 0))
    ]
    n_ref_b = {}
    for (cl, mx), g in ref_methods.groupby(["cluster", "mix"]):
        n_ref_b[(cl, mx)] = math.floor(0.75 * g["count"].min())
        
    return n_ref_a, n_ref_b

def compute_cluster_totals():
    """Return dictionary of total GPUs per type (A, B, C, D) for each cluster preset."""
    cluster_totals = {}
    for kind in CLUSTER_KINDS:
        spec = preset(kind)
        counts = {"A": 0, "B": 0, "C": 0, "D": 0}
        for t in spec.rack_types:
            counts[GPU_TYPES[t].short] += spec.gpus_per_rack
        cluster_totals[kind] = counts
    return cluster_totals

def compute_idle_series(df_steps, cluster_totals):
    """Compute idle TDP for each row in df_steps."""
    idle_tdps = []
    for _, r in df_steps.iterrows():
        tot = cluster_totals[r["cluster"]]
        idle_tdp = (
            (tot["A"] - r["n_A"]) * TDP_W["A"] +
            (tot["B"] - r["n_B"]) * TDP_W["B"] +
            (tot["C"] - r["n_C"]) * TDP_W["C"] +
            (tot["D"] - r["n_D"]) * TDP_W["D"]
        )
        idle_tdps.append(idle_tdp)
    df = df_steps.copy()
    df["idle_tdp"] = idle_tdps
    return df

def get_explorer_data(df_runs, df_steps, n_ref_a, sel_env, sel_lam):
    """Extract metrics and plot series for the Results Explorer."""
    # RAMPA runs with links=0
    r_sub = df_runs[(df_runs["method"] == "RAMPA") & (df_runs["links"] == 0)]
    s_sub = df_steps[(df_steps["method"] == "RAMPA") & (df_steps["links"] == 0)]
    
    envs = sorted(list(set(zip(r_sub["cluster"], r_sub["mix"]))))
    
    if sel_env == "All pooled":
        pct_tdps = []
        pct_cnts = []
        type_0_totals = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
        type_lam_totals = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
        
        # Min common steps across all envs for cumulative plot
        min_step = min(n_ref_a.values())
        diff_accum = {st: [] for st in range(1, min_step + 1)}
        
        for (cl, mx) in envs:
            n_ref = n_ref_a[(cl, mx)]
            s0 = s_sub[(s_sub["cluster"] == cl) & (s_sub["mix"] == mx) & (s_sub["lam"] == 0.0)]
            slam = s_sub[(s_sub["cluster"] == cl) & (s_sub["mix"] == mx) & (s_sub["lam"] == sel_lam)]
            
            p0 = s0[s0["step"] == n_ref]["alloc_tdp"].mean()
            plam = slam[slam["step"] == n_ref]["alloc_tdp"].mean()
            pct_tdps.append((plam - p0) / p0 * 100.0)
            
            c0 = r_sub[(r_sub["cluster"] == cl) & (r_sub["mix"] == mx) & (r_sub["lam"] == 0.0)]["count"].mean()
            clam = r_sub[(r_sub["cluster"] == cl) & (r_sub["mix"] == mx) & (r_sub["lam"] == sel_lam)]["count"].mean()
            pct_cnts.append((clam - c0) / c0 * 100.0)
            
            # Types at n_ref
            s0_ref = s0[s0["step"] == n_ref]
            slam_ref = slam[slam["step"] == n_ref]
            for t in ["A", "B", "C", "D"]:
                type_0_totals[t] += s0_ref[f"n_{t}"].mean()
                type_lam_totals[t] += slam_ref[f"n_{t}"].mean()
                
            # Diff over steps up to min_step
            s0_m = s0.groupby("step")["alloc_tdp"].mean()
            slam_m = slam.groupby("step")["alloc_tdp"].mean()
            for st in range(1, min_step + 1):
                if st in s0_m and st in slam_m:
                    diff_accum[st].append(slam_m[st] - s0_m[st])
                    
        diff_series = [{"step": st, "diff_w": float(np.mean(diff_accum[st]))} for st in range(1, min_step + 1) if diff_accum[st]]
        n_ref_str = f"per-environment $N_{{\\mathrm{{ref}},A}}$ (13 to 48 requests)"
        
        return {
            "pct_tdp": float(np.mean(pct_tdps)),
            "pct_cnt": float(np.mean(pct_cnts)),
            "n_ref_str": n_ref_str,
            "type_0": type_0_totals,
            "type_lam": type_lam_totals,
            "diff_series": diff_series
        }
    else:
        cl, mx = sel_env.split(" / ")
        n_ref = n_ref_a[(cl, mx)]
        s0 = s_sub[(s_sub["cluster"] == cl) & (s_sub["mix"] == mx) & (s_sub["lam"] == 0.0)]
        slam = s_sub[(s_sub["cluster"] == cl) & (s_sub["mix"] == mx) & (s_sub["lam"] == sel_lam)]
        
        p0 = s0[s0["step"] == n_ref]["alloc_tdp"].mean()
        plam = slam[slam["step"] == n_ref]["alloc_tdp"].mean()
        pct_tdp = (plam - p0) / p0 * 100.0
        
        c0 = r_sub[(r_sub["cluster"] == cl) & (r_sub["mix"] == mx) & (r_sub["lam"] == 0.0)]["count"].mean()
        clam = r_sub[(r_sub["cluster"] == cl) & (r_sub["mix"] == mx) & (r_sub["lam"] == sel_lam)]["count"].mean()
        pct_cnt = (clam - c0) / c0 * 100.0
        
        s0_ref = s0[s0["step"] == n_ref]
        slam_ref = slam[slam["step"] == n_ref]
        type_0 = {t: float(s0_ref[f"n_{t}"].mean()) for t in ["A", "B", "C", "D"]}
        type_lam = {t: float(slam_ref[f"n_{t}"].mean()) for t in ["A", "B", "C", "D"]}
        
        s0_m = s0.groupby("step")["alloc_tdp"].mean()
        slam_m = slam.groupby("step")["alloc_tdp"].mean()
        common_steps = sorted(list(set(s0_m.index).intersection(set(slam_m.index))))
        diff_series = [{"step": st, "diff_w": float(slam_m[st] - s0_m[st])} for st in common_steps]
        
        return {
            "pct_tdp": float(pct_tdp),
            "pct_cnt": float(pct_cnt),
            "n_ref_str": f"{n_ref} requests",
            "type_0": type_0,
            "type_lam": type_lam,
            "diff_series": diff_series
        }

def get_method_comparison_data(df_runs, df_steps, n_ref_b, sel_env):
    """Return method comparison table at equal load point N_ref_B."""
    variants = [
        (r"RAMPA ($\lambda=0$)", "RAMPA", -1, 0.0, 0),
        (r"e-RAMPA ($\lambda=1$)", "RAMPA", -1, 1.0, 0),
        (r"e-RAMPA ($\lambda=5$)", "RAMPA", -1, 5.0, 0),
        (r"NCAR ($p=0$)", "NCAR", 0, 0.0, 0),
        (r"PipeEdge ($p=0$)", "PE", 0, 0.0, 0),
    ]
    
    envs = sorted(list(set(zip(df_runs["cluster"], df_runs["mix"]))))
    
    if sel_env == "All pooled":
        # Average metrics across environments
        rows = []
        base_tdp_map = {}
        for (cl, mx) in envs:
            nb = n_ref_b[(cl, mx)]
            s0 = df_steps[(df_steps["method"] == "RAMPA") & (df_steps["lam"] == 0.0) & (df_steps["links"] == 0) & 
                          (df_steps["cluster"] == cl) & (df_steps["mix"] == mx) & (df_steps["step"] == nb)]
            base_tdp_map[(cl, mx)] = s0["alloc_tdp"].mean()
            
        for label, m, pat, l, lnk in variants:
            cnt_list, tdp_list, tpg_list, pct_list = [], [], [], []
            for (cl, mx) in envs:
                nb = n_ref_b[(cl, mx)]
                sub_r = df_runs[(df_runs["method"] == m) & (df_runs["pattern"] == pat) & (df_runs["lam"] == l) &
                                (df_runs["links"] == lnk) & (df_runs["cluster"] == cl) & (df_runs["mix"] == mx)]
                sub_s = df_steps[(df_steps["method"] == m) & (df_steps["pattern"] == pat) & (df_steps["lam"] == l) &
                                 (df_steps["links"] == lnk) & (df_steps["cluster"] == cl) & (df_steps["mix"] == mx) & 
                                 (df_steps["step"] == nb)]
                c = sub_r["count"].mean()
                p = sub_s["alloc_tdp"].mean()
                g = sub_s["gpus_in_use"].mean()
                cnt_list.append(c)
                tdp_list.append(p)
                tpg_list.append(p / g if g > 0 else np.nan)
                
                b_tdp = base_tdp_map[(cl, mx)]
                pct_list.append((p - b_tdp) / b_tdp * 100.0)
                
            delta_str = "baseline" if label == r"RAMPA ($\lambda=0$)" else f"{np.mean(pct_list):+.2f}%"
            rows.append({
                "Method": label,
                "Services Admitted": round(float(np.mean(cnt_list)), 1),
                "Allocated TDP (W)": round(float(np.mean(tdp_list)), 1),
                "TDP per GPU (W)": round(float(np.mean(tpg_list)), 1),
                "%Δ TDP vs RAMPA": delta_str
            })
        return pd.DataFrame(rows), "pooled $N_{\\mathrm{ref},B}$ (11 to 39 requests)"
    else:
        cl, mx = sel_env.split(" / ")
        nb = n_ref_b[(cl, mx)]
        
        # Get base tdp
        s0 = df_steps[(df_steps["method"] == "RAMPA") & (df_steps["lam"] == 0.0) & (df_steps["links"] == 0) & 
                      (df_steps["cluster"] == cl) & (df_steps["mix"] == mx) & (df_steps["step"] == nb)]
        b_tdp = s0["alloc_tdp"].mean()
        
        rows = []
        for label, m, pat, l, lnk in variants:
            sub_r = df_runs[(df_runs["method"] == m) & (df_runs["pattern"] == pat) & (df_runs["lam"] == l) &
                            (df_runs["links"] == lnk) & (df_runs["cluster"] == cl) & (df_runs["mix"] == mx)]
            sub_s = df_steps[(df_steps["method"] == m) & (df_steps["pattern"] == pat) & (df_steps["lam"] == l) &
                             (df_steps["links"] == lnk) & (df_steps["cluster"] == cl) & (df_steps["mix"] == mx) & 
                             (df_steps["step"] == nb)]
            c = sub_r["count"].mean()
            p = sub_s["alloc_tdp"].mean()
            g = sub_s["gpus_in_use"].mean()
            delta_str = "baseline" if label == r"RAMPA ($\lambda=0$)" else f"{(p - b_tdp) / b_tdp * 100.0:+.2f}%"
            rows.append({
                "Method": label,
                "Services Admitted": round(float(c), 1),
                "Allocated TDP (W)": round(float(p), 1),
                "TDP per GPU (W)": round(float(p / g if g > 0 else np.nan), 1),
                "%Δ TDP vs RAMPA": delta_str
            })
        return pd.DataFrame(rows), f"{nb} requests"

def get_idle_efficiency_data(df_steps, cluster_totals, f_val, sel_env):
    """Compute endpoint services/kW with idle leakage fraction f at first rejection."""
    last_steps = df_steps.loc[df_steps.groupby(["cluster", "mix", "seed", "method", "pattern", "lam", "links"])["step"].idxmax()].copy()
    
    idle_tdps = []
    for _, r in last_steps.iterrows():
        tot = cluster_totals[r["cluster"]]
        idle_tdp = (
            (tot["A"] - r["n_A"]) * TDP_W["A"] +
            (tot["B"] - r["n_B"]) * TDP_W["B"] +
            (tot["C"] - r["n_C"]) * TDP_W["C"] +
            (tot["D"] - r["n_D"]) * TDP_W["D"]
        )
        idle_tdps.append(idle_tdp)
    last_steps["idle_tdp"] = idle_tdps
    
    variants = [
        (r"RAMPA ($\lambda=0$)", "RAMPA", -1, 0.0, 0),
        (r"e-RAMPA ($\lambda=0.5$)", "RAMPA", -1, 0.5, 0),
        (r"e-RAMPA ($\lambda=1$)", "RAMPA", -1, 1.0, 0),
        (r"e-RAMPA ($\lambda=2$)", "RAMPA", -1, 2.0, 0),
        (r"e-RAMPA ($\lambda=5$)", "RAMPA", -1, 5.0, 0),
        (r"NCAR ($p=0$)", "NCAR", 0, 0.0, 0),
        (r"PipeEdge ($p=0$)", "PE", 0, 0.0, 0),
    ]
    
    v_dfs = []
    for label, m, pat, l, lnk in variants:
        sub = last_steps[(last_steps["method"] == m) & (last_steps["pattern"] == pat) & (last_steps["lam"] == l) & (last_steps["links"] == lnk)].copy()
        sub["variant"] = label
        v_dfs.append(sub)
    df_v = pd.concat(v_dfs, ignore_index=True)
    
    df_v["total_power"] = df_v["alloc_tdp"] + f_val * df_v["idle_tdp"]
    df_v["spk"] = df_v["step"] / (df_v["total_power"] / 1000.0)
    
    if sel_env == "All pooled":
        env_means = df_v.groupby(["cluster", "mix", "variant"])[["step", "total_power", "spk"]].mean().reset_index()
        base = env_means[env_means["variant"] == r"RAMPA ($\lambda=0$)"].set_index(["cluster", "mix"])["spk"]
        
        rows = []
        for label, _, _, _, _ in variants:
            sub = env_means[env_means["variant"] == label].set_index(["cluster", "mix"])
            c = sub["step"].mean()
            p = sub["total_power"].mean()
            s = sub["spk"].mean()
            if label == r"RAMPA ($\lambda=0$)":
                pct_str = "baseline"
            else:
                pct = (sub["spk"] - base) / base * 100.0
                pct_str = f"{pct.mean():+.2f}%"
            rows.append({
                "Variant": label,
                "Admitted Count": round(float(c), 1),
                "Total Power (W)": round(float(p), 1),
                "Services / kW": round(float(s), 2),
                "%Δ Svc/kW vs RAMPA": pct_str
            })
        return pd.DataFrame(rows)
    else:
        cl, mx = sel_env.split(" / ")
        sub_env = df_v[(df_v["cluster"] == cl) & (df_v["mix"] == mx)]
        env_means = sub_env.groupby("variant")[["step", "total_power", "spk"]].mean().reset_index()
        base_s = env_means[env_means["variant"] == r"RAMPA ($\lambda=0$)"]["spk"].iloc[0]
        
        rows = []
        for label, _, _, _, _ in variants:
            sub = env_means[env_means["variant"] == label]
            if sub.empty:
                continue
            c = sub["step"].iloc[0]
            p = sub["total_power"].iloc[0]
            s = sub["spk"].iloc[0]
            pct_str = "baseline" if label == r"RAMPA ($\lambda=0$)" else f"{(s - base_s) / base_s * 100.0:+.2f}%"
            rows.append({
                "Variant": label,
                "Admitted Count": round(float(c), 1),
                "Total Power (W)": round(float(p), 1),
                "Services / kW": round(float(s), 2),
                "%Δ Svc/kW vs RAMPA": pct_str
            })
        return pd.DataFrame(rows)

def plot_gpu_types_stacked(type_0, type_lam, lam_val):
    """Plot stacked bar chart comparing GPU types at equal load between lam=0 and chosen lam."""
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    x = np.array([0, 1])
    labels = [r"$\lambda=0$", rf"$\lambda={lam_val}$"]
    
    types = ["A", "B", "C", "D"]
    names = {"A": "L4", "B": "V100", "C": "A30", "D": "T4"}
    
    bottom = np.zeros(2)
    for t in types:
        vals = np.array([type_0.get(t, 0.0), type_lam.get(t, 0.0)])
        ax.bar(x, vals, bottom=bottom, label=f"{names[t]} ({TDP_W[t]:.0f}W)",
               color=TYPE_COL[t], width=0.52, edgecolor="white", linewidth=0.8)
        bottom += vals
        
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, fontweight="medium")
    ax.set_ylabel("GPUs in Use at Equal Load", fontsize=9.5)
    ax.set_title("GPU Type Breakdown", fontsize=10.5, fontweight="bold")
    viz._style(ax)
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)
    fig.tight_layout()
    return fig

def plot_cum_tdp_diff(diff_series, lam_val):
    """Plot cumulative allocated TDP difference vs lam=0 over request steps."""
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    steps = [d["step"] for d in diff_series]
    diffs = [d["diff_w"] for d in diff_series]
    
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.plot(steps, diffs, color=C_RAMPA, linewidth=2.0, marker="o", markersize=3.5,
            label=rf"$\lambda={lam_val}$ minus $\lambda=0$")
    
    ax.set_xlabel("Admitted Service Index (Requests)", fontsize=9.5)
    ax.set_ylabel("TDP Difference (W)", fontsize=9.5)
    ax.set_title(r"Allocated TDP Difference vs $\lambda=0$", fontsize=10.5, fontweight="bold")
    viz._style(ax)
    ax.legend(frameon=False, loc="best", fontsize=8.5)
    fig.tight_layout()
    return fig

def plot_live_comparison(series_0, series_lam, lam_val):
    """Line chart of allocated TDP vs requests for live simulation."""
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    
    steps_0 = [d["step"] for d in series_0]
    tdp_0 = [d["alloc_tdp"] for d in series_0]
    
    steps_lam = [d["step"] for d in series_lam]
    tdp_lam = [d["alloc_tdp"] for d in series_lam]
    
    ax.plot(steps_0, tdp_0, color=C_NCAR, linewidth=2.0, linestyle="--", label=r"RAMPA ($\lambda=0$)")
    ax.plot(steps_lam, tdp_lam, color=C_RAMPA, linewidth=2.2, label=rf"e-RAMPA ($\lambda={lam_val}$)")
    
    ax.set_xlabel("Admitted Services", fontsize=9.5)
    ax.set_ylabel("Allocated TDP (W)", fontsize=9.5)
    ax.set_title("Allocated TDP Growth Over Requests", fontsize=10.5, fontweight="bold")
    viz._style(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    fig.tight_layout()
    return fig

def plot_live_gpu_bar(counts_0, counts_lam, lam_val):
    """Bar chart comparing final GPU types for live simulation."""
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    types = ["A", "B", "C", "D"]
    names = ["L4", "V100", "A30", "T4"]
    x = np.arange(len(types))
    width = 0.35
    
    vals_0 = [counts_0.get(f"n_{t}", counts_0.get(t, 0)) for t in types]
    vals_lam = [counts_lam.get(f"n_{t}", counts_lam.get(t, 0)) for t in types]
    
    ax.bar(x - width/2, vals_0, width, label=r"$\lambda=0$", color=C_NCAR, alpha=0.85)
    ax.bar(x + width/2, vals_lam, width, label=rf"$\lambda={lam_val}$", color=C_RAMPA, alpha=0.9)
    
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9.5)
    ax.set_ylabel("Final GPUs in Use", fontsize=9.5)
    ax.set_title("Final GPU Allocation by Type", fontsize=10.5, fontweight="bold")
    viz._style(ax)
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)
    fig.tight_layout()
    return fig
