"""idle_curve_v2.py -- Two-panel idle power sensitivity curve for endpoint efficiency (services/kW).

Panel (a): Full range comparing all methods (e-RAMPA, NCAR, PE) across f in [0, 0.5].
Panel (b): Zoomed view of e-RAMPA variants with y in [-3%, +8%].
"""
import os
import math
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core import preset, GPU_TYPES, TDP_W, CLUSTER_KINDS

def holm_adjust(pvals):
    pvals = np.array(pvals)
    n = len(pvals)
    if n == 0:
        return np.array([])
    sorted_indices = np.argsort(pvals)
    sorted_pvals = pvals[sorted_indices]
    adjusted = np.zeros(n)
    running_max = 0.0
    for i, p in enumerate(sorted_pvals):
        mult = n - i
        adj = p * mult
        running_max = max(running_max, adj)
        adjusted[i] = min(1.0, running_max)
    out = np.zeros(n)
    out[sorted_indices] = adjusted
    return out

def find_zero_crossings(f_arr, y_arr):
    crossings = []
    for i in range(len(y_arr) - 1):
        y1, y2 = y_arr[i], y_arr[i+1]
        f1, f2 = f_arr[i], f_arr[i+1]
        if (y1 > 0 and y2 < 0) or (y1 < 0 and y2 > 0):
            # linear interpolation: y = y1 + (f - f1)/(f2 - f1) * (y2 - y1) = 0
            f_star = f1 - y1 * (f2 - f1) / (y2 - y1)
            crossings.append(float(f_star))
        elif y1 == 0:
            crossings.append(float(f1))
    return crossings

def main():
    runs_path = "results/energy/energy_runs.csv"
    steps_path = "results/energy/energy_steps.csv"
    out_dir = "results/energy/diag"
    os.makedirs(out_dir, exist_ok=True)
    
    csv_out_path = os.path.join(out_dir, "idle_curve_v2.csv")
    png_out_path = os.path.join(out_dir, "idle_curve_v2.png")
    
    df_steps = pd.read_csv(steps_path)
    
    # Compute total GPUs per type per cluster
    cluster_totals = {}
    for kind in CLUSTER_KINDS:
        spec = preset(kind)
        counts = {"A": 0, "B": 0, "C": 0, "D": 0}
        for t in spec.rack_types:
            counts[GPU_TYPES[t].short] += spec.gpus_per_rack
        cluster_totals[kind] = counts
        
    # Last step per run (state at first rejection)
    last_steps = df_steps.loc[df_steps.groupby(["cluster", "mix", "seed", "method", "pattern", "lam", "links"])["step"].idxmax()].copy()
    
    # Idle TDP per run (unallocated GPUs * TDP)
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
        ("RAMPA lam=0", "RAMPA", -1, 0.0, 0),
        ("RAMPA lam=0.5", "RAMPA", -1, 0.5, 0),
        ("RAMPA lam=1", "RAMPA", -1, 1.0, 0),
        ("RAMPA lam=5", "RAMPA", -1, 5.0, 0),
        ("NCAR p0", "NCAR", 0, 0.0, 0),
        ("PE p0", "PE", 0, 0.0, 0),
    ]
    
    v_dfs = []
    for label, m, pat, l, lnk in variants:
        sub = last_steps[(last_steps["method"] == m) & (last_steps["pattern"] == pat) & (last_steps["lam"] == l) & (last_steps["links"] == lnk)].copy()
        sub["variant"] = label
        v_dfs.append(sub)
    df_v = pd.concat(v_dfs, ignore_index=True)
    
    f_steps = np.round(np.arange(0.0, 0.51, 0.05), 2)
    comp_labels = ["RAMPA lam=0.5", "RAMPA lam=1", "RAMPA lam=5", "NCAR p0", "PE p0"]
    
    rows = []
    for f in f_steps:
        total_p = df_v["alloc_tdp"] + f * df_v["idle_tdp"]
        df_v["spk"] = df_v["step"] / (total_p / 1000.0)
        
        env_means = df_v.groupby(["cluster", "mix", "variant"])["spk"].mean().reset_index()
        base = env_means[env_means["variant"] == "RAMPA lam=0"].set_index(["cluster", "mix"])["spk"]
        
        p_raw = []
        cur_f_rows = []
        for label in comp_labels:
            cur = env_means[env_means["variant"] == label].set_index(["cluster", "mix"])["spk"]
            pct = (cur - base) / base * 100.0
            vals = pct.values
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals, ddof=1))
            ci_v = float(stats.t.ppf(0.975, df=len(vals) - 1) * (std_v / math.sqrt(len(vals))))
            w_res = stats.wilcoxon(vals, alternative="two-sided")
            p_val = float(w_res.pvalue)
            p_raw.append(p_val)
            cur_f_rows.append({
                "f": f,
                "variant": label,
                "mean_pct": mean_v,
                "ci_95": ci_v,
                "ci_lower": mean_v - ci_v,
                "ci_upper": mean_v + ci_v,
                "p_raw": p_val
            })
        p_adj = holm_adjust(p_raw)
        for i, padj in enumerate(p_adj):
            cur_f_rows[i]["p_holm"] = padj
        rows.extend(cur_f_rows)
        
    df_res = pd.DataFrame(rows)
    df_res.to_csv(csv_out_path, index=False)
    
    # Plotting setup
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.6), dpi=300)
    
    style_dict = {
        "RAMPA lam=0.5": {"color": "#1f77b4", "label": r"e-RAMPA $\lambda=0.5$", "marker": "o", "ls": "-"},
        "RAMPA lam=1":   {"color": "#ff7f0e", "label": r"e-RAMPA $\lambda=1$",   "marker": "s", "ls": "-"},
        "RAMPA lam=5":   {"color": "#d62728", "label": r"e-RAMPA $\lambda=5$",   "marker": "^", "ls": "-"},
        "NCAR p0":       {"color": "#2ca02c", "label": r"NCAR ($p=0$)",         "marker": "D", "ls": "--"},
        "PE p0":         {"color": "#9467bd", "label": r"PE ($p=0$)",           "marker": "v", "ls": ":"},
    }
    
    # -------------------------------------------------------------
    # Panel 1: Full Range (All Methods)
    # -------------------------------------------------------------
    ax1.axhline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.7)
    
    for label in ["PE p0", "NCAR p0", "RAMPA lam=5", "RAMPA lam=1", "RAMPA lam=0.5"]:
        sub = df_res[df_res["variant"] == label].sort_values("f")
        f_arr = sub["f"].values
        y_arr = sub["mean_pct"].values
        ci_arr = sub["ci_95"].values
        st = style_dict[label]
        
        ax1.plot(f_arr, y_arr, label=st["label"], color=st["color"], linestyle=st["ls"],
                 marker=st["marker"], markersize=4.5, linewidth=1.6)
        ax1.fill_between(f_arr, y_arr - ci_arr, y_arr + ci_arr, color=st["color"], alpha=0.15)
        
    # Crossings on Panel 1
    # PE p0 crossover
    sub_pe = df_res[df_res["variant"] == "PE p0"].sort_values("f")
    pe_cross = find_zero_crossings(sub_pe["f"].values, sub_pe["mean_pct"].values)
    if pe_cross:
        f_star_pe = pe_cross[0]
        ax1.plot(f_star_pe, 0, marker="X", markersize=8, color="#9467bd",
                 markeredgecolor="black", markeredgewidth=1.1, zorder=6)
        ax1.annotate(
            r"PE: $f^* \approx " + f"{f_star_pe:.3f}" + r"$",
            xy=(f_star_pe, 0),
            xytext=(0.06, -16),
            fontsize=8.5,
            fontweight="semibold",
            color="#9467bd",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#9467bd", alpha=0.9, lw=1.0),
            arrowprops=dict(arrowstyle="->", color="#9467bd", lw=1.2, shrinkB=4)
        )
        
    # NCAR p0 crossover
    sub_ncar = df_res[df_res["variant"] == "NCAR p0"].sort_values("f")
    ncar_cross = find_zero_crossings(sub_ncar["f"].values, sub_ncar["mean_pct"].values)
    if ncar_cross:
        f_star_ncar = ncar_cross[0]
        ax1.plot(f_star_ncar, 0, marker="X", markersize=8, color="#2ca02c",
                 markeredgecolor="black", markeredgewidth=1.1, zorder=6)
        ax1.annotate(
            r"NCAR: $f^* \approx " + f"{f_star_ncar:.3f}" + r"$",
            xy=(f_star_ncar, 0),
            xytext=(0.26, 15),
            fontsize=8.5,
            fontweight="semibold",
            color="#2ca02c",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#2ca02c", alpha=0.9, lw=1.0),
            arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.2, shrinkB=4)
        )
        
    # e-RAMPA lam=5 crossover
    sub_lam5 = df_res[df_res["variant"] == "RAMPA lam=5"].sort_values("f")
    lam5_cross = find_zero_crossings(sub_lam5["f"].values, sub_lam5["mean_pct"].values)
    if lam5_cross:
        f_star_lam5 = lam5_cross[0]
        ax1.plot(f_star_lam5, 0, marker="X", markersize=8, color="#d62728",
                 markeredgecolor="black", markeredgewidth=1.1, zorder=6)
        ax1.annotate(
            r"e-RAMPA $\lambda=5$: $f^* \approx " + f"{f_star_lam5:.3f}" + r"$",
            xy=(f_star_lam5, 0),
            xytext=(0.26, 7.5),
            fontsize=8.5,
            fontweight="semibold",
            color="#d62728",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#d62728", alpha=0.9, lw=1.0),
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2, shrinkB=4)
        )
        
    ax1.set_xlabel(r"Unallocated GPU Idle Power Fraction $f$", fontsize=10.5)
    ax1.set_ylabel(r"Mean % Change in Services / kW vs RAMPA $\lambda=0$", fontsize=10.5)
    ax1.set_title(r"(a) All Evaluated Methods (Full Range)", fontsize=11.5, fontweight="bold")
    ax1.set_xticks(f_steps)
    ax1.set_xlim(-0.02, 0.52)
    ax1.set_ylim(-28, 48)
    ax1.grid(True, linestyle="--", alpha=0.45)
    ax1.legend(frameon=True, loc="upper right", fontsize=8.8, edgecolor="lightgray")
    
    # -------------------------------------------------------------
    # Panel 2: Zoomed View (e-RAMPA Only)
    # -------------------------------------------------------------
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.7)
    
    for label in ["RAMPA lam=5", "RAMPA lam=1", "RAMPA lam=0.5"]:
        sub = df_res[df_res["variant"] == label].sort_values("f")
        f_arr = sub["f"].values
        y_arr = sub["mean_pct"].values
        ci_arr = sub["ci_95"].values
        st = style_dict[label]
        
        ax2.plot(f_arr, y_arr, label=st["label"], color=st["color"], linestyle=st["ls"],
                 marker=st["marker"], markersize=5, linewidth=1.8)
        ax2.fill_between(f_arr, y_arr - ci_arr, y_arr + ci_arr, color=st["color"], alpha=0.15)
        
    # Crossover on Panel 2 (e-RAMPA lam=5)
    if lam5_cross:
        f_star_lam5 = lam5_cross[0]
        ax2.plot(f_star_lam5, 0, marker="X", markersize=9, color="#d62728",
                 markeredgecolor="black", markeredgewidth=1.2, zorder=6)
        ax2.annotate(
            r"e-RAMPA $\lambda=5$: $f^* \approx " + f"{f_star_lam5:.3f}" + r"$",
            xy=(f_star_lam5, 0),
            xytext=(0.26, -2.0),
            fontsize=8.5,
            fontweight="semibold",
            color="#d62728",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#d62728", alpha=0.9, lw=1.0),
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2, shrinkB=4)
        )
        
    ax2.set_xlabel(r"Unallocated GPU Idle Power Fraction $f$", fontsize=10.5)
    ax2.set_ylabel(r"Mean % Change in Services / kW vs RAMPA $\lambda=0$", fontsize=10.5)
    ax2.set_title(r"(b) e-RAMPA Variants (Zoomed: $y \in [-3\%, +8\%]$)", fontsize=11.5, fontweight="bold")
    ax2.set_xticks(f_steps)
    ax2.set_xlim(-0.02, 0.52)
    ax2.set_ylim(-3.0, 8.0)
    ax2.grid(True, linestyle="--", alpha=0.45)
    ax2.legend(frameon=True, loc="upper right", fontsize=9.2, edgecolor="lightgray")
    
    # Caption at the bottom stating 95% CI across 16 environments uncorrected
    fig.text(
        0.5, 0.015,
        "Note: Shaded bands indicate 95% confidence intervals across the 16 environments (uncorrected). State measured at first admission rejection.",
        ha="center", fontsize=9, style="italic", color="#222222"
    )
    
    plt.subplots_adjust(bottom=0.14, top=0.92, left=0.07, right=0.97, wspace=0.22)
    fig.savefig(png_out_path, dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    main()
