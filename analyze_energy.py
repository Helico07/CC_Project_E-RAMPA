"""analyze_energy.py -- Statistical analysis and visualization for e-RAMPA."""
import os, sys, math
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Styling
C_RAMPA, C_NCAR, C_PE, INK, GRID = "#E4572E", "#2E86AB", "#8D99AE", "#1F2933", "#E4E7EB"
TYPE_COL = {"A": "#E4572E", "B": "#F3A712", "C": "#2E86AB", "D": "#6C757D"} # A=L4, B=V100, C=A30, D=T4
plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.edgecolor": "#9AA5B1", "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 11, "figure.dpi": 150
})

def holm_adjust(pvals):
    """Holm-Bonferroni step-down adjustment."""
    pvals = np.array(pvals)
    n = len(pvals)
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

def load_data():
    runs_path = "results/energy/energy_runs.csv"
    steps_path = "results/energy/energy_steps.csv"
    if not os.path.exists(runs_path) or not os.path.exists(steps_path):
        raise FileNotFoundError(f"Missing CSVs: {runs_path} or {steps_path}")
    df_runs = pd.read_csv(runs_path)
    df_steps = pd.read_csv(steps_path)
    return df_runs, df_steps

def compute_reference_points(df_runs):
    # N_ref_A: per env, floor(0.75 * min count over all RAMPA runs, all lam and seeds, in that env)
    rampa_runs = df_runs[df_runs["method"] == "RAMPA"]
    n_ref_a = {}
    for (cl, mx), g in rampa_runs.groupby(["cluster", "mix"]):
        n_ref_a[(cl, mx)] = math.floor(0.75 * g["count"].min())
        
    # N_ref_B: per env, floor(0.75 * min count over {RAMPA lam=0, NCAR p0, PE p0} runs in that env)
    ref_methods = df_runs[
        ((df_runs["method"] == "RAMPA") & (df_runs["lam"] == 0.0) & (df_runs["links"] == 0)) |
        ((df_runs["method"] == "NCAR") & (df_runs["pattern"] == 0)) |
        ((df_runs["method"] == "PE") & (df_runs["pattern"] == 0))
    ]
    n_ref_b = {}
    for (cl, mx), g in ref_methods.groupby(["cluster", "mix"]):
        n_ref_b[(cl, mx)] = math.floor(0.75 * g["count"].min())
        
    return n_ref_a, n_ref_b

def extract_metrics_at_step(df_steps, df_runs, n_ref_map, rampa_only=True):
    # For each run in df_runs, extract the row from df_steps at step = n_ref_map[(cl, mx)]
    records = []
    
    # Pre-index steps for speed
    grouped = df_steps.groupby(["cluster", "mix", "seed", "method", "pattern", "lam", "links"])
    
    for _, row in df_runs.iterrows():
        key = (row["cluster"], row["mix"], row["seed"], row["method"], row["pattern"], row["lam"], row["links"])
        if key not in grouped.groups:
            continue
        g = grouped.get_group(key)
        target_step = n_ref_map[(row["cluster"], row["mix"])]
        
        step_row = g[g["step"] == target_step]
        if step_row.empty:
            continue
        sr = step_row.iloc[0]
        
        gpus = sr["gpus_in_use"]
        tdp = sr["alloc_tdp"]
        records.append({
            "cluster": row["cluster"],
            "mix": row["mix"],
            "seed": row["seed"],
            "method": row["method"],
            "pattern": row["pattern"],
            "lam": row["lam"],
            "links": row["links"],
            "count": row["count"],
            "step": target_step,
            "gpus_in_use": gpus,
            "alloc_tdp": tdp,
            "tdp_per_gpu": tdp / gpus if gpus > 0 else np.nan,
            "n_A": sr["n_A"],
            "n_B": sr["n_B"],
            "n_C": sr["n_C"],
            "n_D": sr["n_D"],
        })
    return pd.DataFrame(records)

def run_statistics(df_metrics_a):
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS: RAMPA lam vs lam=0 (penalize_links=False)")
    print("=" * 80)
    
    df_p0 = df_metrics_a[df_metrics_a["links"] == 0]
    base = df_p0[df_p0["lam"] == 0.0].set_index(["cluster", "mix", "seed"])
    
    lams = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    metrics = ["alloc_tdp", "tdp_per_gpu", "gpus_in_use", "count"]
    
    envs = sorted(list(set(zip(df_p0["cluster"], df_p0["mix"]))))
    
    # 1. Environment-level analysis
    env_diffs = {m: {l: [] for l in lams} for m in metrics}
    env_pcts = {m: {l: [] for l in lams} for m in metrics}
    
    for l in lams:
        cur = df_p0[df_p0["lam"] == l].set_index(["cluster", "mix", "seed"])
        for cl, mx in envs:
            b_sub = base.loc[(cl, mx)]
            c_sub = cur.loc[(cl, mx)]
            for m in metrics:
                diff = c_sub[m].values - b_sub[m].values
                mean_diff = float(np.mean(diff))
                env_diffs[m][l].append(mean_diff)
                
                pct = (diff / b_sub[m].values) * 100
                env_pcts[m][l].append(float(np.mean(pct)))
                
    # 2. Pooled analysis across 16 environments
    pooled_rows = []
    
    for m in metrics:
        p_raw = []
        for l in lams:
            diffs = np.array(env_diffs[m][l])
            pcts = np.array(env_pcts[m][l])
            n = len(diffs) # 16
            mean_d = np.mean(diffs)
            std_d = np.std(diffs, ddof=1)
            t_crit = stats.t.ppf(0.975, df=n-1)
            ci_low = mean_d - t_crit * (std_d / math.sqrt(n))
            ci_high = mean_d + t_crit * (std_d / math.sqrt(n))
            
            # Wilcoxon signed-rank
            # If all diffs are 0, wilcoxon cannot run
            if np.all(diffs == 0):
                w_stat, w_p = np.nan, 1.0
            else:
                try:
                    w_res = stats.wilcoxon(diffs, alternative="two-sided")
                    w_stat, w_p = w_res.statistic, w_res.pvalue
                except Exception:
                    w_stat, w_p = np.nan, 1.0
                    
            # Sign test (binomial on positive vs negative)
            n_pos = np.sum(diffs > 0)
            n_neg = np.sum(diffs < 0)
            n_eff = n_pos + n_neg
            if n_eff > 0:
                s_res = stats.binomtest(min(n_pos, n_neg), n_eff, p=0.5, alternative="two-sided")
                sign_p = s_res.pvalue
            else:
                sign_p = 1.0
                
            p_raw.append(w_p)
            pooled_rows.append({
                "metric": m,
                "lam": l,
                "mean_diff": mean_d,
                "mean_pct": np.mean(pcts),
                "ci_95_low": ci_low,
                "ci_95_high": ci_high,
                "wilcoxon_stat": w_stat,
                "wilcoxon_p": w_p,
                "sign_test_pos": n_pos,
                "sign_test_neg": n_neg,
                "sign_test_p": sign_p
            })
            
        # Holm adjustment across the lams for this metric
        p_adj = holm_adjust(p_raw)
        # update pooled_rows for this metric
        start_idx = len(pooled_rows) - len(lams)
        for i, padj in enumerate(p_adj):
            pooled_rows[start_idx + i]["wilcoxon_p_holm"] = padj

    df_pooled_stats = pd.DataFrame(pooled_rows)
    print("\n--- Pooled Statistics Across 16 Environments (vs lam=0) ---")
    print(df_pooled_stats[["metric", "lam", "mean_pct", "ci_95_low", "ci_95_high", "wilcoxon_p", "wilcoxon_p_holm", "sign_test_pos", "sign_test_neg"]].to_string(index=False))
    
    df_pooled_stats.to_csv("results/energy/pooled_statistics.csv", index=False)
    return df_pooled_stats, env_diffs, env_pcts

def make_figures(df_runs, df_steps, df_metrics_a, df_metrics_b):
    print("\nGenerating Figures 1 - 7...")
    os.makedirs("results/energy", exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Fig 1: services vs lam (mean +- 95% CI over environments, links=0)
    # -------------------------------------------------------------------------
    rampa_p0 = df_runs[(df_runs["method"] == "RAMPA") & (df_runs["links"] == 0)]
    # Per env, compute mean count across seeds
    env_means = rampa_p0.groupby(["cluster", "mix", "lam"])["count"].mean().reset_index()
    
    fig1_data = []
    lams = sorted(env_means["lam"].unique())
    for l in lams:
        sub = env_means[env_means["lam"] == l]["count"]
        m = sub.mean()
        s = sub.std(ddof=1)
        ci = stats.t.ppf(0.975, df=len(sub)-1) * (s / math.sqrt(len(sub)))
        fig1_data.append({"lam": l, "mean_services": m, "ci_95": ci, "min": sub.min(), "max": sub.max()})
    df_fig1 = pd.DataFrame(fig1_data)
    df_fig1.to_csv("results/energy/fig1_services_vs_lam.csv", index=False)
    
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.errorbar(df_fig1["lam"], df_fig1["mean_services"], yerr=df_fig1["ci_95"],
                fmt="-o", color=C_RAMPA, capsize=4, lw=2, markersize=6)
    ax.set_xlabel("Energy Penalty Weight ($\lambda$)")
    ax.set_ylabel("Services Admitted (Mean ± 95% CI)")
    ax.set_title("Figure 1: Services Admitted vs $\lambda$ (Pooled over 16 Envs)")
    ax.grid(axis="y", color=GRID, lw=0.8)
    fig.tight_layout()
    fig.savefig("results/energy/fig1_services_vs_lam.png")
    plt.close(fig)
    print("  Saved Figure 1")
    
    # -------------------------------------------------------------------------
    # Fig 2: alloc_tdp at N_ref_A vs lam as percent change vs lam=0
    # Small multiples (4x4) and pooled
    # -------------------------------------------------------------------------
    df_p0 = df_metrics_a[df_metrics_a["links"] == 0]
    base_tdp = df_p0[df_p0["lam"] == 0.0].set_index(["cluster", "mix", "seed"])["alloc_tdp"]
    
    df_p0 = df_p0.copy()
    pct_changes = []
    for _, row in df_p0.iterrows():
        b = base_tdp.loc[(row["cluster"], row["mix"], row["seed"])]
        pct_changes.append((row["alloc_tdp"] - b) / b * 100)
    df_p0["pct_change_tdp"] = pct_changes
    
    df_p0.to_csv("results/energy/fig2_alloc_tdp_vs_lam.csv", index=False)
    
    clusters = ["Base", "High-bandwidth", "High-performance", "Large"]
    mixes = ["Same demand", "High-throughput heavy", "Low-delay heavy", "Huge-model heavy"]
    
    fig, axes = plt.subplots(4, 4, figsize=(14, 11), sharex=True, sharey=True)
    for i, cl in enumerate(clusters):
        for j, mx in enumerate(mixes):
            ax = axes[i, j]
            sub = df_p0[(df_p0["cluster"] == cl) & (df_p0["mix"] == mx)]
            mean_pct = sub.groupby("lam")["pct_change_tdp"].mean()
            ax.plot(mean_pct.index, mean_pct.values, "-o", color=C_RAMPA, markersize=4, lw=1.5)
            ax.axhline(0, color="gray", ls="--", lw=0.8)
            ax.set_title(f"{cl}\n{mx}", fontsize=8)
            ax.grid(color=GRID, lw=0.5)
            if i == 3: ax.set_xlabel("$\lambda$")
            if j == 0: ax.set_ylabel("% $\Delta$ TDP")
            
    fig.suptitle("Figure 2: % Change in Allocated TDP at $N_{ref}$ vs $\lambda$ (Small Multiples by Environment)", y=0.99, fontweight="bold")
    fig.tight_layout()
    fig.savefig("results/energy/fig2_alloc_tdp_vs_lam.png")
    plt.close(fig)
    print("  Saved Figure 2")
    
    # -------------------------------------------------------------------------
    # Fig 3: stacked GPU-type usage at N_ref_A vs lam (pooled mean)
    # -------------------------------------------------------------------------
    df_p0_mean = df_p0.groupby("lam")[["n_A", "n_B", "n_C", "n_D"]].mean().reset_index()
    df_p0_mean.to_csv("results/energy/fig3_gpu_types_vs_lam.csv", index=False)
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(df_p0_mean))
    width = 0.55
    
    b_A = np.zeros(len(df_p0_mean))
    b_B = df_p0_mean["n_A"].values
    b_C = b_B + df_p0_mean["n_B"].values
    b_D = b_C + df_p0_mean["n_C"].values
    
    ax.bar(x, df_p0_mean["n_A"], width, label="L4 (A, 72W)", color=TYPE_COL["A"])
    ax.bar(x, df_p0_mean["n_B"], width, bottom=b_B, label="V100 (B, 250W)", color=TYPE_COL["B"])
    ax.bar(x, df_p0_mean["n_C"], width, bottom=b_C, label="A30 (C, 165W)", color=TYPE_COL["C"])
    ax.bar(x, df_p0_mean["n_D"], width, bottom=b_D, label="T4 (D, 70W)", color=TYPE_COL["D"])
    
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in df_p0_mean["lam"]])
    ax.set_xlabel("Energy Penalty Weight ($\lambda$)")
    ax.set_ylabel("Average Active GPUs at $N_{ref}$")
    ax.set_title("Figure 3: Active GPUs by Type at $N_{ref}$ vs $\lambda$ (Pooled Mean)")
    ax.legend(frameon=True, facecolor="white", edgecolor=GRID)
    ax.grid(axis="y", color=GRID, lw=0.8)
    fig.tight_layout()
    fig.savefig("results/energy/fig3_gpu_types_vs_lam.png")
    plt.close(fig)
    print("  Saved Figure 3")
    
    # -------------------------------------------------------------------------
    # Fig 4: cumulative alloc_tdp difference to lam=0 over requests (normalised fraction)
    # -------------------------------------------------------------------------
    # Using df_steps for RAMPA links=0
    # Normalize each run by step / count_lam0 (up to 1.0)
    rampa_steps = df_steps[(df_steps["method"] == "RAMPA") & (df_steps["links"] == 0)]
    
    # Merge step with lam=0 step to compute difference
    base_steps = rampa_steps[rampa_steps["lam"] == 0.0][["cluster", "mix", "seed", "step", "alloc_tdp"]].rename(columns={"alloc_tdp": "tdp_base"})
    merged_steps = pd.merge(rampa_steps, base_steps, on=["cluster", "mix", "seed", "step"])
    merged_steps["tdp_diff"] = merged_steps["alloc_tdp"] - merged_steps["tdp_base"]
    
    # Get total steps (count) of lam=0 for normalisation
    base_counts = df_runs[(df_runs["method"] == "RAMPA") & (df_runs["lam"] == 0.0) & (df_runs["links"] == 0)][["cluster", "mix", "seed", "count"]].rename(columns={"count": "max_step"})
    merged_steps = pd.merge(merged_steps, base_counts, on=["cluster", "mix", "seed"])
    merged_steps["step_frac"] = (merged_steps["step"] / merged_steps["max_step"]).round(2)
    
    # Bin by step_frac (e.g. 0.1, 0.2, ..., 0.8)
    merged_steps["frac_bin"] = pd.cut(merged_steps["step_frac"], bins=np.linspace(0.05, 0.85, 9), labels=np.linspace(0.1, 0.8, 8).round(1))
    
    fig4_data = merged_steps.groupby(["lam", "frac_bin"], observed=False)["tdp_diff"].mean().reset_index()
    fig4_data.to_csv("results/energy/fig4_cum_tdp_diff_over_requests.csv", index=False)
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_lams = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(plot_lams)))
    for idx, l in enumerate(plot_lams):
        sub = fig4_data[fig4_data["lam"] == l]
        ax.plot(sub["frac_bin"].astype(float), sub["tdp_diff"], "-o", label=f"$\lambda$={l}", color=cmap[idx], lw=1.8, markersize=5)
        
    ax.axhline(0, color="black", ls="--", lw=0.8)
    ax.set_xlabel("Normalized Request Progress (Fraction of lam=0 capacity)")
    ax.set_ylabel("Allocated TDP Difference (W vs $\lambda$=0)")
    ax.set_title("Figure 4: Cumulative Energy Trajectory Over Arrival Stream")
    ax.legend(frameon=True, facecolor="white", edgecolor=GRID)
    ax.grid(color=GRID, lw=0.8)
    fig.tight_layout()
    fig.savefig("results/energy/fig4_cum_tdp_diff_over_requests.png")
    plt.close(fig)
    print("  Saved Figure 4")
    
    # -------------------------------------------------------------------------
    # Fig 5: Heatmap of % change in alloc_tdp at lam=1 and lam=5 per environment
    # -------------------------------------------------------------------------
    fig5_rows = []
    for l in [1.0, 5.0]:
        sub = df_p0[df_p0["lam"] == l]
        for cl in clusters:
            for mx in mixes:
                val = sub[(sub["cluster"] == cl) & (sub["mix"] == mx)]["pct_change_tdp"].mean()
                fig5_rows.append({"lam": l, "cluster": cl, "mix": mx, "pct_change_tdp": val})
    df_fig5 = pd.DataFrame(fig5_rows)
    df_fig5.to_csv("results/energy/fig5_heatmap_alloc_tdp_change.csv", index=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, l in [(ax1, 1.0), (ax2, 5.0)]:
        piv = df_fig5[df_fig5["lam"] == l].pivot(index="cluster", columns="mix", values="pct_change_tdp")
        piv = piv.reindex(index=clusters, columns=mixes)
        im = ax.imshow(piv.values, cmap="RdYlGn_r", vmin=-25, vmax=10)
        ax.set_xticks(range(len(mixes)))
        ax.set_xticklabels(["Same", "HT", "LD", "HM"], fontsize=9)
        ax.set_yticks(range(len(clusters)))
        ax.set_yticklabels(clusters, fontsize=9)
        ax.set_title(f"$\lambda$ = {l}")
        for r in range(len(clusters)):
            for c in range(len(mixes)):
                v = piv.iloc[r, c]
                color = "white" if abs(v) > 15 else "black"
                ax.text(c, r, f"{v:+.1f}%", ha="center", va="center", color=color, fontweight="bold", fontsize=8.5)
                
    fig.colorbar(im, ax=[ax1, ax2], fraction=0.03, pad=0.04, label="% Change in Allocated TDP at $N_{ref}$")
    fig.suptitle("Figure 5: Heatmap of Energy Savings (% $\Delta$ TDP at $N_{ref}$)", y=0.98, fontweight="bold")
    fig.tight_layout()
    fig.savefig("results/energy/fig5_heatmap_alloc_tdp_change.png")
    plt.close(fig)
    print("  Saved Figure 5")
    
    # -------------------------------------------------------------------------
    # Fig 6: trade-off scatter: % change in services vs % change in alloc_tdp
    # -------------------------------------------------------------------------
    base_cnt = df_runs[(df_runs["method"] == "RAMPA") & (df_runs["lam"] == 0.0) & (df_runs["links"] == 0)].set_index(["cluster", "mix", "seed"])["count"]
    
    fig6_data = []
    for l in [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]:
        sub_r = df_runs[(df_runs["method"] == "RAMPA") & (df_runs["lam"] == l) & (df_runs["links"] == 0)]
        sub_m = df_p0[df_p0["lam"] == l]
        for cl in clusters:
            for mx in mixes:
                # mean count change
                c_cur = sub_r[(sub_r["cluster"] == cl) & (sub_r["mix"] == mx)]["count"].values
                c_base = base_cnt.loc[(cl, mx)].values
                pct_cnt = float(np.mean((c_cur - c_base) / c_base * 100))
                
                # mean tdp change
                pct_tdp = float(sub_m[(sub_m["cluster"] == cl) & (sub_m["mix"] == mx)]["pct_change_tdp"].mean())
                fig6_data.append({"lam": l, "cluster": cl, "mix": mx, "pct_services": pct_cnt, "pct_tdp": pct_tdp})
                
    df_fig6 = pd.DataFrame(fig6_data)
    df_fig6.to_csv("results/energy/fig6_tradeoff_services_vs_tdp.csv", index=False)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_lams = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(plot_lams)))
    for idx, l in enumerate(plot_lams):
        sub = df_fig6[df_fig6["lam"] == l]
        ax.scatter(sub["pct_tdp"], sub["pct_services"], label=f"$\lambda$={l}", color=cmap[idx], s=40, alpha=0.85)
        
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("% Change in Allocated TDP at $N_{ref}$ (Negative = Less Power)")
    ax.set_ylabel("% Change in Services Admitted (Positive = More Services)")
    ax.set_title("Figure 6: Energy-Capacity Tradeoff Scatter Across Environments")
    ax.legend(frameon=True, facecolor="white", edgecolor=GRID, loc="best")
    ax.grid(color=GRID, lw=0.8)
    fig.tight_layout()
    fig.savefig("results/energy/fig6_tradeoff_services_vs_tdp.png")
    plt.close(fig)
    print("  Saved Figure 6")
    
    # -------------------------------------------------------------------------
    # Fig 7: method comparison at N_ref_B
    # alloc_tdp and tdp_per_gpu for RAMPA lam=0, RAMPA lam=1, RAMPA lam=5, NCAR p0, PE p0
    # -------------------------------------------------------------------------
    fig7_variants = [
        ("RAMPA $\lambda$=0", "RAMPA", 0.0, -1),
        ("RAMPA $\lambda$=1", "RAMPA", 1.0, -1),
        ("RAMPA $\lambda$=5", "RAMPA", 5.0, -1),
        ("NCAR p0", "NCAR", 0.0, 0),
        ("PE p0", "PE", 0.0, 0),
    ]
    
    fig7_rows = []
    for label, m, l, pat in fig7_variants:
        sub = df_metrics_b[
            (df_metrics_b["method"] == m) &
            (df_metrics_b["lam"] == l) &
            (df_metrics_b["pattern"] == pat) &
            (df_metrics_b["links"] == 0)
        ]
        fig7_rows.append({
            "variant": label,
            "mean_tdp": sub["alloc_tdp"].mean(),
            "ci_tdp": stats.t.ppf(0.975, df=len(sub)-1) * (sub["alloc_tdp"].std(ddof=1) / math.sqrt(len(sub))),
            "mean_tdp_per_gpu": sub["tdp_per_gpu"].mean(),
            "ci_tdp_per_gpu": stats.t.ppf(0.975, df=len(sub)-1) * (sub["tdp_per_gpu"].std(ddof=1) / math.sqrt(len(sub))),
            "mean_gpus": sub["gpus_in_use"].mean(),
            "mean_services": sub["count"].mean(),
        })
    df_fig7 = pd.DataFrame(fig7_rows)
    df_fig7.to_csv("results/energy/fig7_method_comparison.csv", index=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    x = np.arange(len(fig7_rows))
    colors = [C_RAMPA, "#D9381E", "#900C3F", C_NCAR, C_PE]
    
    ax1.bar(x, df_fig7["mean_tdp"], yerr=df_fig7["ci_tdp"], capsize=4, color=colors, width=0.55)
    ax1.set_xticks(x)
    ax1.set_xticklabels([r["variant"] for r in fig7_rows], rotation=25, ha="right", fontsize=9)
    ax1.set_ylabel("Total Allocated TDP at $N_{ref,B}$ (W)")
    ax1.set_title("Total Allocated Power (W)")
    ax1.grid(axis="y", color=GRID, lw=0.8)
    
    ax2.bar(x, df_fig7["mean_tdp_per_gpu"], yerr=df_fig7["ci_tdp_per_gpu"], capsize=4, color=colors, width=0.55)
    ax2.set_xticks(x)
    ax2.set_xticklabels([r["variant"] for r in fig7_rows], rotation=25, ha="right", fontsize=9)
    ax2.set_ylabel("Average TDP per GPU (W)")
    ax2.set_title("Average Power per Active GPU (W)")
    ax2.grid(axis="y", color=GRID, lw=0.8)
    
    fig.suptitle("Figure 7: Method Comparison at Common Reference Stage ($N_{ref,B}$)", y=0.98, fontweight="bold")
    fig.tight_layout()
    fig.savefig("results/energy/fig7_method_comparison.png")
    plt.close(fig)
    print("  Saved Figure 7")

def honest_reporting_checks(df_runs, df_p0, df_metrics_a):
    print("\n" + "=" * 80)
    print("HONEST-REPORTING CHECKS")
    print("=" * 80)
    
    clusters = ["Base", "High-bandwidth", "High-performance", "Large"]
    mixes = ["Same demand", "High-throughput heavy", "Low-delay heavy", "Huge-model heavy"]
    
    # 1. Environments where lam>0 REDUCES services admitted
    base_cnt = df_runs[(df_runs["method"] == "RAMPA") & (df_runs["lam"] == 0.0) & (df_runs["links"] == 0)].set_index(["cluster", "mix", "seed"])["count"]
    
    print("\n[CHECK 1] Environments where lam > 0 REDUCES services admitted:")
    reduced_cases = []
    for l in [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]:
        sub = df_runs[(df_runs["method"] == "RAMPA") & (df_runs["lam"] == l) & (df_runs["links"] == 0)]
        for cl in clusters:
            for mx in mixes:
                c_cur = sub[(sub["cluster"] == cl) & (sub["mix"] == mx)]["count"].values
                c_base = base_cnt.loc[(cl, mx)].values
                diff = c_cur - c_base
                mean_diff = np.mean(diff)
                if mean_diff < -0.2: # non-trivial decrease
                    reduced_cases.append((cl, mx, l, mean_diff, float(np.mean(diff/c_base*100))))
                    
    if reduced_cases:
        df_red = pd.DataFrame(reduced_cases, columns=["cluster", "mix", "lam", "mean_delta_services", "pct_delta_services"])
        print(df_red.to_string(index=False))
    else:
        print("  None found where mean delta < -0.2.")
        
    # 2. Environments where alloc_tdp does NOT fall
    print("\n[CHECK 2] Environments where alloc_tdp does NOT fall (at N_ref_A):")
    non_falling_tdp = []
    for l in [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]:
        sub = df_p0[df_p0["lam"] == l]
        for cl in clusters:
            for mx in mixes:
                v = sub[(sub["cluster"] == cl) & (sub["mix"] == mx)]["pct_change_tdp"].mean()
                if v >= 0.0:
                    non_falling_tdp.append((cl, mx, l, v))
    if non_falling_tdp:
        df_nft = pd.DataFrame(non_falling_tdp, columns=["cluster", "mix", "lam", "pct_change_tdp"])
        print(df_nft.to_string(index=False))
    else:
        print("  Allocated TDP decreased across all tested lambda > 0.")
        
    # 3. Whether penalize_links=True behaves differently from False
    print("\n[CHECK 3] Comparison: penalize_links=True vs penalize_links=False (lam in {1, 5}):")
    rampa_runs = df_runs[df_runs["method"] == "RAMPA"]
    link_comp = []
    for l in [1.0, 5.0]:
        sub_f = rampa_runs[(rampa_runs["lam"] == l) & (rampa_runs["links"] == 0)].set_index(["cluster", "mix", "seed"])["count"]
        sub_t = rampa_runs[(rampa_runs["lam"] == l) & (rampa_runs["links"] == 1)].set_index(["cluster", "mix", "seed"])["count"]
        diff = sub_t.values - sub_f.values
        
        # also check TDP at N_ref
        tdp_f = df_metrics_a[(df_metrics_a["lam"] == l) & (df_metrics_a["links"] == 0)].set_index(["cluster", "mix", "seed"])["alloc_tdp"]
        tdp_t = df_metrics_a[(df_metrics_a["lam"] == l) & (df_metrics_a["links"] == 1)].set_index(["cluster", "mix", "seed"])["alloc_tdp"]
        diff_tdp = tdp_t.values - tdp_f.values
        
        link_comp.append({
            "lam": l,
            "mean_count_diff (True - False)": float(np.mean(diff)),
            "count_identical_runs": int(np.sum(diff == 0)),
            "total_runs": len(diff),
            "mean_tdp_diff_W": float(np.mean(diff_tdp)),
            "pct_tdp_diff": float(np.mean(diff_tdp / tdp_f.values * 100))
        })
    df_lc = pd.DataFrame(link_comp)
    print(df_lc.to_string(index=False))
    
    # 4. Whether effects saturate between lam=1 and lam=10
    print("\n[CHECK 4] Saturation analysis (lam=1 vs lam=2 vs lam=5 vs lam=10):")
    sat_data = []
    for l in [1.0, 2.0, 5.0, 10.0]:
        sub_tdp = df_p0[df_p0["lam"] == l]["pct_change_tdp"].mean()
        sub_cnt = df_runs[(df_runs["method"] == "RAMPA") & (df_runs["lam"] == l) & (df_runs["links"] == 0)]["count"].mean()
        sat_data.append({"lam": l, "mean_pct_delta_tdp": sub_tdp, "mean_services": sub_cnt})
    df_sat = pd.DataFrame(sat_data)
    print(df_sat.to_string(index=False))
    
    # 5. Mechanism: change in GPU types at N_ref_A
    print("\n[CHECK 5] Mechanism: GPU Type Allocation Shifts at N_ref_A (Pooled Mean Counts):")
    base_gpus = df_p0[df_p0["lam"] == 0.0][["n_A", "n_B", "n_C", "n_D"]].mean()
    mech_rows = []
    for l in [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]:
        m = df_p0[df_p0["lam"] == l][["n_A", "n_B", "n_C", "n_D"]].mean()
        mech_rows.append({
            "lam": l,
            "L4 (A, 72W)": m["n_A"],
            "V100 (B, 250W)": m["n_B"],
            "A30 (C, 165W)": m["n_C"],
            "T4 (D, 70W)": m["n_D"],
            "delta_L4": m["n_A"] - base_gpus["n_A"],
            "delta_V100": m["n_B"] - base_gpus["n_B"],
            "delta_A30": m["n_C"] - base_gpus["n_C"],
            "delta_T4": m["n_D"] - base_gpus["n_D"],
        })
    df_mech = pd.DataFrame(mech_rows)
    print(df_mech.to_string(index=False))

def main():
    df_runs, df_steps = load_data()
    n_ref_a, n_ref_b = compute_reference_points(df_runs)
    
    print(f"Extracted N_ref_A for 16 environments (min services across all RAMPA runs * 0.75):")
    for (cl, mx), v in n_ref_a.items():
        print(f"  {cl:<18} {mx:<22}: N_ref_A = {v}")
        
    df_metrics_a = extract_metrics_at_step(df_steps, df_runs[df_runs["method"] == "RAMPA"], n_ref_a)
    df_metrics_b = extract_metrics_at_step(df_steps, df_runs, n_ref_b)
    
    df_pooled_stats, env_diffs, env_pcts = run_statistics(df_metrics_a)
    
    make_figures(df_runs, df_steps, df_metrics_a, df_metrics_b)
    
    df_p0 = df_metrics_a[df_metrics_a["links"] == 0]
    base_tdp = df_p0[df_p0["lam"] == 0.0].set_index(["cluster", "mix", "seed"])["alloc_tdp"]
    pct_changes = []
    for _, row in df_p0.iterrows():
        b = base_tdp.loc[(row["cluster"], row["mix"], row["seed"])]
        pct_changes.append((row["alloc_tdp"] - b) / b * 100)
    df_p0 = df_p0.copy()
    df_p0["pct_change_tdp"] = pct_changes
    
    honest_reporting_checks(df_runs, df_p0, df_metrics_a)

if __name__ == "__main__":
    main()
