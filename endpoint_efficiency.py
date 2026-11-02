"""endpoint_efficiency.py -- Endpoint and equal-load efficiency comparisons for e-RAMPA."""
import os, sys, math
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Styling
C_RAMPA, C_NCAR, C_PE, INK, GRID = "#E4572E", "#2E86AB", "#8D99AE", "#1F2933", "#E4E7EB"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.edgecolor": "#9AA5B1", "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 11, "figure.dpi": 150
})

def holm_adjust(pvals):
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

def main():
    runs_path = "results/energy/energy_runs.csv"
    steps_path = "results/energy/energy_steps.csv"
    out_dir = "results/energy/diag"
    os.makedirs(out_dir, exist_ok=True)
    
    df_runs = pd.read_csv(runs_path)
    df_steps = pd.read_csv(steps_path)
    
    # =========================================================================
    # 1. Endpoint Analysis (state at first rejection)
    # =========================================================================
    # For every run, take the last step (step == count)
    last_steps = df_steps.loc[df_steps.groupby(["cluster", "mix", "seed", "method", "pattern", "lam", "links"])["step"].idxmax()].copy()
    last_steps["alloc_tdp_end"] = last_steps["alloc_tdp"]
    last_steps["gpus_in_use_end"] = last_steps["gpus_in_use"]
    last_steps["tdp_per_service"] = last_steps["alloc_tdp_end"] / last_steps["step"]
    last_steps["services_per_kW"] = last_steps["step"] / (last_steps["alloc_tdp_end"] / 1000.0)
    
    variants = [
        ("RAMPA lam=0", "RAMPA", -1, 0.0, 0),
        ("RAMPA lam=0.5", "RAMPA", -1, 0.5, 0),
        ("RAMPA lam=1", "RAMPA", -1, 1.0, 0),
        ("RAMPA lam=2", "RAMPA", -1, 2.0, 0),
        ("RAMPA lam=5", "RAMPA", -1, 5.0, 0),
        ("NCAR p0", "NCAR", 0, 0.0, 0),
        ("PE p0", "PE", 0, 0.0, 0),
    ]
    
    end_dfs = []
    for label, m, pat, l, lnk in variants:
        sub = last_steps[(last_steps["method"] == m) & (last_steps["pattern"] == pat) & (last_steps["lam"] == l) & (last_steps["links"] == lnk)].copy()
        sub["variant"] = label
        end_dfs.append(sub)
    df_end = pd.concat(end_dfs, ignore_index=True)
    
    end_metrics = ["alloc_tdp_end", "gpus_in_use_end", "step", "tdp_per_service", "services_per_kW"]
    env_means_end = df_end.groupby(["cluster", "mix", "variant"])[end_metrics].mean().reset_index()
    
    # % change vs RAMPA lam=0
    base_end = env_means_end[env_means_end["variant"] == "RAMPA lam=0"].set_index(["cluster", "mix"])[end_metrics]
    
    pct_rows = []
    for label, _, _, _, _ in variants[1:]:
        cur = env_means_end[env_means_end["variant"] == label].set_index(["cluster", "mix"])[end_metrics]
        pct = (cur - base_end) / base_end * 100.0
        pct["variant"] = label
        pct_rows.append(pct.reset_index())
    df_end_pct_per_env = pd.concat(pct_rows, ignore_index=True)
    
    # Save endpoint per-env CSV
    df_end_all_per_env = pd.merge(env_means_end, df_end_pct_per_env, on=["cluster", "mix", "variant"], suffixes=("", "_pct_vs_rampa0"))
    df_end_all_per_env.to_csv(os.path.join(out_dir, "endpoint_per_env.csv"), index=False)
    
    # Pooled endpoint stats across 16 envs
    pooled_end = []
    comp_labels = [v[0] for v in variants[1:]]
    for m in end_metrics:
        p_raw = []
        rows_m = []
        for label in comp_labels:
            vals = df_end_pct_per_env[df_end_pct_per_env["variant"] == label][m].values
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals, ddof=1))
            ci_v = float(stats.t.ppf(0.975, df=len(vals) - 1) * (std_v / math.sqrt(len(vals))))
            w_res = stats.wilcoxon(vals, alternative="two-sided")
            p_val = float(w_res.pvalue)
            p_raw.append(p_val)
            rows_m.append({
                "variant": label,
                "metric": m,
                "mean_pct": mean_v,
                "ci_95": ci_v,
                "p_raw": p_val
            })
        p_adj = holm_adjust(p_raw)
        for i, padj in enumerate(p_adj):
            rows_m[i]["p_holm"] = padj
        pooled_end.extend(rows_m)
        
    df_end_pooled = pd.DataFrame(pooled_end)
    df_end_pooled.to_csv(os.path.join(out_dir, "endpoint_pooled.csv"), index=False)
    
    # =========================================================================
    # 2. Fig. 7 Equal-Load (N_ref_B) Comparison
    # =========================================================================
    ref_runs = df_runs[
        ((df_runs["method"] == "RAMPA") & (df_runs["lam"] == 0.0) & (df_runs["links"] == 0)) |
        ((df_runs["method"] == "NCAR") & (df_runs["pattern"] == 0)) |
        ((df_runs["method"] == "PE") & (df_runs["pattern"] == 0))
    ]
    n_ref_b = {}
    for (cl, mx), g in ref_runs.groupby(["cluster", "mix"]):
        n_ref_b[(cl, mx)] = math.floor(0.75 * g["count"].min())
        
    b_records = []
    step_grouped = df_steps.groupby(["cluster", "mix", "seed", "method", "pattern", "lam", "links"])
    for (cl, mx, seed, m, pat, l, lnk), g in step_grouped:
        tgt = n_ref_b[(cl, mx)]
        sub = g[g["step"] == tgt]
        if not sub.empty:
            r = sub.iloc[0]
            b_records.append({
                "cluster": cl, "mix": mx, "seed": seed, "method": m, "pattern": pat, "lam": l, "links": lnk,
                "alloc_tdp": r["alloc_tdp"], "gpus_in_use": r["gpus_in_use"],
                "tdp_per_gpu": r["alloc_tdp"] / r["gpus_in_use"] if r["gpus_in_use"] > 0 else np.nan,
                "services_per_kW": tgt / (r["alloc_tdp"] / 1000.0)
            })
    df_b = pd.DataFrame(b_records)
    
    b_v_dfs = []
    for label, m, pat, l, lnk in variants:
        sub = df_b[(df_b["method"] == m) & (df_b["pattern"] == pat) & (df_b["lam"] == l) & (df_b["links"] == lnk)].copy()
        sub["variant"] = label
        b_v_dfs.append(sub)
    df_b_var = pd.concat(b_v_dfs, ignore_index=True)
    
    b_metrics = ["alloc_tdp", "tdp_per_gpu", "services_per_kW"]
    env_means_b = df_b_var.groupby(["cluster", "mix", "variant"])[b_metrics].mean().reset_index()
    base_b = env_means_b[env_means_b["variant"] == "RAMPA lam=0"].set_index(["cluster", "mix"])[b_metrics]
    
    b_pct_rows = []
    for label, _, _, _, _ in variants[1:]:
        cur = env_means_b[env_means_b["variant"] == label].set_index(["cluster", "mix"])[b_metrics]
        pct = (cur - base_b) / base_b * 100.0
        pct["variant"] = label
        b_pct_rows.append(pct.reset_index())
    df_b_pct_per_env = pd.concat(b_pct_rows, ignore_index=True)
    
    df_b_all_per_env = pd.merge(env_means_b, df_b_pct_per_env, on=["cluster", "mix", "variant"], suffixes=("", "_pct_vs_rampa0"))
    df_b_all_per_env.to_csv(os.path.join(out_dir, "fig7_equal_load_pct_per_env.csv"), index=False)
    
    pooled_b = []
    for m in b_metrics:
        p_raw = []
        rows_m = []
        for label in comp_labels:
            vals = df_b_pct_per_env[df_b_pct_per_env["variant"] == label][m].values
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals, ddof=1))
            ci_v = float(stats.t.ppf(0.975, df=len(vals) - 1) * (std_v / math.sqrt(len(vals))))
            w_res = stats.wilcoxon(vals, alternative="two-sided")
            p_val = float(w_res.pvalue)
            p_raw.append(p_val)
            rows_m.append({
                "variant": label,
                "metric": m,
                "mean_pct": mean_v,
                "ci_95": ci_v,
                "p_raw": p_val
            })
        p_adj = holm_adjust(p_raw)
        for i, padj in enumerate(p_adj):
            rows_m[i]["p_holm"] = padj
        pooled_b.extend(rows_m)
        
    df_b_pooled = pd.DataFrame(pooled_b)
    df_b_pooled.to_csv(os.path.join(out_dir, "fig7_equal_load_pct_pooled.csv"), index=False)
    
    # =========================================================================
    # 3. Plots
    # =========================================================================
    # Plot 1: Endpoint Efficiency (% change in services/kW and alloc_tdp_end)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(comp_labels))
    colors = [C_RAMPA, "#D9381E", "#900C3F", "#581845", C_NCAR, C_PE]
    
    sub_spk = df_end_pooled[df_end_pooled["metric"] == "services_per_kW"]
    ax1.bar(x, sub_spk["mean_pct"], yerr=sub_spk["ci_95"], capsize=4, color=colors, width=0.55)
    ax1.axhline(0, color="gray", ls="--", lw=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(comp_labels, rotation=25, ha="right", fontsize=9)
    ax1.set_ylabel("% Δ in Services admitted / kW vs RAMPA λ=0")
    ax1.set_title("Endpoint Services per kW (% Δ)")
    ax1.grid(axis="y", color=GRID, lw=0.8)
    
    sub_tdp = df_end_pooled[df_end_pooled["metric"] == "alloc_tdp_end"]
    ax2.bar(x, sub_tdp["mean_pct"], yerr=sub_tdp["ci_95"], capsize=4, color=colors, width=0.55)
    ax2.axhline(0, color="gray", ls="--", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(comp_labels, rotation=25, ha="right", fontsize=9)
    ax2.set_ylabel("% Δ in Total Allocated TDP at Rejection")
    ax2.set_title("Endpoint Allocated Power (% Δ)")
    ax2.grid(axis="y", color=GRID, lw=0.8)
    
    fig.suptitle("Endpoint Full-Capacity Efficiency Comparison (% Δ vs RAMPA λ=0)", y=0.98, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_endpoint_efficiency.png"))
    plt.close(fig)
    
    # Plot 2: Equal-Load (N_ref_B) Comparison (% change)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    
    sub_b_tdp = df_b_pooled[df_b_pooled["metric"] == "alloc_tdp"]
    ax1.bar(x, sub_b_tdp["mean_pct"], yerr=sub_b_tdp["ci_95"], capsize=4, color=colors, width=0.55)
    ax1.axhline(0, color="gray", ls="--", lw=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(comp_labels, rotation=25, ha="right", fontsize=9)
    ax1.set_ylabel("% Δ in Allocated TDP at $N_{ref,B}$")
    ax1.set_title("Allocated Power at Equal Load (% Δ)")
    ax1.grid(axis="y", color=GRID, lw=0.8)
    
    sub_b_tpg = df_b_pooled[df_b_pooled["metric"] == "tdp_per_gpu"]
    ax2.bar(x, sub_b_tpg["mean_pct"], yerr=sub_b_tpg["ci_95"], capsize=4, color=colors, width=0.55)
    ax2.axhline(0, color="gray", ls="--", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(comp_labels, rotation=25, ha="right", fontsize=9)
    ax2.set_ylabel("% Δ in TDP per Active GPU at $N_{ref,B}$")
    ax2.set_title("TDP per Active GPU (% Δ)")
    ax2.grid(axis="y", color=GRID, lw=0.8)
    
    fig.suptitle("Figure 7 (Re-expressed): Equal-Load Efficiency (% Δ vs RAMPA λ=0)", y=0.98, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig7_equal_load_pct.png"))
    plt.close(fig)
    
    # =========================================================================
    # 4. Append table to SUMMARY.md (at most 30 lines)
    # =========================================================================
    summary_path = os.path.join(out_dir, "SUMMARY.md")
    
    # Format table lines
    lines = [
        "",
        "## 5. Relative Efficiency Comparison (% Change vs RAMPA λ=0, Pooled over 16 Envs)",
        "| Variant | Endpoint Svc/kW (%Δ ± 95% CI) | Endpoint TDP (%Δ) | Equal-Load TDP (%Δ) | Equal-Load TDP/GPU (%Δ) | Holm p (Svc/kW) | Holm p (Eq-TDP) |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    
    for label in comp_labels:
        spk = df_end_pooled[(df_end_pooled["variant"] == label) & (df_end_pooled["metric"] == "services_per_kW")].iloc[0]
        end_t = df_end_pooled[(df_end_pooled["variant"] == label) & (df_end_pooled["metric"] == "alloc_tdp_end")].iloc[0]
        eq_t = df_b_pooled[(df_b_pooled["variant"] == label) & (df_b_pooled["metric"] == "alloc_tdp")].iloc[0]
        eq_g = df_b_pooled[(df_b_pooled["variant"] == label) & (df_b_pooled["metric"] == "tdp_per_gpu")].iloc[0]
        
        sign_spk = "+" if spk["mean_pct"] > 0 else ""
        sign_endt = "+" if end_t["mean_pct"] > 0 else ""
        sign_eqt = "+" if eq_t["mean_pct"] > 0 else ""
        sign_eqg = "+" if eq_g["mean_pct"] > 0 else ""
        
        row_str = f"| {label} | {sign_spk}{spk['mean_pct']:.2f}% ± {spk['ci_95']:.2f}% | {sign_endt}{end_t['mean_pct']:.2f}% | {sign_eqt}{eq_t['mean_pct']:.2f}% | {sign_eqg}{eq_g['mean_pct']:.2f}% | p = {spk['p_holm']:.4f} | p = {eq_t['p_holm']:.4f} |"
        lines.append(row_str)
        
    lines.extend([
        "",
        "*Note: Endpoint is measured at first rejection (`step == count`). Equal-load is measured at common stage $N_{ref,B}$. All statistics are mean % changes relative to RAMPA λ=0 across the 16 environments, with Wilcoxon p-values Holm-adjusted across variants.*"
    ])
    
    # Read existing SUMMARY.md
    with open(summary_path, "r", encoding="utf-8") as f:
        existing_content = f.read()
        
    new_content = existing_content.rstrip() + "\n" + "\n".join(lines) + "\n"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(new_content)

if __name__ == "__main__":
    main()
