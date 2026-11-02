"""viz.py -- all charts (used by the experiment report, the slides, and the Streamlit dashboard)."""
from __future__ import annotations
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import Rectangle
from core import GPU_TYPES, SERVICES, CLUSTER_KINDS, MIXES

C_RAMPA, C_NCAR, C_PE, INK, GRID = "#E4572E", "#2E86AB", "#8D99AE", "#1F2933", "#E4E7EB"
TYPE_COL = {"A": "#E4572E", "B": "#F3A712", "C": "#2E86AB", "D": "#6C757D"}
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#9AA5B1", "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 12, "figure.dpi": 130})

def _style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)

# ------------------------------------------------------------------ main result
def summarize_fill(fill: pd.DataFrame) -> pd.DataFrame:
    """One row per (cluster, mix): RAMPA vs best/avg/worst fixed-stage pattern of NCAR and PE (mean over seeds)."""
    rows = []
    for (cl, mx), g in fill.groupby(["cluster", "mix"]):
        r = dict(cluster=cl, mix=mx)
        rr = g[g.method == "RAMPA"]["count"]
        r["RAMPA"], r["RAMPA_std"] = rr.mean(), rr.std(ddof=0)
        for m in ("NCAR", "PE"):
            per = g[g.method == m].groupby("pattern")["count"].mean()
            r[f"{m}_best"], r[f"{m}_avg"], r[f"{m}_worst"] = per.max(), per.mean(), per.min()
        r["gain_vs_NCAR_best_%"] = 100 * (r["RAMPA"] / r["NCAR_best"] - 1)
        r["gain_vs_PE_best_%"] = 100 * (r["RAMPA"] / r["PE_best"] - 1)
        r["gain_vs_PE_avg_%"] = 100 * (r["RAMPA"] / r["PE_avg"] - 1)
        rows.append(r)
    return pd.DataFrame(rows)

def plot_main(fill: pd.DataFrame, clusters=None):
    s = summarize_fill(fill)
    clusters = clusters or [c for c in CLUSTER_KINDS if c in set(s.cluster)]
    mixes = list(MIXES)
    ncol = 2 if len(clusters) > 1 else 1
    nrow = int(np.ceil(len(clusters) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 3.5 * nrow), squeeze=False)
    series = [("PE_avg", "PipeEdge (avg)", "#C5CBD3"), ("PE_best", "PipeEdge (best)", C_PE),
              ("NCAR_avg", "NCAR (avg)", "#9CCBE0"), ("NCAR_best", "NCAR (best)", C_NCAR), ("RAMPA", "RAMPA", C_RAMPA)]
    for ax, cl in zip(axes.ravel(), clusters):
        d = s[s.cluster == cl].set_index("mix").loc[mixes]
        w = 0.16
        for i, (col, lab, colr) in enumerate(series):
            ax.bar(np.arange(len(mixes)) + (i - 2) * w, d[col], w, color=colr, label=lab,
                   yerr=d["RAMPA_std"] if col == "RAMPA" else None, error_kw=dict(lw=1, capsize=2, ecolor=INK))
        ax.set_xticks(range(len(mixes))); ax.set_xticklabels([m.replace(" heavy", "\nheavy").replace("Same demand", "Same\ndemand") for m in mixes], fontsize=8)
        ax.set_title(f"{cl} cluster"); ax.set_ylabel("services admitted"); _style(ax)
    for ax in axes.ravel()[len(clusters):]:
        ax.axis("off")
    h, l = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig

def plot_gain_heat(fill: pd.DataFrame):
    s = summarize_fill(fill)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    for ax, col, ttl in zip(axes, ["gain_vs_NCAR_best_%", "gain_vs_PE_best_%"], ["RAMPA gain vs NCAR (best pattern)", "RAMPA gain vs PipeEdge (best pattern)"]):
        M = s.pivot(index="cluster", columns="mix", values=col).loc[[c for c in CLUSTER_KINDS if c in set(s.cluster)], list(MIXES)]
        im = ax.imshow(M.values, cmap="YlOrRd", vmin=0)
        ax.set_xticks(range(M.shape[1])); ax.set_xticklabels([m.replace(" heavy", "\nheavy").replace("Same demand", "Same\ndemand") for m in M.columns], fontsize=7)
        ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(M.index, fontsize=8); ax.set_title(ttl, fontsize=10)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M.values[i, j]:+.0f}%", ha="center", va="center", fontsize=9, color=INK)
        ax.spines[:].set_visible(False)
    fig.tight_layout()
    return fig

# ------------------------------------------------------------------ stage counts (Fig. 8 analogue)
def plot_stage_hist(stages: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    for ax, m, c in zip(axes, ["PE", "NCAR", "RAMPA"], [C_PE, C_NCAR, C_RAMPA]):
        d = stages[stages.method == m]["Np"]
        vc = d.value_counts(normalize=True).sort_index() * 100
        ax.bar(vc.index, vc.values, color=c, width=0.7)
        ax.set_title({"PE": "PipeEdge (fixed N_p)", "NCAR": "NCAR (fixed N_p)", "RAMPA": "RAMPA (adaptive N_p)"}[m], fontsize=10)
        ax.set_xlabel("pipeline stages used per service"); ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True)); _style(ax)
    axes[0].set_ylabel("% of admitted services")
    fig.tight_layout()
    return fig

# ------------------------------------------------------------------ fragmentation / utilisation vs step
def plot_frag(frag: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    cols = {"RAMPA": C_RAMPA, "NCAR": C_NCAR, "PE": C_PE}
    nseed = frag.seed.nunique()
    for m in ["PE", "NCAR", "RAMPA"]:
        d = frag[frag.method == m].groupby("step").agg(util=("util", "mean"), ff=("full_free", "mean"), n=("seed", "count"))
        d = d[d.n >= max(1, nseed // 2)]
        axes[0].plot(d.index, 100 * d.util, color=cols[m], lw=2.2, label=m)
        axes[1].plot(d.index, d.ff, color=cols[m], lw=2.2, label=m)
    axes[0].set_ylabel("GPU utilisation (%)"); axes[1].set_ylabel("fully-free racks left")
    for ax in axes: ax.set_xlabel("services admitted so far"); _style(ax)
    axes[0].legend(frameon=False); fig.tight_layout()
    return fig

# ------------------------------------------------------------------ extension A: profiling error
def plot_noise(noise: pd.DataFrame):
    order = [("PE", "PipeEdge", C_PE, "-"), ("NCAR", "NCAR", C_NCAR, "-"), ("RAMPA m=0%", "RAMPA (no margin)", C_RAMPA, "-"),
             ("RAMPA m=10%", "RAMPA + 10% margin", "#2A9D8F", "--"), ("RAMPA m=20%", "RAMPA + 20% margin", "#264653", "--")]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for key, lab, c, ls in order:
        d = noise[noise.variant == key].groupby("noise").violation.mean() * 100
        if len(d): ax.plot(d.index * 100, d.values, color=c, ls=ls, lw=2.4, marker="o", ms=4, label=lab)
    ax.set_xlabel("profiling error  (stage time off by up to ±x %)"); ax.set_ylabel("deployed services violating SLA (%)")
    ax.set_title("Profiling error vs SLA violations"); _style(ax); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig

def plot_margin(margin: pd.DataFrame, noise: pd.DataFrame, at_noise=0.20):
    names = ["RAMPA m=0%", "RAMPA m=5%", "RAMPA m=10%", "RAMPA m=20%", "RAMPA m=30%"]
    names = [n for n in names if n in set(margin.variant)]
    adm = margin.groupby("variant").admitted.mean().reindex(names)
    vio = noise[(noise.noise.round(2) == at_noise)].groupby("variant").violation.mean().reindex(names) * 100
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.bar(range(len(names)), adm.values, color=C_RAMPA, alpha=0.9, width=0.55)
    ax.set_xticks(range(len(names))); ax.set_xticklabels([n.split("=")[1] for n in names])
    ax.set_xlabel("planning safety margin"); ax.set_ylabel("services admitted", color=C_RAMPA); _style(ax)
    ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
    ax2.plot(range(len(names)), vio.values, color=INK, marker="o", lw=2.2)
    ax2.set_ylabel(f"SLA violations at ±{int(at_noise*100)}% error (%)"); ax2.set_ylim(bottom=0)
    ax.set_title("Safety margin: capacity vs reliability")
    fig.tight_layout()
    return fig

# ------------------------------------------------------------------ extension B: dynamic workload
def plot_dynamic(dyn: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    cols = {"RAMPA": C_RAMPA, "NCAR": C_NCAR, "PE": C_PE}
    for m in ["PE", "NCAR", "RAMPA"]:
        d = dyn[dyn.method == m].groupby("load")[["blocking", "util"]].mean()
        axes[0].plot(d.index, 100 * d.blocking, color=cols[m], lw=2.4, marker="o", label=m)
        axes[1].plot(d.index, 100 * d.util, color=cols[m], lw=2.4, marker="o", label=m)
    axes[0].set_ylabel("requests rejected (%)"); axes[1].set_ylabel("average GPU utilisation (%)")
    for ax in axes: ax.set_xlabel("offered load (avg. concurrent services)"); _style(ax)
    axes[0].legend(frameon=False); axes[0].set_title("Blocking probability"); axes[1].set_title("Utilisation")
    fig.tight_layout()
    return fig

# ------------------------------------------------------------------ cluster picture (heatmap of free GPUs per rack)
def plot_rack_grid(spec, free, title="", ax=None, new_racks=()):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(3.6, 3.6 * spec.rows / spec.cols + 0.4))
    for r in range(spec.rows * spec.cols):
        i, j = divmod(r, spec.cols)
        t = GPU_TYPES[spec.rack_types[r]].short
        frac = free[r] / spec.gpus_per_rack
        ax.add_patch(Rectangle((j, spec.rows - 1 - i), 0.92, 0.92, fc="#F5F7FA", ec="#CBD2D9", lw=1))
        ax.add_patch(Rectangle((j, spec.rows - 1 - i), 0.92, 0.92 * frac, fc=TYPE_COL[t], alpha=0.85, ec="none"))
        if r in new_racks:
            ax.add_patch(Rectangle((j, spec.rows - 1 - i), 0.92, 0.92, fc="none", ec=INK, lw=2.2))
        ax.text(j + 0.46, spec.rows - 1 - i + 0.46, f"{t}\n{free[r]}/{spec.gpus_per_rack}", ha="center", va="center", fontsize=11, color=INK, fontweight="bold")
    ax.set_xlim(-0.05, spec.cols); ax.set_ylim(-0.05, spec.rows); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=10)
    return ax.figure
