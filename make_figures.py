"""Create result figures and a summary JSON file for the slides."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import viz
from core import ACOParams, preset
from sim import run_fill


RESULTS_DIR = Path("results")


def save_figure(figure, name):
    """Save a figure in the results directory and close it."""
    output_path = RESULTS_DIR / f"{name}.png"
    figure.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def create_rack_figure(spec, params):
    """Build snapshots of each method after PipeEdge's admission count."""
    methods = {
        "PE": "PipeEdge",
        "NCAR": "NCAR",
        "RAMPA": "RAMPA",
    }
    runs = {
        method: run_fill(
            spec,
            "Same demand",
            1,
            method,
            params=params,
            snapshots=True,
        )
        for method in methods
    }
    service_count = runs["PE"].count
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))

    for axis, (method, label) in zip(axes, methods.items()):
        records = runs[method].records
        record_index = min(service_count, len(records)) - 1
        viz.plot_rack_grid(
            spec,
            records[record_index]["free"],
            f"{label} after {service_count} services",
            ax=axis,
        )

    figure.tight_layout()
    return figure, runs, service_count


def build_summary(
    fill,
    stages,
    noise,
    margin,
    dynamic,
    fragmentation,
    runs,
    service_count,
):
    """Collect the aggregate metrics used by the slides."""
    summary = viz.summarize_fill(fill)

    return {
        "seeds": int(fill.seed.nunique()),
        "mean_gain_vs_ncar_best": float(summary["gain_vs_NCAR_best_%"].mean()),
        "mean_gain_vs_pe_best": float(summary["gain_vs_PE_best_%"].mean()),
        "min_gain_vs_ncar_best": float(summary["gain_vs_NCAR_best_%"].min()),
        "max_gain_vs_ncar_best": float(summary["gain_vs_NCAR_best_%"].max()),
        "min_gain_vs_pe_best": float(summary["gain_vs_PE_best_%"].min()),
        "max_gain_vs_pe_best": float(summary["gain_vs_PE_best_%"].max()),
        "envs": int(len(summary)),
        "envs_rampa_wins_ncar": int((summary["gain_vs_NCAR_best_%"] > 0).sum()),
        "envs_rampa_wins_pe": int((summary["gain_vs_PE_best_%"] > 0).sum()),
        "pe_count_at_demo": service_count,
        "demo_counts": {method: run.count for method, run in runs.items()},
        "main_base": (
            summary[summary.cluster == "Base"]
            .set_index("mix")
            .round(1)
            .to_dict("index")
        ),
        "noise": (
            noise.groupby(["variant", "noise"])
            .violation.mean()
            .unstack()
            .round(4)
            .to_dict("index")
        ),
        "margin_admitted": margin.groupby("variant").admitted.mean().round(1).to_dict(),
        "dynamic": (
            dynamic.groupby(["method", "load"])[["blocking", "util"]]
            .mean()
            .round(4)
            .reset_index()
            .to_dict("records")
        ),
        "frag": (
            fragmentation.groupby(["method", "step"])
            .full_free.mean()
            .round(2)
            .reset_index()
            .to_dict("records")
        ),
        "stage_share": {
            method: (
                stages[stages.method == method]
                .Np.value_counts(normalize=True)
                .sort_index()
                .round(3)
                .to_dict()
            )
            for method in ("PE", "NCAR", "RAMPA")
        },
    }


def main():
    """Generate all figures and write the slide summary."""
    fill = pd.read_csv(RESULTS_DIR / "fill.csv")
    stages = pd.read_csv(RESULTS_DIR / "stages.csv")
    fragmentation = pd.read_csv(RESULTS_DIR / "fragmentation.csv")
    noise = pd.read_csv(RESULTS_DIR / "noise.csv")
    margin = pd.read_csv(RESULTS_DIR / "margin.csv")
    dynamic = pd.read_csv(RESULTS_DIR / "dynamic.csv")

    save_figure(viz.plot_main(fill), "fig_main")
    save_figure(viz.plot_gain_heat(fill), "fig_gain")
    save_figure(viz.plot_stage_hist(stages), "fig_stages")
    save_figure(viz.plot_frag(fragmentation), "fig_frag")
    save_figure(viz.plot_noise(noise), "fig_noise")
    save_figure(viz.plot_margin(margin, noise), "fig_margin")
    save_figure(viz.plot_dynamic(dynamic), "fig_dynamic")

    spec = preset("Base")
    params = ACOParams(n_ants=20, n_iter=8, extra_stages=6)
    rack_figure, runs, service_count = create_rack_figure(spec, params)
    save_figure(rack_figure, "fig_racks")

    summary = build_summary(
        fill,
        stages,
        noise,
        margin,
        dynamic,
        fragmentation,
        runs,
        service_count,
    )
    summary_path = RESULTS_DIR / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=1, default=float),
        encoding="utf-8",
    )
    print("figures + summary.json written")


if __name__ == "__main__":
    main()
