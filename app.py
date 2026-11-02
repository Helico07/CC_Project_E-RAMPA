"""
RAMPA Lab -- interactive dashboard.   Run:  streamlit run app.py
Tabs: Overview (+cost explorer) | Live simulator | Paper-style results | Extensions | About
"""
import os
import numpy as np, pandas as pd, streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from core import *
from sim import run_fill
import viz
import energy
import energy_dash

st.set_page_config(page_title="RAMPA Lab", page_icon="RAMPA", layout="wide")
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

@st.cache_data(show_spinner=False)
def load(name):
    p = os.path.join(RES, name)
    return pd.read_csv(p) if os.path.exists(p) else None

@st.cache_data(show_spinner=False)
def simulate(kind, mix, seed, pattern, n_ants, n_iter):
    spec = preset(kind)
    prm = ACOParams(n_ants=n_ants, n_iter=n_iter, extra_stages=6)
    out = {}
    for m in METHODS:
        r = run_fill(spec, mix, seed, m, pattern=pattern, params=prm, snapshots=True)
        log = []
        for rec, a in zip(r.records, r.allocs):
            s = SERVICES[a.svc]
            log.append({"step": rec["step"], "service": s.name, "stages": len(a.racks),
                        "GPU types": "-".join(GPU_TYPES[spec.rack_types[x]].short for x in a.racks),
                        "throughput (tps)": round(1 / a.plan.Tu, 1), "need (tps)": s.sla.min_tps,
                        "latency (ms)": round(a.plan.Tr * 1e3, 1), "limit (ms)": s.sla.max_latency * 1e3,
                        "alloc. cost": round(a.cost)})
        out[m] = dict(count=r.count, records=r.records, log=log,
                      failed=SERVICES[r.failed_svc].name if r.failed_svc is not None else "-")
    return spec, out

@st.cache_data(show_spinner=False)
def simulate_energy(kind, mix, seed, lam, n_ants=20, n_iter=8, extra_stages=6):
    spec = preset(kind)
    prm = ACOParams(n_ants=n_ants, n_iter=n_iter, extra_stages=extra_stages)
    r0 = run_fill(spec, mix, seed, "RAMPA", lam=0.0, params=prm, snapshots=False)
    series0 = energy.alloc_series(r0.records, spec)
    if lam == 0.0:
        rlam = r0
        series_lam = series0
    else:
        rlam = run_fill(spec, mix, seed, "RAMPA", lam=lam, params=prm, snapshots=False)
        series_lam = energy.alloc_series(rlam.records, spec)
    return {
        "count_0": r0.count,
        "count_lam": rlam.count,
        "series_0": series0,
        "series_lam": series_lam,
        "failed_0": SERVICES[r0.failed_svc].name if r0.failed_svc is not None else "-",
        "failed_lam": SERVICES[rlam.failed_svc].name if rlam.failed_svc is not None else "-",
    }

@st.cache_data(show_spinner=False)
def load_energy_data():
    runs_p = os.path.join(RES, "energy", "energy_runs.csv")
    steps_p = os.path.join(RES, "energy", "energy_steps.csv")
    if not os.path.exists(runs_p) or not os.path.exists(steps_p):
        return None, None
    return pd.read_csv(runs_p), pd.read_csv(steps_p)

st.title("RAMPA Lab")
st.caption("Resource-aware DL model partitioning & allocation on heterogeneous GPU clusters — "
           "re-implementation and extension of Ikoma, Ohsita & Murata, IEEE Trans. Cloud Computing 13(4), 2025.")
tab_over, tab_live, tab_res, tab_ext, tab_energy, tab_about = st.tabs(
    ["Overview", "Live simulator", "Paper-style results", "Extensions", "Energy (e-RAMPA)", "About & limitations"])

# ------------------------------------------------------------------------------------------ Overview
with tab_over:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("The idea in one minute")
        st.markdown("""
Big models are split into **pipeline stages**, one GPU per stage. The classic goal is *maximum speed for the request in front of you*.
In a shared cluster that is short-sighted: it burns the best GPUs and scatters the free ones, so later requests are **rejected even though GPUs are idle**.

**RAMPA** flips the objective: *meet the SLA (throughput, latency, memory) with the least valuable resources*.
Every GPU and network link gets a **cost = how important it is for future services**, and an ant-colony search finds the cheapest placement
that still satisfies the SLA — including *how many* stages to use.
""")
        st.markdown("""
| | PipeEdge (PE) | NCAR | RAMPA |
|---|---|---|---|
| Objective | max performance | min resource cost | min resource cost |
| #stages | fixed beforehand | fixed beforehand | **searched** |
""")
    with c2:
        st.subheader("Cost explorer  (Eq. 4)")
        st.caption("C_g = f_g · m_g · q_g — the GPU gets cheaper as its rack fills up, so RAMPA drains half-used racks first.")
        t = st.selectbox("GPU type", GPU_TYPES, format_func=lambda g: g.name)
        q = st.slider("free GPUs in the rack (q_g)", 1, 8, 6)
        cost = t.flops / 1e12 * t.mem / 1e9 * q
        st.metric("allocation cost C_g", f"{cost:,.0f}", help="TFLOPS × GB × free GPUs in rack")
        fig, ax = plt.subplots(figsize=(4.2, 2.2))
        qs = np.arange(1, 9)
        ax.bar(qs, t.flops / 1e12 * t.mem / 1e9 * qs, color=["#E4572E" if x == q else "#CBD2D9" for x in qs])
        ax.set_xlabel("free GPUs in rack"); ax.set_ylabel("C_g"); viz._style(ax); fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
    st.subheader("Workload (paper Table V)")
    st.dataframe(pd.DataFrame([{"service": s.name, "category": s.category, "blocks": s.model.n,
                                "min throughput (tps)": s.sla.min_tps, "max latency (s)": s.sla.max_latency,
                                "model memory (GB)": round(s.model.cumW[-1] / 1e9, 1)} for s in SERVICES]),
                 hide_index=True, width="stretch")

# ------------------------------------------------------------------------------------------ Live simulator
with tab_live:
    st.subheader("Same request stream, three allocators")
    a, b, c, d = st.columns(4)
    kind = a.selectbox("Cluster", CLUSTER_KINDS, index=0)
    mix = b.selectbox("Demand mix", list(MIXES), index=0)
    seed = c.number_input("Random seed", 0, 99, 1)
    pattern = d.selectbox("Baseline stage pattern", [0, 1, 2], format_func=lambda p: f"n_min + {p}",
                          help="NCAR/PE use a fixed number of stages = fewest feasible stages + this offset")
    with st.expander("Search effort (ACO)"):
        n_ants = st.slider("ants per iteration", 6, 40, 20); n_iter = st.slider("iterations", 2, 15, 8)
    with st.spinner("Running RAMPA, NCAR and PipeEdge on the same arrivals…"):
        spec, res = simulate(kind, mix, int(seed), int(pattern), n_ants, n_iter)
    m1, m2, m3 = st.columns(3)
    for col, m, lab in zip((m1, m2, m3), ("RAMPA", "NCAR", "PE"), ("RAMPA", "NCAR (cost, fixed stages)", "PipeEdge (perf, fixed stages)")):
        delta = None if m == "RAMPA" else f"{res[m]['count'] - res['RAMPA']['count']:+d} vs RAMPA"
        col.metric(lab + " — services admitted", res[m]["count"], delta, delta_color="off")
        col.caption(f"first rejected: {res[m]['failed']}")
    maxn = max(res[m]["count"] for m in METHODS)
    step = st.slider("Replay: after this many admitted services", 1, max(1, maxn), min(max(1, maxn), 10))
    cols = st.columns(3)
    for col, m, lab in zip(cols, ("PE", "NCAR", "RAMPA"), ("PipeEdge", "NCAR", "RAMPA")):
        recs = res[m]["records"]
        with col:
            if not recs:
                st.write("no service admitted"); continue
            k = min(step, len(recs)) - 1
            rec = recs[k]
            title = f"{lab}: step {rec['step']}" + ("" if step <= len(recs) else f"  (stopped at {len(recs)})")
            st.pyplot(viz.plot_rack_grid(spec, rec["free"], title, new_racks=set(rec["racks"]) if step <= len(recs) else ()), clear_figure=True)
            st.caption(f"GPUs busy {100 * rec['util']:.0f}% · fully-free racks {rec['full_free']}")
    st.caption("Each tile is a rack (letter = GPU type, bar = free GPUs). Outlined racks received the newest service. "
               "Watch PipeEdge empty the A/B racks first.")
    f1, f2 = st.columns(2)
    for ax_col, key, ylabel in ((f1, "util", "GPU utilisation (%)"), (f2, "full_free", "fully-free racks")):
        fig, ax = plt.subplots(figsize=(5.2, 2.8))
        for m, colr in (("PE", viz.C_PE), ("NCAR", viz.C_NCAR), ("RAMPA", viz.C_RAMPA)):
            r = res[m]["records"]
            if r:
                ax.plot([x["step"] for x in r], [x[key] * (100 if key == "util" else 1) for x in r], color=colr, lw=2.2, label=m)
        ax.set_xlabel("services admitted"); ax.set_ylabel(ylabel); viz._style(ax); ax.legend(frameon=False); fig.tight_layout()
        ax_col.pyplot(fig, clear_figure=True)
    who = st.selectbox("Allocation log of", ["RAMPA", "NCAR", "PE"])
    st.dataframe(pd.DataFrame(res[who]["log"]), hide_index=True, width="stretch", height=260)

# ------------------------------------------------------------------------------------------ Paper-style results
with tab_res:
    fill, stages, frag = load("fill.csv"), load("stages.csv"), load("fragmentation.csv")
    if fill is None:
        st.info("Run `python run_experiments.py` first to create the result files.")
    else:
        st.subheader("Services admitted before the first rejection (paper Fig. 7 analogue)")
        st.caption(f"{fill.seed.nunique()} random seeds per environment · baselines evaluated for 3 fixed-stage patterns (best / average).")
        st.pyplot(viz.plot_main(fill), clear_figure=True)
        s = viz.summarize_fill(fill)
        st.pyplot(viz.plot_gain_heat(fill), clear_figure=True)
        st.dataframe(s.round(1), hide_index=True, width="stretch")
        if stages is not None:
            st.subheader("How many stages does each method use? (Fig. 8 analogue)")
            st.pyplot(viz.plot_stage_hist(stages), clear_figure=True)
        if frag is not None:
            st.subheader("Fragmentation over time")
            st.pyplot(viz.plot_frag(frag), clear_figure=True)

# ------------------------------------------------------------------------------------------ Extensions
with tab_ext:
    noise, margin, dyn = load("noise.csv"), load("margin.csv"), load("dynamic.csv")
    st.markdown("Two extensions that address limitations the paper itself lists: *accurate profiling is assumed* (Sec. V-F) "
                "and *services are never released* (fill-up evaluation only).")
    if noise is None:
        st.info("Run `python run_experiments.py` first.")
    else:
        st.subheader("A. What if the profile is wrong?")
        x1, x2 = st.columns(2)
        x1.pyplot(viz.plot_noise(noise), clear_figure=True)
        x2.pyplot(viz.plot_margin(margin, noise), clear_figure=True)
        st.markdown("RAMPA deliberately leaves **no slack**, so estimation error turns into SLA violations. "
                    "Planning against a tightened SLA (a *safety margin*) removes almost all violations for a modest capacity cost.")
        st.subheader("B. Continuous arrivals and departures")
        st.pyplot(viz.plot_dynamic(dyn), clear_figure=True)
        st.markdown("With services leaving over time, RAMPA keeps rejecting fewer requests at the same offered load and keeps GPUs busier.")

# ------------------------------------------------------------------------------------------ Energy (e-RAMPA)
with tab_energy:
    st.subheader("e-RAMPA: Energy-Aware Partitioning and Allocation")
    st.markdown(r"""
**e-RAMPA** introduces an energy penalty to the GPU allocation cost $\mathcal{C}_g$ so that the ant-colony search
prefers lower-power GPU types:
$$\mathcal{C}_{g,\mathrm{pen}} = \mathcal{C}_g \times \left(1 + \lambda \cdot \frac{\mathrm{TDP}_g}{\mathrm{TDP}_{\max}}\right)$$
where $\mathrm{TDP}_{\max} = 250\,\mathrm{W}$ (V100 PCIe). At $\lambda=0$, the cost function and returned allocations are bit-identical to baseline RAMPA.
""")
    tdp_df = pd.DataFrame([
        {"GPU Type": g.name, "Short": g.short, "Nominal TDP (W)": TDP_W[g.short],
         "Ratio (TDP/TDP_MAX)": round(TDP_W[g.short] / TDP_MAX, 3)}
        for g in GPU_TYPES
    ])
    st.dataframe(tdp_df, hide_index=True)
    st.caption("Note: TDP is a nominal thermal design power proxy reflecting maximum continuous power draw, not dynamic per-kernel instrumented power.")

    # 1. Results Explorer
    st.divider()
    st.subheader("1. Results Explorer")
    df_runs, df_steps = load_energy_data()
    if df_runs is None or df_steps is None:
        st.info("Energy experiment results missing. Run `python run_energy.py` to generate `results/energy/energy_runs.csv` and `energy_steps.csv`.")
    else:
        n_ref_a, n_ref_b = energy_dash.compute_reference_points(df_runs)
        cluster_totals = energy_dash.compute_cluster_totals()
        
        lam_vals = sorted(list(df_runs[df_runs["method"] == "RAMPA"]["lam"].unique()))
        def_idx = lam_vals.index(1.0) if 1.0 in lam_vals else 0
        
        envs_list = sorted(list(set(f"{c} / {m}" for c, m in zip(df_runs["cluster"], df_runs["mix"]))))
        env_options = ["Pooled over 16 environments"] + envs_list
        
        c_lam, c_env = st.columns([1, 2])
        sel_lam = c_lam.selectbox("Energy penalty parameter λ", lam_vals, index=def_idx, key="en_lam_sel")
        sel_env = c_env.selectbox("Environment", env_options, index=0, key="en_env_sel")
        
        exp_data = energy_dash.get_explorer_data(df_runs, df_steps, n_ref_a, sel_env, sel_lam)
        
        st.caption(fr"Equal-load request index: **{exp_data['n_ref_str']}** (defined as $\lfloor 0.75 \times \min_{{\text{{RAMPA runs}}}} \text{{count}} \rfloor$).")
        
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Allocated TDP at Equal Load (%Δ vs λ=0)", f"{exp_data['pct_tdp']:+.2f}%",
                      help="Change in cumulative allocated TDP at N_ref_A")
        m_col2.metric("Services Admitted (%Δ vs λ=0)", f"{exp_data['pct_cnt']:+.2f}%",
                      delta_color="off", help="Change in admission capacity before first rejection")
        
        col_gpus, col_diff = st.columns(2)
        fig_gpus = energy_dash.plot_gpu_types_stacked(exp_data["type_0"], exp_data["type_lam"], sel_lam)
        col_gpus.pyplot(fig_gpus, clear_figure=True)
        fig_diff = energy_dash.plot_cum_tdp_diff(exp_data["diff_series"], sel_lam)
        col_diff.pyplot(fig_diff, clear_figure=True)
        
        # 2. Method comparison at equal load
        st.divider()
        scope_comp = st.selectbox("Scope for method comparison", env_options, index=0, key="en_comp_scope")
        st.subheader(f"2. Method Comparison at Equal Load — {scope_comp}")
        df_comp, n_ref_b_str = energy_dash.get_method_comparison_data(df_runs, df_steps, n_ref_b, scope_comp)
        st.dataframe(df_comp, hide_index=True, width="stretch")
        st.caption(f"Evaluated at common load point {n_ref_b_str}. "
                   "PE places work on the fastest GPUs, which also draw least power, but it rejects requests much earlier and leaves more of the cluster unallocated. "
                   "Its efficiency therefore depends on the idle-power assumption (see the idle-power view).")

        # 3. Idle-power view
        st.divider()
        scope_idle = st.selectbox("Scope for idle sensitivity", env_options, index=0, key="en_idle_scope")
        f_val = st.slider("Unallocated GPU idle leakage fraction f", min_value=0.0, max_value=0.5, value=0.10, step=0.05,
                          help="Fraction of unallocated GPU nominal TDP consumed as static power leakage (key reference values: 0 for no idle power, 0.1 for 10% leakage, 0.3 for 30% leakage).",
                          key="en_idle_f_slider")
        st.subheader(f"3. Idle Power Sensitivity at f = {f_val:.2f} — {scope_idle}")
        st.markdown(r"Total cluster power accounting for static leakage on idle GPUs: "
                    r"$P_{\mathrm{total}} = P_{\mathrm{alloc}} + f \cdot \sum_{\text{free}} \mathrm{TDP}$.")
        
        df_idle = energy_dash.get_idle_efficiency_data(df_steps, cluster_totals, f_val, scope_idle)
        st.dataframe(df_idle, hide_index=True, width="stretch")
        
        pe_pct = energy_dash.get_pe_pooled_idle_pct(df_steps, cluster_totals, f_val)
        st.caption(f"At f = {f_val:.2f}, PipeEdge shows {pe_pct:+.2f}% services/kW vs RAMPA (λ=0) pooled across the 16 environments (state at first rejection).")
        st.caption("Note: Services/kW is the mean of per-run ratios, so it can differ slightly from mean count divided by mean power.")
        
        curve_png = os.path.join(RES, "energy", "diag", "idle_curve_v2.png")
        if os.path.exists(curve_png):
            st.image(curve_png, caption="Two-panel sensitivity curves: (a) Full range comparing all methods, (b) Zoomed e-RAMPA variants with 95% CI bands across the 16 environments (uncorrected).")

    # 4. Live demo
    st.divider()
    st.subheader("4. Live Interactive e-RAMPA Demo")
    st.caption("Run RAMPA (λ=0) and e-RAMPA (chosen λ) on the exact same arrival stream.")
    
    d1, d2, d3, d4 = st.columns(4)
    live_kind = d1.selectbox("Cluster", CLUSTER_KINDS, index=0, key="en_live_kind")
    live_mix = d2.selectbox("Demand mix", list(MIXES), index=0, key="en_live_mix")
    live_seed = d3.number_input("Random seed", 0, 99, 1, key="en_live_seed")
    live_lam = d4.slider("Energy penalty λ", 0.0, 10.0, 1.0, 0.5, key="en_live_lam")
    if live_kind == "Large":
        st.warning("⚠️ Large cluster (216 GPUs) takes ~10–15s to simulate.")
        
    with st.spinner(f"Simulating RAMPA (λ=0) and e-RAMPA (λ={live_lam}) on the same seed..."):
        live_res = simulate_energy(live_kind, live_mix, int(live_seed), float(live_lam))
        
    c_live1, c_live2 = st.columns(2)
    s0_last = live_res["series_0"][-1] if live_res["series_0"] else {"alloc_tdp": 0}
    slam_last = live_res["series_lam"][-1] if live_res["series_lam"] else {"alloc_tdp": 0}
    
    cnt0 = live_res["count_0"]
    cnt_lam = live_res["count_lam"]
    tdp0 = s0_last["alloc_tdp"]
    tdp_lam = slam_last["alloc_tdp"]
    
    cnt_delta = f"{cnt_lam - cnt0:+d} vs λ=0" if (live_lam > 0 and cnt_lam != cnt0) else "0 vs λ=0"
    tdp_delta = f"{(tdp_lam - tdp0)/tdp0 * 100:+.1f}% vs λ=0" if tdp0 > 0 and live_lam > 0 else None
    
    c_live1.metric("RAMPA (λ=0) — Services Admitted", cnt0)
    c_live1.caption(f"First rejected: {live_res['failed_0']} · Final Alloc TDP: {tdp0:,.0f} W")
    
    c_live2.metric(f"e-RAMPA (λ={live_lam}) — Services Admitted", cnt_lam, cnt_delta, delta_color="off")
    c_live2.caption(f"First rejected: {live_res['failed_lam']} · Final Alloc TDP: {tdp_lam:,.0f} W ({tdp_delta or 'baseline'})")
    
    lp1, lp2 = st.columns(2)
    if live_res["series_0"] and live_res["series_lam"]:
        fig_l_tdp = energy_dash.plot_live_comparison(live_res["series_0"], live_res["series_lam"], live_lam)
        lp1.pyplot(fig_l_tdp, clear_figure=True)
        fig_l_gpu = energy_dash.plot_live_gpu_bar(s0_last, slam_last, live_lam)
        lp2.pyplot(fig_l_gpu, clear_figure=True)
        
    # 5. Limitations
    st.divider()
    st.subheader("5. Limitations & Methodological Caveats")
    link_reduction_val = energy_dash.get_link_penalty_reduction_val(RES)
    st.info(f"""
* **Nominal TDP Proxy**: TDP represents maximum continuous thermal design power, not measured dynamic power consumption. Dynamic idle power is modeled as a parameterized fraction $f$ of unallocated GPU TDP.
* **Scaled-Down Topology**: Clusters are scaled down from the original paper (96–216 GPUs vs 640–1440) using synthetic DL model FLOP/memory block profiles.
* **Shared Seed Correlation**: The 16 environments share seeds and arrival streams, and Base and High-bandwidth clusters yield near-identical allocation patterns; p-values across environments are indicative rather than independent.
* **L4 Starvation at High λ**: Diagnostic data indicate that in clusters with only L4 and V100 GPUs (e.g., High-performance) and latency-critical workloads (e.g., ViT with a 50 ms SLA that is physically impossible on V100 in this simulator's synthetic profiles), high λ causes earlier non-latency tasks to consume all L4 racks. Latency-critical tasks are then rejected while V100 GPUs sit idle, reducing admission capacity.
* **Link Penalty**: By default only the GPU cost term is penalised, as in the C++ experiment. Penalising link costs as well (penalize_links=True) gave an additional reduction in allocated power in our sweep ({link_reduction_val} at λ=1.0, read from results/energy/pooled_statistics.csv) with a small effect on services admitted.
* **Search Effort**: Evaluated under a fixed ACO search configuration ($N_{{\\mathrm{{ants}}}}=20, N_{{\\mathrm{{iter}}}}=8, \\text{{extra\\_stages}}=6$).
""")

# ------------------------------------------------------------------------------------------ About
with tab_about:
    st.markdown("""
### What was re-implemented
Eq. 4–6 (GPU / Link Cost, Objective), Eq. 1–3 (Pipeline-Cycle Throughput And Latency), The Memory Constraint, And An ACO Solver With
Cost-Bound Pruning And Performance-Check Rejection. Baselines: PipeEdge-Style And NCAR (As Defined In The Paper's Table VIII).

### Honest simplifications
* Models Are Chains Of 12–16 **blocks** (Layer groups) With Synthetic FLOP/Memory Profiles Sized After The Real Models — Not Profiled On Hardware.
* Clusters Are Scaled Down (96–216 GPUs instead of 640–1440); GPUs in a Rack Are Identical So Ants Choose **racks** (Pheromone Per Rack).
* Paths: Cheapest Shortest Feasible Route With Optical-Circuit Capacity Limits, Instead Of The Paper's Link-Level Ant Walk.
* Partition Search Per Candidate Placement Is A DP (Min Bottleneck, Then Min Communication).
* A Feasibility Pre-Filter Restricts Ants To GPU Types That Can Possibly Host A Given Stage Count.
* Baselines Use 3 Fixed Stage Patterns Instead Of The Paper's 288 Combinations.
* Absolute Numbers Differ From The Paper; The *ranking* RAMPA ≥ NCAR ≥ PipeEdge s what is being reproduced.

### Reproduce
`python tests.py` · `python run_experiments.py` · `streamlit run app.py`
""")