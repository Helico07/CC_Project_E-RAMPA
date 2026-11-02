"""
RAMPA Lab -- interactive dashboard.   Run:  streamlit run app.py
Tabs: Overview (+cost explorer) | Live simulator | Paper-style results | Extensions | About
"""
import os
import numpy as np, pandas as pd, streamlit as st
import matplotlib.pyplot as plt
from core import *
from sim import run_fill
import viz

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

st.title("RAMPA Lab")
st.caption("Resource-aware DL model partitioning & allocation on heterogeneous GPU clusters — "
           "re-implementation and extension of Ikoma, Ohsita & Murata, IEEE Trans. Cloud Computing 13(4), 2025.")
tab_over, tab_live, tab_res, tab_ext, tab_about = st.tabs(
    ["Overview", "Live simulator", "Paper-style results", "Extensions", "About & limitations"])

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