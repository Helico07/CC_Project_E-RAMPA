# RAMPA Lab

Re-implementation, evaluation and extension of  
**"Resource Aware Deep Learning Model Partitioning and Allocation for Inference Task in Clusters With Heterogeneous GPUs"**  
(Ikoma, Ohsita, Murata — IEEE Transactions on Cloud Computing 13(4), 2025).

## Quick start
```bash
pip install -r requirements.txt
python tests.py                 # 3 structural checks (Eq.5 vs brute force, DP constraints, alloc/release)
python run_experiments.py       # ~10-15 min, writes results/*.csv   (add --quick for a 3-minute run)
python make_figures.py          # writes results/*.png
streamlit run app.py            # interactive dashboard
```
Results from a full run are already included in `results/`, so the dashboard works immediately.

## Files
| file | purpose |
|---|---|
| `core.py` | cluster model, cost functions (Eq. 4-6), performance model (Eq. 1-3), ACO, the three methods (RAMPA, NCAR, PipeEdge-style) |
| `sim.py` | experiment drivers: fill-until-failure, dynamic arrivals/departures, profiling-noise test |
| `run_experiments.py` | runs everything, writes CSVs |
| `viz.py`, `make_figures.py` | all charts |
| `app.py` | Streamlit dashboard (live simulator + results + extensions) |
| `tests.py` | correctness checks |

## What is reproduced / what is new
* **Reproduced**: RAMPA > NCAR > PipeEdge in number of admitted services (paper Fig. 7 analogue), adaptive stage counts (Fig. 8 analogue).
* **New (extensions)**: (A) sensitivity to profiling error and a planning *safety margin*; (B) continuous arrivals **and departures**
  (blocking probability / utilisation); (C) fragmentation timeline.

## Simplifications (be upfront about these)
Synthetic per-block profiles (sized after the real models, not measured); scaled-down clusters; chain-structured models of 12-16 blocks;
ants pick racks (GPUs in a rack are identical); cheapest-shortest-feasible routing instead of the paper's link-level ant walk;
DP for the partition search; a feasibility pre-filter for GPU types; 3 fixed-stage patterns for baselines instead of 288.
