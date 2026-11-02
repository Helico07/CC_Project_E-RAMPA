# CHANGES.md — e-RAMPA branch (`erampa`)

Branch created from commit `2f604d2` (baseline).

---

## 1. Code Changes

### `core.py` (+17 lines, 0 deleted)
- **TDP constants** (after `GPU_TYPES`, lines 51-52):
  ```python
  TDP_W: dict[str, float] = {"A": 72.0, "B": 250.0, "C": 165.0, "D": 70.0}
  TDP_MAX: float = 250.0
  ```
  V100 PCIe = 250 W = TDP_MAX (penalty factor exactly 2.0 at λ=1).
- **`Cluster.__init__`** — added `self.rack_tdp`, `self.lam = 0.0`, `self.penalize_links = False`.
- **`Cluster.costs()`** — calculates `gpu_cost0`, `penal = 1.0 + self.lam * self.rack_tdp / TDP_MAX`, `gpu_cost_pen = gpu_cost0 * penal`.
  `gpu_for_links` is `gpu_cost_pen` if `self.penalize_links` else `gpu_cost0`. Returns `gpu_cost_pen, access_cost, link_cost`.
  Bit-identical to baseline at λ=0.
- `context()` and `_evaluate()` are untouched and automatically use `gpu_cost_pen`.

### `sim.py` (+5 lines, 1 modified)
- `run_fill(..., lam=0.0, penalize_links=False)` passes `lam` and `penalize_links` to `cl`.
- Forces `cl.lam = 0.0` when `method == "PE"`.

### `tests.py` (+168 lines)
- Preserves original tests `test_eq5()`, `test_plans()`, `test_bookkeeping()`.
- Added T1 (`test_lam0_regression`), T2 (`test_penalty_factor`), T3 (`test_energy_accounting`), T4 (`test_pe_ignores_lam`).

### `energy.py` (new file, 56 lines)
- `alloc_series(records, spec)` calculates cumulative `gpus_in_use`, `alloc_tdp` (nominal W), and GPU counts per type `n_A, n_B, n_C, n_D`.

### `diagnose_part_a.py` (new file)
- Diagnostic script for Part A (A1 comparisons, A2 crossover table, A3 cost decomposition).

### `run_energy.py` (new file)
- Experiment runner for the full 16 envs x 10 seeds grid (1760 runs) using 12-core multiprocessing.
- Outputs `results/energy/energy_runs.csv` and `results/energy/energy_steps.csv`.
- Confirmed: 160 / 160 runs at λ=0 RAMPA match baseline `results/fill.csv` bit-identically.

### `analyze_energy.py` (new file)
- Analysis script computing pooled and environment-level statistics, Holm-adjusted p-values, honest reporting checks, and generating Figures 1–7 (PNG + CSV).

---

## 2. Part A: Diagnostics (Why λ=1 and λ=5 Appeared Identical in Smoke Test)

1. **A1 (Breadth across 16 envs × 5 seeds = 80 pairs):**
   - Placements between λ=1 and λ=5 actually **differ in 91.2% (73/80)** of environments and seeds!
   - Services admitted differ in **36.2% (29/80)** of environments and seeds.
   - The first differing request has median step 7 (mean step 9.5).
   - In the smoke test (Base, Same demand, seeds 0–2), early placement steps happened to coincide.

2. **A2 (Crossover confirmation):**
   - At λ=0: D (777.6) < C (1483.2) < B (2688.0) < A (4363.2). V100 is cheaper than L4 due to lower FLOPS.
   - At λ=1.0: D (995.3) < C (2462.1) < B (5376.0) < A (5619.8). V100 remains cheaper than L4.
   - At λ=1.1703: Exact crossover where Cg_pen(L4) = Cg_pen(V100) = 5833.8.
   - At λ≥2.0: D < C < A < B. L4 becomes cheaper than V100.

3. **A3 (Cost Decomposition):**
   - Multi-stage requests in Base cluster: GPU cost is ~5k–21k, but `access_cost` is ~198k–360k.
   - GPU cost is only **1.4% – 5.9%** of total candidate cost for multi-stage requests.
   - When `penalize_links=False`, access costs are unpenalized, so 95%+ of the candidate score is invariant to λ. This causes saturation between moderate and high λ for multi-stage requests.

---

## 3. Part B: Experiment Execution

- Grid: 4 clusters × 4 mixes × 10 seeds × 11 configurations = 1760 runs.
- Sequential estimate: 53.2 minutes; 12-core parallel execution completed in **506.5 seconds (8.44 minutes)**.
- **Verification:** Exactly 160 / 160 (100%) runs for RAMPA λ=0 (links=0) match `results/fill.csv` exact counts.

---

## 4. Part C: Key Statistical Findings & Figures

### Statistical Significance (Pooled across 16 Environments, N_ref_A):
- **`alloc_tdp`**: Decreases significantly across all tested λ > 0.
  - λ=0.25: −2.1% (p_wilcoxon = 0.00043, Holm p = 0.0026, Sign test: 0 pos, 16 neg, p = 0.00003)
  - λ=0.50: −2.5% (p_wilcoxon = 0.00043, Holm p = 0.0026, Sign test: 0 pos, 16 neg, p = 0.00003)
  - λ=1.00: −4.2% (p_wilcoxon = 0.00043, Holm p = 0.0026, Sign test: 0 pos, 16 neg, p = 0.00003)
  - λ=2.00: −6.5% (p_wilcoxon = 0.00043, Holm p = 0.0026, Sign test: 0 pos, 16 neg, p = 0.00003)
  - λ=5.00: −7.7% (p_wilcoxon = 0.00043, Holm p = 0.0026, Sign test: 0 pos, 16 neg, p = 0.00003)
  - λ=10.0: −9.2% (p_wilcoxon = 0.00043, Holm p = 0.0026, Sign test: 0 pos, 16 neg, p = 0.00003)
- **`tdp_per_gpu`**: Decreases by −4.1% at λ=1 and −7.4% at λ=5 (Holm p = 0.00316).
- **`count` (admitted capacity)**: Mean capacity impact is small at moderate λ (−0.97% at λ=1.0, Holm p = 1.000), increasing to −3.5% at λ=5.0 and −3.9% at λ=10.0 (Holm p = 0.24).

### Honest Reporting Checks:
1. **Capacity Drops**: In High-performance / Low-delay heavy, high λ causes significant service drops (e.g., −28.9% at λ=2, −31.5% at λ=5) because penalizing V100/A30 restricts options where latency requires fast/high-bandwidth GPUs.
2. **Allocated TDP**: Falls in all 16 environments for all λ > 0 (0 exceptions).
3. **`penalize_links=True`**: Slightly further reduces TDP by −2.0% (λ=1) and −3.5% (λ=5), with a negligible capacity drop (−0.4 to −0.6 services). 120/160 runs are identical in admitted count.
4. **Saturation**: TDP reduction shows diminishing returns (−4.2% at λ=1, −6.5% at λ=2, −7.7% at λ=5, −9.2% at λ=10).
5. **Mechanism**:
   - V100 (250W) count drops from 13.6 → 12.7 (λ=1) → 11.4 (λ=10) [Δ = −2.2 GPUs].
   - A30 (165W) drops from 11.6 → 11.1 [Δ = −0.4 GPUs].
   - L4 (72W) increases from 17.8 → 18.4 (λ=1) → 19.7 (λ=10) [Δ = +1.9 GPUs].
   - T4 (70W) increases from 3.2 → 3.9 (λ=1) → 3.7 (λ=10) [Δ = +0.5 GPUs].
   - Thus, V100 is directly substituted by low-power L4 and T4 GPUs.

---

## 5. Artifacts and Output Files Produced

### Data files (`results/energy/`):
- `energy_runs.csv`: 1,760 experiment runs
- `energy_steps.csv`: 73,738 per-admission records
- `pooled_statistics.csv`: Summary statistics with Holm-adjusted p-values

### Figures and behind-the-figure CSVs (`results/energy/`):
- `fig1_services_vs_lam.png` & `.csv`: Services vs λ
- `fig2_alloc_tdp_vs_lam.png` & `.csv`: Small-multiples % Δ TDP vs λ
- `fig3_gpu_types_vs_lam.png` & `.csv`: Stacked GPU type usage
- `fig4_cum_tdp_diff_over_requests.png` & `.csv`: Trajectory over request arrival stream
- `fig5_heatmap_alloc_tdp_change.png` & `.csv`: Heatmap of energy savings across all 16 envs
- `fig6_tradeoff_services_vs_tdp.png` & `.csv`: Energy vs capacity tradeoff scatter
- `fig7_method_comparison.png` & `.csv`: Method comparison (RAMPA vs NCAR vs PE) at common reference stage
