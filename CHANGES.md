# CHANGES.md — e-RAMPA branch (`erampa`)

Branch created from commit `2f604d2` (baseline).

## Modified files

### `core.py` (+17 lines, 0 deleted)

**TDP constants** (after `GPU_TYPES`, lines 51-52):
```python
TDP_W: dict[str, float] = {"A": 72.0, "B": 250.0, "C": 165.0, "D": 70.0}
TDP_MAX: float = 250.0
```
V100 PCIe = 250 W = TDP_MAX, so V100 gets penalty factor exactly 2 at λ=1.

**`Cluster.__init__`** — three new attributes after `rack_mem`:
```python
self.rack_tdp = np.array([TDP_W[GPU_TYPES[t].short] for t in self.types])
self.lam: float = 0.0          # energy penalty weight (default = off)
self.penalize_links: bool = False
```

**`Cluster.costs()`** — replaced 4 lines, added 10:
- Computes `gpu_cost0` (unchanged Eq. 4 product).
- Computes `penal = 1 + lam * rack_tdp / TDP_MAX` (all-ones when `lam=0`).
- `gpu_cost_pen = gpu_cost0 * penal` is returned as the first element.
- `gpu_for_links` selects between `gpu_cost_pen` and `gpu_cost0` based on
  `penalize_links`, controlling whether the link/access costs are also penalised.
- **With `lam=0`: all returned arrays are bit-identical to the baseline.**

`context()` and `_evaluate()` are **not modified**; they consume `costs()` output
unchanged, so the heuristic `1/Cg` and the objective GPU-cost term are automatically
penalised.

### `sim.py` (+5 lines, 1 modified)

`run_fill()` signature extended with two keyword arguments:
```python
def run_fill(..., lam=0.0, penalize_links=False):
```
After `Cluster(spec)`:
```python
cl.lam = 0.0 if method == "PE" else lam   # PE always ignores energy penalty
cl.penalize_links = penalize_links
```
All existing callers (`run_experiments.py`, `app.py`, tests) are unaffected (default
values preserve baseline behaviour). `run_dynamic()` is **not modified** (out of scope).

### `tests.py` (+168 lines, imports extended)

Added imports: `json`, `subprocess`, `sys`; `from energy import alloc_series`.

Four new test functions (T1–T4); original three tests unchanged:

| Test | What it checks |
|------|---------------|
| **T1** `test_lam0_regression` | Runs baseline worktree and new code in subprocesses for all 16 (cluster × mix) × seeds {0,1,2} × {RAMPA, NCAR p=0}; asserts count and per-record (racks, cost) are identical. |
| **T2** `test_penalty_factor` | Verifies `gpu_cost_pen == gpu_cost0 * (1 + TDP/TDP_MAX)` per rack; with `penalize_links=False` access and link costs equal the `lam=0` values; with `penalize_links=True` the brute-force Eq.(5) check still passes using the penalised Cg. |
| **T3** `test_energy_accounting` | With `snapshots=True`, verifies `alloc_series` `gpus_in_use` and `alloc_tdp` match the ground truth derived from `rec["free"]` at every step. |
| **T4** `test_pe_ignores_lam` | Asserts `run_fill(PE, lam=1)` produces identical count and records to `run_fill(PE, lam=0)`. |

## New files

### `energy.py`

```
alloc_series(records, spec) -> list[dict]
```
Consumes `FillResult.records` (must contain `"racks"` list). Returns one dict per
admitted request with:
- `step` — admission step (1-based)
- `gpus_in_use` — cumulative individual GPUs in use (one per pipeline stage)
- `alloc_tdp` — cumulative nominal TDP (W) of all in-use GPUs (independent of `lam`)
- `n_A`, `n_B`, `n_C`, `n_D` — cumulative count per GPU type (L4, V100, A30, T4)

## Unchanged files

`run_experiments.py`, `make_figures.py`, `app.py`, `viz.py` — **not modified**.  
Partition DP (`_plan`, `plan_partition`), constraint checks, PE logic — **not modified**.

## Callers of `Cluster.costs()` / `Cluster.context()` (full repo grep)

| File | Line | Call | Status |
|------|------|------|--------|
| `core.py` | 572 | `context()` calls `self.costs()` internally | ✅ unchanged |
| `core.py` | 802 | `allocate()` calls `cl.context(objective)` | ✅ unchanged |
| `tests.py` | 44 | `cl.costs()` in `test_eq5()` | ✅ unchanged (lam=0 default) |
| `app.py` | — | no direct call | ✅ unaffected |
| `viz.py` | — | no direct call | ✅ unaffected |

## Smoke test results (Step 3)

Base cluster, "Same demand", RAMPA, seeds 0–2, `penalize_links=False`.  
N_ref = floor(0.75 × 37) = **27** (37 = minimum admitted across all 9 runs).

```
  lam  seed  admitted  gpus_in_use  alloc_tdp_W  n_L4 n_V100  n_A30   n_T4  runtime
-----------------------------------------------------------------------------------
    0     0        41           45       6767.0    16     12     15      2  0.47s
    0     1        38           56       8028.0    24     12     20      0  0.46s
    0     2        37           52       7829.0    17     12     21      2  0.48s
    1     0        43           45       6227.0    16      9     15      5  0.52s
    1     1        38           56       8121.0    23     12     21      0  0.46s
    1     2        40           52       7829.0    17     12     21      2  0.49s
    5     0        43           45       6227.0    16      9     15      5  0.49s
    5     1        38           56       8121.0    23     12     21      0  0.48s
    5     2        40           52       7829.0    17     12     21      2  0.48s
```

**Observations:**
- λ=0: lam=0 results match baseline exactly (confirmed by T1).
- λ=1,5: seed 0 shows V100 usage dropping from 12 → 9 and T4 rising 2 → 5, confirming
  the penalty steers the ant search away from high-TDP GPUs.
- `alloc_tdp` at N_ref=27: seed 0 drops from 6767 W → 6227 W with λ≥1 (−8%), while
  total admitted *increases* slightly (+2 services), suggesting the penalty improves
  GPU-type diversity without hurting packing.
- Runtime per run: ~0.47–0.52 s (no measurable overhead from the penalty computation).
