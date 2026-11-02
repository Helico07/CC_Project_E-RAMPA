"""energy.py -- energy accounting helpers for e-RAMPA.

alloc_series: convert run_fill records into a per-step energy/GPU-count summary.
"""
from __future__ import annotations
from core import GPU_TYPES, TDP_W


def alloc_series(records: list[dict], spec) -> list[dict]:
    """Return one dict per admitted request with cumulative GPU and energy accounting.

    Parameters
    ----------
    records : list[dict]
        ``FillResult.records`` produced by ``run_fill(..., snapshots=True)`` or the default.
        Each record must contain ``"racks"`` (list of rack indices, one per stage).
    spec : ClusterSpec
        The cluster spec used for the run (provides rack_types).

    Returns
    -------
    list[dict] with keys:
        step          – admission step (1-based, from record)
        gpus_in_use   – cumulative individual GPUs in use after this step
                        (one GPU per stage, i.e. len(racks) per allocation)
        alloc_tdp     – cumulative nominal TDP (W) of all GPUs in use after this step
                        (each GPU contributes its rack type's TDP, independent of lam)
        n_A, n_B, n_C, n_D – cumulative GPU count per type (L4, V100, A30, T4)
    """
    # Build a lookup: rack index -> GPUType.short and TDP
    rack_short = [GPU_TYPES[t].short for t in spec.rack_types]
    rack_tdp = [TDP_W[s] for s in rack_short]

    cumulative_gpus = 0
    cumulative_tdp = 0.0
    type_counts: dict[str, int] = {"A": 0, "B": 0, "C": 0, "D": 0}

    result = []
    for rec in records:
        for rack in rec["racks"]:           # one GPU consumed per stage
            short = rack_short[rack]
            cumulative_gpus += 1
            cumulative_tdp += rack_tdp[rack]
            type_counts[short] += 1

        result.append(dict(
            step=rec["step"],
            gpus_in_use=cumulative_gpus,
            alloc_tdp=cumulative_tdp,
            n_A=type_counts["A"],
            n_B=type_counts["B"],
            n_C=type_counts["C"],
            n_D=type_counts["D"],
        ))

    return result
