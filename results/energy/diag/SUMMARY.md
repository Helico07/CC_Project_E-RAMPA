# e-RAMPA High-Performance & Energy Diagnosis Summary

## 1. Per-Environment Summary (RAMPA, penalize_links=False)
| Cluster | Mix | Cnt λ=0 | Cnt λ=0.5 (%Δ) | Cnt λ=1 (%Δ, D/S/U) | TDP%Δ λ=1 | Cnt λ=5 (%Δ, D/S/U) | TDP%Δ λ=5 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Base | HT-heavy | 29.8 | 29.9 (+0.3%) | 30.0 (+0.6%, 0/8/2) | -0.3% | 30.2 (+1.1%, 0/8/2) | -2.2% |
| Base | HM-heavy | 30.2 | 30.2 (0.0%) | 30.2 (0.0%, 0/10/0) | -1.1% | 30.2 (+0.0%, 1/8/1) | -2.1% |
| Base | LD-heavy | 42.3 | 42.9 (+1.5%) | 42.9 (+1.5%, 1/7/2) | -1.9% | 42.6 (+0.7%, 2/7/1) | -3.2% |
| Base | Same | 39.4 | 39.5 (+0.5%) | 39.1 (-0.6%, 3/3/4) | -5.6% | 37.4 (-4.8%, 4/5/1) | -6.5% |
| High-bw | HT-heavy | 30.6 | 30.5 (-0.2%) | 30.6 (0.0%, 0/10/0) | -0.3% | 30.6 (0.0%, 0/10/0) | -2.2% |
| High-bw | HM-heavy | 30.2 | 30.2 (0.0%) | 30.2 (0.0%, 0/10/0) | -1.1% | 30.2 (+0.0%, 1/8/1) | -2.1% |
| High-bw | LD-heavy | 42.3 | 42.6 (+1.0%) | 42.9 (+1.5%, 1/7/2) | -1.9% | 42.6 (+0.7%, 2/7/1) | -3.2% |
| High-bw | Same | 39.6 | 39.5 (-0.2%) | 39.1 (-1.3%, 3/5/2) | -5.6% | 37.4 (-5.3%, 4/5/1) | -6.5% |
| High-perf | HT-heavy | 25.7 | 25.7 (0.0%) | 23.7 (-7.9%, 3/7/0) | -7.5% | 23.4 (-9.0%, 6/4/0) | -10.3% |
| High-perf | HM-heavy | 16.6 | 16.7 (+0.6%) | 16.9 (+2.0%, 0/7/3) | -8.4% | 17.1 (+3.2%, 0/5/5) | -19.6% |
| High-perf | LD-heavy | 36.9 | 35.1 (-4.6%) | 32.4 (-12.0%, 5/5/0)| -14.8% | 25.0 (-31.5%, 10/0/0)| -34.6% |
| High-perf | Same | 25.4 | 25.3 (-0.4%) | 25.5 (+0.5%, 0/9/1) | -7.3% | 23.5 (-7.3%, 6/3/1) | -17.3% |
| Large | HT-heavy | 67.9 | 68.6 (+0.8%) | 67.8 (-0.1%, 1/9/0) | -3.0% | 65.0 (-3.4%, 2/7/1) | -4.3% |
| Large | HM-heavy | 68.4 | 68.5 (+0.2%) | 68.9 (+0.8%, 2/4/4) | -0.4% | 69.2 (+1.2%, 0/7/3) | -0.9% |
| Large | LD-heavy | 103.9| 105.2 (+1.4%)| 104.5 (+0.7%, 3/4/3)| -6.5% | 103.9 (+0.2%, 4/3/3)| -6.8% |
| Large | Same | 91.7 | 90.0 (-1.7%) | 90.3 (-1.3%, 4/4/2) | -1.8% | 89.7 (-2.1%, 5/5/0) | -1.6% |

## 2. First-Rejection Findings (Low-Delay Heavy)
- **High-performance**: λ=0 admits 36.9 avg; first rejection is S3 ViT (70%) or S4/S5 (30%) with 0 free L4, 4.1 free V100. At λ=1, count drops to 32.4; 90% fail on S3 ViT with 0 free L4, 8.4 free V100. At λ=5, count collapses to 25.0; 100% fail on S3 ViT with 0 free L4 and 16.7 free V100 (up to 23/24 idle). S3 ViT SLA is impossible on V100 (V100 SLA = False).
- **Base**: At λ=0, 1, 5, 100% of seeds reject S3 ViT when L4 reaches 0 free. Mean count is invariant (~42.6) because non-latency tasks route to free T4 (70W) and A30 (165W) racks rather than displacing L4s.

## 3. Verdict
1. YES: Data proves high λ causes premature L4 exhaustion, rejecting latency-critical tasks while V100s idle.
2. At λ=0, V100 has lower allocation cost than L4 (2688 vs 4363), naturally absorbing non-latency requests.
3. At λ>=1, V100 power penalty (250W vs 72W) steers earlier requests to L4, draining all 24 L4 GPUs (0 free).
4. S3 ViT (50ms limit) physically cannot satisfy throughput/latency on V100 (V100 SLA = False for all stages).
5. In High-performance at λ=5, this starves S3 ViT while 12-23 V100 GPUs (up to 96%) sit idle, cutting admissions by 31.5%.

## 4. Fig. 7 Data at Reference Point N_ref_B
| Method / Variant | Pooled Alloc TDP (W) | Pooled TDP / GPU (W) | Base/Same TDP | HP/LD TDP | Large/HM TDP |
|:---|:---:|:---:|:---:|:---:|:---:|
| RAMPA λ=0 | 3838.0 ± 1543.7 | 139.5 ± 24.2 | 2618.1 W | 2634.4 W | 12774.5 W |
| RAMPA λ=1 | 3713.9 ± 1537.3 | 133.9 ± 22.5 | 2599.1 W | 2346.2 W | 12678.7 W |
| RAMPA λ=5 | 3605.6 ± 1562.7 | 127.8 ± 21.5 | 2590.6 W | 1769.4 W | 12686.2 W |
| NCAR p0 | 3859.5 ± 1547.7 | 141.0 ± 25.6 | 2653.1 W | 2652.2 W | 12794.2 W |
| PE p0 | 2919.4 ± 1193.8 | 104.7 ± 13.7 | 1922.4 W | 1495.2 W | 9777.6 W |

## 5. Relative Efficiency Comparison (% Change vs RAMPA λ=0, Pooled over 16 Envs)
| Variant | Endpoint Svc/kW (%Δ ± 95% CI) | Endpoint TDP (%Δ) | Equal-Load TDP (%Δ) | Equal-Load TDP/GPU (%Δ) | Holm p (Svc/kW) | Holm p (Eq-TDP) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| RAMPA lam=0.5 | +1.32% ± 0.96% | -1.30% | -1.78% | -1.73% | p = 0.0012 | p = 0.0049 |
| RAMPA lam=1 | +3.17% ± 2.77% | -3.05% | -3.70% | -3.65% | p = 0.0002 | p = 0.0049 |
| RAMPA lam=2 | +4.62% ± 4.73% | -4.81% | -6.06% | -6.02% | p = 0.0002 | p = 0.0049 |
| RAMPA lam=5 | +6.66% ± 5.24% | -7.49% | -7.38% | -7.31% | p = 0.0002 | p = 0.0049 |
| NCAR p0 | +18.04% ± 20.36% | -25.93% | +0.53% | +0.63% | p = 0.1591 | p = 0.1238 |
| PE p0 | +42.57% ± 22.40% | -56.00% | -22.97% | -22.55% | p = 0.0002 | p = 0.0026 |

*Note: Endpoint is measured at first rejection (`step == count`). Equal-load is measured at common stage $N_{ref,B}$. All statistics are mean % changes relative to RAMPA λ=0 across the 16 environments, with Wilcoxon p-values Holm-adjusted across variants.*

## 6. Idle Power Sensitivity: Services/kW (% Change vs RAMPA λ=0, Pooled over 16 Envs)
| Variant | f=0 (%Δ ± 95% CI, Holm p) | f=0.1 (%Δ ± 95% CI, Holm p) | f=0.3 (%Δ ± 95% CI, Holm p) |
|:---|:---:|:---:|:---:|
| RAMPA λ=0.5 | +1.32% ± 0.96% (p = 0.0012) | +1.07% ± 0.79% (p = 0.0009) | +0.69% ± 0.59% (p = 0.0101) |
| RAMPA λ=1 | +3.17% ± 2.77% (p = 0.0002) | +2.29% ± 1.74% (p = 0.0002) | +1.09% ± 0.61% (p = 0.0008) |
| RAMPA λ=2 | +4.62% ± 4.73% (p = 0.0002) | +3.11% ± 2.85% (p = 0.0002) | +1.11% ± 0.53% (p = 0.0004) |
| RAMPA λ=5 | +6.66% ± 5.24% (p = 0.0002) | +4.44% ± 3.03% (p = 0.0002) | +1.45% ± 0.62% (p = 0.0008) |
| NCAR p0 | +18.04% ± 20.36% (p = 0.1591) | +7.48% ± 12.66% (p = 0.1928) | -3.88% ± 6.89% (p = 0.8209) |
| PE p0 | +42.57% ± 22.40% (p = 0.0002) | +16.12% ± 13.57% (p = 0.1478) | -9.93% ± 7.96% (p = 0.0580) |

- **Top 3 environments where λ=5 changed services/kW most at f=0.3**: (1) Large / Low-delay heavy (+4.37%), (2) High-performance / Huge-model heavy (+3.24%), (3) High-performance / High-throughput heavy (+2.30%).
- **Findings**: At f=0.3, PE collapses from +42.57% to -9.93% (Holm p=0.058), losing its apparent advantage due to idle leakage on unallocated GPUs. RAMPA's λ>0 gain holds across all f (e.g. λ=1 +1.09%, λ=5 +1.45%, Holm p=0.0008).
