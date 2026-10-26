# SGVehicle results: joint space and task projections

Full 20-epoch run; seed 42; 4,812 test images. All methods use the same model and splits.

- Coverage is joint tuple coverage in the Joint table and marginal projection coverage in the task tables.
- Efficiency is mean prediction-set cardinality (lower is better): joint tuples out of 132, color labels out of 12, type labels out of 11.
- Informativeness is the percentage of singleton sets. Singleton predictions can still be incorrect.
- Δ coverage (pp) = empirical coverage − nominal target, in percentage points.
- Classwise CovGap (pp) = 100 × mean over observed classes of |class-conditional coverage − nominal target|. Joint classes are complete tuples. This equally weights observed classes rather than weighting by their sample counts.
- Both gap columns use the common nominal target (90% or 95%) for every scope. Bonferroni uses stricter allocated task targets internally; positive task Δ values are therefore expected.
- Empty sets count as uncovered and nonsingleton. Task sets here are exact projections of the returned joint region.
- Results are descriptive for one seed. Neither singleton rate nor classwise CovGap is a class-conditional guarantee.

Definitions of informativeness and classwise CovGap follow the [reference repository](https://github.com/tmb2k01/multioutput-conformal/blob/ab6c2483cd07adc0ec3b29abcbe9eb356ac3cb67/doc/metrics.md). The original run’s `joint_coverage_gap` is the signed Δ metric, not classwise CovGap.

## Target 90% — Joint (set size out of 132)

| Method | Coverage (%) | Efficiency | Informativeness (%) | Δ coverage (pp) | Classwise CovGap (pp) |
|---|---:|---:|---:|---:|---:|
| Bonferroni | 89.61 | 6.19 | 1.29 | -0.39 | 14.64 |
| Weighted Bonferroni | 89.44 | 6.19 | 1.29 | -0.56 | 18.13 |
| Max-norm | 89.21 | 6.29 | 1.27 | -0.79 | 19.48 |
| Mahalanobis absolute | 90.86 | 8.27 | 1.50 | +0.86 | 17.07 |
| Mahalanobis diagonal | 90.65 | 10.19 | 1.14 | +0.65 | 16.82 |
| Mahalanobis signed | 90.19 | 9.46 | 1.18 | +0.19 | 18.28 |
| Mahalanobis centered | 90.11 | 9.27 | 1.16 | +0.11 | 18.34 |

## Target 90% — Color (set size out of 12)

| Method | Coverage (%) | Efficiency | Informativeness (%) | Δ coverage (pp) | Classwise CovGap (pp) |
|---|---:|---:|---:|---:|---:|
| Bonferroni | 94.72 | 1.42 | 63.99 | +4.72 | 5.55 |
| Weighted Bonferroni | 96.51 | 1.60 | 55.22 | +6.51 | 6.53 |
| Max-norm | 97.40 | 1.74 | 50.08 | +7.40 | 7.42 |
| Mahalanobis absolute | 94.47 | 1.55 | 59.00 | +4.47 | 4.74 |
| Mahalanobis diagonal | 93.62 | 1.74 | 63.42 | +3.62 | 4.60 |
| Mahalanobis signed | 93.79 | 1.49 | 58.62 | +3.79 | 5.08 |
| Mahalanobis centered | 93.81 | 1.48 | 59.58 | +3.81 | 5.02 |

## Target 90% — Type (set size out of 11)

| Method | Coverage (%) | Efficiency | Informativeness (%) | Δ coverage (pp) | Classwise CovGap (pp) |
|---|---:|---:|---:|---:|---:|
| Bonferroni | 94.58 | 4.33 | 2.10 | +4.58 | 11.78 |
| Weighted Bonferroni | 92.52 | 3.83 | 2.54 | +2.52 | 15.31 |
| Max-norm | 91.44 | 3.58 | 2.81 | +1.44 | 16.42 |
| Mahalanobis absolute | 96.36 | 6.80 | 3.18 | +6.36 | 13.43 |
| Mahalanobis diagonal | 97.17 | 8.29 | 2.18 | +7.17 | 14.22 |
| Mahalanobis signed | 96.01 | 7.80 | 2.37 | +6.01 | 16.09 |
| Mahalanobis centered | 95.91 | 7.64 | 2.33 | +5.91 | 16.35 |

## Target 95% — Joint (set size out of 132)

| Method | Coverage (%) | Efficiency | Informativeness (%) | Δ coverage (pp) | Classwise CovGap (pp) |
|---|---:|---:|---:|---:|---:|
| Bonferroni | 94.20 | 9.07 | 0.79 | -0.80 | 9.18 |
| Weighted Bonferroni | 94.49 | 9.47 | 0.85 | -0.51 | 10.20 |
| Max-norm | 94.37 | 9.52 | 0.85 | -0.63 | 10.48 |
| Mahalanobis absolute | 94.87 | 12.34 | 0.96 | -0.13 | 13.67 |
| Mahalanobis diagonal | 95.05 | 16.92 | 0.69 | +0.05 | 13.42 |
| Mahalanobis signed | 95.12 | 13.34 | 0.83 | +0.12 | 13.83 |
| Mahalanobis centered | 95.05 | 13.13 | 0.81 | +0.05 | 13.91 |

## Target 95% — Color (set size out of 12)

| Method | Coverage (%) | Efficiency | Informativeness (%) | Δ coverage (pp) | Classwise CovGap (pp) |
|---|---:|---:|---:|---:|---:|
| Bonferroni | 96.90 | 1.64 | 53.37 | +1.90 | 2.86 |
| Weighted Bonferroni | 98.40 | 1.90 | 46.01 | +3.40 | 3.41 |
| Max-norm | 98.63 | 1.96 | 44.18 | +3.63 | 3.64 |
| Mahalanobis absolute | 97.09 | 2.06 | 50.83 | +2.09 | 2.28 |
| Mahalanobis diagonal | 96.99 | 2.99 | 48.00 | +1.99 | 2.57 |
| Mahalanobis signed | 96.99 | 2.00 | 46.40 | +1.99 | 2.35 |
| Mahalanobis centered | 96.99 | 1.99 | 47.34 | +1.99 | 2.31 |

## Target 95% — Type (set size out of 11)

| Method | Coverage (%) | Efficiency | Informativeness (%) | Δ coverage (pp) | Classwise CovGap (pp) |
|---|---:|---:|---:|---:|---:|
| Bonferroni | 97.15 | 5.40 | 1.31 | +2.15 | 5.74 |
| Weighted Bonferroni | 95.99 | 4.88 | 1.66 | +0.99 | 7.79 |
| Max-norm | 95.64 | 4.74 | 1.81 | +0.64 | 8.29 |
| Mahalanobis absolute | 98.25 | 8.66 | 1.97 | +3.25 | 11.87 |
| Mahalanobis diagonal | 98.61 | 9.42 | 1.39 | +3.61 | 11.86 |
| Mahalanobis signed | 98.30 | 8.79 | 1.64 | +3.30 | 11.77 |
| Mahalanobis centered | 98.25 | 8.76 | 1.62 | +3.25 | 11.88 |

## Provenance

Source: `outputs/sgvehicle-mahalanobis-ablation/20261019T083707Z-4b9ae3d8/metrics.csv` and the saved `seed-42/METHOD/alpha-A/prediction_sets.npz` membership arrays.

Task singleton rates and classwise CovGap were recomputed from these saved arrays; no model retraining or recalibration was performed. All coverages, efficiencies, and joint singleton rates were checked against the saved CSV.

Observed class counts: Joint 132/132, Color 12/12, Type 11/11.
