# Validation performed

Initial CPU validation performed on 2026-10-19 with Python 3.12.3, NumPy 1.26.4, PyYAML 6.0.1,
Pillow 10.2.0, gdown 5.2.2, PyTorch 2.14.1+cpu and torchvision 0.29.1+cpu.
The project virtual environment is `.venv/`. No test files were created.

## Actual data acquisition

Both datasets were downloaded and fully preprocessed using the project CLI:

```bash
.venv/bin/mocp prepare experiments/sgvehicle.yaml
.venv/bin/mocp prepare experiments/utkface.yaml
```

| Dataset | Retained | Malformed skipped | Repeated copies removed | Additional conflicting unique images excluded |
|---|---:|---:|---:|---:|
| SGVehicle | 32,080 | 0 | 0 | 0 |
| UTKFace | 23,682 | 4 | 329 | 91 |

Duplicate counts include repeated copies of conflicting images; the last column
counts unique originals also removed because their hash had conflicting labels.
Conflicts concern the selected task annotations (gender/race), not excluded age.
Manifests record the conflict details and raw source hashes. Byte deduplication
does not identify all repeated subjects or perceptually similar images.

Observed archive SHA-256 hashes (UTKFace hashes are measured, not upstream-published):

```text
MDC_dataset.tar.gz       a7ceabfe8c1ebfad554a955842cfb85a1281a63c264215174df7d19776476893
UTKFace_dataset_1.tar.gz 257a0396403395f54bb3d60dc920404c95619f9aaf6992c4990081420de5089c
UTKFace_dataset_2.tar.gz 20b1115b9f0395e37ee8372b7cab98907f5d3eb76764210edf3c93682640d09d
UTKFace_dataset_3.tar.gz 64264819b916eddd9ba609ba280adbe41c07daae3085e02b145aeebedaff7930
```

SGVehicle's actual vocabulary is 12 colors and 11 vehicle types; UTKFace's is
2 gender codes and 5 race codes, matching the reference experiment schemas.

## Training and conformal runs

```bash
.venv/bin/mocp run experiments/synthetic.yaml
.venv/bin/mocp run experiments/sgvehicle.yaml \
  --set name=sgvehicle-smoke --set dataset.max_samples=512 \
  --set training.epochs=1 --set model.width=8
.venv/bin/mocp run experiments/utkface.yaml \
  --set name=utkface-smoke --set dataset.max_samples=512 \
  --set training.epochs=1 --set model.width=8
```

All completed successfully, including model checkpointing, shape fitting,
calibration at alpha 0.05/0.1, all four methods, prediction sets, and CSV/JSON
outputs. The synthetic run used three seeds and 4,000 examples per seed.
The image runs used 256 training, 51 validation, 51 shape, 76 calibration and
78 test examples, and one epoch. They verify pipeline execution, not model quality,
coverage superiority, or benchmark performance. A subsequent full 20-epoch
SGVehicle GPU ablation is recorded below. Full-length UTKFace training has not been run.

The seven-method SGVehicle ablation also completed using saved image probabilities:
equal and weighted Bonferroni, max-norm, absolute Mahalanobis, diagonal covariance,
signed residuals and centered signed residuals. Recalibrating a synthetic cache
reproduced its complete metrics exactly. Restoring all four fitted methods
reproduced saved membership masks exactly. Restored Bonferroni also recalibrated
successfully. ResNet18's two-head forward pass was checked without pretrained
weights; pretrained weight download and full ResNet training were not run.

## Mathematical and implementation checks

Direct inline Python assertions checked:

- The finite-sample order statistic, ties, and the infinity sentinel for
  calibration sets too small to support a finite threshold.
- Full candidate inclusion with infinite thresholds for every built-in method.
- Agreement of true-label scores with the corresponding enumerated candidate
  scores for every method.
- Positive definiteness of regularized covariance, covariance/precision consistency,
  and finite precision for a singular empirical covariance with zero shrinkage.
- Disjoint split IDs, rejection of invalid probability vectors, and explicit
  failure when the candidate-space limit is exceeded.
- A rectangular predictor with one empty head: native per-task coverage and
  projection coverage differ as expected, and the joint set is empty.

All YAML files were loaded successfully. Source lint/format checks:

```bash
.venv/bin/ruff check src
.venv/bin/ruff format --check src
```

Coverage on a particular test split can fall below the nominal level because
split conformal validity is a probability statement over exchangeable calibration
and future observations. These execution checks do not constitute an empirical
validation of coverage under distribution shift or a proof of research superiority.

## Full SGVehicle GPU ablation

The requested `experiments/mahalanobis-ablation.yaml` run completed on 2026-10-19
using an NVIDIA GeForce RTX 5060 Ti (16 GB), PyTorch 2.11.0+cu128,
torchvision 0.26.0+cu128, and CUDA runtime 12.8. The configuration now uses
`training.device: auto`, selecting CUDA when available and CPU otherwise.
GPU access required execution outside the restricted agent sandbox.

The original CPU run was interrupted after two epochs. The GPU run restarted
training from the beginning and completed all 20 epochs on the full dataset,
with seed 42, followed by seven methods at alpha 0.05 and 0.1.
The lowest validation loss was 1.7698 at epoch 20.

Artifacts are under:

```text
outputs/sgvehicle-mahalanobis-ablation/20261019T083707Z-4b9ae3d8/
```

`hardware.json` records the actual GPU and CUDA/PyTorch versions. The checkpoint,
cached probabilities, training history, all 14 method/alpha artifact directories,
and aggregate metrics/summary CSV files were verified after completion.

| Method | Joint coverage, target 90% | Mean tuple count |
|---|---:|---:|
| Bonferroni | 89.6% | 6.19 |
| Weighted Bonferroni | 89.4% | 6.19 |
| Max-norm | 89.2% | 6.29 |
| Mahalanobis absolute | 90.9% | 8.27 |
| Mahalanobis diagonal | 90.6% | 10.19 |
| Mahalanobis signed | 90.2% | 9.46 |
| Mahalanobis centered | 90.1% | 9.27 |

These are empirical results from one seed and one test split. They do not
establish that a method dominates others; compare efficiency alongside coverage
and repeat across seeds before drawing research conclusions.
