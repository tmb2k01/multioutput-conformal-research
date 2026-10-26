# Multi-output conformal prediction research

A Python research foundation for **categorical multi-output classification**, with an
emphasis on covariance-aware Mahalanobis prediction sets. It includes SGVehicle
(color, vehicle type), UTKFace (gender, race), a synthetic dataset, model training,
split conformal calibration, and joint/marginal coverage and set-size comparisons.

## Install and run

Run commands from this repository root. Python 3.10+ is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
mocp run experiments/synthetic.yaml
```

The synthetic experiment needs no downloads or PyTorch. For image experiments:

```bash
pip install -e '.[images]'
mocp prepare experiments/sgvehicle.yaml
mocp run experiments/sgvehicle.yaml
mocp prepare experiments/utkface.yaml
mocp run experiments/utkface.yaml
```

For CPU-only PyTorch, install `torch torchvision` from the official
`https://download.pytorch.org/whl/cpu` index before installing the image extra.
For CUDA, install a compatible build using the [official PyTorch installer](https://pytorch.org/get-started/locally/).
The local RTX 5060 Ti was verified with:

```bash
pip install 'torch==2.11.0+cu128' 'torchvision==0.26.0+cu128' \
  --index-url https://download.pytorch.org/whl/cu128
```

`experiments/mahalanobis-ablation.yaml` uses `training.device: auto`: CUDA when
available, otherwise CPU. For other configurations, add `--set training.device=auto`
to choose the same behavior. GPU use requires access to the host NVIDIA driver;
a restricted execution sandbox may hide an otherwise available GPU.

Full image training may take substantial time; first check the pipeline with:

```bash
mocp run experiments/sgvehicle.yaml \
  --set name=sgvehicle-smoke --set dataset.max_samples=512 \
  --set training.epochs=1 --set model.width=8
```

This small run checks execution; it does not establish research performance.
`python -m mocp` provides the same CLI as `mocp`.

## Configure experiments

YAML files support `extends` relative to their own location. Dictionaries merge
recursively; lists replace the parent list. Runtime paths (`dataset.root` and
`output_dir`) are relative to the working directory. Unknown configuration keys
are rejected. Repeat `--set dotted.key=value` for YAML-valued overrides.

```bash
mocp check-config experiments/utkface.yaml
mocp run experiments/sgvehicle.yaml --set 'seeds=[42,43,44]' \
  --set 'conformal.alphas=[0.05,0.1,0.2]'
mocp run experiments/mahalanobis-ablation.yaml
mocp run experiments/sgvehicle-resnet.yaml
```

Each seed trains one model, then all methods and alpha levels reuse its
probabilities in memory. Probabilities are also saved by default for later
recalibration. Disable this cache explicitly with:

```bash
mocp run experiments/sgvehicle.yaml --set artifacts.save_probabilities=false
```

To change conformal methods using a previously saved cache:

```bash
mocp run experiments/mahalanobis-ablation.yaml --set 'seeds=[42]' \
  --predictions outputs/sgvehicle/RUN_ID/seed-42/probabilities.npz
```

The cache records the original dataset, seed, class vocabulary and split IDs.
Recalibration requires the same dataset and a matching single seed. Training and
dataset preprocessing settings in the new YAML do not retrain or alter this cache;
the calibration bundle records the cache path and hash. Use the original run's
configuration for its actual split fractions and preprocessing settings; a full
split manifest is available when diagnostic exports are enabled.

## Data and preprocessing

- **SGVehicle**: downloads the 233 MB Git LFS archive from `pappd90/mdc_dataset`
  at a pinned commit and verifies the published SHA-256. Color/type are read from
  `color_type_id` filenames. Class IDs map to the sorted observed vocabularies,
  recorded in each manifest. A vocabulary entry is an annotation category, not an
  ordinal coordinate in the conformal score.
- **UTKFace**: downloads the three Google Drive archives specified by the reference
  repository. Gender/race labels come from `age_gender_race_timestamp` filenames;
  age is excluded from prediction. The dataset's numeric annotation codes are
  preserved in the manifest. Both ordinary images and `*.jpg.chip.jpg` are accepted.
- Raw images remain intact. Archives are downloaded atomically, extracted with
  traversal/link checks, and cached. Images are verified, malformed filenames and
  unreadable images are counted, and identical image bytes are deduplicated before
  splitting. All copies of any image hash with conflicting labels are excluded
  and recorded. Set `dataset.conflict_policy: error` to fail instead.
- Image models convert to RGB, resize, and apply ImageNet channel normalization.
  With `training.augment: true` (the default), training images also receive random
  horizontal flips (50% probability) and mild affine transforms: rotation up to
  5 degrees, translation up to 5% per axis, and scaling from 0.95 to 1.05. Affine
  borders use the normalization mean. These transforms preserve color labels;
  no color jitter or grayscale is applied. Validation, shape, calibration, and
  test preprocessing stays deterministic. Disable augmentation for a comparison
  with `--set training.augment=false`. Augmentation uses the training seed and
  runs on image loading; raw files remain intact. The linear baseline ignores
  this option and standardizes using training data only.
- `max_samples` selects a seeded random subset **before** splitting. Splits are
  label-independent random partitions: train/validation/shape/calibration/test =
  50/10/10/15/15% by default. Every split needs at least two observations.
  The shape split estimates covariance; validation chooses the model checkpoint.
  This deliberately differs from the source repository's stratified split scheme.

For local archives, place them under `data/sgvehicle/archives/` or
`data/utkface/archives/` using the names in `src/mocp/data.py`, and run `prepare`
with `--set dataset.download=false`. Alternatively, put extracted images anywhere
below `data/DATASET/raw/` and disable downloading. Accessible mirrors can be set in
YAML (the list replaces the defaults):

```yaml
dataset:
  sources:
    - filename: UTKFace_dataset_1.tar.gz
      url: https://your-mirror.example/part1.tar.gz
      sha256: YOUR_ARCHIVE_SHA256
```

Include all desired archives in that list. Google Drive availability/quotas are
external dependencies. UTKFace has no upstream checksum in the reference scripts;
downloaded hashes are recorded, and user-specified hashes are enforced. Cached
extractions carry their source metadata. Use a fresh data root when changing a source.

## Conformal methods

| YAML name | Calibration | Guarantee under exchangeability |
|---|---|---|
| `marginal` | One threshold per task, each at alpha | Per-task only, for native task sets |
| `bonferroni` | Task thresholds at alpha times fixed weights summing to 1 | Joint, without independence |
| `max_norm` | One threshold for max over task scores `1-p_k(y_k)` | Joint |
| `mahalanobis` | One threshold for covariance-weighted one-hot residual distance | Joint |

Bonferroni uses equal weights by default; override with `params.weights`.
Mahalanobis defaults to absolute residuals and no centering, following the
multi-label paper's score construction. Signed and centered residuals are
configurable ablations. Shrinkage toward the covariance diagonal and a positive
ridge stabilize the covariance inverse, including categorical simplex constraints.

All thresholds use the **ceil((n+1)(1-alpha))** order statistic with inclusive
membership and an infinite threshold if the rank exceeds n. No interpolation,
chi-square cutoff, independence assumption or Gaussian assumption is used.
Ties can make coverage conservative. Coverage guarantees are marginal over fresh
exchangeable examples and calibration data, not per-class or conditional guarantees.

The Mahalanobis method enumerates categorical tuples, embeds each as concatenated
one-hot vectors, and retains those inside its residual-space region. SGVehicle
has 132 possible tuples; UTKFace has 10. Enumeration includes combinations absent
from training. Exact enumeration is exponential in task count; a configurable
candidate limit fails explicitly rather than silently pruning valid labels.

Joint coverage of a region implies at least that coverage for each projection.
This also applies to max-norm. None of these methods universally improves empirical
marginal coverage, tightness, or efficiency over all others. These are measured
research objectives; see [the research notes](docs/research.md).

## Outputs

Each invocation creates a new `outputs/NAME/RUN_ID/`. The default output contains
only the core artifacts, with six files for a single training seed:

- `config.yaml`: resolved settings.
- `metrics.csv`: all seed/method/alpha results, including classwise coverage gaps.
- `class_metrics.csv`: per-class support, covered counts, and coverage for joint
  labels, task projections, and native task sets. `set_kind` identifies the scope.
- `seed-N/model.pt` or `model.npz`: selected model checkpoint.
- `seed-N/calibrations.npz`: all method/alpha calibration states and their metadata
  in one compressed bundle, including task names and class vocabularies.
- `seed-N/probabilities.npz`: prediction cache for later recalibration.

Runs with multiple seeds also produce `summary.csv` with seed means and standard
deviations. Cache replay runs reference the existing predictions and produce no
new model checkpoint or prediction cache. All NumPy archives load with
`allow_pickle=False`. Set `artifacts.save_probabilities=false` to omit the
prediction cache from training runs.

Additional exports are opt-in:

| Setting | Additional files |
|---|---|
| `evaluation.save_prediction_sets: true` | Per-method/alpha `prediction_sets.npz` with per-example memberships |
| `artifacts.save_diagnostics: true` | Provenance, split manifests, training history, and per-seed/per-method diagnostic files |

Both settings default to `false`. Enable them independently with `--set`.
Saved diagnostic and training-history files remain disabled by default; normal
console progress remains enabled.
The explicit `mocp prepare` command still writes its requested dataset manifest.

Key results are `joint_coverage`, per-task `*_coverage`, `mean_joint_set_size`,
per-task `*_mean_set_size`, empty/singleton rates, and coverage gaps to the target.
Class balance is measured by `joint_classwise_coverage_gap` and per-task
`*_classwise_coverage_gap`: the mean absolute difference between each observed
class's coverage and `1-alpha`, giving every observed class equal weight. Joint
classes are complete label tuples; task metrics use the exact projections.
These fields are fractions; multiply by 100 for percentage points. This differs
from `joint_coverage_gap`, which measures overall signed deviation from target.
Also reported are `*_classwise_undercoverage_gap` (mean shortfall below target),
`*_worst_class_coverage`, and `*_observed_classes`. All classwise gaps use the
same nominal `1-alpha`, including native Bonferroni sets, to match the results
tables. These descriptive metrics do not imply class-conditional guarantees.
`class_metrics.csv` retains per-class counts and coverage even when prediction
sets are not exported. Unobserved classes have zero support and blank coverage
and are excluded from classwise averages. Optional prediction-set NPZ files store
the same class summaries with NaN coverage for unobserved classes. Counts expose
uncertainty for rare classes.
`rectangular_joint_coverage` evaluates the Cartesian product of projections. It can
exceed the coverage of the actual Mahalanobis set; both are reported separately.
Wilson intervals describe test-sample uncertainty. Summary standard deviations are
across seeds; overlapping splits mean they are not independent confidence intervals.

For `marginal` and `bonferroni`, `*_native_coverage` and `*_native_mean_set_size`
also report the original task sets. Their per-task calibration guarantees apply
to these native sets. When one task's set is empty, the joint product and all its
exact projections are empty, even if other native task sets are nonempty.
The ordinary `*_coverage` and `*_mean_set_size` fields consistently describe exact
projections for every method.

Restore a fitted conformal method and predict from saved model probabilities:

```python
from mocp.conformal import load_method

method = load_method(
    "outputs/NAME/RUN_ID/seed-42/calibrations.npz",
    method="mahalanobis",
    alpha=0.1,
)
# probabilities: K arrays of shape (N, C_k), candidates: (J, K) integer tuples
membership = method.predict(probabilities, candidates)
```

Select an ablation by its configured label, for example `mahalanobis-absolute`.
`load_method` also accepts a seed directory containing the bundle, and continues
to support older per-method/alpha directories.

## Extend the project

`src/mocp/registry.py` exposes dataset, model, and conformal registries. Add a
`@METHODS.register("new_method")` class implementing `ConformalMethod` in
`conformal.py`, or import your plugin module before calling `experiment.run`.
Fit all score parameters on training-side data, then leave the final calibration
set untouched until computing thresholds. The evaluator needs only `predict` and
`state`, so more complex methods can override the calibration/prediction interface.
Model adapters implement `Model.fit` and `predict_proba`; dataset adapters return
the `Dataset` record. Copulas, CRC, learned allocation and optimized templates are
future extensions and are not claimed as implemented.

No test files are introduced. Verification commands and observed limitations are
recorded in [docs/validation.md](docs/validation.md).

## Sources

Dataset identification and archive locations were inspected in
[tmb2k01/multioutput-conformal](https://github.com/tmb2k01/multioutput-conformal)
at commit `ab6c2483cd07adc0ec3b29abcbe9eb356ac3cb67`. This repository contains
an independent implementation, with links to the original data sources.

- Katsios & Papadopulos (2024), [Multi-label Conformal Prediction with a Mahalanobis
  Distance Nonconformity Measure](https://proceedings.mlr.press/v230/katsios24a.html).
- Tumu et al. (2024), [Multi-modal conformal prediction regions by optimizing convex
  shape templates](https://proceedings.mlr.press/v242/tumu24a.html).
- [SGVehicle archive source](https://github.com/pappd90/mdc_dataset).

The categorical Mahalanobis implementation is an adaptation of the first paper,
not a reproduction of its experiments or of the second paper's template optimizer.
Dataset usage terms remain those of the respective sources. UTKFace annotation
categories reflect the dataset; results on them do not establish demographic fairness.
