# Synthetic multi-output conformal experiments

Generate labels and controlled model probability vectors directly. The current
studies focus on **standard conformal prediction**, varying label balance,
prediction loss at fixed accuracy, and calibration sample size. Each group is
defined in **one YAML file**, including all its cases. Defaults
match SGVehicle's output dimensions: **12 colors and 11 vehicle types**, giving
132 possible joint labels and a concatenated probability vector of length 23.
No image downloads, features, model training, or PyTorch are needed.

This replaces the image-training, Mahalanobis, KDE, and bandwidth-tuning workflow.
Existing datasets and output directories are left in place. Legacy image/KDE
configuration files and caches do not work with the new synthetic schema.

## Run

From the repository root, with Python 3.10+:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[plots]'
python -m src study experiments/study-label-balance.yaml --check
python -m src study experiments/study-label-balance.yaml
```

The installed `mocp` command is equivalent to `python -m src`. Reinstall the
editable package if your existing installation still points to `mocp.cli`.

```bash
mocp study experiments/study-prediction-loss.yaml
mocp study experiments/study-calibration-size.yaml
```

In VS Code, select the `.venv` Python interpreter and use **Run and Debug**:
**Standard studies: choose group**, **Standard studies: active group YAML**,
or **Standard studies: validate group**.
The configurations are in [.vscode/launch.json](.vscode/launch.json).

## Controlled groups: one YAML each

| Group definition | Comparison | Results |
|---|---|---|
| [study-label-balance.yaml](experiments/study-label-balance.yaml) | Uniform, mild, moderate, and severe imbalance, with both prior orientations; repeat with homogeneous and fixed class-specific score laws. 14 cases. | [HTML](docs/standard-studies/label-balance/report.html), [PDF](docs/standard-studies/label-balance/report.pdf) |
| [study-prediction-loss.yaml](experiments/study-prediction-loss.yaml) | Same labels, argmax predictions, and accuracy; vary error confidence, true-label probability on mistakes, and loss concentrated in half the classes. 17 cases. | [HTML](docs/standard-studies/prediction-loss/report.html), [PDF](docs/standard-studies/prediction-loss/report.pdf) |
| [study-calibration-size.yaml](experiments/study-calibration-size.yaml) | Same probability vectors and test examples; use nested calibration budgets of 100, 500, 2,000, 6,000, and 12,000. 5 cases. | [HTML](docs/standard-studies/calibration-size/report.html), [PDF](docs/standard-studies/calibration-size/report.pdf) |

All three groups use standard CP exclusively, 30,000 observations per seed,
five seeds `[42,43,44,45,46]`, and nominal coverages 75/80/85/90/95%. Expected
per-head and joint top-1 accuracy is 80%. The default conformal method in the
Python configuration is now standard. Earlier methods and reports remain
available for later comparisons.

The study command runs every case, verifies its controls against the saved
probabilities for **all three splits**, compares the results, and writes plots
plus self-contained HTML and PDF reports. Loss cases have exactly matched
labels and top-1 predictions across the entire group; calibration-budget cases
also have exactly identical probability arrays. Balance cases have exactly
matched correctness patterns while the true labels change.

Each YAML contains `base` settings and `blocks`. A block fixes its own settings
and lists its `cases`; `overrides` use the ordinary experiment schema. To change
label balance, edit a case's `balance.exponent` and `balance.reverse`:
per-head class probabilities are proportional to `rank ** (-exponent)`, and
reversal changes which labels are common. Exponent zero is uniform. A YAML
anchor reuses the same balance levels in both blocks.

For loss cases, edit `predictions.incorrect_confidence.mean` or
`predictions.true_class_weight` inside the case overrides. The localized-loss
block applies changes to a fixed class range. Dimensions are set in
`base.dataset.num_classes`; update explicit class-profile ranges when changing
them. Calibration cases set `conformal.calibration_samples`, which selects a
prefix of the full calibration pool without changing the test set.

Validate all cases before running, or override a shared setting:

```bash
python -m src study experiments/study-prediction-loss.yaml --check
python -m src study experiments/study-prediction-loss.yaml --set 'seeds=[42,43,44]'
```

Rebuild a report from its exact saved runs without new simulations:

```bash
python -m src study experiments/study-label-balance.yaml \
  --manifest docs/standard-studies/label-balance/study.json
```

Keep the same YAML and `--set` overrides for replay. `--output-dir` chooses a
different report destination. Raw runs are organized under
`outputs/GROUP/BLOCK-CASE/RUN_ID/`.

The focused comparisons measure:

- **Efficiency:** mean joint set size, alongside actual coverage.
- **Informativeness:** `log2(132 / set_size)` for nonempty sets; empty sets
  receive zero. `mean_covered_information_bits` additionally assigns zero when
  the truth is missing. This is candidate reduction against a uniform reference,
  rather than posterior entropy. Singleton, correct-singleton, empty, full,
  and nonempty non-full set rates are also exported.
- **Class coverage:** macro absolute gap, macro undercoverage gap, and worst
  class coverage. The balance group also uses the **same balanced test audit**
  for every case within a block and seed, keeping per-class support fixed.
  Population coverage is measured on each case's own label distribution;
  the audit evaluates class coverage under a changed prior and carries no
  separate marginal coverage claim.

The completed runs contain **900 standard-CP evaluations**, plus 350 evaluations
on the balanced audit. Each report directory includes `comparison.csv`,
`summary.csv`, `paired_differences.csv`, `source_runs.csv`, `controls.json`,
`study.json`, and figures in PNG/SVG/vector PDF. The balance report also includes
`audit_class_metrics.csv`. Plot whiskers are seed ranges; numerical tables use
mean ± sample SD, rather than confidence intervals.

Initial findings at 90% nominal coverage: in the homogeneous balance control,
set sizes stay around 76–77 labels and balanced-audit gaps around 2.5 percentage
points, while raw gaps rise with sparse class support. With fixed class-specific
score laws, reversing the same severe imbalance changes mean set size from
19.73 to 124.38 labels. In the loss group, near-equal NLL values of 2.261 and
2.268 coexist with class gaps of 2.53 and 8.46 percentage points: loss alone
does not determine how coverage is distributed across classes. These are
descriptive findings from the configured synthetic cases.

## Earlier research comparisons

Except for the small-dimensions example and smoke run, each preset uses 30,000
samples, seeds `[42, 43, 44]`, and alpha levels `[0.05, 0.1, 0.2]`.
Each method sees exactly the same clustering, calibration, and test observations.

| YAML in `experiments/` | Question |
|---|---|
| `synthetic.yaml` | Baseline: correct and incorrect predictions both confidently score about 0.9. |
| `synthetic-uncertain-errors.yaml` | What changes when errors have confidence about 0.6? |
| `synthetic-broad-confidence.yaml` | Same accuracy and mean confidence, much wider Beta distributions. |
| `synthetic-uniform-confidence.yaml` | Uniform confidence distributions, still with mean 0.9. |
| `synthetic-constant-confidence.yaml` | Fixed winning confidence of 0.9; study ties. |
| `synthetic-underconfident-correct.yaml` | Correct confidence is 0.65 while error confidence is 0.9. |
| `synthetic-class-profiles.yaml` | Same accuracy for every class, different confidence/tail distributions by class. |
| `synthetic-class-accuracy.yaml` | Half the joint classes have 95% accuracy, half 65%; balanced average is 80%. |
| `synthetic-imbalanced.yaml` | Long-tailed label frequencies and rare calibration classes. |
| `synthetic-independent-errors.yaml` | Independent errors between heads: 80% per head, 64% jointly. |
| `synthetic-small-dimensions.yaml` | Custom 3 × 2 dimensions, correlated labels, and asymmetric confusion matrices. |
| `synthetic-smoke.yaml` | Fast pipeline/debug run with 2,000 observations and one seed. |

Run a set of comparisons with:

```bash
for config in experiments/synthetic*.yaml; do
  python -m src run "$config"
done
```

## Configure the generated predictions

YAML `extends` paths resolve relative to the child YAML. Mappings merge recursively;
lists replace the parent list. Unknown configuration fields are rejected.
Runtime output paths are relative to your working directory.

```bash
python -m src run experiments/synthetic.yaml \
  --set name=my-comparison --set dataset.n_samples=50000 \
  --set 'seeds=[42]' --set predictions.incorrect_confidence.mean=0.7

python -m src run experiments/synthetic.yaml \
  --set name=three-heads --set 'dataset.num_classes=[5,4,3]' \
  --set dataset.task_names=null --set predictions.error_correlation=0.5
```

`dataset.num_classes` determines both the number of heads and each head's width.
`task_names: null` gives automatic names (`color/type` for `[12,11]`, otherwise
`task_0`, `task_1`, ...). Change class probability vectors, confusion matrices,
and profile ranges to match whenever you change the dimensions.

- **Label frequencies:** `dataset.class_probabilities` supplies one probability
  vector per head. Alternatively, `dataset.joint_probabilities` supplies one vector
  over complete label tuples, allowing correlations between labels. These options
  are mutually exclusive. `null` for both means independent uniform labels.
- **Top-1 accuracy:** `predictions.accuracy` is a scalar or a list per head.
  Correctness is sampled as a Bernoulli event, so finite samples fluctuate around
  the target. “80% precision” is interpreted here as **80% top-1 accuracy**;
  macro precision is reported separately, especially because imbalance changes it.
- **Dependence between errors:** `error_correlation` is the probability of using
  one shared uniform correctness draw for all heads instead of independent draws.
  For equal head accuracy `a`, expected joint accuracy is
  `rho*a + (1-rho)*a**K`. Thus the default `a=0.8, rho=1` gives about 80% for each
  head and for the complete tuple. With different per-head targets, replace the
  shared term `a` by their minimum and `a**K` by their product. This parameter is
  a mixture weight; it need not equal the Pearson correlation of the error flags.
- **Confidence conditional on correctness:** `correct_confidence` and
  `incorrect_confidence` control the actual winning softmax probability separately.
  `beta` uses a Beta distribution rescaled to `[low,high]`, with its expectation at
  `mean`; larger `concentration` makes it narrower. `uniform` samples between
  `low/high` and ignores `mean/concentration`. `constant` uses `mean` only.
  Bounds must be at least the largest `1/C_k`; constant confidence must exceed it.
- **The remaining probabilities:** a Dirichlet tail divides `1-confidence` among
  other labels. `tail_concentration` controls its spread. On errors,
  `true_class_weight` changes the true label's share of this tail. Smaller values
  make the true class especially implausible. The tail is shrunk toward uniform
  if necessary to preserve the intended argmax, and outputs are normalized by
  softmax. Finite logits can be recovered with `log(probabilities)`.
- **Which wrong labels win:** by default choose uniformly from incorrect labels.
  `confusion_matrices` supplies one error-only matrix per head; rows correspond to
  true labels, rows sum to one, and diagonals must be zero.
- **Class-specific behavior:** `predictions.class_profiles` overrides accuracy,
  correct/incorrect confidence, and tail parameters for `[start,stop)` ranges of
  flattened joint labels. Indexing is Cartesian order, e.g. `color*11+type`.
  Ranges must not overlap; unspecified labels use the global defaults. Confidence
  mappings inherit unspecified global fields. See `synthetic-class-profiles.yaml`.

Independent random streams for labels, correctness, wrong labels, score generation,
and splitting keep confidence-only ablations paired on the same examples, point
predictions, and split assignments. No empirical accuracy quota is enforced.

## Conformal methods

A **class** means a complete label tuple, e.g. `(color=2, type=5)`.
All methods use the same nonconformity score:

```text
s(x, y) = 1 - product_k p_k(y_k | x)
```

The product defines a normalized joint scoring distribution. This choice does
not require independence of the heads for the conformal coverage guarantee.
For calibration group size `n`, use the `ceil((n+1)*(1-alpha))` order statistic,
include ties with `score <= threshold`, and use infinity if the rank exceeds `n`.
Empty groups also get infinity and their candidate labels are always included.

| Method | Threshold groups | Coverage under exchangeability |
|---|---|---|
| `standard` | All labels share one threshold. | Joint marginal |
| `classwise` | One threshold per complete label tuple. | Joint class-conditional |
| `clustered` | Pool classes with similar true-label score distributions. | Joint cluster-conditional |

Clustered CP follows the score-distribution grouping idea of
[Ding et al. (NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/file/cb931eddd563f8d473c355518ce8601c-Paper-Conference.pdf).
It embeds each joint class using empirical score quantiles from the independent
clustering split and applies deterministic k-means. Classes below
`min_class_samples` share a separate fallback cluster. The mapping is then frozen
before calibration. Parameters are `n_clusters`, `quantiles`, `min_class_samples`,
and `max_iter`. This is a basic implementation, not the paper's complete tuning
and rare-class selection procedure. Its guarantee is at the cluster level;
per-class performance remains an empirical measurement.

Splits are label-independent and disjoint: **20% clustering, 40% calibration,
40% test** by default. Standard and classwise do not learn anything from clustering;
all methods use the same calibration budget for comparison. Test examples never
choose thresholds or clusters. Nominal coverage is a distributional guarantee;
a particular test run can fall below it. Ties and sparse classes can be conservative.

The evaluator enumerates every joint label and measures the actual joint set plus
its exact per-head projections. Projection sets may contain extra combinations;
`rectangular_joint_coverage` reports coverage of their Cartesian product separately.
`evaluation.max_candidates` prevents accidentally enumerating a huge label space.

## Plots and report

Open the [illustrated HTML report](docs/synthetic-report/report.html) or the
[PDF report](docs/synthetic-report/report.pdf). They compare distributions
**within each fixed conformal method**, with conclusions, numerical tables,
experimental settings, limitations, and source-run hashes.

To regenerate the report from the recorded comparison:

```bash
pip install -e '.[plots]'
python -m src report --results outputs/synthetic-comparison.csv \
  --output-dir docs/synthetic-report
```

`python -m src report` uses these same paths by default. VS Code also provides
**Synthetic: plots and report**. The command reads the exact run paths recorded
in the comparison CSV, verifies the metrics against their original runs, and
checks matching labels and point predictions for the six confidence controls.
To analyze a different set of runs, supply a comparison CSV with `experiment`,
`run`, `seed`, `method`, `alpha`, and the original metric columns. Duplicate
seed/method/alpha rows and incomplete or incompatible preset groups are rejected.

Eight figures show input confidence distributions, true-label plausibility,
prediction-set sizes at 90% nominal coverage, detailed curves for each method,
class behavior, and rare-class support. Individual figures are saved as PNG,
SVG, and vector PDF under `docs/synthetic-report/figures/`. The HTML embeds its
plots and can be read offline as a standalone file.

Plots show the mean and observed **seed range**; tables show mean ± sample
standard deviation. Neither represents a confidence interval. `summary.csv`
and `paired_differences.csv` retain aggregate results and changes from the
baseline. `source_runs.csv`, `source_caches.csv`, and `manifest.json` record
the inputs and verification details.

## Results and diagnostics

Each run creates `outputs/NAME/UTC_TIMESTAMP-ID/`:

| File | Purpose |
|---|---|
| `config.yaml`, `provenance.json` | Resolved configuration, package versions, source hashes, Git state. |
| `metrics.csv` | Joint/projected coverage, Wilson intervals, sizes, empty/singleton rates, classwise gaps, correct/error subsets. |
| `summary.csv` | Means and sample standard deviations across seeds, when multiple seeds run. |
| `class_metrics.csv` | Per-class support, covered count, coverage, and mean set size for joint labels and projections. |
| `group_metrics.csv` | Coverage and set size by learned/calibration group, with calibration support. |
| `subgroup_metrics.csv` | Joint coverage and set size by point correctness and configured class profile. |
| `prediction_metrics.csv` | Actual/target accuracy, macro precision, correct/error confidence, ECE, NLL, Brier score, for every split/head/joint. |
| `prediction_class_metrics.csv` | Realized label frequencies, per-class accuracy/precision, confidence, true-label probability. |
| `confidence_bins.csv` | Reliability bins: confidence, accuracy, support, true-label probability. |
| `calibration.csv` | Each class's group, calibration support, threshold, and always-included flag. |
| `seed-N/probabilities.npz` | Generated scores, labels, profile IDs, targets, and original settings. |
| `seed-N/calibrations.npz` | Fitted group mappings and thresholds for each method/alpha. |

Macro precision averages all configured classes, with zero for never-predicted
classes. Classwise coverage gaps average only classes observed in test, giving
each equal weight. Gaps are fractions: multiply by 100 for percentage points.
Inspect class support alongside worst-class coverage, especially under imbalance.
Empty correctness/profile subgroups have blank metrics rather than artificial zeros.

Use `subgroup_metrics.csv` to see whether larger sets recover confident errors,
and `class_metrics.csv` to see which classes standard CP undercovers. Use
`prediction_metrics.csv` to verify that the intended accuracy/confidence contrast
actually occurred. Under class profiles, changing label frequencies can change
average target accuracy; the empirical weighted target is exported too.

Set `artifacts.save_probabilities: false` to skip prediction caches, or
`evaluation.save_prediction_sets: true` to export full memberships and projections'
coverage/size arrays for individual examples. NumPy bundles use `allow_pickle=False`.

Replay a **new synthetic** cache with the original generation settings and seed:

```bash
python -m src run experiments/synthetic.yaml --set 'seeds=[42]' \
  --set 'conformal.alphas=[0.025,0.1]' \
  --predictions outputs/NAME/RUN_ID/seed-42/probabilities.npz
```

Replay can change conformal/evaluation settings, but rejects changes to dataset,
prediction-generation, or split settings to avoid mislabelling cached simulations.

```python
from src.conformal import candidate_space, load_method

method = load_method("outputs/NAME/RUN_ID/seed-42", method="clustered", alpha=0.1)
membership = method.predict(probabilities, candidate_space([12, 11]))
```

No test files are introduced. Verification uses full preset runs, inline numerical
checks, cache replay, calibration restoration, package/CLI checks, and Ruff.

## Initial observed patterns

The full presets were run with seeds 42/43/44. At `alpha=0.1`, these are means
across seeds; sizes count complete label tuples out of 132:

| Scenario | Top-1 joint accuracy | Standard coverage | Standard size | Classwise size | Clustered size |
|---|---:|---:|---:|---:|---:|
| Confident errors | 80.06% | 89.89% | 76.46 | 80.13 | 76.53 |
| Uncertain errors | 80.06% | 89.89% | 32.65 | 35.29 | 32.67 |
| Underconfident correct predictions | 80.06% | 89.75% | 112.07 | 112.26 | 112.15 |
| Different confidence profiles by class | 80.06% | 89.97% | 59.00 | 71.03 | 70.64 |
| Different accuracy by class | 79.89% | 89.95% | 76.84 | 51.91 | 51.30 |
| Imbalanced labels | 80.06% | 89.67% | 76.02 | 94.48 | 75.80 |

Accuracy alone does not determine prediction-set efficiency. Lower-confidence
errors substantially reduced set size here while retaining the same point
predictions and nearly the same coverage. Making correct predictions less
confident than mistakes made sets much larger.

Class-specific score distributions also matter: in the confidence-profile preset,
the mean absolute classwise coverage gap was **9.71 percentage points** for
standard, **3.76** for classwise, and **2.53** for clustered. The larger sets of the
latter methods came with more balanced class coverage. Under class-specific
accuracy, clustered CP improved both average set size and the classwise gap in
these runs. These observations depend on the synthetic generator and score;
they do not establish a universal ordering of the methods.

The verification run also produced `outputs/synthetic-comparison.csv`, combining
all 300 method/seed/alpha results from the 12 presets with source configuration
and run paths. Full per-run `summary.csv` files retain the standard deviations.
