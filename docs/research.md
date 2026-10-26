# Research formulation and experimental protocol

For K categorical tasks, the goal is

`P(Y in Gamma(X)) >= 1-alpha`.

For rectangular sets `Gamma(X) = C_1(X) x ... x C_K(X)`, this is the event that
every task is covered. Task-wise 90% coverage does not guarantee 90% joint
coverage. If the coverage events are independent, `0.9**5 = 0.59049`.
Outputs being independent does not automatically make learned coverage events
independent: they may share features, training data and calibration data.

## Categorical Mahalanobis adaptation

Represent a candidate tuple y by
`e(y) = concat(one_hot(y_1), ..., one_hot(y_K))`, and predictions by
`p(x) = concat(p_1(x), ..., p_K(x))`.

The default error embedding is `r(x,y) = abs(e(y)-p(x))`. Estimate the covariance
of true-label errors on a disjoint **shape split**. Regularize as

`Sigma_reg = (1-lambda)*Sigma + lambda*diag(Sigma) + ridge*I`.

The score is `s(x,y) = sqrt(r(x,y).T @ inv(Sigma_reg) @ r(x,y))`.
With `center: true`, subtract the shape-split residual mean. With
`residual: signed`, use `e(y)-p(x)` without taking absolute values.

An ellipsoid is convex in residual space. For absolute residuals, its preimage
in embedding space need not be a single ellipsoid. In all variants the final
region is a **discrete set of tuples**, rather than a continuous regression region.
Class codes are never treated as ordinal coordinates, so category renaming does
not introduce an artificial Euclidean ordering.

Binary and multiclass one-hot residuals have linear dependencies. Shrinkage and
ridge ensure positive definiteness; a naive unregularized inverse can be singular.
Each task contributes its full class probability vector. Tasks with more classes
therefore have larger embedding blocks; block weighting is a useful future ablation.

For n calibration observations and k = ceil((n+1)*(1-alpha)), choose the kth
smallest calibration score (infinity when k > n), then define
`Gamma(x) = {y: s(x,y) <= q}` over the **entire** product label space. The model,
error embedding, covariance and any shape hyperparameters must be fixed independently
of the final calibration labels. The usual rank argument then provides joint coverage
under exchangeability. No multivariate normality is required.

## Marginal objectives and projection semantics

The exact projection is `C_k(x) = {y_k: some y in Gamma(x) has coordinate y_k}`.
The event `Y in Gamma(X)` implies every `Y_k in C_k(X)`, hence each projected
task has a coverage lower bound of `1-alpha`. This is true of **all** joint-valid
methods here, including max-norm. Mahalanobis does not have a separate theorem
that it will improve task-wise coverage compared with these baselines.

The Cartesian product of the projections can strictly contain Gamma. Report
the actual joint-set coverage, projected rectangular joint coverage, projected
marginal coverages and both set cardinalities. Comparing only projected coverage
would hide whether the dependence-aware joint region actually contains the truth.

The primary comparison with Bonferroni is classwise coverage balance, alongside
joint validity and prediction-set size. For each scope, compute
`CovGap = mean_c abs(coverage_c - (1-alpha))` over observed true classes, weighting
each class equally. Joint classes are complete tuples; task classes are evaluated
using the exact projections. Also report mean classwise shortfall below target,
worst observed class coverage, and each class's sample count and coverage. Gaps
are stored as fractions and displayed as percentage points. Unobserved classes
are excluded from averages and explicitly recorded with zero support. Rare-class
estimates are noisy; repeated splits of one dataset are not independent replications.

Higher coverage alone can be obtained by returning the full label space. Retain
the joint guarantee, reduce classwise undercoverage and excess coverage, and
compare joint and projected set sizes. Smaller sets alone do not establish an
improvement on this objective. Tune any score or hyperparameters for classwise
coverage gap and size on development data, freeze them before final calibration,
and report all three scopes rather than choosing a favorable scope after evaluation.
All projected marginal coverages need not simultaneously equal the joint target.
Class-conditional or group-conditional coverage would require additional methods;
it is not guaranteed by these split conformal constructions.

For rectangular predictors, an empty set on one task makes the whole Cartesian
product empty. Its exact projections then differ from the other original task
sets. Report native per-task sets separately (`*_native_*` metrics); the marginal
baseline's per-task guarantees refer to these native sets, not to projections of
an empty product.

## Initial study

1. Run marginal, equal-allocation Bonferroni, max-norm and absolute Mahalanobis on
   identical model predictions and fixed splits for both real datasets.
2. Repeat over predeclared seeds and alpha values. Report uncertainty and both
   coverage and set-size metrics, including empty sets rather than forcing a label in.
3. Use synthetic shared-noise strengths 0, 0.5 and 0.8 to study sensitivity to
   residual dependence. This is a controlled latent-noise ablation, not a claim
   that task coverage events are unconditionally independent at zero strength.
4. Compare full covariance with diagonal covariance (`shrinkage: 1`), signed
   residuals, centering and fixed Bonferroni weights. The ablation YAML trains once.
5. For tuned shape templates or allocation, optimize on a separate development
   partition, freeze the choice, then calibrate once on untouched data. Avoid
   choosing the best method, alpha or hyperparameters using the reported test set.
6. Evaluate class/group-specific performance descriptively. UTKFace may contain
   repeated identities not identified by filenames; byte deduplication does not
   establish subject independence. Establish identity-aware splits before making
   subject-generalization claims.

The supplied UTKFace archives also contain byte-identical images with conflicting
gender/race annotations. The default preprocessing excludes every image hash
involved in such a conflict, records the conflict annotations, and deduplicates
the remaining images before splitting. This changes the evaluation population;
report this policy with any results. `dataset.conflict_policy: error` provides a
strict alternative. No annotation is arbitrarily chosen as the ground truth.

These are exchangeable-example guarantees, not guarantees under dataset shift.
Label-independent splits support the basic argument, but do not repair dependence
already present among the underlying collected observations.

The Katsios & Papadopulos paper motivates absolute residuals and error covariance.
The Tumu et al. paper motivates fitting region shapes separately from final
calibration; its multi-modal convex-template optimization is outside this initial
implementation. A possible extension is development-set optimization of a
positive-definite shape matrix with discrete candidate cardinality as the objective.
