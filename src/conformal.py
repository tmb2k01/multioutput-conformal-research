"""Standard, classwise, and clustered split conformal on joint labels.

All methods use s(x,y) = 1 - product_k p_k(y_k). The product is a scoring
choice; validity does not require statistically independent output heads.
"""

import json
import math
from itertools import product
from pathlib import Path

import numpy as np


def validate_probabilities(probabilities, labels=None):
    if not probabilities:
        raise ValueError("At least one probability head is required")
    arrays = [np.asarray(p, dtype=float) for p in probabilities]
    n = len(arrays[0])
    if n == 0:
        raise ValueError("Empty probability batch")
    for p in arrays:
        if p.ndim != 2 or len(p) != n or p.shape[1] < 2:
            raise ValueError("Heads must have shape (same nonzero N, C_k >= 2)")
        if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
            raise ValueError("Probabilities must be finite and in [0, 1]")
        if not np.allclose(p.sum(axis=1), 1, atol=1e-6, rtol=0):
            raise ValueError("Each probability row must sum to one")
    if labels is not None:
        labels = np.asarray(labels)
        if labels.shape != (n, len(arrays)) or labels.dtype.kind not in "iu":
            raise ValueError("Labels must be an integer array of shape (N, K)")
        for k, p in enumerate(arrays):
            if (labels[:, k] < 0).any() or (labels[:, k] >= p.shape[1]).any():
                raise ValueError(f"Labels out of bounds in task {k}")
    return arrays


def conformal_quantile(scores, alpha):
    """ceil((n+1)(1-alpha))-th order statistic, with an infinity sentinel."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Calibration scores must be a nonempty finite vector")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    return np.inf if rank > len(scores) else float(np.partition(scores, rank - 1)[rank - 1])


def candidate_space(num_classes, max_candidates=100_000):
    count = math.prod(num_classes)
    if count > max_candidates:
        raise ValueError(
            f"Exact joint prediction needs {count:,} candidates; limit is {max_candidates:,}. "
            "Increase evaluation.max_candidates deliberately or implement exact structured search."
        )
    return np.asarray(list(product(*(range(c) for c in num_classes))), dtype=np.int64)


def joint_scores(probabilities, candidates):
    product_probability = np.ones((len(probabilities[0]), len(candidates)))
    for k, p in enumerate(probabilities):
        product_probability *= p[:, candidates[:, k]]
    return 1 - product_probability


def true_scores(probabilities, labels):
    values = np.ones(len(labels))
    for k, p in enumerate(probabilities):
        values *= p[np.arange(len(labels)), labels[:, k]]
    return 1 - values


class Standard:
    guarantee = "joint marginal"

    def fit(self, probabilities, labels):
        arrays = validate_probabilities(probabilities, labels)
        self.num_classes = [p.shape[1] for p in arrays]
        self.group_ids = np.zeros(math.prod(self.num_classes), dtype=int)
        self.thresholds = None
        return self

    def _validate(self, probabilities, labels=None):
        arrays = validate_probabilities(probabilities, labels)
        if [p.shape[1] for p in arrays] != self.num_classes:
            raise ValueError("Probability dimensions differ from the fitted method")
        return arrays

    def calibrate(self, probabilities, labels, alpha):
        arrays = self._validate(probabilities, labels)
        self.alpha = alpha
        self.calibration_scores = true_scores(arrays, labels)
        classes = np.ravel_multi_index(labels.T, self.num_classes)
        groups = self.group_ids[classes]
        self.calibration_counts = np.bincount(groups, minlength=int(self.group_ids.max()) + 1)
        self.thresholds = np.asarray(
            [
                conformal_quantile(self.calibration_scores[groups == g], alpha) if count else np.inf
                for g, count in enumerate(self.calibration_counts)
            ]
        )
        return self

    def predict(self, probabilities, candidates):
        arrays = self._validate(probabilities)
        if self.thresholds is None:
            raise RuntimeError("Calibrate before prediction")
        candidates = np.asarray(candidates)
        if (
            candidates.ndim != 2
            or candidates.shape[1] != len(self.num_classes)
            or candidates.dtype.kind not in "iu"
        ):
            raise ValueError("Candidates must have shape (J, K) and integer labels")
        ids = np.ravel_multi_index(candidates.T, self.num_classes)
        return joint_scores(arrays, candidates) <= self.thresholds[self.group_ids[ids]]

    def state(self):
        return {
            "num_classes": np.asarray(self.num_classes),
            "alpha": np.asarray(self.alpha),
            "group_ids": self.group_ids,
            "thresholds": self.thresholds,
            "calibration_counts": self.calibration_counts,
        }


class Classwise(Standard):
    guarantee = "joint class-conditional"

    def fit(self, probabilities, labels):
        super().fit(probabilities, labels)
        self.group_ids = np.arange(math.prod(self.num_classes))
        return self


class Clustered(Standard):
    """Cluster classes by independent empirical score-quantile vectors.

    Rare/unseen classes share a separate fallback cluster. The mapping is frozen
    before calibration. Coverage is cluster-conditional, not per-class in general.
    """

    guarantee = "joint cluster-conditional"

    def __init__(self, n_clusters=3, quantiles=None, min_class_samples=10, max_iter=100):
        self.quantiles = [0.1, 0.3, 0.5, 0.7, 0.9] if quantiles is None else quantiles
        for value in [n_clusters, min_class_samples, max_iter]:
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(
                    "Cluster counts, min_class_samples, max_iter must be positive ints"
                )
        levels = np.asarray(self.quantiles)
        if (
            levels.ndim != 1
            or not levels.size
            or levels.dtype.kind not in "fiu"
            or not np.isfinite(levels).all()
            or (levels < 0).any()
            or (levels > 1).any()
            or (np.diff(levels) <= 0).any()
        ):
            raise ValueError("Cluster quantiles must be increasing values in [0,1]")
        self.n_clusters = n_clusters
        self.min_class_samples = min_class_samples
        self.max_iter = max_iter

    def fit(self, probabilities, labels):
        super().fit(probabilities, labels)
        scores = true_scores(probabilities, labels)
        ids = np.ravel_multi_index(labels.T, self.num_classes)
        n_classes = len(self.group_ids)
        self.clustering_counts = np.bincount(ids, minlength=n_classes)
        eligible = np.flatnonzero(self.clustering_counts >= self.min_class_samples)
        self.embeddings = np.full((n_classes, len(self.quantiles)), np.nan)
        for index in eligible:
            self.embeddings[index] = np.quantile(scores[ids == index], self.quantiles)
        if not len(eligible):
            # No class has sufficient fitting support: pool all into the fallback.
            self.group_ids[:] = 0
            return self
        features = self.embeddings[eligible]
        n_centers = min(self.n_clusters, len(np.unique(features, axis=0)))
        # Deterministic farthest-point initialization, then ordinary Lloyd updates.
        first = np.argmax(np.sum((features - features.mean(axis=0)) ** 2, axis=1))
        centers = [features[first]]
        for _ in range(1, n_centers):
            distances = ((features[:, None] - np.asarray(centers)) ** 2).sum(axis=2).min(axis=1)
            centers.append(features[distances.argmax()])
        centers = np.asarray(centers)
        assignments = np.full(len(features), -1)
        for _ in range(self.max_iter):
            updated = ((features[:, None] - centers) ** 2).sum(axis=2).argmin(axis=1)
            if np.array_equal(updated, assignments):
                break
            assignments = updated
            for g in range(n_centers):
                if np.any(assignments == g):
                    centers[g] = features[assignments == g].mean(axis=0)
        # Compact group numbers if a centroid became unused.
        _, assignments = np.unique(assignments, return_inverse=True)
        fallback = int(assignments.max()) + 1
        self.group_ids[:] = fallback
        self.group_ids[eligible] = assignments
        return self

    def state(self):
        return {
            **super().state(),
            "clustering_counts": self.clustering_counts,
            "embeddings": self.embeddings,
        }


METHODS = {"standard": Standard, "classwise": Classwise, "clustered": Clustered}


def create_method(name, **params):
    if name not in METHODS:
        raise ValueError(f"Unknown conformal method {name!r}; choose from {list(METHODS)}")
    try:
        return METHODS[name](**params)
    except TypeError as error:
        raise ValueError(f"Invalid parameters for {name}: {error}") from error


def load_method(path, method, alpha):
    """Restore a selected method/alpha from a synthetic calibrations.npz bundle."""
    path = Path(path)
    if path.is_dir():
        path = path / "calibrations.npz"
    with np.load(path, allow_pickle=False) as cache:
        meta = json.loads(str(cache["metadata"]))
        matches = [
            entry
            for entry in meta["entries"]
            if entry["method"] == method and entry["alpha"] == alpha
        ]
        if len(matches) != 1:
            raise ValueError("Select one saved method label and alpha")
        entry = matches[0]
        result = create_method(entry["name"], **entry["params"])
        prefix = entry["prefix"]
        for key in cache.files:
            if key.startswith(prefix):
                setattr(result, key[len(prefix) :], cache[key].copy())
        result.num_classes = result.num_classes.tolist()
        result.alpha = float(result.alpha)
    return result
