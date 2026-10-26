"""Split conformal predictors over the full Cartesian categorical label space."""

import json
import math
from abc import ABC, abstractmethod
from itertools import product
from pathlib import Path

import numpy as np

from .registry import METHODS


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
        if not np.allclose(p.sum(axis=1), 1, atol=1e-6):
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


def true_task_scores(probabilities, labels):
    return np.column_stack(
        [1 - p[np.arange(len(labels)), labels[:, k]] for k, p in enumerate(probabilities)]
    )


class ConformalMethod(ABC):
    """fit uses shape data; calibrate alone may access calibration labels.

    New methods implement score_true and score_candidates (lower is better).
    Candidates have shape (J, K); candidate scores have shape (N, J).
    """

    guarantee = "joint"

    def fit(self, probabilities, labels):
        arrays = validate_probabilities(probabilities, labels)
        self.num_classes = [p.shape[1] for p in arrays]
        self.threshold = None
        return self

    def _validate(self, probabilities, labels=None):
        arrays = validate_probabilities(probabilities, labels)
        if [p.shape[1] for p in arrays] != self.num_classes:
            raise ValueError("Probability head dimensions differ from fitted method")
        return arrays

    def calibrate(self, probabilities, labels, alpha):
        self._validate(probabilities, labels)
        self.alpha = alpha
        self.calibration_scores = self.score_true(probabilities, labels)
        self.threshold = conformal_quantile(self.calibration_scores, alpha)
        return self

    @abstractmethod
    def score_true(self, probabilities, labels):
        pass

    @abstractmethod
    def score_candidates(self, probabilities, candidates):
        pass

    def predict(self, probabilities, candidates):
        self._validate(probabilities)
        candidates = self._validate_candidates(candidates)
        if self.threshold is None:
            raise RuntimeError("Calibrate the method before prediction")
        return self.score_candidates(probabilities, candidates) <= self.threshold

    def _validate_candidates(self, candidates):
        candidates = np.asarray(candidates)
        if (
            candidates.ndim != 2
            or candidates.shape[1] != len(self.num_classes)
            or candidates.dtype.kind not in "iu"
        ):
            raise ValueError("Candidates must be an integer array of shape (J, K)")
        for k, c in enumerate(self.num_classes):
            if (candidates[:, k] < 0).any() or (candidates[:, k] >= c).any():
                raise ValueError(f"Candidate label outside task {k} vocabulary")
        return candidates

    def state(self):
        return {
            "threshold": np.asarray(self.threshold),
            "alpha": np.asarray(self.alpha),
            "num_classes": np.asarray(self.num_classes),
            "calibration_scores": self.calibration_scores,
        }


@METHODS.register("max_norm")
class MaxNorm(ConformalMethod):
    """Maximum per-task 1-p(true class) score; one joint threshold."""

    def score_true(self, probabilities, labels):
        return true_task_scores(probabilities, labels).max(axis=1)

    def score_candidates(self, probabilities, candidates):
        return np.maximum.reduce([1 - p[:, candidates[:, k]] for k, p in enumerate(probabilities)])


@METHODS.register("marginal")
class Marginal(ConformalMethod):
    """Uncorrected per-task split CP: intentionally has no joint guarantee."""

    guarantee = "per-task only (native sets)"

    def __init__(self, weights=None):
        if weights is not None:
            raise ValueError("weights are supported by bonferroni, not marginal")

    def task_alphas(self, alpha):
        return np.full(len(self.num_classes), alpha)

    def calibrate(self, probabilities, labels, alpha):
        self._validate(probabilities, labels)
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie strictly between zero and one")
        self.alpha = alpha
        self.calibration_scores = true_task_scores(probabilities, labels)
        self.allocated_alphas = self.task_alphas(alpha)
        self.threshold = np.asarray(
            [
                conformal_quantile(self.calibration_scores[:, k], a)
                for k, a in enumerate(self.allocated_alphas)
            ]
        )
        return self

    def score_true(self, probabilities, labels):
        return true_task_scores(probabilities, labels)

    def score_candidates(self, probabilities, candidates):
        return np.stack([1 - p[:, candidates[:, k]] for k, p in enumerate(probabilities)], axis=-1)

    def predict(self, probabilities, candidates):
        self._validate(probabilities)
        candidates = self._validate_candidates(candidates)
        if self.threshold is None:
            raise RuntimeError("Calibrate the method before prediction")
        return (self.score_candidates(probabilities, candidates) <= self.threshold).all(axis=-1)

    def state(self):
        return {**super().state(), "allocated_alphas": self.allocated_alphas}

    def native_sets(self, probabilities):
        """Keep task sets even when another task's set makes their product empty."""
        self._validate(probabilities)
        if self.threshold is None:
            raise RuntimeError("Calibrate the method before prediction")
        return [1 - p <= self.threshold[k] for k, p in enumerate(probabilities)]


@METHODS.register("bonferroni")
class Bonferroni(Marginal):
    """Fixed positive error allocation; sum(alpha_k)=alpha, no independence assumption."""

    guarantee = "joint"

    def __init__(self, weights=None):
        self.weights = weights

    def fit(self, probabilities, labels):
        super().fit(probabilities, labels)
        k = len(self.num_classes)
        self.allocation = (
            np.full(k, 1 / k) if self.weights is None else np.asarray(self.weights, dtype=float)
        )
        if (
            self.allocation.shape != (k,)
            or not np.isfinite(self.allocation).all()
            or (self.allocation <= 0).any()
            or not np.isclose(self.allocation.sum(), 1, atol=1e-10, rtol=0)
        ):
            raise ValueError("Bonferroni weights must be positive, sum to 1, and match K")
        return self

    def task_alphas(self, alpha):
        return alpha * self.allocation

    def state(self):
        return {**super().state(), "allocation": self.allocation}


@METHODS.register("mahalanobis")
class Mahalanobis(ConformalMethod):
    """Ellipsoid in concatenated one-hot residual space, restricted to categorical tuples.

    Default: |one_hot(y)-p| and uncentered distance as in Katsios & Papadopulos (2024).
    Multiclass heads, a disjoint shape split, shrinkage and ridge are extensions.
    """

    def __init__(self, residual="absolute", shrinkage=0.1, ridge=1e-5, center=False):
        if residual not in {"absolute", "signed"}:
            raise ValueError("residual must be absolute or signed")
        if not np.isfinite(shrinkage) or not 0 <= shrinkage <= 1:
            raise ValueError("shrinkage must lie in [0, 1]")
        if not np.isfinite(ridge) or ridge <= 0:
            raise ValueError("ridge must be positive")
        if not isinstance(center, bool):
            raise ValueError("center must be boolean")
        self.residual, self.shrinkage, self.ridge, self.center = residual, shrinkage, ridge, center

    def _residuals(self, probabilities, labels):
        result = np.concatenate(
            [np.eye(c)[labels[:, k]] - probabilities[k] for k, c in enumerate(self.num_classes)],
            axis=-1,
        )
        return np.abs(result) if self.residual == "absolute" else result

    def fit(self, probabilities, labels):
        super().fit(probabilities, labels)
        if len(labels) < 2:
            raise ValueError("Mahalanobis covariance estimation requires >=2 shape examples")
        residuals = self._residuals(probabilities, labels)
        covariance = np.cov(residuals, rowvar=False, ddof=1)
        target = np.diag(np.diag(covariance))
        covariance = (1 - self.shrinkage) * covariance + self.shrinkage * target
        covariance += self.ridge * np.eye(len(covariance))
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, self.ridge)
        self.covariance = (eigenvectors * eigenvalues) @ eigenvectors.T
        self.precision = (eigenvectors / eigenvalues) @ eigenvectors.T
        self.offset = residuals.mean(axis=0) if self.center else np.zeros(residuals.shape[1])
        self.condition_number = float(eigenvalues.max() / eigenvalues.min())
        return self

    def _distance(self, residuals):
        residuals = residuals - self.offset
        squared = np.einsum(
            "...i,ij,...j->...", residuals, self.precision, residuals, optimize=True
        )
        return np.sqrt(np.maximum(squared, 0))

    def score_true(self, probabilities, labels):
        return self._distance(self._residuals(probabilities, labels))

    def score_candidates(self, probabilities, candidates):
        embedding = np.concatenate(
            [np.eye(c)[candidates[:, k]] for k, c in enumerate(self.num_classes)], axis=-1
        )
        residuals = embedding[None, :, :] - np.concatenate(probabilities, axis=1)[:, None, :]
        if self.residual == "absolute":
            residuals = np.abs(residuals)
        return self._distance(residuals)

    def state(self):
        return {
            **super().state(),
            "covariance": self.covariance,
            "precision": self.precision,
            "offset": self.offset,
            "condition_number": np.asarray(self.condition_number),
        }


def load_method(directory, method=None, alpha=None):
    """Restore from a calibration bundle or a legacy per-method directory.

    For bundles, select by configured method identifier and alpha. Selectors may
    be omitted when the remaining entry is unique. A seed directory is accepted
    as shorthand for its calibrations.npz file.
    """
    path = Path(directory)
    if path.is_dir() and (path / "method.json").exists():
        metadata = json.loads((path / "method.json").read_text())
        if method is not None and method not in {metadata["name"], path.parent.name}:
            raise ValueError(f"Method {method!r} does not match {path}")
        archive_path, prefix, keys = path / "calibration.npz", "", None
    else:
        archive_path = path / "calibrations.npz" if path.is_dir() else path
        with np.load(archive_path, allow_pickle=False) as archive:
            bundle = json.loads(str(archive["metadata"]))
        if bundle.get("kind") != "mocp-calibrations" or bundle.get("schema_version") != 1:
            raise ValueError("Unsupported calibration bundle schema")
        entries = bundle["entries"]
        if method is not None:
            exact = [entry for entry in entries if entry["identifier"] == method]
            entries = exact or [entry for entry in entries if entry["name"] == method]
        if alpha is not None:
            entries = [entry for entry in entries if entry["alpha"] == float(alpha)]
        if len(entries) != 1:
            available = ", ".join(
                f"{entry['identifier']}@{entry['alpha']:g}" for entry in bundle["entries"]
            )
            raise ValueError(
                f"Select one calibration using method and alpha; available: {available}"
            )
        metadata = entries[0]
        prefix, keys = metadata["prefix"], metadata["state_keys"]
    restored = METHODS.create(metadata["name"], **metadata["params"])
    with np.load(archive_path, allow_pickle=False) as state:
        for key in state.files if keys is None else keys:
            value = state[f"{prefix}{key}"]
            if key == "num_classes":
                value = value.tolist()
            elif value.ndim == 0:
                value = value.item()
            setattr(restored, key, value)
    if alpha is not None and restored.alpha != float(alpha):
        raise ValueError(f"Alpha {alpha!r} does not match the saved calibration")
    return restored
