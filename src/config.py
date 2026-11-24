"""Small, strictly validated YAML schema for synthetic prediction experiments."""

import math
import re
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def integer(value, minimum=1):
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def probability_vector(values, size, name):
    array = np.asarray(values)
    if (
        array.shape != (size,)
        or array.dtype.kind not in "fiu"
        or not np.isfinite(array).all()
        or (array < 0).any()
        or not np.isclose(array.sum(), 1, atol=1e-10, rtol=0)
    ):
        raise ValueError(f"{name} must contain {size} nonnegative probabilities summing to one")


@dataclass
class ConfidenceConfig:
    distribution: str = "beta"
    mean: float = 0.9
    concentration: float = 40.0
    low: float = 0.5
    high: float = 1.0

    def validate(self, floor):
        if self.distribution not in {"beta", "uniform", "constant"}:
            raise ValueError("confidence.distribution must be beta, uniform or constant")
        if not all(number(v) for v in [self.low, self.high, self.mean, self.concentration]):
            raise ValueError("Confidence parameters must be finite numbers")
        if self.distribution == "constant":
            if not floor < self.mean < 1:
                raise ValueError(f"Constant confidence must satisfy {floor} < mean < 1")
            return
        if not floor <= self.low < self.high <= 1:
            raise ValueError(f"Confidence bounds must satisfy {floor} <= low < high <= 1")
        if self.distribution == "beta" and (
            not self.low < self.mean < self.high or self.concentration <= 0
        ):
            raise ValueError("Confidence requires low < mean < high and concentration > 0")


@dataclass
class DatasetConfig:
    n_samples: int = 30000
    num_classes: list[int] = field(default_factory=lambda: [12, 11])
    task_names: list[str] | None = None
    class_probabilities: list[list[float]] | None = None
    # Flattened Cartesian order: for [12, 11], joint index = color * 11 + type.
    joint_probabilities: list[float] | None = None


@dataclass
class ClassProfile:
    name: str
    class_range: list[int]
    accuracy: float | list[float] | None = None
    correct_confidence: ConfidenceConfig | None = None
    incorrect_confidence: ConfidenceConfig | None = None
    tail_concentration: float | None = None
    true_class_weight: float | None = None


@dataclass
class PredictionConfig:
    accuracy: float | list[float] = 0.8
    error_correlation: float = 1.0
    correct_confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    incorrect_confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    tail_concentration: float = 1.0
    true_class_weight: float = 1.0
    confusion_matrices: list[list[list[float]]] | None = None
    class_profiles: list[ClassProfile] = field(default_factory=list)


@dataclass
class SplitConfig:
    clustering: float = 0.2
    calibration: float = 0.4
    test: float = 0.4


@dataclass
class MethodConfig:
    name: str
    label: str | None = None
    params: dict = field(default_factory=dict)

    @property
    def identifier(self):
        return self.label or self.name


@dataclass
class ConformalConfig:
    alphas: list[float] = field(default_factory=lambda: [0.05, 0.1, 0.2])
    methods: list[MethodConfig] = field(default_factory=lambda: [MethodConfig("standard")])
    calibration_samples: int | None = None


@dataclass
class EvaluationConfig:
    batch_size: int = 256
    max_candidates: int = 100000
    candidate_batch_size: int = 256
    save_prediction_sets: bool = False


@dataclass
class ArtifactConfig:
    save_probabilities: bool = True


@dataclass
class Config:
    name: str = "synthetic"
    seeds: list[int] = field(default_factory=lambda: [42, 43, 44])
    output_dir: str = "outputs"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    predictions: PredictionConfig = field(default_factory=PredictionConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    conformal: ConformalConfig = field(default_factory=ConformalConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)

    def to_dict(self):
        return asdict(self)

    def validate(self):
        if not isinstance(self.name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", self.name):
            raise ValueError("name must contain only letters, digits, underscores and hyphens")
        if not isinstance(self.output_dir, str) or not self.output_dir:
            raise ValueError("output_dir must be a nonempty path")
        if (
            not self.seeds
            or any(not integer(seed, 0) for seed in self.seeds)
            or len(set(self.seeds)) != len(self.seeds)
        ):
            raise ValueError("seeds must be distinct nonnegative integers")
        data, pred = self.dataset, self.predictions
        if not integer(data.n_samples) or not data.num_classes:
            raise ValueError("n_samples must be positive; num_classes must be nonempty")
        if any(not integer(c, 2) for c in data.num_classes):
            raise ValueError("num_classes must contain integers >= 2")
        tasks, classes = len(data.num_classes), math.prod(data.num_classes)
        if data.task_names is not None and (
            len(data.task_names) != tasks
            or any(
                not isinstance(t, str)
                or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", t)
                or t == "joint"
                for t in data.task_names
            )
            or len(set(data.task_names)) != tasks
        ):
            raise ValueError("task_names must be distinct names matching the number of heads")
        if data.class_probabilities is not None and data.joint_probabilities is not None:
            raise ValueError("Choose class_probabilities OR joint_probabilities")
        if data.class_probabilities is not None:
            if len(data.class_probabilities) != tasks:
                raise ValueError("class_probabilities needs one probability vector per head")
            for k, c in enumerate(data.num_classes):
                probability_vector(data.class_probabilities[k], c, f"class_probabilities[{k}]")
        if data.joint_probabilities is not None:
            probability_vector(data.joint_probabilities, classes, "joint_probabilities")
        fractions = list(asdict(self.splits).values())
        if any(not number(f) or f <= 0 for f in fractions) or not math.isclose(
            sum(fractions), 1, abs_tol=1e-10
        ):
            raise ValueError("The three split fractions must be positive and sum to one")
        if min(split_counts(data.n_samples, self.splits)) < 2:
            raise ValueError("Each split needs at least two observations")
        budget = self.conformal.calibration_samples
        if budget is not None and (
            not integer(budget, 2) or budget > split_counts(data.n_samples, self.splits)[1]
        ):
            raise ValueError(
                "calibration_samples must be >=2 and no larger than the calibration pool"
            )
        floor = max(1 / c for c in data.num_classes)

        def accuracy(value):
            values = value if isinstance(value, list) else [value] * tasks
            if len(values) != tasks or any(not number(a) or not 0 <= a <= 1 for a in values):
                raise ValueError("accuracy must be in [0,1], either a scalar or one value per head")

        accuracy(pred.accuracy)
        if not number(pred.error_correlation) or not 0 <= pred.error_correlation <= 1:
            raise ValueError("error_correlation must lie in [0,1]")
        for cfg in [pred.correct_confidence, pred.incorrect_confidence]:
            cfg.validate(floor)
        for value in [pred.tail_concentration, pred.true_class_weight]:
            if not number(value) or value <= 0:
                raise ValueError("Tail concentration and true class weight must be positive")
        used, names = set(), {"default"}
        for profile in pred.class_profiles:
            if not isinstance(profile.name, str) or not profile.name or profile.name in names:
                raise ValueError(
                    "Class profile names must be nonempty, distinct, and not 'default'"
                )
            names.add(profile.name)
            bounds = profile.class_range
            if (
                len(bounds) != 2
                or any(not integer(b, 0) for b in bounds)
                or not 0 <= bounds[0] < bounds[1] <= classes
            ):
                raise ValueError("class_range must be [start, stop] within flattened joint labels")
            selected = set(range(*bounds))
            if used.intersection(selected):
                raise ValueError("Class profile ranges must not overlap")
            used.update(selected)
            if profile.accuracy is not None:
                accuracy(profile.accuracy)
            for cfg in [profile.correct_confidence, profile.incorrect_confidence]:
                if cfg is not None:
                    cfg.validate(floor)
            for value in [profile.tail_concentration, profile.true_class_weight]:
                if value is not None and (not number(value) or value <= 0):
                    raise ValueError("Profile tail parameters must be positive")
        if pred.confusion_matrices is not None:
            if len(pred.confusion_matrices) != tasks:
                raise ValueError("confusion_matrices needs one matrix per head")
            for k, c in enumerate(data.num_classes):
                matrix = np.asarray(pred.confusion_matrices[k])
                if matrix.shape != (c, c):
                    raise ValueError(f"confusion_matrices[{k}] must have shape ({c}, {c})")
                for row in matrix:
                    probability_vector(row, c, f"confusion_matrices[{k}] row")
                if np.any(np.diag(matrix) != 0):
                    raise ValueError("Confusion matrix diagonals must be zero (errors only)")
        if (
            not self.conformal.alphas
            or any(not number(a) or not 0 < a < 1 for a in self.conformal.alphas)
            or len(set(self.conformal.alphas)) != len(self.conformal.alphas)
        ):
            raise ValueError("alphas must be distinct values strictly between zero and one")
        ids = [m.identifier for m in self.conformal.methods]
        if (
            not ids
            or len(set(ids)) != len(ids)
            or any(not isinstance(x, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", x) for x in ids)
        ):
            raise ValueError("Methods require distinct names or safe labels")
        for value in [
            self.evaluation.batch_size,
            self.evaluation.candidate_batch_size,
            self.evaluation.max_candidates,
        ]:
            if not integer(value):
                raise ValueError("Evaluation batch sizes and candidate limit must be positive")
        if classes > self.evaluation.max_candidates:
            raise ValueError(f"{classes} joint labels exceed evaluation.max_candidates")
        for value in [self.evaluation.save_prediction_sets, self.artifacts.save_probabilities]:
            if not isinstance(value, bool):
                raise ValueError("Boolean settings require YAML true/false")
        from .conformal import create_method

        for method in self.conformal.methods:
            create_method(method.name, **method.params)


def split_counts(n, cfg):
    counts = np.floor(np.asarray(list(asdict(cfg).values())) * n).astype(int)
    counts[-1] += n - counts.sum()
    return counts


def deep_merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def read_yaml(path, seen=()):
    path = Path(path).resolve()
    if path in seen:
        raise ValueError(f"Cyclic YAML inheritance: {path}")
    with path.open() as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    parent = value.pop("extends", None)
    if parent:
        value = deep_merge(read_yaml(path.parent / parent, (*seen, path)), value)
    return value


def load_config(path, overrides=()):
    return config_from_dict(read_yaml(path), overrides)


def config_from_dict(mapping, overrides=()):
    """Use the same schema for individual experiments and cases within a study."""
    raw = deep_merge(Config().to_dict(), mapping)
    for override in overrides:
        key, separator, value = override.partition("=")
        if not separator:
            raise ValueError("Overrides use dotted.key=YAML-value")
        parts, target = key.split("."), raw
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                raise ValueError(f"Unknown configuration path: {key}")
            target = target[part]
        if parts[-1] not in target:
            raise ValueError(f"Unknown configuration key: {key}")
        target[parts[-1]] = yaml.safe_load(value)
    try:
        pred = raw["predictions"]
        for key in ["correct_confidence", "incorrect_confidence"]:
            pred[key] = ConfidenceConfig(**pred[key])
        for profile in pred["class_profiles"]:
            for key in ["correct_confidence", "incorrect_confidence"]:
                if profile.get(key) is not None:
                    profile[key] = ConfidenceConfig(**deep_merge(asdict(pred[key]), profile[key]))
        pred["class_profiles"] = [ClassProfile(**p) for p in pred["class_profiles"]]
        raw["conformal"]["methods"] = [MethodConfig(**m) for m in raw["conformal"]["methods"]]
        for key, cls in [
            ("dataset", DatasetConfig),
            ("predictions", PredictionConfig),
            ("splits", SplitConfig),
            ("conformal", ConformalConfig),
            ("evaluation", EvaluationConfig),
            ("artifacts", ArtifactConfig),
        ]:
            raw[key] = cls(**raw[key])
        result = Config(**raw)
    except (TypeError, KeyError, AttributeError) as error:
        raise ValueError(f"Invalid configuration: {error}") from error
    result.validate()
    return result
