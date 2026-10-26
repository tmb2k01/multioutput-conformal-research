"""Typed YAML configuration, inheritance, and CLI overrides."""

import re
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml


@dataclass
class DatasetConfig:
    name: str = "synthetic"
    root: str = "data"
    download: bool = True
    verify_images: bool = True
    conflict_policy: str = "exclude"
    max_samples: int | None = None
    image_size: int = 64
    n_samples: int = 3000
    n_features: int = 12
    num_classes: list[int] = field(default_factory=lambda: [3, 4, 2])
    dependence: float = 0.7
    sources: list[dict] | None = None


@dataclass
class SplitConfig:
    train: float = 0.5
    validation: float = 0.1
    shape: float = 0.1
    calibration: float = 0.15
    test: float = 0.15


@dataclass
class ModelConfig:
    name: str = "linear"
    width: int = 32
    pretrained: bool = False
    freeze_encoder: bool = False


@dataclass
class TrainingConfig:
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 0.01
    weight_decay: float = 0.0001
    num_workers: int = 0
    num_threads: int = 2
    device: str = "cpu"
    augment: bool = True


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
    alphas: list[float] = field(default_factory=lambda: [0.1])
    methods: list[MethodConfig] = field(default_factory=lambda: [MethodConfig("mahalanobis")])


@dataclass
class EvaluationConfig:
    batch_size: int = 128
    max_candidates: int = 100_000
    candidate_batch_size: int = 256
    save_prediction_sets: bool = False


@dataclass
class ArtifactConfig:
    save_probabilities: bool = True
    save_diagnostics: bool = False


@dataclass
class Config:
    name: str = "experiment"
    seeds: list[int] = field(default_factory=lambda: [42])
    output_dir: str = "outputs"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    conformal: ConformalConfig = field(default_factory=ConformalConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)

    def to_dict(self):
        return asdict(self)

    def validate(self):
        def positive_int(value):
            return isinstance(value, int) and not isinstance(value, bool) and value > 0

        def finite_number(value):
            return (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and np.isfinite(value)
            )

        for value in [
            self.dataset.download,
            self.dataset.verify_images,
            self.model.pretrained,
            self.model.freeze_encoder,
            self.training.augment,
            self.evaluation.save_prediction_sets,
            self.artifacts.save_probabilities,
            self.artifacts.save_diagnostics,
        ]:
            if not isinstance(value, bool):
                raise ValueError("Boolean settings require YAML true/false without quotes")

        if self.dataset.conflict_policy not in {"exclude", "error"}:
            raise ValueError("dataset.conflict_policy must be exclude or error")

        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.name):
            raise ValueError("name must contain only letters, digits, underscores and hyphens")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be a nonempty list of distinct integers")
        if any(not isinstance(s, int) or isinstance(s, bool) or s < 0 for s in self.seeds):
            raise ValueError("seeds must be nonnegative integers")
        fractions = list(asdict(self.splits).values())
        if any(not finite_number(x) or x <= 0 for x in fractions):
            raise ValueError("All five split fractions must be finite and positive")
        if not np.isclose(sum(fractions), 1, atol=1e-10, rtol=0):
            raise ValueError("Split fractions must sum to one")
        if not self.conformal.alphas or any(
            not finite_number(a) or not 0 < a < 1 for a in self.conformal.alphas
        ):
            raise ValueError("alphas must be nonempty and strictly between zero and one")
        if len(set(self.conformal.alphas)) != len(self.conformal.alphas):
            raise ValueError("alphas must be distinct")
        ids = [m.identifier for m in self.conformal.methods]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("Methods require distinct names or labels")
        if any(not re.fullmatch(r"[A-Za-z0-9_-]+", x) for x in ids):
            raise ValueError("Method labels must be safe directory names")
        for value in [
            self.training.epochs,
            self.training.batch_size,
            self.training.num_threads,
            self.dataset.image_size,
            self.dataset.n_samples,
            self.dataset.n_features,
            self.model.width,
            self.evaluation.batch_size,
            self.evaluation.candidate_batch_size,
            self.evaluation.max_candidates,
        ]:
            if not positive_int(value):
                raise ValueError("Counts, dimensions and batch sizes must be positive integers")
        if self.dataset.max_samples is not None and not positive_int(self.dataset.max_samples):
            raise ValueError("max_samples must be null or a positive integer")
        if not self.dataset.num_classes or any(
            not positive_int(c) or c < 2 for c in self.dataset.num_classes
        ):
            raise ValueError("num_classes must contain integers >= 2")
        if not finite_number(self.dataset.dependence) or not 0 <= self.dataset.dependence <= 1:
            raise ValueError("dependence must lie in [0, 1]")
        if (
            not finite_number(self.training.learning_rate)
            or self.training.learning_rate <= 0
            or not finite_number(self.training.weight_decay)
            or self.training.weight_decay < 0
        ):
            raise ValueError("Invalid learning rate or weight decay")
        if not isinstance(self.training.num_workers, int) or self.training.num_workers < 0:
            raise ValueError("num_workers must be a nonnegative integer")


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
    raw = deep_merge(Config().to_dict(), read_yaml(path))
    for override in overrides:
        key, separator, value = override.partition("=")
        if not separator:
            raise ValueError("Overrides use dotted.key=YAML-value")
        parts = key.split(".")
        target = raw
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                raise ValueError(f"Unknown configuration path: {key}")
            target = target[part]
        if parts[-1] not in target:
            raise ValueError(f"Unknown configuration key: {key}")
        target[parts[-1]] = yaml.safe_load(value)
    try:
        for key, cls in [
            ("dataset", DatasetConfig),
            ("splits", SplitConfig),
            ("model", ModelConfig),
            ("training", TrainingConfig),
            ("evaluation", EvaluationConfig),
            ("artifacts", ArtifactConfig),
        ]:
            raw[key] = cls(**raw[key])
        raw["conformal"]["methods"] = [MethodConfig(**m) for m in raw["conformal"]["methods"]]
        raw["conformal"] = ConformalConfig(**raw["conformal"])
        result = Config(**raw)
    except TypeError as error:
        raise ValueError(f"Invalid configuration: {error}") from error
    result.validate()
    return result
