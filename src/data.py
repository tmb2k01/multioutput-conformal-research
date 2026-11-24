"""IID labels and controllable softmax outputs; no features or model training."""

from dataclasses import dataclass

import numpy as np

from .config import split_counts


@dataclass
class Predictions:
    probabilities: list[np.ndarray]
    labels: np.ndarray
    profile_ids: np.ndarray
    target_accuracy: np.ndarray

    def subset(self, indices):
        return Predictions(
            [p[indices] for p in self.probabilities],
            self.labels[indices],
            self.profile_ids[indices],
            self.target_accuracy[indices],
        )


def confidence_samples(cfg, n, rng):
    if cfg.distribution == "constant":
        return np.full(n, cfg.mean)
    if cfg.distribution == "uniform":
        values = rng.uniform(cfg.low, cfg.high, n)
    else:
        mean = (cfg.mean - cfg.low) / (cfg.high - cfg.low)
        values = cfg.low + (cfg.high - cfg.low) * rng.beta(
            mean * cfg.concentration, (1 - mean) * cfg.concentration, n
        )
    # Keep logits finite and the winning class unique, including extreme Beta draws.
    return np.clip(values, cfg.low + 1e-12, cfg.high - 1e-12)


def profile_settings(cfg):
    settings = [
        dict(
            name="default",
            accuracy=cfg.accuracy,
            correct_confidence=cfg.correct_confidence,
            incorrect_confidence=cfg.incorrect_confidence,
            tail_concentration=cfg.tail_concentration,
            true_class_weight=cfg.true_class_weight,
        )
    ]
    for profile in cfg.class_profiles:
        settings.append(
            {
                key: getattr(profile, key, None)
                if getattr(profile, key, None) is not None
                else value
                for key, value in settings[0].items()
            }
        )
    return settings


def generate(config, seed):
    """Sample correctness independently of confidence; never force an empirical quota.

    error_correlation mixes a shared uniform draw with independent draws per head.
    Each head retains its requested Bernoulli accuracy. At equal accuracy a,
    P(all heads correct) = rho*a + (1-rho)*a**K.
    """
    # Separate streams keep labels/correctness paired across confidence ablations.
    streams = np.random.SeedSequence(seed).spawn(4)
    label_rng, correct_rng, error_rng, score_rng = [np.random.default_rng(s) for s in streams]
    data, pred = config.dataset, config.predictions
    n, dimensions = data.n_samples, data.num_classes
    if data.joint_probabilities is not None:
        ids = label_rng.choice(int(np.prod(dimensions)), n, p=data.joint_probabilities)
        labels = np.column_stack(np.unravel_index(ids, dimensions))
    else:
        labels = np.column_stack(
            [
                label_rng.choice(
                    c,
                    n,
                    p=None if data.class_probabilities is None else data.class_probabilities[k],
                )
                for k, c in enumerate(dimensions)
            ]
        )
        ids = np.ravel_multi_index(labels.T, dimensions)
    profile_ids = np.zeros(n, dtype=int)
    for index, profile in enumerate(pred.class_profiles, start=1):
        start, stop = profile.class_range
        profile_ids[(ids >= start) & (ids < stop)] = index
    settings = profile_settings(pred)
    target_accuracy = np.empty((n, len(dimensions)))
    for index, setting in enumerate(settings):
        target_accuracy[profile_ids == index] = setting["accuracy"]
    shared = correct_rng.random((n, 1))
    independent = correct_rng.random((n, len(dimensions)))
    use_shared = correct_rng.random((n, 1)) < pred.error_correlation
    correct = np.where(use_shared, shared, independent) < target_accuracy
    # Wrong-label draws also stay fixed across confidence-only ablations.
    winners = labels.copy()
    for k, c in enumerate(dimensions):
        wrong = ~correct[:, k]
        if pred.confusion_matrices is None:
            values = error_rng.integers(0, c - 1, n)
            values += values >= labels[:, k]
        else:
            matrix = np.asarray(pred.confusion_matrices[k])
            draws = error_rng.random(n)
            values = (draws[:, None] >= matrix[labels[:, k]].cumsum(axis=1)).sum(axis=1)
            values = values.clip(max=c - 1)
        winners[wrong, k] = values[wrong]
    probabilities = []
    for k, c in enumerate(dimensions):
        p = np.empty((n, c))
        for index, setting in enumerate(settings):
            rows = np.flatnonzero(profile_ids == index)
            count = len(rows)
            if not count:
                continue
            right = correct[rows, k]
            confidence = np.empty(count)
            for mask, key in [(right, "correct_confidence"), (~right, "incorrect_confidence")]:
                confidence[mask] = confidence_samples(setting[key], int(mask.sum()), score_rng)
            alpha = np.full((count, c), setting["tail_concentration"], dtype=float)
            local = np.arange(count)
            alpha[local[~right], labels[rows[~right], k]] *= setting["true_class_weight"]
            tail = score_rng.gamma(alpha).clip(min=np.finfo(float).tiny)
            winner = winners[rows, k]
            tail[local, winner] = 0
            tail /= tail.sum(axis=1, keepdims=True)
            # Shrink toward a uniform tail only as needed to retain the chosen argmax.
            uniform = 1 / (c - 1)
            cap = confidence / (1 - confidence)
            mixing = np.minimum(
                1, 0.999 * (cap - uniform) / np.maximum(tail.max(axis=1) - uniform, 1e-15)
            )
            tail = uniform + mixing[:, None] * (tail - uniform)
            tail[local, winner] = 0
            values = (1 - confidence[:, None]) * tail
            values[local, winner] = confidence
            # These are actual softmax scores; log(probability) is a valid logit vector.
            logits = np.log(values.clip(min=np.finfo(float).tiny))
            exp = np.exp(logits - logits.max(axis=1, keepdims=True))
            p[rows] = exp / exp.sum(axis=1, keepdims=True)
        probabilities.append(p)
    return Predictions(probabilities, labels, profile_ids, target_accuracy)


def split_predictions(predictions, cfg, seed):
    # A separate stream means changes to score generation cannot change split membership.
    order = np.random.default_rng(np.random.SeedSequence([seed, 8317])).permutation(
        len(predictions.labels)
    )
    bounds = np.cumsum(split_counts(len(order), cfg))[:-1]
    return {
        name: predictions.subset(indices)
        for name, indices in zip(
            ["clustering", "calibration", "test"], np.split(order, bounds), strict=True
        )
    }


def metadata(config, seed):
    names = config.dataset.task_names
    if names is None:
        names = (
            ["color", "type"]
            if config.dataset.num_classes == [12, 11]
            else [f"task_{k}" for k in range(len(config.dataset.num_classes))]
        )
    return {
        "schema_version": 2,
        "dataset": "synthetic",
        "seed": seed,
        "task_names": names,
        "class_names": [[str(j) for j in range(c)] for c in config.dataset.num_classes],
        "profile_names": [p["name"] for p in profile_settings(config.predictions)],
    }
