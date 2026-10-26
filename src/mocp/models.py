"""Model adapters. The linear baseline runs without PyTorch."""

from abc import ABC, abstractmethod

import numpy as np

from .registry import MODELS


class Model(ABC):
    @abstractmethod
    def fit(self, dataset, splits, training, output, seed):
        """Train on train; select a checkpoint using validation; return history rows."""

    @abstractmethod
    def predict_proba(self, dataset, indices, training):
        """Return a list of K arrays with shape (N, C_k)."""


def softmax(logits):
    exponentials = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exponentials / exponentials.sum(axis=1, keepdims=True)


@MODELS.register("linear")
class Linear(Model):
    def __init__(self, cfg, dataset):
        if dataset.kind != "tabular":
            raise ValueError(
                "linear requires tabular features; use tiny_cnn or resnet18 for images"
            )
        if cfg.pretrained or cfg.freeze_encoder:
            raise ValueError("linear has no pretrained or frozen encoder")
        self.num_classes = dataset.num_classes

    def fit(self, dataset, splits, training, output, seed):
        train = splits["train"]
        self.mean = dataset.x[train].mean(axis=0)
        self.scale = dataset.x[train].std(axis=0).clip(1e-8)
        x = self._features(dataset.x)
        rng = np.random.default_rng(seed)
        self.weights = [rng.normal(0, 0.01, (x.shape[1], c)) for c in self.num_classes]
        best, best_weights, history = np.inf, None, []
        for epoch in range(training.epochs):
            order = rng.permutation(train)
            for start in range(0, len(order), training.batch_size):
                idx = order[start : start + training.batch_size]
                for k, weights in enumerate(self.weights):
                    gradient = softmax(x[idx] @ weights)
                    gradient[np.arange(len(idx)), dataset.y[idx, k]] -= 1
                    gradient = x[idx].T @ gradient / len(idx)
                    penalty = training.weight_decay * weights
                    penalty[-1] = 0  # Do not penalize intercepts.
                    weights -= training.learning_rate * (gradient + penalty)
            train_loss = self._loss(x[train], dataset.y[train])
            val = splits["validation"]
            val_loss = self._loss(x[val], dataset.y[val])
            history.append(
                {"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": val_loss}
            )
            if val_loss < best:
                best, best_weights = val_loss, [w.copy() for w in self.weights]
        self.weights = best_weights
        np.savez_compressed(
            output / "model.npz",
            mean=self.mean,
            scale=self.scale,
            num_classes=np.asarray(self.num_classes),
            **{f"weights_{k}": w for k, w in enumerate(self.weights)},
        )
        return history

    def _features(self, x):
        return np.column_stack([(x - self.mean) / self.scale, np.ones(len(x))])

    def _loss(self, x, y):
        return float(
            sum(
                -np.log(softmax(x @ w)[np.arange(len(y)), y[:, k]].clip(1e-15)).mean()
                for k, w in enumerate(self.weights)
            )
        )

    def predict_proba(self, dataset, indices, training):
        x = self._features(dataset.x[indices])
        return [softmax(x @ w) for w in self.weights]


@MODELS.register("tiny_cnn")
@MODELS.register("resnet18")
def image_model(cfg, dataset):
    if dataset.kind != "image":
        raise ValueError(f"{cfg.name} requires an image dataset")
    try:
        from .image_models import ImageModel
    except ImportError as error:
        raise ImportError("Image models require: pip install -e '.[images]'") from error
    return ImageModel(cfg, dataset)
