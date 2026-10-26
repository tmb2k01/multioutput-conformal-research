"""Shared image encoders with one categorical classifier head per task."""

import copy
import os
import random

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from .models import Model


class ImageRows(Dataset):
    def __init__(self, dataset, indices, image_size, augment=False):
        self.dataset, self.indices = dataset, indices
        steps = [transforms.Resize((image_size, image_size))]
        if augment:
            # Preserve color targets and most of the object with mild geometry changes.
            steps.extend(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomAffine(
                        degrees=5,
                        translate=(0.05, 0.05),
                        scale=(0.95, 1.05),
                        interpolation=transforms.InterpolationMode.BILINEAR,
                        fill=(124, 116, 104),  # Approximately zero after normalization.
                    ),
                ]
            )
        steps.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.transform = transforms.Compose(steps)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, row):
        idx = self.indices[row]
        with Image.open(self.dataset.x[idx]) as image:
            features = self.transform(image.convert("RGB"))
        return features, torch.as_tensor(self.dataset.y[idx], dtype=torch.long)


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


class MultiHeadNetwork(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        if cfg.name == "resnet18":
            # Respect project-local storage for automatically fetched model weights.
            torch.hub.set_dir(str(os.path.abspath("data/pretrained")))
            weights = models.ResNet18_Weights.DEFAULT if cfg.pretrained else None
            self.encoder = models.resnet18(weights=weights)
            dimension = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        else:
            if cfg.pretrained:
                raise ValueError("tiny_cnn does not have pretrained weights")
            w = cfg.width
            self.encoder = nn.Sequential(
                nn.Conv2d(3, w, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(w, 2 * w, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(2 * w, 4 * w, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            dimension = 4 * w
        if cfg.freeze_encoder:
            if not cfg.pretrained:
                raise ValueError("freeze_encoder requires pretrained: true")
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)
        self.frozen = cfg.freeze_encoder
        self.heads = nn.ModuleList(nn.Linear(dimension, c) for c in num_classes)

    def forward(self, x):
        features = self.encoder(x)
        return [head(features) for head in self.heads]

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self


class ImageModel(Model):
    def __init__(self, cfg, dataset):
        self.cfg, self.num_classes = cfg, dataset.num_classes

    def _loader(self, dataset, indices, training, shuffle=False, seed=0, *, augment=False):
        return DataLoader(
            ImageRows(dataset, indices, self.image_size, augment=augment),
            batch_size=training.batch_size,
            shuffle=shuffle,
            num_workers=training.num_workers,
            generator=torch.Generator().manual_seed(seed),
            worker_init_fn=seed_worker,
        )

    def fit(self, dataset, splits, training, output, seed):
        torch.set_num_threads(training.num_threads)
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.use_deterministic_algorithms(True)
        if training.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(training.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable; choose training.device=cpu")
        self.image_size = dataset.metadata["image_size"]
        if self.image_size < 4:
            raise ValueError("Image models require image_size >=4")
        self.network = MultiHeadNetwork(self.cfg, self.num_classes).to(self.device)
        optimizer = torch.optim.AdamW(
            [p for p in self.network.parameters() if p.requires_grad],
            lr=training.learning_rate,
            weight_decay=training.weight_decay,
        )
        train_loader = self._loader(
            dataset, splits["train"], training, shuffle=True, seed=seed, augment=training.augment
        )
        val_loader = self._loader(dataset, splits["validation"], training)
        best, best_state, history = np.inf, None, []
        for epoch in range(training.epochs):
            self.network.train()
            train_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = sum(
                    nn.functional.cross_entropy(head, y[:, k])
                    for k, head in enumerate(self.network(x))
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "Nonfinite training loss; check training hyperparameters"
                    )
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(y)
            self.network.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    loss = sum(
                        nn.functional.cross_entropy(head, y[:, k])
                        for k, head in enumerate(self.network(x))
                    )
                    val_loss += loss.item() * len(y)
            row = {
                "epoch": epoch + 1,
                "train_loss": train_loss / len(splits["train"]),
                "validation_loss": val_loss / len(splits["validation"]),
            }
            history.append(row)
            print(
                f"epoch {epoch + 1}: train={row['train_loss']:.4f} "
                f"validation={row['validation_loss']:.4f}",
                flush=True,
            )
            if row["validation_loss"] < best:
                best = row["validation_loss"]
                best_state = copy.deepcopy(self.network.state_dict())
        if best_state is None:
            raise FloatingPointError("No finite validation checkpoint was produced")
        self.network.load_state_dict(best_state)
        torch.save(
            {
                "state_dict": {k: v.cpu() for k, v in best_state.items()},
                "model_config": vars(self.cfg),
                "num_classes": self.num_classes,
                "image_size": self.image_size,
                "task_names": dataset.task_names,
                "class_names": dataset.class_names,
                "seed": seed,
            },
            output / "model.pt",
        )
        return history

    def predict_proba(self, dataset, indices, training):
        self.network.eval()
        batches = [[] for _ in self.num_classes]
        with torch.no_grad():
            for x, _ in self._loader(dataset, indices, training):
                for k, logits in enumerate(self.network(x.to(self.device))):
                    batches[k].append(logits.softmax(dim=1).cpu().numpy())
        return [np.concatenate(parts) for parts in batches]
