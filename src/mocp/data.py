"""Dataset acquisition, immutable raw images, manifests and random disjoint splits."""

import hashlib
import json
import re
import shutil
import tarfile
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .registry import DATASETS

SGVEHICLE_SOURCE = {
    "filename": "MDC_dataset.tar.gz",
    "url": "https://media.githubusercontent.com/media/pappd90/mdc_dataset/"
    "98c360d16ae91e11779ac3f5ef3b7c0a2af54156/data/MDC_dataset.tar.gz",
    "sha256": "a7ceabfe8c1ebfad554a955842cfb85a1281a63c264215174df7d19776476893",
}
UTKFACE_SOURCES = [
    {"filename": f"UTKFace_dataset_{i}.tar.gz", "gdrive_id": file_id}
    for i, file_id in enumerate(
        [
            "1mb5Z24TsnKI3ygNIlX6ZFiwUj0_PmpAW",
            "19vdaXVRtkP-nyxz1MYwXiFsh_m_OL72b",
            "1oj9ZWsLV2-k2idoW_nRSrLQLUP3hus3b",
        ],
        start=1,
    )
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(source, directory, allow_download):
    """Atomic downloads; cache is checked against a known hash when available."""
    directory.mkdir(parents=True, exist_ok=True)
    filename = source["filename"]
    if Path(filename).name != filename or filename in {".", ".."}:
        raise ValueError("Archive filename must be a plain basename")
    path = directory / filename
    if not path.exists():
        if not allow_download:
            raise FileNotFoundError(f"Missing archive {path}; enable dataset.download or supply it")
        temporary = path.with_suffix(path.suffix + ".part")
        try:
            if "gdrive_id" in source:
                import gdown

                result = gdown.download(
                    id=source["gdrive_id"], output=str(temporary), quiet=False, use_cookies=False
                )
                if result is None:
                    raise OSError("Google Drive did not return a file")
            else:
                import requests

                print(f"Downloading {source['url']}", flush=True)
                with requests.get(source["url"], stream=True, timeout=(30, 120)) as response:
                    response.raise_for_status()
                    with temporary.open("wb") as stream:
                        for chunk in response.iter_content(1024 * 1024):
                            stream.write(chunk)
            with temporary.open("rb") as stream:
                if stream.read(100).startswith(b"version https://git-lfs.github.com/spec"):
                    raise OSError("URL returned a Git LFS pointer; use media.githubusercontent.com")
            checksum = sha256_file(temporary)
            if source.get("sha256") and checksum != source["sha256"]:
                raise OSError(f"SHA-256 mismatch for {filename}")
            # Check format before placing an HTML/quota response in the cache.
            with tarfile.open(temporary, "r:gz") as archive:
                next(iter(archive), None)
            temporary.replace(path)
        except Exception as error:
            temporary.unlink(missing_ok=True)
            raise OSError(
                f"Could not acquire {filename}: {error}. Supply a local archive in {directory}, "
                "or override dataset.sources with an accessible mirror."
            ) from error
    checksum = sha256_file(path)
    if source.get("sha256") and checksum != source["sha256"]:
        raise OSError(f"SHA-256 mismatch for cached {path}; replace the corrupted archive")
    return path, checksum


def extract_archive(path, destination):
    """Reject traversal, links and special files; publish only a complete extraction."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="extract-", dir=destination.parent.parent) as temp:
        temp = Path(temp).resolve()
        with tarfile.open(path, "r:gz") as archive:
            for member in archive:
                target = (temp / member.name).resolve()
                if not target.is_relative_to(temp):
                    raise ValueError(f"Unsafe archive member: {member.name}")
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                else:
                    raise ValueError(f"Archive links/special files are not allowed: {member.name}")
        if destination.exists():
            raise FileExistsError(
                f"Incomplete extraction exists at {destination}; inspect it first"
            )
        shutil.move(str(temp), str(destination))


def acquire(cfg, sources):
    base = Path(cfg.root).resolve() / cfg.name
    raw = base / "raw"
    if not cfg.download and raw.exists():
        return raw
    for source in cfg.sources if cfg.sources is not None else sources:
        # Validate names before building paths, even if extraction is already cached.
        filename = source["filename"]
        if Path(filename).name != filename or filename in {".", ".."}:
            raise ValueError("Archive filename must be a plain basename")
        target = raw / filename.removesuffix(".tar.gz")
        marker = target / ".source.json"
        if marker.exists():
            recorded = json.loads(marker.read_text())
            if recorded["source"] != source:
                raise ValueError(f"Source changed for {target}; use a different dataset.root")
            continue
        archive, checksum = download_archive(source, base / "archives", cfg.download)
        extract_archive(archive, target)
        marker.write_text(json.dumps({"source": source, "sha256": checksum}, indent=2))
    if not raw.exists():
        raise FileNotFoundError(f"No raw data at {raw}")
    return raw


@dataclass
class Dataset:
    x: np.ndarray
    y: np.ndarray
    task_names: list[str]
    class_names: list[list[str]]
    ids: list[str]
    metadata: dict
    kind: str = "tabular"

    @property
    def num_classes(self):
        return [len(c) for c in self.class_names]


def prepare_images(cfg, parser, task_names, sources, fixed_classes=None):
    from PIL import Image, UnidentifiedImageError

    raw = acquire(cfg, sources)
    records, seen, conflicts = [], {}, []
    conflicting_hashes = set()
    skipped, duplicates = 0, 0
    for path in sorted(raw.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        labels = parser(path.name)
        if labels is None:
            skipped += 1
            continue
        try:
            if cfg.verify_images:
                with Image.open(path) as image:
                    image.verify()
        except (UnidentifiedImageError, OSError, ValueError):
            skipped += 1
            continue
        checksum = sha256_file(path)
        if checksum in seen:
            if seen[checksum] != labels:
                if cfg.conflict_policy == "error":
                    raise ValueError(f"Identical image bytes have conflicting annotations: {path}")
                conflicting_hashes.add(checksum)
                conflicts.append(
                    {
                        "sha256": checksum,
                        "path": str(path),
                        "first_labels": seen[checksum],
                        "conflicting_labels": labels,
                    }
                )
            duplicates += 1
            continue
        seen[checksum] = labels
        records.append({"path": str(path), "labels": labels, "sha256": checksum})
    records = [record for record in records if record["sha256"] not in conflicting_hashes]
    if not records:
        raise ValueError(f"No valid labeled images found below {raw}")
    class_names = fixed_classes or [
        sorted(
            {record["labels"][k] for record in records},
            key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
        )
        for k in range(len(task_names))
    ]
    if any(len(classes) < 2 for classes in class_names):
        raise ValueError("Each output must contain at least two classes")
    encoders = [{label: i for i, label in enumerate(classes)} for classes in class_names]
    labels = np.asarray(
        [[encoders[k][label] for k, label in enumerate(record["labels"])] for record in records],
        dtype=np.int64,
    )
    fingerprint = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    metadata = {
        "name": cfg.name,
        "fingerprint": fingerprint,
        "skipped": skipped,
        "duplicates_removed": duplicates,
        "conflicting_unique_images_removed": len(conflicting_hashes),
        "annotation_conflicts": conflicts,
        "total": len(records),
        "raw_root": str(raw),
        "records": records,
        "sources": [json.loads(p.read_text()) for p in sorted(raw.rglob(".source.json"))],
    }
    print(
        f"{cfg.name}: {len(records)} images, {skipped} skipped, {duplicates} duplicates removed, "
        f"{len(conflicting_hashes)} conflicting unique images excluded",
        flush=True,
    )
    return Dataset(
        np.asarray([r["path"] for r in records]),
        labels,
        task_names,
        class_names,
        [r["sha256"] for r in records],
        metadata,
        "image",
    )


@DATASETS.register("sgvehicle")
def sgvehicle(cfg, seed):
    def parse(filename):
        parts = Path(filename).stem.split("_")
        return parts[:2] if len(parts) >= 3 and all(parts[:2]) else None

    return prepare_images(cfg, parse, ["color", "type"], [SGVEHICLE_SOURCE])


@DATASETS.register("utkface")
def utkface(cfg, seed):
    def parse(filename):
        # Also accepts the commonly distributed *.jpg.chip.jpg filenames.
        match = re.match(
            r"^\d{1,3}_([01])_([0-4])_.+\.(?:jpg|jpeg|png|bmp|webp)$", filename, re.IGNORECASE
        )
        return list(match.groups()) if match else None

    return prepare_images(
        cfg, parse, ["gender", "race"], UTKFACE_SOURCES, [["0", "1"], ["0", "1", "2", "3", "4"]]
    )


@DATASETS.register("synthetic")
def synthetic(cfg, seed):
    """Categorical tasks with shared unobserved noise; dependence=0 removes shared noise."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(cfg.n_samples, cfg.n_features))
    shared = rng.normal(size=(cfg.n_samples, 1))
    labels = []
    for c in cfg.num_classes:
        logits = x @ rng.normal(size=(cfg.n_features, c)) / np.sqrt(cfg.n_features)
        logits += np.sqrt(cfg.dependence) * shared * np.linspace(-2, 2, c)
        logits += np.sqrt(1 - cfg.dependence) * rng.normal(size=logits.shape)
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        u = rng.random(cfg.n_samples)
        labels.append((u[:, None] > probabilities.cumsum(axis=1)).sum(axis=1).clip(0, c - 1))
    return Dataset(
        x,
        np.column_stack(labels),
        [f"task_{k}" for k in range(len(labels))],
        [[str(j) for j in range(c)] for c in cfg.num_classes],
        [f"synthetic-{seed}-{i}" for i in range(cfg.n_samples)],
        {"name": "synthetic", "seed": seed, "dependence": cfg.dependence},
    )


def split_dataset(dataset, cfg, seed, max_samples=None):
    """Label-independent random split preserves the usual exchangeability argument."""
    indices = np.random.default_rng(seed).permutation(len(dataset.y))
    if max_samples is not None:
        indices = indices[:max_samples]
    fractions = asdict(cfg)
    counts = np.floor(np.asarray(list(fractions.values())) * len(indices)).astype(int)
    counts[-1] += len(indices) - counts.sum()
    if counts.min() < 2:
        raise ValueError("Each split needs >=2 examples; increase data or adjust split fractions")
    splits, start = {}, 0
    for (name, _), count in zip(fractions.items(), counts, strict=True):
        splits[name] = indices[start : start + count]
        start += count
    return splits


def split_manifest(dataset, splits):
    return {
        "task_names": dataset.task_names,
        "class_names": dataset.class_names,
        "metadata": dataset.metadata,
        "splits": {
            name: [
                {"index": int(i), "id": dataset.ids[i], "labels": dataset.y[i].tolist()}
                for i in indices
            ]
            for name, indices in splits.items()
        },
    }
