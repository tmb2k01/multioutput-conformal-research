"""Training, prediction caching, calibration and reproducible evaluation artifacts."""

import csv
import hashlib
import json
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import yaml

from . import data, models  # noqa: F401 - register built-in providers
from .conformal import candidate_space, validate_probabilities
from .data import sha256_file, split_dataset, split_manifest
from .metrics import evaluate
from .registry import DATASETS, METHODS, MODELS


def write_json(path, value):
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return convert(obj.tolist())
        if isinstance(obj, np.generic):
            return convert(obj.item())
        if isinstance(obj, float) and not np.isfinite(obj):
            return str(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (tuple, list)):
            return [convert(v) for v in obj]
        return obj

    path.write_text(json.dumps(convert(value), indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def provenance(config):
    packages = {}
    for name in ["numpy", "PyYAML", "Pillow", "requests", "gdown", "torch", "torchvision"]:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    root = Path(__file__).resolve().parents[2]

    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "git_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--porcelain"),
        "source_sha256": {
            str(path.relative_to(root)): sha256_file(path)
            for path in sorted((root / "src" / "mocp").glob("*.py"))
        },
        "config_sha256": hashlib.sha256(
            json.dumps(config.to_dict(), sort_keys=True).encode()
        ).hexdigest(),
    }


def prepare(config, seed):
    dataset = DATASETS.create(config.dataset.name, config.dataset, seed)
    dataset.metadata["image_size"] = config.dataset.image_size
    splits = split_dataset(dataset, config.splits, seed, config.dataset.max_samples)
    return dataset, splits


def prepare_only(config):
    paths = []
    for seed in config.seeds:
        dataset, splits = prepare(config, seed)
        destination = Path(config.dataset.root) / config.dataset.name / "prepared"
        destination.mkdir(parents=True, exist_ok=True)
        signature = hashlib.sha256(
            json.dumps(config.to_dict(), sort_keys=True).encode()
        ).hexdigest()
        path = destination / f"seed-{seed}-{signature[:12]}.json"
        write_json(path, split_manifest(dataset, splits))
        paths.append(path)
        print(
            f"Prepared {path}; split sizes: { {k: len(v) for k, v in splits.items()} }", flush=True
        )
    return paths


def train_predictions(config, seed, output):
    dataset, splits = prepare(config, seed)
    # Fail on infeasible exact enumeration before spending time on model training.
    candidate_space(dataset.num_classes, config.evaluation.max_candidates)
    if config.artifacts.save_diagnostics:
        write_json(output / "split_manifest.json", split_manifest(dataset, splits))
    model = MODELS.create(config.model.name, config.model, dataset)
    history = model.fit(dataset, splits, config.training, output, seed)
    if config.artifacts.save_diagnostics:
        write_csv(output / "training.csv", history)
    bundle = {}
    for split in ["shape", "calibration", "test"]:
        bundle[split] = (
            model.predict_proba(dataset, splits[split], config.training),
            dataset.y[splits[split]],
        )
    metadata = {
        "schema_version": 1,
        "dataset": config.dataset.name,
        "seed": seed,
        "task_names": dataset.task_names,
        "class_names": dataset.class_names,
    }
    for split, (probabilities, labels) in bundle.items():
        validate_probabilities(probabilities, labels)
    if config.artifacts.save_probabilities:
        metadata["split_ids"] = {
            name: [dataset.ids[i] for i in indices] for name, indices in splits.items()
        }
        arrays = {"metadata": np.asarray(json.dumps(metadata))}
        for split, (probabilities, labels) in bundle.items():
            arrays[f"{split}_y"] = labels
            arrays.update({f"{split}_p{k}": p for k, p in enumerate(probabilities)})
        np.savez_compressed(output / "probabilities.npz", **arrays)
    return bundle, metadata


def load_predictions(path, config):
    with np.load(path, allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata"]))
        if metadata["schema_version"] != 1:
            raise ValueError("Unsupported probability cache schema")
        if metadata["dataset"] != config.dataset.name or config.seeds != [metadata["seed"]]:
            raise ValueError("Cached dataset and single seed must match this configuration")
        num_classes = [len(c) for c in metadata["class_names"]]
        if len(metadata["task_names"]) != len(num_classes):
            raise ValueError("Cached task metadata does not match class metadata")
        bundle, seen = {}, set()
        for split, ids in metadata["split_ids"].items():
            if len(ids) != len(set(ids)) or seen.intersection(ids):
                raise ValueError("Cached split IDs overlap")
            seen.update(ids)
        for split in ["shape", "calibration", "test"]:
            labels = cache[f"{split}_y"]
            probabilities = [cache[f"{split}_p{k}"] for k in range(len(num_classes))]
            validate_probabilities(probabilities, labels)
            if (
                [p.shape[1] for p in probabilities] != num_classes
                or len(labels) != len(metadata["split_ids"][split])
                or len(labels) < 2
            ):
                raise ValueError(f"Invalid dimensions in cached {split}")
            bundle[split] = (probabilities, labels)
    return bundle, metadata


def class_metric_rows(artifacts, metadata):
    """Small class-level metrics table, without per-example prediction exports."""
    scopes = [("joint", "joint", "joint", None)]
    for k, task in enumerate(metadata["task_names"]):
        scopes.append((task, task, "projection", k))
        if f"{task}_native_class_counts" in artifacts:
            scopes.append((f"{task}_native", task, "native", k))
    rows = []
    for prefix, scope, set_kind, task_index in scopes:
        counts = artifacts[f"{prefix}_class_counts"]
        covered_counts = artifacts[f"{prefix}_class_covered_counts"]
        for index, (count, covered) in enumerate(zip(counts, covered_counts, strict=True)):
            if task_index is None:
                label = json.dumps(
                    [
                        metadata["class_names"][k][int(value)]
                        for k, value in enumerate(artifacts["candidates"][index])
                    ]
                )
            else:
                label = metadata["class_names"][task_index][index]
            rows.append(
                {
                    "scope": scope,
                    "set_kind": set_kind,
                    "class_index": index,
                    "class_label": label,
                    "count": int(count),
                    "covered_count": int(covered),
                    "coverage": float(covered / count) if count else None,
                }
            )
    return rows


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["method"], row["alpha"]), []).append(row)
    result = []
    for (method, alpha), group in groups.items():
        summary = {
            "method": method,
            "alpha": alpha,
            "n_seeds": len(group),
            "guarantee": group[0]["guarantee"],
        }
        keys = [
            k
            for k, v in group[0].items()
            if isinstance(v, (int, float)) and k not in {"seed", "alpha"}
        ]
        for key in keys:
            values = [r[key] for r in group]
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        result.append(summary)
    return result


def run(config, predictions=None):
    # Validate plugin names and parameter constructors before downloads/training.
    for method_cfg in config.conformal.methods:
        METHODS.create(method_cfg.name, **method_cfg.params)
    if config.dataset.name not in DATASETS or config.model.name not in MODELS:
        raise ValueError(f"Unknown dataset/model; choices: {list(DATASETS)}, {list(MODELS)}")
    if predictions and len(config.seeds) != 1:
        raise ValueError("One probability cache corresponds to exactly one seed")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = Path(config.output_dir) / config.name / f"{stamp}-{uuid.uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=False)
    (root / "config.yaml").write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))
    if config.artifacts.save_diagnostics:
        write_json(root / "provenance.json", provenance(config))
    rows, class_rows = [], []
    for seed in config.seeds:
        output = root / f"seed-{seed}"
        output.mkdir()
        prediction_source = None
        if predictions:
            bundle, metadata = load_predictions(predictions, config)
            prediction_source = {
                "path": str(Path(predictions).resolve()),
                "sha256": sha256_file(predictions),
            }
            if config.artifacts.save_diagnostics:
                write_json(
                    output / "prediction_source.json",
                    {**prediction_source, "metadata": metadata},
                )
        else:
            bundle, metadata = train_predictions(config, seed, output)
        candidates = candidate_space(
            [len(c) for c in metadata["class_names"]], config.evaluation.max_candidates
        )
        seed_rows, calibration_entries, calibration_arrays = [], [], {}
        for method_cfg in config.conformal.methods:
            method = METHODS.create(method_cfg.name, **method_cfg.params)
            method.fit(*bundle["shape"])
            for alpha in config.conformal.alphas:
                method.calibrate(*bundle["calibration"], alpha)
                result, artifacts = evaluate(
                    method,
                    *bundle["test"],
                    candidates,
                    metadata["task_names"],
                    alpha,
                    config.evaluation,
                )
                row = {
                    "seed": seed,
                    "method": method_cfg.identifier,
                    "alpha": alpha,
                    "guarantee": method.guarantee,
                    **result,
                }
                state = method.state()
                prefix = f"entry_{len(calibration_entries)}_"
                calibration_arrays.update(
                    {f"{prefix}{key}": np.asarray(value).copy() for key, value in state.items()}
                )
                calibration_entries.append(
                    {
                        "identifier": method_cfg.identifier,
                        "name": method_cfg.name,
                        "params": method_cfg.params,
                        "guarantee": method.guarantee,
                        "alpha": alpha,
                        "prefix": prefix,
                        "state_keys": list(state),
                    }
                )
                class_rows.extend(
                    {"seed": seed, "method": method_cfg.identifier, "alpha": alpha, **detail}
                    for detail in class_metric_rows(artifacts, metadata)
                )
                if config.artifacts.save_diagnostics or config.evaluation.save_prediction_sets:
                    destination = output / method_cfg.identifier / f"alpha-{alpha:g}"
                    destination.mkdir(parents=True)
                    if config.evaluation.save_prediction_sets:
                        np.savez_compressed(destination / "prediction_sets.npz", **artifacts)
                    if config.artifacts.save_diagnostics:
                        np.savez_compressed(destination / "calibration.npz", **state)
                        write_json(destination / "metrics.json", row)
                        write_json(
                            destination / "method.json",
                            {
                                "name": method_cfg.name,
                                "params": method_cfg.params,
                                "guarantee": method.guarantee,
                                "threshold": method.threshold,
                                "task_names": metadata["task_names"],
                                "class_names": metadata["class_names"],
                            },
                        )
                seed_rows.append(row)
                print(
                    f"seed={seed} {method_cfg.identifier} alpha={alpha:g}: "
                    f"joint={row['joint_coverage']:.3f} "
                    f"classgap={100 * row['joint_classwise_coverage_gap']:.2f}pp "
                    f"marginal={row['mean_marginal_coverage']:.3f} "
                    f"size={row['mean_joint_set_size']:.2f}",
                    flush=True,
                )
        calibration_metadata = {
            "kind": "mocp-calibrations",
            "schema_version": 1,
            "dataset": metadata["dataset"],
            "seed": seed,
            "task_names": metadata["task_names"],
            "class_names": metadata["class_names"],
            "prediction_source": prediction_source,
            "entries": calibration_entries,
        }
        np.savez_compressed(
            output / "calibrations.npz",
            metadata=np.asarray(json.dumps(calibration_metadata, allow_nan=False)),
            **calibration_arrays,
        )
        if config.artifacts.save_diagnostics:
            write_csv(output / "metrics.csv", seed_rows)
        rows.extend(seed_rows)
    write_csv(root / "metrics.csv", rows)
    write_csv(root / "class_metrics.csv", class_rows)
    if len(config.seeds) > 1:
        write_csv(root / "summary.csv", summarize(rows))
    print(f"Artifacts: {root.resolve()}", flush=True)
    return root
