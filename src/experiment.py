"""Generate controlled outputs, calibrate three methods, and save research tables."""

import csv
import hashlib
import json
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from .conformal import candidate_space, create_method, validate_probabilities
from .data import Predictions, generate, metadata, split_predictions
from .metrics import evaluate, prediction_diagnostics, subgroup_rows


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_predictions(path, splits, meta, config):
    meta = {**meta, "source_config": config.to_dict()}
    arrays = {"metadata": np.asarray(json.dumps(meta))}
    for name, batch in splits.items():
        arrays[f"{name}_y"] = batch.labels
        arrays[f"{name}_profile_ids"] = batch.profile_ids
        arrays[f"{name}_target_accuracy"] = batch.target_accuracy
        for k, p in enumerate(batch.probabilities):
            arrays[f"{name}_p{k}"] = p
    np.savez_compressed(path, **arrays)


def load_predictions(path, config):
    with np.load(path, allow_pickle=False) as cache:
        meta = json.loads(str(cache["metadata"]))
        if meta["schema_version"] != 2:
            raise ValueError("Use a synthetic probability cache with schema_version 2")
        if config.seeds != [meta["seed"]]:
            raise ValueError("Cache replay needs the same single seed as the source run")
        # Avoid silently labelling a cached simulation with different generation parameters.
        source = meta["source_config"]
        for key in ["dataset", "predictions", "splits"]:
            if source[key] != config.to_dict()[key]:
                raise ValueError(
                    f"Cached {key} differs; replay with the original generation settings"
                )
        dimensions = [len(c) for c in meta["class_names"]]
        if dimensions != config.dataset.num_classes:
            raise ValueError("Cached dimensions differ from dataset.num_classes")
        splits = {}
        for name in ["clustering", "calibration", "test"]:
            labels = cache[f"{name}_y"]
            probabilities = [cache[f"{name}_p{k}"] for k in range(len(dimensions))]
            validate_probabilities(probabilities, labels)
            profiles = cache[f"{name}_profile_ids"]
            target = cache[f"{name}_target_accuracy"]
            if (
                [p.shape[1] for p in probabilities] != dimensions
                or len(labels) < 2
                or profiles.shape != (len(labels),)
                or profiles.dtype.kind not in "iu"
                or (profiles < 0).any()
                or (profiles >= len(meta["profile_names"])).any()
                or target.shape != labels.shape
                or not np.isfinite(target).all()
                or (target < 0).any()
                or (target > 1).any()
            ):
                raise ValueError(f"Invalid cached arrays in {name}")
            splits[name] = Predictions(probabilities, labels, profiles, target)
    return splits, meta


def class_metric_rows(artifacts, meta):
    rows = []
    labels, candidates = artifacts["labels"], artifacts["candidates"]
    dimensions = [len(c) for c in meta["class_names"]]
    scopes = [
        (
            "joint",
            np.ravel_multi_index(labels.T, dimensions),
            "joint_covered",
            "joint_set_size",
            [json.dumps(list(map(int, y))) for y in candidates],
        )
    ]
    scopes.extend(
        (task, labels[:, k], f"{task}_covered", f"{task}_set_size", meta["class_names"][k])
        for k, task in enumerate(meta["task_names"])
    )
    for scope, ids, event_key, size_key, names in scopes:
        counts = artifacts[f"{scope}_class_counts"]
        covered = artifacts[f"{scope}_class_covered_counts"]
        totals = np.bincount(ids, weights=artifacts[size_key], minlength=len(names))
        for index, name in enumerate(names):
            count = int(counts[index])
            rows.append(
                {
                    "scope": scope,
                    "class_index": index,
                    "class_label": name,
                    "count": count,
                    "covered_count": int(covered[index]),
                    "coverage": float(covered[index] / count) if count else None,
                    "mean_set_size": float(totals[index] / count) if count else None,
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
        for key in group[0]:
            if key in {"seed", "alpha"}:
                continue
            values = [r[key] for r in group if isinstance(r[key], (int, float))]
            if values:
                summary[f"{key}_mean"] = float(np.mean(values))
                summary[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        result.append(summary)
    return result


def provenance():
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True
    )
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pyyaml": yaml.__version__,
        "git_commit": commit.stdout.strip(),
        "git_status": status.stdout,
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root / "src").glob("*.py"))
        },
    }


def run(config, predictions=None):
    config.validate()
    if predictions and len(config.seeds) != 1:
        raise ValueError("One probability cache corresponds to exactly one seed")
    # Check replay before creating a run directory.
    replay = load_predictions(predictions, config) if predictions else None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = Path(config.output_dir) / config.name / f"{stamp}-{uuid.uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=False)
    (root / "config.yaml").write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))
    info = provenance()
    if predictions:
        info["prediction_source"] = str(Path(predictions).resolve())
        info["prediction_sha256"] = hashlib.sha256(Path(predictions).read_bytes()).hexdigest()
    (root / "provenance.json").write_text(json.dumps(info, indent=2) + "\n")
    tables = {
        name: []
        for name in [
            "metrics",
            "class_metrics",
            "group_metrics",
            "subgroup_metrics",
            "prediction_metrics",
            "prediction_class_metrics",
            "confidence_bins",
            "calibration",
        ]
    }
    for seed in config.seeds:
        if replay:
            splits, meta = replay
        else:
            splits = split_predictions(generate(config, seed), config.splits, seed)
            meta = metadata(config, seed)
        output = root / f"seed-{seed}"
        output.mkdir()
        if config.artifacts.save_probabilities and not replay:
            save_predictions(output / "probabilities.npz", splits, meta, config)
        for name, batch in splits.items():
            diagnostics = prediction_diagnostics(batch, meta, config.predictions.error_correlation)
            for table, entries in zip(
                ["prediction_metrics", "confidence_bins", "prediction_class_metrics"],
                diagnostics,
                strict=True,
            ):
                tables[table].extend({"seed": seed, "split": name, **r} for r in entries)
        test_point = next(
            r
            for r in tables["prediction_metrics"]
            if r["seed"] == seed and r["split"] == "test" and r["scope"] == "joint"
        )
        print(
            f"seed={seed} test joint top-1 accuracy={test_point['point_accuracy']:.3f} "
            f"(target={test_point['target_point_accuracy']:.3f})",
            flush=True,
        )
        candidates = candidate_space(config.dataset.num_classes, config.evaluation.max_candidates)
        calibration, test = splits["calibration"], splits["test"]
        if config.conformal.calibration_samples is not None:
            calibration = calibration.subset(np.arange(config.conformal.calibration_samples))
        states, entries = {}, []
        for method_cfg in config.conformal.methods:
            method = create_method(method_cfg.name, **method_cfg.params)
            method.fit(splits["clustering"].probabilities, splits["clustering"].labels)
            for alpha in config.conformal.alphas:
                method.calibrate(calibration.probabilities, calibration.labels, alpha)
                result, artifacts = evaluate(
                    method,
                    test.probabilities,
                    test.labels,
                    candidates,
                    meta["task_names"],
                    alpha,
                    config.evaluation,
                )
                prefix = {"seed": seed, "method": method_cfg.identifier, "alpha": alpha}
                tables["metrics"].append(
                    {
                        **prefix,
                        "guarantee": method.guarantee,
                        "n_calibration": len(calibration.labels),
                        **result,
                    }
                )
                tables["class_metrics"].extend(
                    {**prefix, **r} for r in class_metric_rows(artifacts, meta)
                )
                tables["subgroup_metrics"].extend(
                    {**prefix, **r} for r in subgroup_rows(test, artifacts, meta)
                )
                test_ids = np.ravel_multi_index(test.labels.T, config.dataset.num_classes)
                test_groups = method.group_ids[test_ids]
                for group in range(len(method.thresholds)):
                    selected = test_groups == group
                    count = int(selected.sum())
                    tables["group_metrics"].append(
                        {
                            **prefix,
                            "group": group,
                            "count": count,
                            "n_classes": int((method.group_ids == group).sum()),
                            "calibration_count": int(method.calibration_counts[group]),
                            "coverage": float(artifacts["joint_covered"][selected].mean())
                            if count
                            else None,
                            "mean_set_size": float(artifacts["joint_set_size"][selected].mean())
                            if count
                            else None,
                        }
                    )
                ids = np.ravel_multi_index(calibration.labels.T, config.dataset.num_classes)
                counts = np.bincount(ids, minlength=len(candidates))
                for index, candidate in enumerate(candidates):
                    group = int(method.group_ids[index])
                    threshold = method.thresholds[group]
                    tables["calibration"].append(
                        {
                            **prefix,
                            "class_index": index,
                            "class_label": json.dumps(candidate.tolist()),
                            "group": group,
                            "class_calibration_count": int(counts[index]),
                            "group_calibration_count": int(method.calibration_counts[group]),
                            "threshold": float(threshold),
                            "always_included": bool(threshold >= 1),
                        }
                    )
                state = method.state()
                state_prefix = f"entry_{len(entries)}_"
                states.update(
                    {state_prefix + key: np.asarray(value).copy() for key, value in state.items()}
                )
                entries.append(
                    {
                        "method": method_cfg.identifier,
                        "name": method_cfg.name,
                        "params": method_cfg.params,
                        "alpha": alpha,
                        "prefix": state_prefix,
                    }
                )
                if config.evaluation.save_prediction_sets:
                    destination = output / method_cfg.identifier / f"alpha-{alpha:g}"
                    destination.mkdir(parents=True)
                    np.savez_compressed(destination / "prediction_sets.npz", **artifacts)
                print(
                    f"  {method_cfg.identifier} alpha={alpha:g}: "
                    f"coverage={result['joint_coverage']:.3f} "
                    f"size={result['mean_joint_set_size']:.2f} "
                    f"classgap={100 * result['joint_classwise_coverage_gap']:.2f}pp",
                    flush=True,
                )
        np.savez_compressed(
            output / "calibrations.npz",
            **states,
            metadata=np.asarray(json.dumps({**meta, "entries": entries})),
        )
        for name, rows in tables.items():
            write_csv(root / f"{name}.csv", rows)
    if len(config.seeds) > 1:
        write_csv(root / "summary.csv", summarize(tables["metrics"]))
    print(f"Artifacts: {root.resolve()}", flush=True)
    return root
