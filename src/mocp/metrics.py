"""Evaluate the discrete joint region and its exact task-wise projections."""

import numpy as np

from .conformal import validate_probabilities


def coverage_stats(events):
    """Wilson interval describes evaluation uncertainty, not a CP validity guarantee."""
    n, p, z = len(events), float(np.mean(events)), 1.959963984540054
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half_width = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return {
        "coverage": p,
        "coverage_se": float(np.sqrt(p * (1 - p) / n)),
        "coverage_ci_low": float(center - half_width),
        "coverage_ci_high": float(center + half_width),
    }


def classwise_coverage_stats(events, labels, num_classes, target):
    """Macro-average gaps over observed classes; retain support for rare-class analysis.

    Gaps are fractions, like the other coverage metrics. Unobserved classes have
    NaN coverage in the artifacts and do not contribute to summary statistics.
    """
    counts = np.bincount(labels, minlength=num_classes)
    covered_counts = np.bincount(labels, weights=events, minlength=num_classes).astype(int)
    observed = counts > 0
    coverage = np.divide(
        covered_counts,
        counts,
        out=np.full(num_classes, np.nan),
        where=observed,
    )
    observed_coverage = coverage[observed]
    return (
        {
            "classwise_coverage_gap": float(np.abs(observed_coverage - target).mean()),
            "classwise_undercoverage_gap": float(np.maximum(target - observed_coverage, 0).mean()),
            "worst_class_coverage": float(observed_coverage.min()),
            "observed_classes": int(observed.sum()),
        },
        {
            "class_counts": counts,
            "class_covered_counts": covered_counts,
            "class_coverage": coverage,
        },
    )


def evaluate(method, probabilities, labels, candidates, task_names, alpha, cfg):
    probabilities = validate_probabilities(probabilities, labels)
    sizes = [p.shape[1] for p in probabilities]
    n, j = len(labels), len(candidates)
    true_indices = np.ravel_multi_index(labels.T, sizes)
    joint_events, joint_sizes, rectangular_events, rectangular_sizes = [], [], [], []
    marginal_events = [[] for _ in sizes]
    marginal_sizes = [[] for _ in sizes]
    saved_masks = []
    for start in range(0, n, cfg.batch_size):
        stop = min(start + cfg.batch_size, n)
        batch_probabilities = [p[start:stop] for p in probabilities]
        mask = np.empty((stop - start, j), dtype=bool)
        for first in range(0, j, cfg.candidate_batch_size):
            last = min(first + cfg.candidate_batch_size, j)
            mask[:, first:last] = method.predict(batch_probabilities, candidates[first:last])
        joint_events.append(mask[np.arange(stop - start), true_indices[start:stop]])
        joint_sizes.append(mask.sum(axis=1))
        covered, projected_sizes = [], []
        for k, c in enumerate(sizes):
            projection = np.column_stack(
                [mask[:, candidates[:, k] == value].any(axis=1) for value in range(c)]
            )
            events = projection[np.arange(stop - start), labels[start:stop, k]]
            cardinality = projection.sum(axis=1)
            marginal_events[k].append(events)
            marginal_sizes[k].append(cardinality)
            covered.append(events)
            projected_sizes.append(cardinality)
        rectangular_events.append(np.column_stack(covered).all(axis=1))
        rectangular_sizes.append(np.prod(np.column_stack(projected_sizes), axis=1))
        if cfg.save_prediction_sets:
            saved_masks.append(mask)
    joint_events, joint_sizes = np.concatenate(joint_events), np.concatenate(joint_sizes)
    stats = coverage_stats(joint_events)
    class_stats, class_details = classwise_coverage_stats(joint_events, true_indices, j, 1 - alpha)
    class_artifacts = {f"joint_{key}": value for key, value in class_details.items()}
    result = {
        "n_test": n,
        "n_candidates": j,
        **{f"joint_{key}": value for key, value in stats.items()},
        **{f"joint_{key}": value for key, value in class_stats.items()},
        "target_joint_coverage": 1 - alpha,
        "joint_coverage_gap": stats["coverage"] - (1 - alpha),
        "joint_coverage_absolute_gap": abs(stats["coverage"] - (1 - alpha)),
        "mean_joint_set_size": float(joint_sizes.mean()),
        "mean_joint_set_fraction": float(joint_sizes.mean() / j),
        "empty_set_rate": float(np.mean(joint_sizes == 0)),
        "singleton_rate": float(np.mean(joint_sizes == 1)),
        "rectangular_joint_coverage": float(np.concatenate(rectangular_events).mean()),
        "mean_projection_cartesian_size": float(np.concatenate(rectangular_sizes).mean()),
    }
    point = np.column_stack([p.argmax(axis=1) for p in probabilities])
    result["point_joint_accuracy"] = float(np.mean((point == labels).all(axis=1)))
    marginal_coverages = []
    for k, task in enumerate(task_names):
        task_events = np.concatenate(marginal_events[k])
        task_stats = coverage_stats(task_events)
        class_stats, class_details = classwise_coverage_stats(
            task_events, labels[:, k], sizes[k], 1 - alpha
        )
        result.update({f"{task}_{key}": value for key, value in task_stats.items()})
        result.update({f"{task}_{key}": value for key, value in class_stats.items()})
        class_artifacts.update({f"{task}_{key}": value for key, value in class_details.items()})
        result[f"{task}_mean_set_size"] = float(np.concatenate(marginal_sizes[k]).mean())
        result[f"{task}_point_accuracy"] = float(np.mean(point[:, k] == labels[:, k]))
        marginal_coverages.append(task_stats["coverage"])
    result["mean_marginal_coverage"] = float(np.mean(marginal_coverages))
    result["min_marginal_coverage"] = float(np.min(marginal_coverages))
    artifacts = {
        "candidates": candidates,
        "labels": labels,
        "joint_covered": joint_events,
        "joint_set_size": joint_sizes,
        **class_artifacts,
    }
    if cfg.save_prediction_sets:
        artifacts["membership"] = np.concatenate(saved_masks)
    if hasattr(method, "native_sets"):
        # Projecting an empty Cartesian product loses nonempty sets of other tasks.
        # Report the native per-task guarantees as well as exact joint projections.
        native_coverages = []
        for k, (task, native) in enumerate(
            zip(task_names, method.native_sets(probabilities), strict=True)
        ):
            native_events = native[np.arange(n), labels[:, k]]
            native_stats = coverage_stats(native_events)
            class_stats, class_details = classwise_coverage_stats(
                native_events, labels[:, k], sizes[k], 1 - alpha
            )
            result.update({f"{task}_native_{key}": value for key, value in native_stats.items()})
            result.update({f"{task}_native_{key}": value for key, value in class_stats.items()})
            artifacts.update(
                {f"{task}_native_{key}": value for key, value in class_details.items()}
            )
            result[f"{task}_native_mean_set_size"] = float(native.sum(axis=1).mean())
            native_coverages.append(native_stats["coverage"])
            if cfg.save_prediction_sets:
                artifacts[f"native_membership_{k}"] = native
        result["mean_native_marginal_coverage"] = float(np.mean(native_coverages))
    return result, artifacts
