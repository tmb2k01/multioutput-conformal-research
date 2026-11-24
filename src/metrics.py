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
    nonempty = joint_sizes > 0
    information = np.zeros(n)
    information[nonempty] = np.log2(j / joint_sizes[nonempty])
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
        "mean_false_class_inclusions": float((joint_sizes - joint_events.astype(int)).mean()),
        "mean_joint_set_fraction": float(joint_sizes.mean() / j),
        "empty_set_rate": float(np.mean(joint_sizes == 0)),
        "singleton_rate": float(np.mean(joint_sizes == 1)),
        "full_set_rate": float(np.mean(joint_sizes == j)),
        "informative_set_rate": float(np.mean(nonempty & (joint_sizes < j))),
        "mean_set_information_bits": float(information.mean()),
        "mean_covered_information_bits": float((information * joint_events).mean()),
        "correct_singleton_rate": float(np.mean((joint_sizes == 1) & joint_events)),
        "rectangular_joint_coverage": float(np.concatenate(rectangular_events).mean()),
        "mean_projection_cartesian_size": float(np.concatenate(rectangular_sizes).mean()),
    }
    point = np.column_stack([p.argmax(axis=1) for p in probabilities])
    result["point_joint_accuracy"] = float(np.mean((point == labels).all(axis=1)))
    point_correct = (point == labels).all(axis=1)
    for name, selected in [("correct_point", point_correct), ("incorrect_point", ~point_correct)]:
        result[f"{name}_count"] = int(selected.sum())
        result[f"{name}_joint_coverage"] = mean_or_none(joint_events[selected])
        result[f"{name}_mean_joint_set_size"] = mean_or_none(joint_sizes[selected])
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
    for k, task in enumerate(task_names):
        artifacts[f"{task}_covered"] = np.concatenate(marginal_events[k])
        artifacts[f"{task}_set_size"] = np.concatenate(marginal_sizes[k])
    return result, artifacts


def mean_or_none(values):
    return float(np.mean(values)) if len(values) else None


def prediction_diagnostics(batch, metadata, error_correlation):
    """Summaries reveal the realized experiment, including accuracy and overconfidence."""
    probabilities, labels = batch.probabilities, batch.labels
    point = np.column_stack([p.argmax(axis=1) for p in probabilities])
    dimensions = [p.shape[1] for p in probabilities]
    true_joint = np.ravel_multi_index(labels.T, dimensions)
    point_joint = np.ravel_multi_index(point.T, dimensions)
    true_probability = np.prod(
        [p[np.arange(len(labels)), labels[:, k]] for k, p in enumerate(probabilities)], axis=0
    )
    top_probability = np.prod([p.max(axis=1) for p in probabilities], axis=0)
    target_joint = error_correlation * batch.target_accuracy.min(axis=1) + (
        1 - error_correlation
    ) * batch.target_accuracy.prod(axis=1)
    scopes = [
        (
            "joint",
            true_joint,
            point_joint,
            int(np.prod(dimensions)),
            true_probability,
            top_probability,
            target_joint,
            np.prod([(p * p).sum(axis=1) for p in probabilities], axis=0),
        )
    ]
    for k, task in enumerate(metadata["task_names"]):
        p = probabilities[k]
        scopes.append(
            (
                task,
                labels[:, k],
                point[:, k],
                dimensions[k],
                p[np.arange(len(labels)), labels[:, k]],
                p.max(axis=1),
                batch.target_accuracy[:, k],
                (p * p).sum(axis=1),
            )
        )
    summaries, bins, classes = [], [], []
    for scope, truth, predicted, c, true_p, top_p, target, square_sum in scopes:
        correct = truth == predicted
        counts = np.bincount(truth, minlength=c)
        predicted_counts = np.bincount(predicted, minlength=c)
        true_positive = np.bincount(truth[correct], minlength=c)
        precision = np.divide(
            true_positive, predicted_counts, out=np.zeros(c), where=predicted_counts > 0
        )
        ece = 0.0
        bin_ids = np.minimum((top_p * 10).astype(int), 9)
        for index in range(10):
            selected = bin_ids == index
            count = int(selected.sum())
            acc, conf = mean_or_none(correct[selected]), mean_or_none(top_p[selected])
            if count:
                ece += count / len(labels) * abs(acc - conf)
            bins.append(
                {
                    "scope": scope,
                    "bin_low": index / 10,
                    "bin_high": (index + 1) / 10,
                    "count": count,
                    "point_accuracy": acc,
                    "mean_confidence": conf,
                    "mean_true_probability": mean_or_none(true_p[selected]),
                }
            )
        summaries.append(
            {
                "scope": scope,
                "count": len(labels),
                "target_point_accuracy": float(target.mean()),
                "point_accuracy": float(correct.mean()),
                # Zero precision for labels with no predictions; all labels count equally.
                "macro_precision": float(precision.mean()),
                "mean_confidence": float(top_p.mean()),
                "ece_10_bins": float(ece),
                "mean_confidence_correct": mean_or_none(top_p[correct]),
                "mean_confidence_incorrect": mean_or_none(top_p[~correct]),
                "mean_true_probability_correct": mean_or_none(true_p[correct]),
                "mean_true_probability_incorrect": mean_or_none(true_p[~correct]),
                "negative_log_likelihood": float(-np.log(true_p.clip(1e-300)).mean()),
                "brier_score": float((square_sum - 2 * true_p + 1).mean()),
            }
        )
        for index in range(c):
            selected = truth == index
            classes.append(
                {
                    "scope": scope,
                    "class_index": index,
                    "count": int(counts[index]),
                    "frequency": float(counts[index] / len(labels)),
                    "target_point_accuracy": mean_or_none(target[selected]),
                    "point_accuracy": mean_or_none(correct[selected]),
                    "predicted_count": int(predicted_counts[index]),
                    "precision": float(precision[index]) if predicted_counts[index] else None,
                    "mean_confidence": mean_or_none(top_p[selected]),
                    "mean_true_probability": mean_or_none(true_p[selected]),
                }
            )
    return summaries, bins, classes


def subgroup_rows(batch, artifacts, metadata):
    point = np.column_stack([p.argmax(axis=1) for p in batch.probabilities])
    correct = (point == batch.labels).all(axis=1)
    groups = [
        ("correctness", "all_correct", correct),
        ("correctness", "any_wrong", ~correct),
        ("correctness", "all_wrong", (point != batch.labels).all(axis=1)),
        ("correctness", "partly_correct", (point == batch.labels).any(axis=1) & ~correct),
    ]
    groups.extend(
        ("profile", name, batch.profile_ids == index)
        for index, name in enumerate(metadata["profile_names"])
    )
    rows = []
    for kind, name, selected in groups:
        rows.append(
            {
                "group_kind": kind,
                "group": name,
                "count": int(selected.sum()),
                "point_joint_accuracy": mean_or_none(correct[selected]),
                "joint_coverage": mean_or_none(artifacts["joint_covered"][selected]),
                "mean_joint_set_size": mean_or_none(artifacts["joint_set_size"][selected]),
            }
        )
    return rows
