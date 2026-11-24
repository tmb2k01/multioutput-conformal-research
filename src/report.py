"""Reproducible plots and a self-contained research report from saved preset runs."""

import base64
import csv
import hashlib
import html
import json
import textwrap
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from .conformal import validate_probabilities
from .experiment import write_csv

METHODS = ["standard", "classwise", "clustered"]
CONTROLS = [
    "synthetic-confident-errors",
    "synthetic-uncertain-errors",
    "synthetic-broad-confidence",
    "synthetic-uniform-confidence",
    "synthetic-constant-confidence",
    "synthetic-underconfident-correct",
]
HETEROGENEITY = [
    CONTROLS[0],
    "synthetic-class-profiles",
    "synthetic-class-accuracy",
    "synthetic-imbalanced",
]
LABELS = dict(
    zip(
        CONTROLS,
        [
            "Confident errors",
            "Uncertain errors",
            "Broad Beta confidence",
            "Uniform confidence",
            "Constant confidence",
            "Underconfident correct",
        ],
        strict=True,
    )
)
LABELS.update(
    {
        "synthetic-class-profiles": "Class-specific confidence",
        "synthetic-class-accuracy": "Class-specific accuracy",
        "synthetic-imbalanced": "Imbalanced labels",
        "synthetic-independent-errors": "Independent errors",
    }
)
COLORS = dict(
    zip(CONTROLS, ["#2563eb", "#0f766e", "#9333ea", "#d97706", "#64748b", "#dc2626"], strict=True)
)
METRICS = [
    "point_joint_accuracy",
    "joint_coverage",
    "joint_coverage_gap",
    "mean_joint_set_size",
    "joint_classwise_coverage_gap",
    "joint_classwise_undercoverage_gap",
    "joint_worst_class_coverage",
    "correct_point_joint_coverage",
    "incorrect_point_joint_coverage",
    "mean_false_class_inclusions",
]
BASELINE = CONTROLS[0]
MAIN_ALPHA = 0.1


def report_date():
    return datetime.now(ZoneInfo("Europe/Budapest")).strftime("%d %B %Y")


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def moments(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0,
        "min": float(values.min()),
        "max": float(values.max()),
    }


class Results:
    def __init__(self, source):
        self.source = Path(source).resolve()
        self.rows = read_csv(self.source)
        required_columns = set(METRICS) | {"experiment", "run", "method", "alpha", "seed"}
        if not self.rows:
            raise ValueError("The comparison CSV contains no metric rows")
        if missing := required_columns - self.rows[0].keys():
            raise ValueError(f"Comparison CSV is missing columns: {sorted(missing)}")
        self.index, self.runs, self.configs = {}, {}, {}
        self.manifest, self.cache_manifest = [], []
        self.confidence, self.plausibility = {}, {}
        for row in self.rows:
            key = (row["experiment"], row["method"], float(row["alpha"]), int(row["seed"]))
            if key in self.index:
                raise ValueError(f"Duplicate experiment/method/alpha/seed in comparison: {key}")
            self.index[key] = row
            paths = [Path(row["run"]), self.source.parent.parent / row["run"]]
            path = next((p.resolve() for p in paths if p.is_dir()), None)
            if path is None:
                raise ValueError(f"Missing source run: {row['run']}")
            name = row["experiment"]
            if name in self.runs and self.runs[name] != path:
                raise ValueError(f"Multiple runs supplied for the same experiment: {name}")
            self.runs[name] = path
        for name, run in self.runs.items():
            cfg = yaml.safe_load((run / "config.yaml").read_text())
            if cfg["name"] != name:
                raise ValueError(f"Experiment name does not match source config: {run}")
            self.configs[name] = cfg
            original = {
                (r["method"], float(r["alpha"]), int(r["seed"])): r
                for r in read_csv(run / "metrics.csv")
            }
            for key, row in self.index.items():
                if key[0] != name:
                    continue
                other = original.get(key[1:])
                if other is None or any(row[m] != other[m] for m in METRICS):
                    raise ValueError(f"Comparison metrics differ from original run: {key}")
            self.manifest.append(
                {
                    "experiment": name,
                    "run": str(run),
                    "config_sha256": sha256(run / "config.yaml"),
                    "metrics_sha256": sha256(run / "metrics.csv"),
                    "class_metrics_sha256": sha256(run / "class_metrics.csv"),
                    "calibration_sha256": sha256(run / "calibration.csv"),
                    "provenance_sha256": sha256(run / "provenance.json"),
                    "report_role": (
                        "confidence control"
                        if name in CONTROLS
                        else "class behavior"
                        if name in HETEROGENEITY
                        else "separate dependence comparison"
                        if name.endswith("independent-errors")
                        else "excluded: different dimensions or sample size"
                    ),
                }
            )
        required = set(CONTROLS + HETEROGENEITY + ["synthetic-independent-errors"])
        if missing := required - self.runs.keys():
            raise ValueError(f"Missing full preset results: {sorted(missing)}")
        baseline = self.configs[BASELINE]
        self.seeds = sorted(baseline["seeds"])
        self.alphas = sorted(baseline["conformal"]["alphas"], reverse=True)
        if not {0.05, MAIN_ALPHA, 0.2}.issubset(self.alphas):
            raise ValueError("This preset report needs alpha levels 0.05, 0.1, and 0.2")
        for name in required:
            cfg = self.configs[name]
            if (
                cfg["dataset"]["num_classes"] != [12, 11]
                or cfg["dataset"]["n_samples"] != baseline["dataset"]["n_samples"]
                or cfg["splits"] != baseline["splits"]
                or sorted(cfg["seeds"]) != self.seeds
                or cfg["conformal"] != baseline["conformal"]
            ):
                raise ValueError(f"Preset {name} differs in dimensions, budget, seeds or methods")
            if name in CONTROLS and cfg["dataset"] != baseline["dataset"]:
                raise ValueError(f"Confidence control {name} changes the label distribution")
            if name in CONTROLS:
                keys = set(cfg["predictions"]) - {"correct_confidence", "incorrect_confidence"}
                if any(cfg["predictions"][k] != baseline["predictions"][k] for k in keys):
                    raise ValueError(f"Confidence control {name} changes other prediction settings")
            for method in METHODS:
                for alpha in self.alphas:
                    for seed in self.seeds:
                        if (name, method, alpha, seed) not in self.index:
                            raise ValueError(
                                f"Incomplete results for {name}/{method}/{alpha}/{seed}"
                            )
        self.load_distributions()

    def values(self, name, method, alpha, metric):
        return np.asarray([float(self.index[name, method, alpha, s][metric]) for s in self.seeds])

    def stats(self, name, method, metric, alpha=MAIN_ALPHA):
        return moments(self.values(name, method, alpha, metric))

    def load_distributions(self):
        reference = {}
        for name in CONTROLS:
            confidence, plausibility = [[], []], [[], []]
            for seed in self.seeds:
                path = self.runs[name] / f"seed-{seed}" / "probabilities.npz"
                self.cache_manifest.append(
                    {"experiment": name, "seed": seed, "path": str(path), "sha256": sha256(path)}
                )
                with np.load(path, allow_pickle=False) as cache:
                    meta = json.loads(str(cache["metadata"]))
                    if meta["seed"] != seed:
                        raise ValueError(f"Cache seed mismatch: {path}")
                    for key in ["dataset", "predictions", "splits"]:
                        if meta["source_config"][key] != self.configs[name][key]:
                            raise ValueError(f"Cache {key} does not match run config: {path}")
                    y = cache["test_y"]
                    p = [cache[f"test_p{k}"] for k in range(2)]
                    validate_probabilities(p, y)
                    point = np.column_stack([a.argmax(axis=1) for a in p])
                    if name == BASELINE:
                        reference[seed] = (y.copy(), point.copy())
                    elif not (
                        np.array_equal(y, reference[seed][0])
                        and np.array_equal(point, reference[seed][1])
                    ):
                        raise ValueError(f"Confidence control is not paired: {name}, seed {seed}")
                    right = point == y
                    for index, mask in enumerate([right, ~right]):
                        confidence[index].append(np.column_stack([a.max(1) for a in p])[mask])
                    nll = -sum(
                        np.log10(a[np.arange(len(y)), y[:, k]].clip(1e-300))
                        for k, a in enumerate(p)
                    )
                    joint_right = right.all(axis=1)
                    for index, mask in enumerate([joint_right, ~joint_right]):
                        plausibility[index].append(nll[mask])
                    expected = self.index[name, "standard", MAIN_ALPHA, seed]
                    if not np.isclose(
                        joint_right.mean(),
                        float(expected["point_joint_accuracy"]),
                        rtol=0,
                        atol=1e-12,
                    ):
                        raise ValueError(f"Cached accuracy differs from reported accuracy: {path}")
            self.confidence[name] = [np.concatenate(v) for v in confidence]
            self.plausibility[name] = [np.concatenate(v) for v in plausibility]

    def summary(self):
        rows = []
        for name in LABELS:
            for method in METHODS:
                for alpha in self.alphas:
                    row = {
                        "experiment": name,
                        "method": method,
                        "alpha": alpha,
                        "n_seeds": len(self.seeds),
                    }
                    for metric in METRICS:
                        for stat, value in self.stats(name, method, metric, alpha).items():
                            row[f"{metric}_{stat}"] = value
                    rows.append(row)
        return rows


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#cbd5e1",
            "axes.labelcolor": "#334155",
            "text.color": "#0f172a",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "grid.color": "#e2e8f0",
            "grid.linewidth": 0.7,
            "axes.axisbelow": True,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def seed_error(stats):
    return np.maximum(
        0,
        np.asarray([[s["mean"] - s["min"] for s in stats], [s["max"] - s["mean"] for s in stats]]),
    )


def confidence_figure(data):
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.8), layout="constrained")
    bins = np.arange(-0.01, 1.02, 0.02)
    for ax, name in zip(axes.flat, CONTROLS, strict=True):
        for values, color, label, ls in zip(
            data.confidence[name],
            ["#2563eb", "#ea580c"],
            ["Correct top-1", "Incorrect top-1"],
            ["-", "--"],
            strict=True,
        ):
            ax.hist(
                values,
                bins=bins,
                weights=np.full(len(values), 1 / len(values)),
                histtype="step",
                linewidth=2,
                color=color,
                linestyle=ls,
                label=label,
            )
        means = [v.mean() for v in data.confidence[name]]
        ax.set_title(f"{LABELS[name]}\nMeans: correct {means[0]:.3f} / error {means[1]:.3f}")
        ax.set(xlim=(0.49, 1.01), xlabel="Winning softmax probability", ylabel="Fraction per bin")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y")
    axes[0, 0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle(
        "Input distributions: identical point predictions, different confidence", fontsize=16
    )
    return fig


def plausibility_figure(data):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6), layout="constrained")
    quantiles = np.linspace(0, 1, 1001)
    for index, ax in enumerate(axes):
        xmax = max(np.quantile(data.plausibility[n][index], 0.995) for n in CONTROLS)
        for name in CONTROLS:
            ax.plot(
                np.quantile(data.plausibility[name][index], quantiles),
                quantiles,
                color=COLORS[name],
                label=LABELS[name],
                linewidth=1.8,
            )
        ax.set(
            xlim=(0, xmax * 1.03),
            ylim=(0, 1.02),
            title=["All heads correct", "At least one head wrong"][index],
            xlabel="−log₁₀(product of true-label probabilities)",
            ylabel="Cumulative fraction",
        )
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid()
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("True-label plausibility exposes what the winning score cannot", fontsize=16)
    return fig


def efficiency_figure(data):
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharex=True, sharey=True, layout="constrained")
    for ax, method in zip(axes, METHODS, strict=True):
        stats = [data.stats(n, method, "mean_joint_set_size") for n in CONTROLS]
        positions = np.arange(len(CONTROLS))
        ax.barh(
            positions,
            [s["mean"] for s in stats],
            xerr=seed_error(stats),
            color=[COLORS[n] for n in CONTROLS],
            height=0.64,
            capsize=3,
        )
        ax.set_yticks(positions, [LABELS[n] for n in CONTROLS])
        ax.set(title=method.title(), xlim=(0, 132), xlabel="Mean number of joint labels")
        for index, s in enumerate(stats):
            ax.text(s["max"] + 1.5, index, f"{s['mean']:.1f}", va="center", fontsize=9)
        ax.grid(axis="x")
    axes[0].invert_yaxis()
    fig.suptitle("At 90% nominal coverage: confidence changes prediction-set size", fontsize=16)
    return fig


def method_figure(data, method):
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.6))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.16, hspace=0.42, wspace=0.30)
    targets = np.asarray([100 * (1 - a) for a in data.alphas])
    panels = [
        (axes[0, 0], "mean_joint_set_size", 1, "Prediction-set efficiency", "Mean joint set size"),
        (
            axes[0, 1],
            "joint_coverage_gap",
            100,
            "Deviation from nominal coverage",
            "Coverage gap (pp)",
        ),
        (
            axes[1, 0],
            "joint_classwise_coverage_gap",
            100,
            "Balance across true joint classes",
            "Mean absolute classwise gap (pp)",
        ),
    ]
    markers = ["o", "s", "^", "D", "v", "X"]
    for ax, metric, scale, title, ylabel in panels:
        for name, marker in zip(CONTROLS, markers, strict=True):
            stats = [data.stats(name, method, metric, a) for a in data.alphas]
            ax.errorbar(
                targets,
                np.asarray([s["mean"] for s in stats]) * scale,
                yerr=seed_error(stats) * scale,
                color=COLORS[name],
                marker=marker,
                markersize=5,
                linewidth=1.6,
                capsize=3,
                label=LABELS[name],
            )
        ax.set(title=title, xlabel="Nominal joint coverage (%)", ylabel=ylabel)
        ax.set_xticks(targets)
        ax.grid()
    axes[0, 0].set_ylim(0, 134)
    axes[0, 1].set_ylim(-3, 4)
    axes[0, 1].axhline(0, color="#0f172a", linewidth=1, linestyle=":")
    axes[1, 0].set_ylim(0, 9)
    ax = axes[1, 1]
    positions = np.arange(len(CONTROLS))
    for offset, metric, label, color in [
        (-0.17, "correct_point_joint_coverage", "All heads correct", "#2563eb"),
        (0.17, "incorrect_point_joint_coverage", "Any head wrong", "#ea580c"),
    ]:
        stats = [data.stats(n, method, metric) for n in CONTROLS]
        ax.barh(
            positions + offset,
            np.asarray([s["mean"] for s in stats]) * 100,
            xerr=seed_error(stats) * 100,
            height=0.3,
            color=color,
            capsize=2,
            label=label,
        )
    ax.set_yticks(positions, [LABELS[n] for n in CONTROLS], fontsize=8)
    ax.invert_yaxis()
    ax.set(
        xlim=(0, 103),
        xlabel="Observed joint coverage (%)",
        title="Correct vs incorrect point predictions at 90% target",
    )
    ax.grid(axis="x")
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, -0.25), ncol=2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.02)
    )
    fig.suptitle(
        f"{method.title()} conformal: compare confidence distributions", fontsize=18, y=0.97
    )
    fig.text(
        0.5,
        0.925,
        "Points are seed means; whiskers show the observed seed range, not confidence intervals.",
        ha="center",
        fontsize=10,
        color="#475569",
    )
    return fig


def heterogeneity_figure(data):
    fig, axes = plt.subplots(
        2, 3, figsize=(13, 8), sharex="col", sharey="row", layout="constrained"
    )
    colors = ["#2563eb", "#b45309", "#7c3aed", "#0f766e"]
    ticks = [
        "Uniform\nclasses",
        "Class-specific\nconfidence",
        "Class-specific\naccuracy",
        "Imbalanced\nlabels",
    ]
    for column, method in enumerate(METHODS):
        for row, metric, scale, ylabel in [
            (0, "joint_classwise_coverage_gap", 100, "Mean absolute classwise gap (pp)"),
            (1, "mean_joint_set_size", 1, "Mean joint set size"),
        ]:
            ax = axes[row, column]
            stats = [data.stats(n, method, metric) for n in HETEROGENEITY]
            ax.bar(
                np.arange(4),
                np.asarray([s["mean"] for s in stats]) * scale,
                yerr=seed_error(stats) * scale,
                color=colors,
                capsize=4,
                width=0.65,
            )
            ax.set_xticks(np.arange(4), ticks, fontsize=8)
            ax.set_ylabel(ylabel)
            ax.grid(axis="y")
        axes[0, column].set_title(method.title(), fontsize=14)
    fig.suptitle(
        "Class behavior at 90% nominal coverage: compare scenarios within each method", fontsize=15
    )
    return fig


def support_figure(data):
    run = data.runs["synthetic-imbalanced"]
    class_rows = read_csv(run / "class_metrics.csv")
    calibration = read_csv(run / "calibration.csv")
    fig, axes = plt.subplots(
        1, 3, figsize=(13, 4.8), sharex=True, sharey=True, layout="constrained"
    )
    for ax, method in zip(axes, METHODS, strict=True):
        counts, covered, support, full = (defaultdict(list) for _ in range(4))
        for row in class_rows:
            if (
                row["method"] == method
                and float(row["alpha"]) == MAIN_ALPHA
                and row["scope"] == "joint"
            ):
                index = int(row["class_index"])
                counts[index].append(int(row["count"]))
                covered[index].append(int(row["covered_count"]))
        for row in calibration:
            if row["method"] == method and float(row["alpha"]) == MAIN_ALPHA:
                index = int(row["class_index"])
                support[index].append(int(row["class_calibration_count"]))
                full[index].append(row["always_included"] == "True")
        indices = [i for i in counts if sum(counts[i])]
        points = ax.scatter(
            [np.mean(support[i]) for i in indices],
            [100 * sum(covered[i]) / sum(counts[i]) for i in indices],
            c=[np.mean(full[i]) for i in indices],
            cmap="plasma",
            vmin=0,
            vmax=1,
            s=22,
            linewidths=0.3,
            edgecolors="white",
            alpha=0.85,
        )
        ax.axhline(90, color="#0f172a", linestyle="--", linewidth=1)
        ax.set(xscale="log", title=method.title(), xlabel="Mean calibration count per joint class")
        ax.grid(alpha=0.6)
    axes[0].set(ylabel="Observed per-class coverage (%)", ylim=(0, 103))
    fig.colorbar(
        points, ax=list(axes), label="Fraction of seeds always including this candidate", shrink=0.8
    )
    fig.suptitle("Rare-class support under imbalance (90% target)", fontsize=16)
    return fig


def pm(data, name, method, metric, scale=1, digits=2, alpha=MAIN_ALPHA):
    stats = data.stats(name, method, metric, alpha)
    return f"{stats['mean'] * scale:.{digits}f} ± {stats['std'] * scale:.{digits}f}"


def result_table(data, method):
    headers = [
        "Distribution",
        "Top-1 (%)",
        "Coverage (%)",
        "Joint set size",
        "Classwise gap (pp)",
        "Error coverage (%)",
    ]
    rows = []
    for name in LABELS:
        rows.append(
            [
                LABELS[name],
                pm(data, name, method, "point_joint_accuracy", 100),
                pm(data, name, method, "joint_coverage", 100),
                pm(data, name, method, "mean_joint_set_size"),
                pm(data, name, method, "joint_classwise_coverage_gap", 100),
                pm(data, name, method, "incorrect_point_joint_coverage", 100),
            ]
        )
    return headers, rows


def narrative(data):
    def mean(name, method, metric):
        return data.stats(name, method, metric)["mean"]

    accuracy = mean(BASELINE, "standard", "point_joint_accuracy") * 100
    baseline_size = mean(BASELINE, "standard", "mean_joint_set_size")
    uncertain_size = mean(CONTROLS[1], "standard", "mean_joint_set_size")
    under_size = mean(CONTROLS[-1], "standard", "mean_joint_set_size")
    reduction = 100 * (1 - uncertain_size / baseline_size)
    gap = {
        m: mean("synthetic-class-profiles", m, "joint_classwise_coverage_gap") * 100
        for m in METHODS
    }
    seed_sizes = data.values(BASELINE, "standard", 0.2, "mean_joint_set_size")
    profile_sizes = data.stats("synthetic-class-profiles", "standard", "mean_joint_set_size")
    cfg = data.configs[BASELINE]
    n_test = int(data.index[BASELINE, "standard", MAIN_ALPHA, data.seeds[0]]["n_test"])
    scope = [
        f"The source comparison contains {len(data.rows)} method/seed/alpha rows. "
        "The main controlled "
        f"comparison uses six confidence distributions at identical {accuracy:.2f}% realized joint "
        "top-1 accuracy. Labels and argmax predictions were checked directly in every saved test "
        "cache and match exactly across these six presets for each seed.",
        f"All comparable full presets have {cfg['dataset']['n_samples']:,} observations per seed, "
        "12 color and 11 type classes (132 joint candidates). "
        f"Seeds: {', '.join(map(str, data.seeds))}. "
        f"Test size is {n_test:,} per seed. Splits are "
        f"{cfg['splits']['clustering']:.0%} clustering, "
        f"{cfg['splits']['calibration']:.0%} calibration, {cfg['splits']['test']:.0%} test. "
        "All methods share the same splits and calibration budget.",
        "One panel/figure fixes the conformal method while varying the synthetic distribution. "
        "The principal numerical comparison is at alpha=0.1 (90% nominal joint coverage), with "
        "80% and 95% targets also plotted. Whiskers show minimum–maximum across seeds; numerical "
        "tables show mean ± sample standard deviation. Neither is a confidence interval.",
        "The class-specific confidence, class-specific accuracy, and imbalance presets form a "
        "separate comparison of class behavior. Independent errors retain 80% per-head accuracy "
        "but about 64% joint accuracy, and are included only in the additional results tables. "
        "The smoke and 3×2-dimension runs are excluded from comparative plots because their sample "
        "budgets or candidate spaces differ.",
        "All methods use s(x,y)=1−product_k p_k(y_k), with inclusive finite-sample quantiles. "
        "Standard uses one global threshold, classwise uses a threshold per joint label, and "
        "clustered groups classes using score quantiles from an independent clustering split. "
        "Their guarantees are respectively joint marginal, joint class-conditional, and joint "
        "cluster-conditional under exchangeability. Correct/error subgroups have no separate "
        "coverage guarantee.",
    ]
    executive = [
        f"At the same {accuracy:.2f}% top-1 accuracy, standard conformal's mean joint set size is "
        f"{baseline_size:.2f} with confident errors, {uncertain_size:.2f} with uncertain errors "
        f"({reduction:.1f}% smaller), and {under_size:.2f} when correct predictions are "
        "underconfident. Similar changes occur within classwise and clustered conformal.",
        f"With class-specific confidence distributions, the macro absolute classwise coverage "
        f"gap is {gap['standard']:.2f} pp for standard, {gap['classwise']:.2f} pp for classwise, "
        f"and {gap['clustered']:.2f} pp for clustered. This improved balance has a higher mean "
        "set size in that scenario. Other scenarios exhibit different tradeoffs.",
        f"The 80% target is a sensitive boundary because it equals the generator's expected "
        f"joint accuracy. Standard baseline set sizes range from {seed_sizes.min():.2f} to "
        f"{seed_sizes.max():.2f} across seeds there. Seed variability must be retained when "
        "interpreting the distribution effect.",
    ]
    conclusions = [
        "1. Accuracy alone is insufficient to predict conformal efficiency. These paired "
        "confidence ablations hold every test label and point prediction fixed, yet require "
        "very different set sizes to recover the same nominal fraction of true labels.",
        "2. The probability assigned to the true label matters beyond the winning confidence. "
        "On wrong predictions, reducing the winning probability leaves more mass for the other "
        "labels, including the truth. Conformal can then recover errors with smaller sets in "
        "this generator. The true-label plausibility plot makes this mechanism visible.",
        "3. At a 90% target, standard CP covers essentially all correct point predictions and "
        "roughly half the errors in these controls. The identity coverage = accuracy × "
        "coverage(correct) + (1−accuracy) × coverage(error) explains why 80% accuracy and "
        "90% overall coverage do not imply 90% coverage among mistakes.",
        "4. Within these runs, changing confidence from narrow Beta to broad Beta, uniform, "
        "or constant while preserving its mean has a smaller effect at the 90% target than "
        "changing confidence conditional on correctness. This is a descriptive comparison "
        "over the tested distributions, not a statement about all distributions of equal mean.",
        "5. Adaptation helps when classes have different score distributions. Clustered CP "
        "substantially improves the measured classwise gap in the class-profile examples. "
        "Classwise CP can become conservative on rare classes, and clustered CP does not "
        "provide a general per-class guarantee. Inspect support and set size alongside gaps.",
        "6. At the 80% target, calibration accuracy can fall on either side of 80%. The selected "
        "quantile then moves between the correct and wrong score components. Reporting only "
        "a mean conceals this instability; the seed-range whiskers expose it.",
        "The same sensitivity can occur at another mixture boundary. In the class-confidence "
        "preset at a 90% target, standard set sizes range from "
        f"{profile_sizes['min']:.2f} to {profile_sizes['max']:.2f}. "
        "This is consistent with the global quantile lying near the boundary between diffuse "
        "and sharp error components, each with approximately 10% of the population. The "
        "interpretation is a mixture-boundary explanation, not a separate statistical test.",
    ]
    limitations = [
        f"Only {len(data.seeds)} seeds were used. Differences, ranges, and standard deviations "
        "are exploratory; no significance tests or universal ranking of the methods is claimed. "
        "Finite test samples also add noise to per-class coverage, especially for rare labels.",
        "The generator defaults to shared correctness across heads: both are right or both "
        "wrong. Confidence is bounded below by 0.5, and wrong-label tails are generated by a "
        "Dirichlet mechanism. Results depend on these choices and the product score. They are "
        "not evidence that the same effect size occurs on actual SGVehicle model outputs.",
        "Next, repeat the strongest paired contrasts with more seeds, sweep error dependence "
        "and true_class_weight separately, and inspect how much calibration support is needed "
        "for classwise/clustered thresholds. Use held-out development data if selecting cluster "
        "count or other hyperparameters; these test results were used only for description.",
    ]
    return scope, executive, conclusions, limitations


def plot_sections(data):
    sections = [
        (
            "efficiency",
            "1. Set size with the method held fixed",
            efficiency_figure,
            "Six confidence-only presets have exactly matched labels and point predictions. "
            "Bars show the mean at 90% nominal coverage; whiskers show the seed range. "
            "Every panel fixes one conformal method. Set size counts joint tuples out of 132.",
        ),
        (
            "confidence_distributions",
            "2. The generated confidence distributions",
            confidence_figure,
            "Winning softmax probabilities pool both heads and all test seeds, conditional "
            "on whether that head is correct. Each conditional histogram sums to one, with width "
            "0.02. Heads and paired scenarios are not independent replicates. Constant values are "
            "shown as probability mass in the bin centered at 0.9.",
        ),
        (
            "true_label_plausibility",
            "3. What changes in the true-label scores",
            plausibility_figure,
            "Empirical CDFs use the transform −log10(product of true-label probabilities); "
            "larger values mean the true joint label is less plausible. Axes stop at the largest "
            "99.5th percentile among controls to keep the main distributions legible; CDFs are not "
            "renormalized after clipping. The raw CP score remains 1−product, unchanged.",
        ),
    ]
    for method in METHODS:
        base = data.stats(BASELINE, method, "mean_joint_set_size")["mean"]
        uncertain = data.stats(CONTROLS[1], method, "mean_joint_set_size")["mean"]
        high = data.stats(CONTROLS[-1], method, "mean_joint_set_size")["mean"]
        caption = (
            f"At the 90% target: confident errors require {base:.2f} labels on average; "
            f"uncertain errors require {uncertain:.2f} "
            f"({100 * (1 - uncertain / base):.1f}% fewer); "
            f"underconfident correct predictions require {high:.2f}. Lines fix this method "
            "and vary only the confidence preset. Whiskers are the observed seed range. "
            "The bottom-right panel separates coverage on correct and incorrect predictions."
        )
        sections.append(
            (
                method,
                f"{len(sections) + 1}. {method.title()} conformal across distributions",
                lambda d, m=method: method_figure(d, m),
                caption,
            )
        )
    sections += [
        (
            "class_behavior",
            "7. Class heterogeneity and imbalance",
            heterogeneity_figure,
            "At 90% nominal coverage, compare four class-distribution scenarios within each fixed "
            "method. Gaps average absolute deviations from 90% over observed joint classes, "
            "computed per seed before averaging. These presets are separate from the "
            "six confidence-only controls. Whiskers show the seed range.",
        ),
        (
            "rare_class_support",
            "8. Why rare-class support matters",
            support_figure,
            "Each point is one joint class in the imbalanced preset at 90% nominal coverage. Test "
            "coverage pools covered/support counts across seeds; calibration support is the mean "
            "per seed. Color shows the fraction of seeds always including that candidate because "
            "its threshold is at least one. Small test counts make class estimates noisy.",
        ),
    ]
    return sections


def text_pages(pdf, title, paragraphs):
    fig = None
    y = 0
    for paragraph in paragraphs:
        lines = textwrap.wrap(paragraph, width=94) or [""]
        for index, line in enumerate(lines):
            if fig is None or y < 0.08:
                if fig is not None:
                    pdf.savefig(fig)
                    plt.close(fig)
                fig = plt.figure(figsize=(8.27, 11.69))
                fig.text(0.08, 0.94, title, fontsize=17, weight="bold", va="top")
                fig.text(
                    0.08,
                    0.035,
                    f"Synthetic multi-output conformal research · {report_date()}",
                    fontsize=8,
                    color="#64748b",
                )
                y = 0.875
            fig.text(0.08, y, line, fontsize=10.4, va="top")
            y -= 0.019
        y -= 0.013
    if fig is not None:
        pdf.savefig(fig)
        plt.close(fig)


def table_page(pdf, data, method):
    headers, rows = result_table(data, method)
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colWidths=[0.25, 0.14, 0.15, 0.15, 0.16, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.25)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_facecolor("#dbeafe" if row == 0 else "#f1f5f9" if row % 2 else "white")
        if row == 0:
            cell.set_text_props(weight="bold")
    fig.suptitle(f"{method.title()}: full comparisons at 90% nominal coverage", fontsize=17, y=0.97)
    fig.text(
        0.5,
        0.89,
        f"Mean ± sample SD over {len(data.seeds)} seeds. Independent errors change "
        "joint accuracy and are not a confidence-only control.",
        ha="center",
        fontsize=10,
    )
    fig.text(
        0.06,
        0.04,
        "Classwise gap = macro absolute deviation from 90% (percentage points). "
        "Error coverage = joint coverage among incorrect point predictions.",
        fontsize=9,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def html_table(headers, rows):
    head = "".join(f"<th>{html.escape(v)}</th>" for v in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>" for row in rows
    )
    return (
        f'<div class="table-scroll"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def paragraphs_html(paragraphs):
    return "".join(f"<p>{html.escape(p)}</p>" for p in paragraphs)


def build_report(results="outputs/synthetic-comparison.csv", output_dir="docs/synthetic-report"):
    data = Results(results)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    style()
    scope, executive, conclusions, limitations = narrative(data)
    write_csv(output / "summary.csv", data.summary())
    write_csv(output / "source_runs.csv", data.manifest)
    write_csv(output / "source_caches.csv", data.cache_manifest)
    paired = []
    for name in CONTROLS:
        for method in METHODS:
            for alpha in data.alphas:
                row = {
                    "experiment": name,
                    "reference": BASELINE,
                    "method": method,
                    "alpha": alpha,
                    "n_paired_seeds": len(data.seeds),
                }
                for metric in METRICS:
                    delta = data.values(name, method, alpha, metric) - data.values(
                        BASELINE, method, alpha, metric
                    )
                    for stat, value in moments(delta).items():
                        row[f"{metric}_delta_{stat}"] = value
                paired.append(row)
    write_csv(output / "paired_differences.csv", paired)
    document = [
        "<header><p class='eyebrow'>SYNTHETIC MULTI-OUTPUT CONFORMAL RESEARCH</p>",
        "<h1>Same accuracy.<br>Different confidence distributions.</h1>",
        "<p class='subtitle'>Comparisons within standard, classwise, and clustered conformal "
        f"prediction · {report_date()}</p></header>",
        "<section class='findings'><h2>Main findings</h2>",
        paragraphs_html(executive),
        "</section><section><h2>Design and interpretation</h2>",
        paragraphs_html(scope),
        "</section>",
    ]
    with PdfPages(output / "report.pdf") as pdf:
        pdf.infodict().update(
            {
                "Title": "Synthetic prediction distributions and conformal behavior",
                "Subject": "Comparisons within each conformal method",
                "Author": "Multi-output conformal research",
            }
        )
        text_pages(pdf, "Prediction distributions and conformal behavior", executive + scope[:2])
        text_pages(pdf, "Experimental design and interpretation", scope[2:])
        for key, title, plotter, caption in plot_sections(data):
            fig = plotter(data)
            for extension in ["png", "pdf", "svg"]:
                fig.savefig(figures / f"{key}.{extension}", dpi=180, bbox_inches="tight")
            note = fig.text(
                0.02,
                -0.025,
                textwrap.fill(caption, width=155),
                va="top",
                fontsize=9,
                color="#475569",
            )
            pdf.savefig(fig, bbox_inches="tight")
            note.remove()
            plt.close(fig)
            encoded = base64.b64encode((figures / f"{key}.png").read_bytes()).decode()
            document.append(
                f"<section><h2>{html.escape(title)}</h2><figure>"
                f"<img alt='{html.escape(title)}' src='data:image/png;base64,{encoded}'>"
                f"<figcaption>{html.escape(caption)}</figcaption></figure></section>"
            )
            print(f"Rendered {key}", flush=True)
        for method in METHODS:
            table_page(pdf, data, method)
            document.append(
                f"<section><h2>{method.title()}: numerical results at 90%</h2>"
                "<p>Mean ± sample standard deviation across seeds. Independent errors "
                "change joint accuracy and are shown as a separate comparison.</p>"
                + html_table(*result_table(data, method))
                + "</section>"
            )
        text_pages(pdf, "Conclusions", conclusions)
        text_pages(pdf, "Limits and next experiments", limitations)
        source_text = [
            f"Input comparison: {data.source}",
            f"Input SHA-256: {sha256(data.source)}",
            "Source run/config/metrics hashes are in source_runs.csv; probability cache "
            "hashes are in source_caches.csv. The report validates original metric rows, "
            "cache configurations, and exact labels/argmax pairing before plotting.",
        ]
        source_text += [
            f"{row['experiment']}: {Path(row['run']).name}. Role: {row['report_role']}."
            for row in data.manifest
        ]
        text_pages(pdf, "Source runs and reproducibility", source_text)
    document += [
        "<section><h2>Conclusions</h2>",
        paragraphs_html(conclusions),
        "</section>",
        "<section><h2>Limits and next experiments</h2>",
        paragraphs_html(limitations),
        "</section><section><h2>Sources and reproducibility</h2>",
        paragraphs_html(source_text[:3]),
        html_table(
            ["Experiment", "Run ID", "Role"],
            [[r["experiment"], Path(r["run"]).name, r["report_role"]] for r in data.manifest],
        ),
        "<p>Rebuild from the repository root:</p><pre>python -m src report "
        "--results outputs/synthetic-comparison.csv --output-dir docs/synthetic-report</pre>",
        "<p>This HTML embeds every plot and needs no network access. Individual figures "
        "are available as PNG, SVG, and vector PDF in the figures directory.</p></section>",
    ]
    css = """
    :root{color-scheme:light;font-family:system-ui,-apple-system,BlinkMacSystemFont,sans-serif;
      color:#14213a;background:#eef2f7;font-size:16px;line-height:1.7}
    body{margin:0}main{max-width:1200px;margin:0 auto;padding:56px;background:white}
    header{padding:20px 0 36px;border-bottom:2px solid #dbeafe}.eyebrow{font-size:.8rem;
      letter-spacing:.12em;color:#2563eb;font-weight:700}h1{font-size:3rem;line-height:1.13;
      letter-spacing:-.04em;margin:20px 0}h2{font-size:1.55rem;line-height:1.3;color:#0f172a}
    p{max-width:100ch}.subtitle,figcaption{color:#526176}.findings{background:#eff6ff;
      padding:24px 30px;border-left:4px solid #2563eb}section{margin-top:42px}
    figure{margin:16px 0}img{width:100%;height:auto;display:block}figcaption{font-size:.88rem;
      padding:14px 12px;background:#f8fafc;border-top:1px solid #e2e8f0}.table-scroll{overflow:auto}
    table{border-collapse:collapse;width:100%;font-size:.83rem}th,td{padding:12px 10px;
      border-bottom:1px solid #e2e8f0;text-align:right;white-space:nowrap}
    th:first-child,td:first-child{text-align:left}th{background:#eff6ff}
    tbody tr:nth-child(even){background:#f8fafc}pre{padding:18px;background:#f1f5f9;
      overflow:auto;font-size:.85rem}a{color:#2563eb}
    @media(max-width:700px){main{padding:22px}h1{font-size:2.1rem}.findings{padding:16px}}
    @media print{main{max-width:none;padding:0}body{background:white}section{break-inside:avoid}
      .findings{background:white}.table-scroll{overflow:visible}h1{font-size:2rem}}
    """
    (output / "report.html").write_text(
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Synthetic distributions and conformal behavior</title><style>"
        + css
        + "</style></head><body><main>"
        + "".join(document)
        + "</main></body></html>",
        encoding="utf-8",
    )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison": str(data.source),
        "comparison_sha256": sha256(data.source),
        "report_source_sha256": sha256(__file__),
        "matplotlib": matplotlib.__version__,
        "numpy": np.__version__,
        "n_metric_rows": len(data.rows),
        "seeds": data.seeds,
        "confidence_presets": CONTROLS,
        "paired_labels_and_predictions_verified": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Report: {(output / 'report.html').resolve()}", flush=True)
    print(f"PDF: {(output / 'report.pdf').resolve()}", flush=True)
    return output
