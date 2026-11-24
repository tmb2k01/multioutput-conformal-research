"""Figures and reports for the controlled studies defined in one YAML each."""

import base64
import hashlib
import html
import json
import textwrap
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter

from .experiment import write_csv
from .report import html_table, moments, paragraphs_html, report_date, seed_error, style, text_pages
from .study import joint_prior, read_rows

MAIN_ALPHA = 0.1
METRICS = [
    "point_joint_accuracy",
    "joint_nll",
    "mean_joint_set_size",
    "joint_coverage",
    "joint_coverage_gap",
    "mean_set_information_bits",
    "mean_covered_information_bits",
    "joint_classwise_coverage_gap",
    "joint_classwise_undercoverage_gap",
    "joint_worst_class_coverage",
    "joint_observed_classes",
    "informative_set_rate",
    "singleton_rate",
    "correct_singleton_rate",
    "empty_set_rate",
    "full_set_rate",
    "n_calibration",
    "effective_joint_classes",
]


class StudyResults:
    def __init__(self, spec, plans, rows, runs, output):
        self.spec, self.plans, self.rows, self.runs, self.output = spec, plans, rows, runs, output
        self.seeds = plans[0]["config"].seeds
        self.alphas = sorted(plans[0]["config"].conformal.alphas, reverse=True)
        self.metrics = list(METRICS)
        if spec["kind"] == "label_balance":
            self.metrics += ["audit_" + m for m in METRICS if "audit_" + m in rows[0]]
        self.index = {(r["block"], r["case"], float(r["alpha"]), int(r["seed"])): r for r in rows}
        if len(self.index) != len(rows):
            raise ValueError("Duplicate study result rows")
        self.colors = plt.get_cmap("tab10").colors

    def values(self, block, case, metric, alpha=MAIN_ALPHA):
        return np.asarray(
            [float(self.index[block, case, alpha, seed][metric]) for seed in self.seeds]
        )

    def stat(self, block, case, metric, alpha=MAIN_ALPHA):
        return moments(self.values(block, case, metric, alpha))

    def mean(self, block, case, metric, alpha=MAIN_ALPHA):
        return self.stat(block, case, metric, alpha)["mean"]

    @property
    def gap_metric(self):
        return (
            "audit_joint_classwise_coverage_gap"
            if self.spec["kind"] == "label_balance"
            else "joint_classwise_coverage_gap"
        )

    def export(self):
        summaries, paired = [], []
        for block in self.spec["blocks"]:
            for case in block["cases"]:
                for alpha in self.alphas:
                    row = {
                        "block": block["name"],
                        "case": case["name"],
                        "label": case["label"],
                        "alpha": alpha,
                        "n_seeds": len(self.seeds),
                    }
                    delta = dict(row, reference=block["reference"])
                    for metric in self.metrics:
                        values = self.values(block["name"], case["name"], metric, alpha)
                        reference = self.values(block["name"], block["reference"], metric, alpha)
                        row.update({f"{metric}_{k}": v for k, v in moments(values).items()})
                        delta.update(
                            {
                                f"{metric}_delta_{k}": v
                                for k, v in moments(values - reference).items()
                            }
                        )
                    summaries.append(row)
                    paired.append(delta)
        write_csv(self.output / "summary.csv", summaries)
        write_csv(self.output / "paired_differences.csv", paired)


def overview(data, block):
    cases = block["cases"]
    fig, axes = plt.subplots(
        1, 3, figsize=(13, max(4.4, len(cases) * 0.6 + 1)), sharey=True, layout="constrained"
    )
    panels = [
        ("mean_joint_set_size", 1, "Efficiency: mean joint set size"),
        ("mean_covered_information_bits", 1, "Informativeness: covered information (bits)"),
        (
            data.gap_metric,
            100,
            "Balanced-audit class gap (pp)"
            if data.spec["kind"] == "label_balance"
            else "Mean absolute class coverage gap (pp)",
        ),
    ]
    for ax, (metric, scale, title) in zip(axes, panels, strict=True):
        stats = [data.stat(block["name"], c["name"], metric) for c in cases]
        ax.barh(
            np.arange(len(cases)),
            np.asarray([s["mean"] for s in stats]) * scale,
            xerr=seed_error(stats) * scale,
            capsize=3,
            height=0.65,
            color=[data.colors[i % 10] for i in range(len(cases))],
        )
        ax.set_yticks(np.arange(len(cases)), [c["label"] for c in cases], fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="x")
        for i, s in enumerate(stats):
            ax.annotate(
                f"{s['mean'] * scale:.2f}",
                (s["max"] * scale, i),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
            )
        ax.margins(x=0.18)
        maximum = max(data.stat(p["block"], p["case"], metric)["max"] for p in data.plans)
        ax.set_xlim(0, maximum * scale * 1.18)
    axes[0].invert_yaxis()
    fig.suptitle(f"{block['title']} · 90% nominal coverage", fontsize=15)
    return fig


def target_curves(data, block):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.20, hspace=0.38, wspace=0.25)
    panels = [
        ("mean_joint_set_size", 1, "Mean joint set size"),
        ("mean_covered_information_bits", 1, "Covered information (bits)"),
        (
            data.gap_metric,
            100,
            "Balanced-audit class gap (pp)"
            if data.spec["kind"] == "label_balance"
            else "Mean absolute class coverage gap (pp)",
        ),
        ("joint_coverage_gap", 100, "Population coverage minus target (pp)"),
    ]
    x = [100 * (1 - a) for a in data.alphas]
    for ax, (metric, scale, title) in zip(axes.flat, panels, strict=True):
        for i, case in enumerate(block["cases"]):
            stats = [data.stat(block["name"], case["name"], metric, a) for a in data.alphas]
            ax.errorbar(
                x,
                np.asarray([s["mean"] for s in stats]) * scale,
                yerr=seed_error(stats) * scale,
                color=data.colors[i % 10],
                marker="o",
                markersize=4,
                linewidth=1.5,
                capsize=2,
                label=case["label"],
            )
        ax.set(title=title, xlabel="Nominal joint coverage (%)")
        ax.set_xticks(x)
        ax.grid()
        if metric == "joint_coverage_gap":
            ax.axhline(0, color="#0f172a", linestyle=":", linewidth=1)
        else:
            ax.set_ylim(bottom=0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.015),
    )
    fig.suptitle(block["title"] + " · standard conformal", fontsize=16, y=0.98)
    fig.text(
        0.5,
        0.93,
        "Means and observed seed ranges; ranges are not confidence intervals.",
        ha="center",
        color="#475569",
    )
    return fig


def class_heatmap(data):
    fig, axes = plt.subplots(
        len(data.spec["blocks"]),
        1,
        figsize=(13, 3.0 * len(data.spec["blocks"]) + 1),
        squeeze=False,
        layout="constrained",
    )
    if data.spec["kind"] == "label_balance":
        rows = read_rows(data.output / "audit_class_metrics.csv")
    else:
        rows = []
        for plan in data.plans:
            path = data.runs[plan["block"], plan["case"]]
            rows.extend(
                dict(r, block=plan["block"], case=plan["case"])
                for r in read_rows(Path(path) / "class_metrics.csv")
                if r["scope"] == "joint"
            )
    n_classes = int(data.rows[0]["n_candidates"])
    for ax, block in zip(axes.flat, data.spec["blocks"], strict=True):
        counts, covered = defaultdict(int), defaultdict(int)
        for r in rows:
            if r["block"] == block["name"] and float(r["alpha"]) == MAIN_ALPHA:
                key = (r["case"], int(r["class_index"]))
                counts[key] += int(r["count"])
                covered[key] += int(r["covered_count"])
        values = np.full((len(block["cases"]), n_classes), np.nan)
        for i, case in enumerate(block["cases"]):
            for c in range(n_classes):
                key = case["name"], c
                if counts[key]:
                    values[i, c] = covered[key] / counts[key]
        image = ax.imshow(values, aspect="auto", vmin=0.65, vmax=1, cmap="RdYlBu")
        ax.set_yticks(
            np.arange(len(block["cases"])), [c["label"] for c in block["cases"]], fontsize=9
        )
        ax.set(title=block["title"], xlabel="Flattened joint class index")
        if n_classes == 132:
            ax.axvline(65.5, color="black", linestyle="--", linewidth=0.8)
    colorbar = fig.colorbar(image, ax=list(axes.flat), label="Per-class coverage", shrink=0.85)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle("Joint class coverage at 90% nominal coverage", fontsize=16)
    return fig


def balance_design(data):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    block = data.spec["blocks"][0]
    for i, case in enumerate(block["cases"]):
        plan = next(
            p for p in data.plans if p["block"] == block["name"] and p["case"] == case["name"]
        )
        p = joint_prior(plan["config"])
        axes[0].plot(
            np.arange(len(p)), p, color=data.colors[i % 10], label=case["label"], linewidth=1.5
        )
        axes[1].barh(i, np.exp(-(p * np.log(p)).sum()), color=data.colors[i % 10], height=0.65)
    axes[0].set(
        yscale="log",
        xlabel="Joint class index",
        ylabel="Configured joint label probability",
        title="Prior weights (product of per-head power laws)",
    )
    axes[1].set_yticks(
        np.arange(len(block["cases"])), [c["label"] for c in block["cases"]], fontsize=8
    )
    axes[1].invert_yaxis()
    axes[1].set(
        xlabel="exp(entropy of joint label prior)", title="Effective number of joint classes"
    )
    axes[0].grid()
    axes[1].grid(axis="x")
    fig.suptitle(
        "Label-balance settings; all cases retain 80% expected top-1 accuracy", fontsize=15
    )
    return fig


def loss_relationship(data):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), layout="constrained")
    for ax, (metric, scale, title) in zip(
        axes,
        [
            ("mean_joint_set_size", 1, "Efficiency"),
            ("mean_covered_information_bits", 1, "Covered information (bits)"),
            ("joint_classwise_coverage_gap", 100, "Mean absolute class gap (pp)"),
        ],
        strict=True,
    ):
        for i, block in enumerate(data.spec["blocks"]):
            x = [data.stat(block["name"], c["name"], "joint_nll") for c in block["cases"]]
            y = [data.stat(block["name"], c["name"], metric) for c in block["cases"]]
            ax.errorbar(
                [s["mean"] for s in x],
                np.asarray([s["mean"] for s in y]) * scale,
                xerr=seed_error(x),
                yerr=seed_error(y) * scale,
                marker=["o", "s", "^"][i % 3],
                linewidth=1.5,
                capsize=2,
                color=data.colors[i],
                label=block["title"],
            )
        ax.set(xlabel="Joint cross-entropy / NLL (natural-log units)", title=title)
        ax.grid()
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle(
        "Same exact point predictions: loss and conformal behavior at 90% target", fontsize=15
    )
    return fig


def support_comparison(data):
    blocks = data.spec["blocks"]
    fig, axes = plt.subplots(
        1, len(blocks), figsize=(13, 5.8), sharey=True, sharex=True, squeeze=False
    )
    fig.subplots_adjust(left=0.23, right=0.98, top=0.82, bottom=0.19, wspace=0.15)
    for ax, block in zip(axes.flat, blocks, strict=True):
        for offset, metric, label, color in [
            (-0.17, "joint_classwise_coverage_gap", "Own population test", "#ea580c"),
            (0.17, "audit_joint_classwise_coverage_gap", "Fixed balanced audit", "#2563eb"),
        ]:
            stats = [data.stat(block["name"], c["name"], metric) for c in block["cases"]]
            ax.barh(
                np.arange(len(stats)) + offset,
                [100 * s["mean"] for s in stats],
                xerr=100 * seed_error(stats),
                height=0.3,
                capsize=3,
                label=label,
                color=color,
            )
        ax.set_yticks(
            np.arange(len(block["cases"])), [c["label"] for c in block["cases"]], fontsize=9
        )
        ax.set(title=block["title"], xlabel="Mean absolute class gap (pp)")
        ax.grid(axis="x")
    largest = max(
        data.stat(block["name"], case["name"], metric)["max"]
        for block in blocks
        for case in block["cases"]
        for metric in ["joint_classwise_coverage_gap", "audit_joint_classwise_coverage_gap"]
    )
    for ax in axes.flat:
        ax.set_xlim(0, 100 * largest * 1.08)
    axes[0, 0].invert_yaxis()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Separate a change in coverage from a change in class support", fontsize=16)
    return fig


def block_table(data, block):
    headers = ["Case", "Joint NLL", "Set size", "Covered bits", "Class gap (pp)", "Coverage (%)"]
    rows = []
    for case in block["cases"]:
        row = [case["label"]]
        for metric, scale in [
            ("joint_nll", 1),
            ("mean_joint_set_size", 1),
            ("mean_covered_information_bits", 1),
            (data.gap_metric, 100),
            ("joint_coverage", 100),
        ]:
            s = data.stat(block["name"], case["name"], metric)
            row.append(f"{s['mean'] * scale:.2f} ± {s['std'] * scale:.2f}")
        rows.append(row)
    return headers, rows


def conclusions(data):
    result = []
    accuracy = np.mean([float(r["point_joint_accuracy"]) for r in data.rows])
    result.append(
        f"Every case has the same realized correctness pattern for every seed and split; "
        f"mean joint top-1 accuracy is {accuracy:.2%}. All comparisons use standard split "
        "conformal with the product score. No method or case was chosen using test results."
    )
    for block in data.spec["blocks"]:
        b = block["name"]
        sizes = [data.mean(b, c["name"], "mean_joint_set_size") for c in block["cases"]]
        gaps = [100 * data.mean(b, c["name"], data.gap_metric) for c in block["cases"]]
        info = [data.mean(b, c["name"], "mean_covered_information_bits") for c in block["cases"]]
        result.append(
            f"{block['title']}: at 90% nominal coverage, mean set sizes span "
            f"{min(sizes):.2f}–{max(sizes):.2f}, covered information spans "
            f"{min(info):.2f}–{max(info):.2f} bits, and mean absolute class coverage "
            f"gaps span {min(gaps):.2f}–{max(gaps):.2f} percentage points."
        )
    if data.spec["kind"] == "label_balance":
        result += [
            "The homogeneous block is a negative control: changing label frequencies leaves the "
            "conditional true-score law unchanged. An efficiency effect is not expected in the "
            "population; residual changes reflect finite samples. Raw observed-class "
            "gaps can change simply because rare classes have noisier or missing test estimates.",
            "The second block fixes two class-score laws while changing their population weights. "
            "A single global threshold pools those laws with different weights. The balanced audit "
            "holds the class mix and examples fixed, so its gap measures the impact on the same "
            "classes instead of conflating that impact with changing test support.",
        ]
        homogeneous = data.spec["blocks"][0]
        b = homogeneous["name"]
        worst = max(
            homogeneous["cases"],
            key=lambda c: data.mean(b, c["name"], "joint_classwise_coverage_gap"),
        )
        result.append(
            f"In the homogeneous block, {worst['label'].lower()} has a raw population class gap "
            f"of {100 * data.mean(b, worst['name'], 'joint_classwise_coverage_gap'):.2f} pp, "
            "but a balanced-audit gap of "
            f"{100 * data.mean(b, worst['name'], data.gap_metric):.2f} pp. "
            f"Only {data.mean(b, worst['name'], 'joint_observed_classes'):.1f} of "
            f"{data.rows[0]['n_candidates']} classes are observed per population test on average. "
            "The fixed audit removes the changing-support comparison artifact."
        )
    elif data.spec["kind"] == "prediction_loss":
        for block in data.spec["blocks"]:
            b = block["name"]
            ordered = sorted(block["cases"], key=lambda c: data.mean(b, c["name"], "joint_nll"))
            low, high = ordered[0]["name"], ordered[-1]["name"]
            result.append(
                f"{block['title']}: joint NLL changes from "
                f"{data.mean(b, low, 'joint_nll'):.2f} to {data.mean(b, high, 'joint_nll'):.2f}; "
                "the corresponding mean set sizes are "
                f"{data.mean(b, low, 'mean_joint_set_size'):.2f} "
                f"and {data.mean(b, high, 'mean_joint_set_size'):.2f}. These are controlled "
                "changes in scores with identical labels and argmax predictions."
            )
        result.append(
            "Cross-entropy summarizes average true-label plausibility; it does not specify "
            "the whole score distribution or how loss is allocated across classes. Inspect "
            "the localized-loss block and class-coverage heatmap before treating loss as a "
            "complete predictor of efficiency or class coverage."
        )
        pairs = []
        for i, first in enumerate(data.plans):
            for second in data.plans[i + 1 :]:
                if first["block"] == second["block"]:
                    continue
                x = data.mean(first["block"], first["case"], "joint_nll")
                y = data.mean(second["block"], second["case"], "joint_nll")
                if abs(x - y) <= 0.01 * max(x, y):
                    gap = abs(
                        data.mean(first["block"], first["case"], data.gap_metric)
                        - data.mean(second["block"], second["case"], data.gap_metric)
                    )
                    pairs.append((gap, first, second, x, y))
        if pairs:
            _, first, second, x, y = max(pairs, key=lambda p: p[0])
            a = 100 * data.mean(first["block"], first["case"], data.gap_metric)
            b = 100 * data.mean(second["block"], second["case"], data.gap_metric)
            result.append(
                f"A descriptive near-matched-loss contrast: {first['label']} and "
                f"{second['label']} have NLL {x:.3f} and {y:.3f} (within 1%), yet class "
                f"gaps {a:.2f} and {b:.2f} pp. This pair was found from the sweep after "
                "evaluation and is exploratory, not a prespecified statistical test."
            )
    else:
        block = data.spec["blocks"][0]
        b = block["name"]
        cases = sorted(block["cases"], key=lambda c: data.mean(b, c["name"], "n_calibration"))
        low, high = cases[0], cases[-1]
        low_budget = int(data.mean(b, low["name"], "n_calibration"))
        high_budget = int(data.mean(b, high["name"], "n_calibration"))
        a = data.stat(b, low["name"], "mean_joint_set_size")
        z = data.stat(b, high["name"], "mean_joint_set_size")
        result.append(
            f"The identical probability arrays and test examples were verified across all "
            f"budgets. With {low_budget:,} calibration examples the across-seed set-size "
            f"SD is {a['std']:.2f}; with {high_budget:,} it is {z['std']:.2f}. "
            "This comparison isolates calibration sampling variability from model quality."
        )
    result.append(
        "Smaller sets are useful only together with coverage. Covered information assigns "
        "zero to empty sets and to sets missing the truth, so an aggressive rule cannot "
        "appear informative merely by returning empty or incorrect singleton sets."
    )
    return result


def build_study_report(spec, plans, rows, runs, output):
    style()
    data = StudyResults(spec, plans, rows, runs, output)
    data.export()
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    claims = conclusions(data)
    cfg = plans[0]["config"]
    design = [
        spec["description"],
        f"One YAML defines this entire group: shared base settings, blocks, their fixed settings, "
        f"and all cases. There are {len(plans)} cases, {len(data.seeds)} seeds "
        f"({', '.join(map(str, data.seeds))}), {cfg.dataset.n_samples:,} examples per seed, "
        f"{int(np.prod(cfg.dataset.num_classes))} joint labels, and alpha levels "
        f"{', '.join(map(str, cfg.conformal.alphas))}. Comparisons are paired by seed; whiskers "
        "are seed ranges; tables show mean ± sample SD. Neither is a confidence interval.",
        "Efficiency is mean joint prediction-set size (lower is better at comparable coverage). "
        "Information per nonempty set is log2(number of candidates / set size); empty sets receive "
        "zero. Mean covered information also assigns zero when the true joint label is missing. "
        "This measures combinatorial candidate reduction under a uniform candidate reference, "
        "not posterior entropy or calibrated uncertainty. Singleton, correct-singleton, empty, "
        "full-set and nonempty non-full-set rates are exported at every alpha.",
        "Class coverage gap averages abs(per-class coverage − nominal coverage) across classes. "
        "Undercoverage gaps and worst-class coverage are exported as well. Standard conformal "
        "guarantees joint marginal coverage under exchangeability, not within every class.",
    ]
    if spec["kind"] == "label_balance":
        design.append(
            "For the main class-gap comparison, each case is additionally evaluated on the "
            "same balanced-reference test set within its block and seed. This audit is "
            "independent of calibration and never selects a threshold. Its class support "
            "is fixed. Audit marginal coverage under this changed label prior is not "
            "claimed as a nominal population guarantee. Population metrics remain separate."
        )
    if spec["kind"] == "calibration_size":
        design.append(
            "Calibration budgets use nested prefixes of the same shuffled calibration "
            "pool. The cached pool and test arrays are identical in every case. Prediction "
            "diagnostics describe the full cached pool; n_calibration records the actual "
            "prefix used to compute the threshold."
        )
    limits = [
        f"These are synthetic experiments with {len(data.seeds)} seeds. Ranges and SDs are "
        "descriptive; no significance test or universal monotonic relationship is claimed.",
        "Expected top-1 accuracy is 80%. Shared correctness across heads is kept fixed. "
        "The 80% coverage target lies near the mixture boundary between correct and wrong "
        "score components, so thresholds and set sizes can be sensitive to seed there.",
        "Study the group-specific control alongside the main contrasts. Population overall "
        "coverage can remain close to nominal while some classes are undercovered. Per-class "
        "estimates still contain finite-test noise; observed-class gaps omit unobserved classes.",
    ]
    plots = []
    if spec["kind"] == "label_balance":
        plots.append(
            (
                "label_priors",
                "Configured label balance",
                lambda: balance_design(data),
                "Per-head power-law priors and their reversed versions have matching entropy "
                "but put different class cohorts in the majority. Predictions are held fixed "
                "in distribution conditional on the true class within each block.",
            )
        )
        plots.append(
            (
                "class_gap_support",
                "Class gap and test support",
                lambda: support_comparison(data),
                "At 90% nominal coverage, contrast gaps on each case's population test "
                "with gaps on the same balanced audit. All audit class counts match within "
                "a block/seed. Population rare classes have noisier estimates and may be absent.",
            )
        )
    if spec["kind"] == "prediction_loss":
        plots.append(
            (
                "loss_relationship",
                "Same accuracy, different loss",
                lambda: loss_relationship(data),
                "Loss is joint cross-entropy in natural-log units. Each line is one controlled "
                "error mechanism; identical loss does not determine the whole score law.",
            )
        )
    for block in spec["blocks"]:
        plots.append(
            (
                block["name"] + "_overview",
                block["title"] + ": primary comparison",
                lambda b=block: overview(data, b),
                "90% nominal coverage; means and observed seed ranges. Efficiency and "
                "covered information use the population test. "
                + (
                    "Class gaps use the fixed balanced audit."
                    if spec["kind"] == "label_balance"
                    else "Class gaps use the same test labels in every case."
                ),
            )
        )
        plots.append(
            (
                block["name"] + "_targets",
                block["title"] + ": coverage targets",
                lambda b=block: target_curves(data, b),
                "Each line is one case of standard conformal. Population coverage gap is "
                "observed coverage minus target. Whiskers show seed ranges.",
            )
        )
    plots.append(
        (
            "class_coverage",
            "Which joint classes are covered",
            lambda: class_heatmap(data),
            "Per-class coverage pools counts across seeds at the 90% target. "
            + (
                "All cases use the balanced reference test within each block."
                if spec["kind"] == "label_balance"
                else "Every case has the same test labels."
            )
            + " The dashed line marks class 66 when there are 132 joint classes.",
        )
    )
    doc = [
        f"<h1>{html.escape(spec['title'])}</h1><p class='subtitle'>Standard conformal only · "
        f"{report_date()}</p><section class='findings'><h2>Findings</h2>",
        paragraphs_html(claims),
        "</section><section><h2>Design and metrics</h2>",
        paragraphs_html(design),
        "</section>",
    ]
    with PdfPages(output / "report.pdf") as pdf:
        pdf.infodict()["Title"] = spec["title"]
        text_pages(pdf, "Standard conformal: study findings", claims)
        text_pages(pdf, "Design and metric definitions", design)
        for key, title, plotter, caption in plots:
            fig = plotter()
            for extension in ["png", "pdf", "svg"]:
                fig.savefig(figures / f"{key}.{extension}", dpi=170, bbox_inches="tight")
            note = fig.text(
                0.02, -0.03, textwrap.fill(caption, 150), fontsize=9, va="top", color="#475569"
            )
            pdf.savefig(fig, bbox_inches="tight")
            note.remove()
            plt.close(fig)
            encoded = base64.b64encode((figures / f"{key}.png").read_bytes()).decode()
            doc.append(
                f"<section><h2>{html.escape(title)}</h2><figure>"
                f"<img alt='{html.escape(title)}' src='data:image/png;base64,{encoded}'>"
                f"<figcaption>{html.escape(caption)}</figcaption></figure></section>"
            )
        for block in spec["blocks"]:
            headers, table_rows = block_table(data, block)
            doc.append(
                f"<section><h2>{html.escape(block['title'])}: numerical results at 90%</h2>"
                + html_table(headers, table_rows)
                + "</section>"
            )
            fig, ax = plt.subplots(figsize=(13, 6))
            ax.axis("off")
            table = ax.table(
                cellText=table_rows,
                colLabels=headers,
                loc="center",
                cellLoc="center",
                colWidths=[0.28, 0.13, 0.14, 0.14, 0.16, 0.15],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2.1)
            for (i, _), cell in table.get_celld().items():
                cell.set_edgecolor("white")
                cell.set_facecolor("#dbeafe" if i == 0 else "#f1f5f9" if i % 2 else "white")
            fig.suptitle(block["title"] + " · 90% target", fontsize=16)
            fig.text(
                0.05,
                0.07,
                "Mean ± sample SD. Class gaps use the balanced audit in the label-balance "
                "group; other columns use the population test.",
                fontsize=9,
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        text_pages(pdf, "Interpretation and limits", limits)
        source = [
            "Exact source run paths and hashes are in source_runs.csv. Control checks are in "
            "controls.json; case definitions and original settings are pinned in study.json. "
            "Raw and aggregated comparisons are in comparison.csv and summary.csv; paired "
            "differences are within each block against its declared reference.",
            "To rebuild without new simulations, run the same study command with "
            "--manifest pointing to this report directory's study.json.",
        ]
        text_pages(pdf, "Sources and reproducibility", source)
    doc += [
        "<section><h2>Interpretation and limits</h2>",
        paragraphs_html(limits),
        "</section><section><h2>Sources</h2>",
        paragraphs_html(source),
        "</section>",
    ]
    css = """body{font:16px/1.65 system-ui,sans-serif;color:#14213a;background:#eef2f7;margin:0}
    main{max-width:1200px;margin:auto;padding:48px;background:white}h1{font-size:2.5rem;line-height:1.15}
    h2{line-height:1.3}section{margin-top:36px}.subtitle,figcaption{color:#526176}
    .findings{padding:24px;background:#eff6ff;border-left:4px solid #2563eb}
    figure{margin:12px 0}img{width:100%;height:auto}
    figcaption{font-size:.9rem;background:#f8fafc;padding:12px}
    .table-scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:.85rem}
    th,td{padding:12px;border-bottom:1px solid #e2e8f0;text-align:right;white-space:nowrap}
    td:first-child,th:first-child{text-align:left}th{background:#eff6ff}
    @media(max-width:700px){main{padding:22px}h1{font-size:2rem}}"""
    (output / "report.html").write_text(
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'><title>"
        + html.escape(spec["title"])
        + "</title><style>"
        + css
        + "</style></head><body><main>"
        + "".join(doc)
        + "</main></body></html>",
        encoding="utf-8",
    )
    (output / "report_manifest.json").write_text(
        json.dumps(
            {
                "study_sha256": hashlib.sha256((output / "study.json").read_bytes()).hexdigest(),
                "comparison_sha256": hashlib.sha256(
                    (output / "comparison.csv").read_bytes()
                ).hexdigest(),
                "report_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "method": "standard",
                "n_cases": len(plans),
                "n_rows": len(rows),
                "n_figures": len(plots),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Study report: {(output / 'report.html').resolve()}", flush=True)
