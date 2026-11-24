"""One YAML per controlled standard-conformal study: run, verify, compare, report."""

import contextlib
import csv
import hashlib
import json
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from .config import config_from_dict, deep_merge
from .conformal import candidate_space, load_method, validate_probabilities
from .experiment import class_metric_rows, run, write_csv
from .metrics import evaluate


def read_rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_study(path, overrides=()):
    spec = yaml.safe_load(Path(path).read_text())
    if not isinstance(spec, dict) or set(spec) - {
        "name",
        "title",
        "kind",
        "description",
        "base",
        "blocks",
    }:
        raise ValueError("Invalid study mapping; use name/title/kind/description/base/blocks")
    if spec.get("kind") not in {"label_balance", "prediction_loss", "calibration_size"}:
        raise ValueError("Study kind must be label_balance, prediction_loss or calibration_size")
    safe = re.compile(r"[A-Za-z0-9_-]+")
    if not isinstance(spec.get("name"), str) or not safe.fullmatch(spec["name"]):
        raise ValueError("Study name must contain letters, digits, hyphens or underscores")
    plans, names = [], set()
    for block in spec.get("blocks", []):
        if set(block) - {"name", "title", "description", "reference", "overrides", "cases"}:
            raise ValueError("Unknown study block field")
        if not safe.fullmatch(block["name"]) or block["name"] in names:
            raise ValueError("Block names must be distinct safe names")
        names.add(block["name"])
        cases = set()
        for case in block["cases"]:
            if set(case) - {"name", "label", "overrides", "balance"}:
                raise ValueError("Unknown study case field")
            if not safe.fullmatch(case["name"]) or case["name"] in cases:
                raise ValueError("Case names must be distinct safe names within a block")
            cases.add(case["name"])
            raw = deep_merge(spec.get("base", {}), block.get("overrides", {}))
            raw = deep_merge(raw, case.get("overrides", {}))
            raw["name"] = f"{block['name']}-{case['name']}"
            raw["output_dir"] = str(Path(raw.get("output_dir", "outputs")) / spec["name"])
            cfg = config_from_dict(raw, overrides)
            if "balance" in case:
                balance = case["balance"]
                exponent, reverse = balance.get("exponent", 0), balance.get("reverse", False)
                if (
                    spec["kind"] != "label_balance"
                    or set(balance) - {"exponent", "reverse"}
                    or not isinstance(exponent, (int, float))
                    or isinstance(exponent, bool)
                    or not np.isfinite(exponent)
                    or exponent < 0
                    or not isinstance(reverse, bool)
                ):
                    raise ValueError("balance requires a nonnegative exponent and boolean reverse")
                if cfg.dataset.joint_probabilities is not None:
                    raise ValueError("Choose case.balance or dataset.joint_probabilities")
                cfg.dataset.class_probabilities = None
                if exponent:
                    priors = []
                    for c in cfg.dataset.num_classes:
                        weights = np.arange(1, c + 1, dtype=float) ** (-exponent)
                        weights /= weights.sum()
                        priors.append((weights[::-1] if reverse else weights).tolist())
                    cfg.dataset.class_probabilities = priors
                cfg.validate()
            if (
                len(cfg.conformal.methods) != 1
                or cfg.conformal.methods[0].name != "standard"
                or cfg.conformal.methods[0].identifier != "standard"
            ):
                raise ValueError("These studies currently compare standard conformal only")
            if not cfg.artifacts.save_probabilities:
                raise ValueError(
                    "Studies need artifacts.save_probabilities=true for control checks"
                )
            if 0.1 not in cfg.conformal.alphas:
                raise ValueError("Study reports need alpha=0.1 for the primary comparison")
            plans.append(
                {
                    "block": block["name"],
                    "case": case["name"],
                    "label": case["label"],
                    "config": cfg,
                }
            )
        if len(cases) < 2 or block["reference"] not in cases:
            raise ValueError("Each block needs >=2 cases and a reference naming one case")
    if not plans:
        raise ValueError("A study needs at least one block")
    first = plans[0]["config"]
    for plan in plans:
        cfg = plan["config"]
        if (
            cfg.seeds != first.seeds
            or cfg.splits != first.splits
            or cfg.dataset.n_samples != first.dataset.n_samples
            or cfg.dataset.num_classes != first.dataset.num_classes
            or cfg.conformal.alphas != first.conformal.alphas
        ):
            raise ValueError(
                "All cases need matching seeds, split fractions, samples and dimensions"
            )
    return spec, plans


def joint_prior(cfg):
    if cfg.dataset.joint_probabilities is not None:
        return np.asarray(cfg.dataset.joint_probabilities)
    priors = cfg.dataset.class_probabilities
    result = np.ones(1)
    for k, c in enumerate(cfg.dataset.num_classes):
        result = np.multiply.outer(
            result, np.full(c, 1 / c) if priors is None else priors[k]
        ).ravel()
    return result


def check_controls(spec, plans, runs):
    """Fail before interpretation if a supposedly fixed aspect changed."""
    references = {}
    first_cfg = plans[0]["config"]
    for block in spec["blocks"]:
        references[block["name"]] = next(
            p for p in plans if p["block"] == block["name"] and p["case"] == block["reference"]
        )
    for plan in plans:
        cfg = plan["config"]
        ref = references[plan["block"]]
        ref_cfg = ref["config"]
        if spec["kind"] == "label_balance":
            if cfg.predictions != ref_cfg.predictions or cfg.conformal != ref_cfg.conformal:
                raise ValueError(
                    "Label-balance cases must hold all prediction and CP settings fixed"
                )
        elif spec["kind"] == "prediction_loss":
            if cfg.dataset != first_cfg.dataset or cfg.conformal != first_cfg.conformal:
                raise ValueError("Loss cases must hold label distribution and CP settings fixed")
        else:
            if cfg.dataset != ref_cfg.dataset or cfg.predictions != ref_cfg.predictions:
                raise ValueError("Calibration-size cases must hold labels and predictions fixed")
        if spec["kind"] == "prediction_loss":
            ref = plans[0]
        for seed in cfg.seeds:
            cache_path = Path(runs[plan["block"], plan["case"]]) / f"seed-{seed}/probabilities.npz"
            ref_path = Path(runs[ref["block"], ref["case"]]) / f"seed-{seed}/probabilities.npz"
            with (
                np.load(cache_path, allow_pickle=False) as cache,
                np.load(ref_path, allow_pickle=False) as baseline,
            ):
                meta = json.loads(str(cache["metadata"]))
                if meta["schema_version"] != 2 or meta["seed"] != seed:
                    raise ValueError("Cached schema or seed differs from the study")
                for key in ["dataset", "predictions", "splits"]:
                    if meta["source_config"][key] != cfg.to_dict()[key]:
                        raise ValueError(f"Cached {key} differs from its source case")
                for split in ["clustering", "calibration", "test"]:
                    y, base_y = cache[f"{split}_y"], baseline[f"{split}_y"]
                    p = [cache[f"{split}_p{k}"] for k in range(len(cfg.dataset.num_classes))]
                    base_p = [
                        baseline[f"{split}_p{k}"] for k in range(len(cfg.dataset.num_classes))
                    ]
                    validate_probabilities(p, y)
                    point, base_point = (
                        np.column_stack([v.argmax(1) for v in arrays]) for arrays in [p, base_p]
                    )
                    if not np.array_equal(point == y, base_point == base_y):
                        raise ValueError("Cases do not have exactly paired correctness patterns")
                    if spec["kind"] != "label_balance" and (
                        not np.array_equal(y, base_y) or not np.array_equal(point, base_point)
                    ):
                        raise ValueError("Loss/budget controls changed labels or point predictions")
                    if spec["kind"] == "calibration_size" and any(
                        not np.array_equal(a, b) for a, b in zip(p, base_p, strict=True)
                    ):
                        raise ValueError("Calibration-size controls changed model probabilities")
    return references


def compare(spec, plans, runs, output):
    refs = check_controls(spec, plans, runs)
    output.mkdir(parents=True, exist_ok=True)
    rows, audit_classes, sources = [], [], []
    for plan in plans:
        cfg = plan["config"]
        run_path = Path(runs[plan["block"], plan["case"]])
        prediction = {
            (int(r["seed"]), r["scope"]): r
            for r in read_rows(run_path / "prediction_metrics.csv")
            if r["split"] == "test"
        }
        prior = joint_prior(cfg)
        positive = prior[prior > 0]
        effective = float(np.exp(-np.sum(positive * np.log(positive))))
        sources.append(
            {
                "block": plan["block"],
                "case": plan["case"],
                "run": str(run_path),
                "config_sha256": digest(run_path / "config.yaml"),
                "metrics_sha256": digest(run_path / "metrics.csv"),
                "prediction_metrics_sha256": digest(run_path / "prediction_metrics.csv"),
            }
        )
        for raw in read_rows(run_path / "metrics.csv"):
            seed, alpha = int(raw["seed"]), float(raw["alpha"])
            row = {
                "block": plan["block"],
                "case": plan["case"],
                "label": plan["label"],
                **raw,
                "effective_joint_classes": effective,
                "joint_nll": float(prediction[seed, "joint"]["negative_log_likelihood"]),
                "joint_brier": float(prediction[seed, "joint"]["brier_score"]),
            }
            if spec["kind"] == "label_balance":
                ref = refs[plan["block"]]
                path = Path(runs[ref["block"], ref["case"]]) / f"seed-{seed}/probabilities.npz"
                with np.load(path, allow_pickle=False) as cache:
                    y = cache["test_y"]
                    p = [cache[f"test_p{k}"] for k in range(len(cfg.dataset.num_classes))]
                    meta = json.loads(str(cache["metadata"]))
                method = load_method(run_path / f"seed-{seed}", "standard", alpha)
                result, artifacts = evaluate(
                    method,
                    p,
                    y,
                    candidate_space(cfg.dataset.num_classes),
                    meta["task_names"],
                    alpha,
                    cfg.evaluation,
                )
                row.update({f"audit_{k}": v for k, v in result.items()})
                audit_classes.extend(
                    {
                        "block": plan["block"],
                        "case": plan["case"],
                        "seed": seed,
                        "alpha": alpha,
                        **r,
                    }
                    for r in class_metric_rows(artifacts, meta)
                    if r["scope"] == "joint"
                )
            rows.append(row)
        print(f"Compared {plan['block']}/{plan['case']}", flush=True)
    write_csv(output / "comparison.csv", rows)
    write_csv(output / "source_runs.csv", sources)
    if audit_classes:
        write_csv(output / "audit_class_metrics.csv", audit_classes)
    (output / "controls.json").write_text(
        json.dumps(
            {
                "all_splits_checked": ["clustering", "calibration", "test"],
                "exact_correctness_patterns_matched": True,
                "exact_labels_and_argmax_matched": spec["kind"] != "label_balance",
                "exact_probabilities_matched": spec["kind"] == "calibration_size",
                "audit": "Fixed balanced reference test per block/seed; not population coverage"
                if spec["kind"] == "label_balance"
                else None,
            },
            indent=2,
        )
        + "\n"
    )
    return rows


def run_study(path, overrides=(), manifest=None, output_dir=None):
    spec, plans = load_study(path, overrides)
    output = Path(output_dir or f"docs/standard-studies/{spec['name']}")
    if manifest:
        saved = json.loads(Path(manifest).read_text())
        if saved["spec"] != spec or saved["overrides"] != list(overrides):
            raise ValueError("Study YAML/overrides changed; use the source YAML or run a new study")
        runs = {(r["block"], r["case"]): r["run"] for r in saved["runs"]}
        for plan in plans:
            run_path = Path(runs[plan["block"], plan["case"]])
            stored = yaml.safe_load((run_path / "config.yaml").read_text())
            if stored != plan["config"].to_dict():
                raise ValueError("Saved case settings differ from the study definition")
    else:
        runs = {}
        for plan in plans:
            print(f"Running {spec['name']}: {plan['block']}/{plan['case']}", flush=True)
            # Keep the detailed runner log with each case; console reports study progress.
            with (
                tempfile.TemporaryFile(mode="w+", encoding="utf-8") as log,
                contextlib.redirect_stdout(log),
            ):
                destination = run(plan["config"])
                log.seek(0)
                (destination / "console.log").write_text(log.read())
            runs[plan["block"], plan["case"]] = str(destination.resolve())
        saved = {
            "schema_version": 1,
            "spec": spec,
            "overrides": list(overrides),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_yaml": str(Path(path).resolve()),
            "source_yaml_sha256": digest(path),
            "runs": [{"block": b, "case": c, "run": p} for (b, c), p in runs.items()],
        }
    rows = compare(spec, plans, runs, output)
    (output / "study.json").write_text(json.dumps(saved, indent=2) + "\n")
    from .study_report import build_study_report

    build_study_report(spec, plans, rows, runs, output)
    return output
