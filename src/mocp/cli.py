"""Command-line entry point."""

import argparse
import sys

from .config import load_config


def main():
    parser = argparse.ArgumentParser(description="Multi-output conformal research experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, description in [
        ("prepare", "Download, inspect and create deterministic dataset split manifests"),
        ("run", "Train, calibrate all configured methods and evaluate joint/marginal coverage"),
        ("check-config", "Validate and print the fully resolved configuration"),
    ]:
        subparser = subparsers.add_parser(command, help=description)
        subparser.add_argument("config", help="YAML experiment file")
        subparser.add_argument(
            "--set",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="Override a dotted config key; repeat as needed",
        )
        if command == "run":
            subparser.add_argument("--predictions", help="Recalibrate saved probabilities.npz")
    args = parser.parse_args()
    try:
        config = load_config(args.config, args.set)
        if args.command == "check-config":
            import yaml

            from . import experiment  # noqa: F401 - register plugins
            from .registry import DATASETS, METHODS, MODELS

            if config.dataset.name not in DATASETS or config.model.name not in MODELS:
                raise ValueError("Unknown dataset or model plugin")
            for method in config.conformal.methods:
                METHODS.create(method.name, **method.params)
            print(yaml.safe_dump(config.to_dict(), sort_keys=False))
        elif args.command == "prepare":
            from .experiment import prepare_only

            prepare_only(config)
        else:
            from .experiment import run

            run(config, args.predictions)
    except (ValueError, OSError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(2)
