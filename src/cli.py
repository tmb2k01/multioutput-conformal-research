"""Command-line entry point for synthetic conformal experiments."""

import argparse
import sys

import yaml

from .config import load_config


def main():
    parser = argparse.ArgumentParser(description="Synthetic multi-output conformal experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ["run", "check-config"]:
        sub = commands.add_parser(command)
        sub.add_argument("config", help="YAML experiment file")
        sub.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
        if command == "run":
            sub.add_argument("--predictions", help="Reuse a saved synthetic probabilities.npz")
    report = commands.add_parser(
        "report", help="Plot saved distribution comparisons and write a report"
    )
    report.add_argument("--results", default="outputs/synthetic-comparison.csv")
    report.add_argument("--output-dir", default="docs/synthetic-report")
    study = commands.add_parser("study", help="Run and report one controlled experiment group")
    study.add_argument("config", help="One YAML containing all blocks and cases")
    study.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    study.add_argument("--manifest", help="Rebuild a saved study.json without new simulations")
    study.add_argument("--check", action="store_true", help="Validate all cases without running")
    study.add_argument(
        "--output-dir", help="Report directory; defaults to docs/standard-studies/NAME"
    )
    args = parser.parse_args()
    try:
        if args.command == "study" and args.check:
            from .study import load_study

            spec, plans = load_study(args.config, args.set)
            print(f"{spec['name']}: {len(plans)} valid standard-conformal cases")
            for plan in plans:
                print(f"  {plan['block']}/{plan['case']}")
            return
        if args.command in {"report", "study"}:
            import os
            import tempfile

            with tempfile.TemporaryDirectory(prefix="mocp-matplotlib-") as cache:
                os.environ.setdefault("MPLCONFIGDIR", cache)
                try:
                    from .report import build_report
                except ImportError as error:
                    raise ImportError("Plotting requires: pip install -e '.[plots]'") from error
                if args.command == "report":
                    build_report(args.results, args.output_dir)
                else:
                    from .study import run_study

                    run_study(args.config, args.set, args.manifest, args.output_dir)
            return
        config = load_config(args.config, args.set)
        if args.command == "check-config":
            print(yaml.safe_dump(config.to_dict(), sort_keys=False))
        else:
            from .experiment import run

            run(config, args.predictions)
    except (ValueError, OSError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(2)
