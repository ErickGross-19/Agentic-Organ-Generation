#!/usr/bin/env python3
"""
Runner script for DesignSpec examples.

This script loads a DesignSpec JSON file and runs it through the DesignSpecRunner
pipeline, saving outputs and printing a stage summary.

Usage:
    python scripts/run_designspec_example.py --spec examples/designspec/01_minimal_box_network.json --out ./output
    python scripts/run_designspec_example.py --spec examples/designspec/01_minimal_box_network.json --out ./output --run-until compile_domains
"""

import argparse
import json
import sys
import time
from pathlib import Path

from designspec.spec import DesignSpec
from designspec.runner import DesignSpecRunner
from designspec.plan import ExecutionPlan


VALID_STAGES = [
    "compile_policies",
    "compile_domains",
    "component_ports",
    "component_build",
    "component_mesh",
    "union_voids",
    "mesh_domain",
    "embed",
    "port_recarve",
    "validity",
    "export",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a DesignSpec example through the DesignSpecRunner pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full pipeline
    python scripts/run_designspec_example.py --spec examples/designspec/01_minimal_box_network.json --out ./output

    # Run until a specific stage (for debugging)
    python scripts/run_designspec_example.py --spec examples/designspec/01_minimal_box_network.json --out ./output --run-until compile_domains

Available stages:
    compile_policies, compile_domains, component_ports, component_build,
    component_mesh, union_voids, mesh_domain, embed, port_recarve, validity, export
        """,
    )
    parser.add_argument(
        "--spec",
        type=str,
        required=True,
        help="Path to the DesignSpec JSON file",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory for artifacts",
    )
    parser.add_argument(
        "--run-until",
        type=str,
        default=None,
        choices=VALID_STAGES,
        help="Stop after this stage (for debugging)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    return parser.parse_args()


def print_stage_summary(result, verbose=False):
    """Print a summary of the runner result."""
    print("\n" + "=" * 60)
    print("EXECUTION SUMMARY")
    print("=" * 60)

    print(f"\nSuccess: {result.success}")
    print(f"Stages completed: {', '.join(result.stages_completed)}")

    if result.stage_reports:
        print("\nStage Reports:")
        for report in result.stage_reports:
            status = "OK" if report.success else "FAILED"
            duration = getattr(report, "duration_s", None)
            duration_str = f" ({duration:.2f}s)" if duration is not None else ""
            print(f"  - {report.stage}: {status}{duration_str}")

            if verbose and hasattr(report, "metadata") and report.metadata:
                for key, value in report.metadata.items():
                    if isinstance(value, dict):
                        print(f"      {key}:")
                        for k, v in value.items():
                            print(f"        {k}: {v}")
                    else:
                        print(f"      {key}: {value}")

            if not report.success and hasattr(report, "error"):
                print(f"      Error: {report.error}")

    if hasattr(result, "warnings") and result.warnings:
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"  - {warning}")

    print("\n" + "=" * 60)


def main():
    args = parse_args()

    spec_path = Path(args.spec)
    if not spec_path.exists():
        print(f"Error: Spec file not found: {spec_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading spec: {spec_path}")
    start_time = time.time()

    try:
        spec = DesignSpec.from_json(str(spec_path))
    except Exception as e:
        print(f"Error loading spec: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Spec loaded: {spec.meta.get('name', 'unnamed')}")
    print(f"Seed: {spec.meta.get('seed', 'not set')}")
    print(f"Input units: {spec.meta.get('input_units', 'not set')}")

    plan = None
    if args.run_until:
        print(f"Running until stage: {args.run_until}")
        plan = ExecutionPlan(run_until=args.run_until)

    print(f"\nOutput directory: {output_dir}")
    print("\nStarting runner...")

    try:
        runner = DesignSpecRunner(spec, plan=plan, output_dir=output_dir)
        result = runner.run()
    except Exception as e:
        print(f"Error during execution: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - start_time
    print(f"\nTotal execution time: {elapsed:.2f}s")

    print_stage_summary(result, verbose=args.verbose)

    run_report_path = output_dir / "run_report.json"
    try:
        result_dict = result.to_dict()
        with open(run_report_path, "w") as f:
            json.dump(result_dict, f, indent=2, default=str)
        print(f"\nRun report saved to: {run_report_path}")
    except Exception as e:
        print(f"Warning: Could not save run report: {e}", file=sys.stderr)

    if not result.success:
        print("\nExecution failed!", file=sys.stderr)
        sys.exit(1)

    print("\nExecution completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
