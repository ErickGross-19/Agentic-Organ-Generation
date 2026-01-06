"""
Command-Line Interface

CLI for running organ generation and validation tasks from the command line.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

from .agent_runner import AgentRunner, AgentConfig, create_agent
from .llm_client import LLMClient, LLMConfig
from .task_templates import (
    generate_structure_prompt,
    generate_liver_prompt,
    validate_structure_prompt,
    iterate_design_prompt,
)


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="agentic-organ",
        description="Agentic Organ Generation - CLI for LLM-driven organ structure generation",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate an organ structure")
    gen_parser.add_argument(
        "--organ", "-o",
        type=str,
        default="liver",
        help="Organ type (default: liver)",
    )
    gen_parser.add_argument(
        "--segments", "-s",
        type=int,
        default=500,
        help="Number of segments (default: 500)",
    )
    gen_parser.add_argument(
        "--plate-size",
        type=str,
        default="200,200,200",
        help="Build plate size in mm (default: 200,200,200)",
    )
    gen_parser.add_argument(
        "--output", "-O",
        type=str,
        default="./output",
        help="Output directory (default: ./output)",
    )
    gen_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    
    # Validate command
    val_parser = subparsers.add_parser("validate", help="Validate a structure")
    val_parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to mesh file to validate",
    )
    val_parser.add_argument(
        "--stage",
        type=str,
        choices=["pre_embedding", "post_embedding", "both"],
        default="both",
        help="Validation stage (default: both)",
    )
    val_parser.add_argument(
        "--min-channel",
        type=float,
        default=0.5,
        help="Minimum channel diameter in mm (default: 0.5)",
    )
    val_parser.add_argument(
        "--min-wall",
        type=float,
        default=0.3,
        help="Minimum wall thickness in mm (default: 0.3)",
    )
    val_parser.add_argument(
        "--output", "-O",
        type=str,
        default="./output",
        help="Output directory for report (default: ./output)",
    )
    
    # Iterate command
    iter_parser = subparsers.add_parser("iterate", help="Iteratively improve a design")
    iter_parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to current design",
    )
    iter_parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Path to validation report (optional)",
    )
    iter_parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum iterations (default: 5)",
    )
    iter_parser.add_argument(
        "--output", "-O",
        type=str,
        default="./output",
        help="Output directory (default: ./output)",
    )
    
    # Interactive command
    int_parser = subparsers.add_parser("interactive", help="Start interactive session")
    int_parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Initial task to start with",
    )
    
    # Common arguments for all commands
    for p in [gen_parser, val_parser, iter_parser, int_parser]:
        p.add_argument(
            "--provider",
            type=str,
            default="openai",
            choices=["openai", "anthropic", "local"],
            help="LLM provider (default: openai)",
        )
        p.add_argument(
            "--model",
            type=str,
            default=None,
            help="Model name (default: auto based on provider)",
        )
        p.add_argument(
            "--api-key",
            type=str,
            default=None,
            help="API key (or set via environment variable)",
        )
        p.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="Enable verbose output",
        )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Create agent
    agent = create_agent(
        provider=args.provider,
        api_key=args.api_key,
        model=args.model,
        output_dir=getattr(args, 'output', './output'),
        verbose=args.verbose,
    )
    
    # Execute command
    if args.command == "generate":
        run_generate(agent, args)
    elif args.command == "validate":
        run_validate(agent, args)
    elif args.command == "iterate":
        run_iterate(agent, args)
    elif args.command == "interactive":
        run_interactive(agent, args)


def run_generate(agent: AgentRunner, args):
    """Run the generate command."""
    # Parse plate size
    plate_size = tuple(float(x) for x in args.plate_size.split(","))
    
    # Build constraints
    constraints = {
        "plate_size": plate_size,
        "num_segments": args.segments,
    }
    if args.seed is not None:
        constraints["seed"] = args.seed
    
    # Generate prompt
    if args.organ.lower() == "liver":
        prompt = generate_liver_prompt(
            arterial_segments=args.segments,
            venous_segments=args.segments,
            plate_size=plate_size,
            seed=args.seed,
        )
    else:
        prompt = generate_structure_prompt(
            organ_type=args.organ,
            constraints=constraints,
        )
    
    print(f"Generating {args.organ} structure with {args.segments} segments...")
    result = agent.run_task(prompt)
    
    print(f"\nStatus: {result.status.value}")
    if result.error:
        print(f"Error: {result.error}")
    if result.artifacts:
        print(f"Artifacts: {result.artifacts}")
    print(f"Tokens used: {result.total_tokens}")


def run_validate(agent: AgentRunner, args):
    """Run the validate command."""
    constraints = {
        "min_channel_diameter": args.min_channel,
        "min_wall_thickness": args.min_wall,
    }
    
    prompt = validate_structure_prompt(
        mesh_path=args.input,
        validation_stage=args.stage,
        manufacturing_constraints=constraints,
    )
    
    print(f"Validating {args.input} ({args.stage})...")
    result = agent.run_task(prompt)
    
    print(f"\nStatus: {result.status.value}")
    if result.error:
        print(f"Error: {result.error}")
    print(f"Tokens used: {result.total_tokens}")


def run_iterate(agent: AgentRunner, args):
    """Run the iterate command."""
    prompt = iterate_design_prompt(
        current_design_path=args.input,
        validation_report_path=args.report,
        max_iterations=args.max_iterations,
    )
    
    print(f"Iterating on design {args.input}...")
    result = agent.run_task(prompt)
    
    print(f"\nStatus: {result.status.value}")
    if result.error:
        print(f"Error: {result.error}")
    print(f"Iterations: {result.iterations}")
    print(f"Tokens used: {result.total_tokens}")


def run_interactive(agent: AgentRunner, args):
    """Run interactive mode."""
    agent.run_interactive(initial_task=args.task)


if __name__ == "__main__":
    main()
