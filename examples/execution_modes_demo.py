"""
Execution Modes Demo

This script demonstrates the three execution modes available in the
Single Agent Organ Generator V2 workflow:

1. WRITE_ONLY: Generate script, don't run
2. REVIEW_THEN_RUN: Generate script, pause for review, then run (default)
3. AUTO_RUN: Generate script and run automatically

Usage:
    python execution_modes_demo.py --mode write_only
    python execution_modes_demo.py --mode review_then_run
    python execution_modes_demo.py --mode auto_run
"""

import sys
import os

sys.path.insert(0, os.path.abspath(".."))

from automation import (
    SingleAgentOrganGeneratorV2,
    ExecutionMode,
    parse_execution_mode,
    DEFAULT_EXECUTION_MODE,
    get_mode_description,
    create_agent,
)


def demo_execution_modes():
    """Demonstrate the available execution modes."""
    print("=" * 60)
    print("  Execution Modes Demo")
    print("=" * 60)
    print()
    
    print("Available execution modes:")
    print()
    
    for mode in ExecutionMode:
        desc = get_mode_description(mode)
        default_marker = " (default)" if mode == DEFAULT_EXECUTION_MODE else ""
        print(f"  {mode.value}{default_marker}")
        print(f"    {desc}")
        print()
    
    print("-" * 60)
    print()


def demo_workflow_with_mode(mode_str: str):
    """
    Demonstrate running the workflow with a specific execution mode.
    
    Parameters
    ----------
    mode_str : str
        Execution mode string: "write_only", "review_then_run", or "auto_run"
    """
    print(f"Running workflow with execution mode: {mode_str}")
    print()
    
    execution_mode = parse_execution_mode(mode_str)
    
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("Warning: OPENAI_API_KEY not set. Using mock mode.")
        print("Set OPENAI_API_KEY to run with actual LLM.")
        return
    
    agent = create_agent(
        provider="openai",
        model="gpt-4",
        output_dir="./outputs",
        verbose=True,
    )
    
    workflow = SingleAgentOrganGeneratorV2(
        agent=agent,
        base_output_dir="./outputs",
        verbose=True,
        execution_mode=execution_mode,
        timeout_seconds=300.0,
    )
    
    print(f"Workflow initialized with execution mode: {execution_mode.value}")
    print()
    print("To run the full workflow interactively, call: workflow.run()")
    print()


def demo_script_contract():
    """Demonstrate the script contract requirements."""
    from automation.task_templates import (
        SCRIPT_CONTRACT_REQUIREMENTS,
        script_contract_prompt,
        validate_script_contract,
    )
    
    print("=" * 60)
    print("  Script Contract Requirements")
    print("=" * 60)
    print()
    print(SCRIPT_CONTRACT_REQUIREMENTS)
    print()
    
    print("-" * 60)
    print("Example prompt generation:")
    print("-" * 60)
    print()
    
    prompt = script_contract_prompt(
        spec_path="./spec_v001.json",
        output_units="mm",
        feedback_context="Make the network denser near the inlet",
    )
    print(prompt[:500] + "...")
    print()
    
    print("-" * 60)
    print("Script validation example:")
    print("-" * 60)
    print()
    
    good_script = '''
import os
OUTPUT_DIR = os.environ.get("ORGAN_AGENT_OUTPUT_DIR", ".")

def main():
    print("Generating network...")
    print("ARTIFACTS_JSON: {\\"files\\": [], \\"metrics\\": {}, \\"status\\": \\"success\\"}")

if __name__ == "__main__":
    main()
'''
    
    bad_script = '''
import subprocess
subprocess.run(["rm", "-rf", "/"])
'''
    
    print("Good script validation:")
    result = validate_script_contract(good_script)
    print(f"  Valid: {result['valid']}")
    print(f"  Warnings: {result['warnings']}")
    print(f"  Errors: {result['errors']}")
    print()
    
    print("Bad script validation:")
    result = validate_script_contract(bad_script)
    print(f"  Valid: {result['valid']}")
    print(f"  Warnings: {result['warnings']}")
    print(f"  Errors: {result['errors']}")
    print()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Execution Modes Demo")
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["write_only", "review_then_run", "auto_run"],
        default=None,
        help="Execution mode to demonstrate",
    )
    parser.add_argument(
        "--contract",
        action="store_true",
        help="Show script contract requirements",
    )
    
    args = parser.parse_args()
    
    demo_execution_modes()
    
    if args.contract:
        demo_script_contract()
    
    if args.mode:
        demo_workflow_with_mode(args.mode)


if __name__ == "__main__":
    main()
