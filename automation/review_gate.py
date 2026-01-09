"""
Review Gate

Implements human checkpoint for reviewing LLM-generated scripts before execution.
Supports both CLI (interactive) and notebook environments.
"""

import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable, List


class ReviewAction(Enum):
    """
    Actions available at the review gate.
    
    Attributes
    ----------
    RUN : str
        Execute the script
    DONE : str
        Skip execution and move to verification (assume script was run manually)
    CANCEL : str
        Cancel the workflow, keep generated files
    EDIT : str
        Open script for editing (if editor available)
    VIEW : str
        View the script contents
    """
    RUN = "run"
    DONE = "done"
    CANCEL = "cancel"
    EDIT = "edit"
    VIEW = "view"


@dataclass
class ReviewResult:
    """
    Result of the review gate interaction.
    
    Attributes
    ----------
    action : ReviewAction
        The action chosen by the user
    should_execute : bool
        Whether the script should be executed
    should_continue : bool
        Whether the workflow should continue
    message : str
        Optional message from the user
    """
    action: ReviewAction
    should_execute: bool
    should_continue: bool
    message: str = ""


def print_review_prompt(
    script_path: str,
    run_command: str,
    warnings: Optional[List[str]] = None,
    object_name: str = "",
    version: int = 1,
) -> None:
    """
    Print the review prompt with script information.
    
    Parameters
    ----------
    script_path : str
        Path to the generated script
    run_command : str
        Command to run the script
    warnings : List[str], optional
        List of warnings about the script
    object_name : str
        Name of the object being generated
    version : int
        Version number of the script
    """
    print()
    print("=" * 60)
    print("  REVIEW GATE - Human Review Required")
    print("=" * 60)
    print()
    
    if object_name:
        print(f"Object: {object_name} (v{version})")
        print()
    
    print(f"Generated script: {script_path}")
    print()
    print("To run manually:")
    print(f"  {run_command}")
    print()
    
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"  - {w}")
        print()
    
    print("Actions:")
    print("  run (or yes/y) - EXECUTE THE SCRIPT NOW to generate mesh files")
    print("  done           - Skip execution (only if you already ran the script manually)")
    print("  cancel (or no/n) - Cancel workflow, keep files")
    print("  view           - View the script contents")
    print("  edit           - Open script in editor (if available)")
    print()
    print(">>> To generate your mesh, choose 'run' <<<")
    print()


def read_script_contents(script_path: str, max_lines: int = 100) -> str:
    """
    Read and return script contents for viewing.
    
    Parameters
    ----------
    script_path : str
        Path to the script file
    max_lines : int
        Maximum number of lines to return
        
    Returns
    -------
    str
        Script contents (possibly truncated)
    """
    try:
        with open(script_path, 'r') as f:
            lines = f.readlines()
        
        if len(lines) > max_lines:
            content = ''.join(lines[:max_lines])
            content += f"\n... ({len(lines) - max_lines} more lines) ..."
        else:
            content = ''.join(lines)
        
        return content
    except Exception as e:
        return f"Error reading script: {e}"


def open_in_editor(script_path: str) -> bool:
    """
    Attempt to open the script in an editor.
    
    Parameters
    ----------
    script_path : str
        Path to the script file
        
    Returns
    -------
    bool
        True if editor was opened successfully
    """
    import subprocess
    import shutil
    
    # Try common editors in order of preference
    editors = ['code', 'vim', 'nano', 'vi', 'notepad']
    
    # Check EDITOR environment variable first
    env_editor = os.environ.get('EDITOR')
    if env_editor:
        editors.insert(0, env_editor)
    
    for editor in editors:
        if shutil.which(editor):
            try:
                subprocess.Popen([editor, script_path])
                return True
            except Exception:
                continue
    
    return False


def interactive_review(
    script_path: str,
    run_command: str,
    warnings: Optional[List[str]] = None,
    object_name: str = "",
    version: int = 1,
    input_func: Optional[Callable[[str], str]] = None,
    expected_outputs: Optional[List[str]] = None,
) -> ReviewResult:
    """
    Run interactive review gate.
    
    Parameters
    ----------
    script_path : str
        Path to the generated script
    run_command : str
        Command to run the script
    warnings : List[str], optional
        List of warnings about the script
    object_name : str
        Name of the object being generated
    version : int
        Version number of the script
    input_func : Callable, optional
        Custom input function (for notebook compatibility)
    expected_outputs : List[str], optional
        List of expected output file paths. If provided and user chooses 'done',
        will warn if these files don't exist.
        
    Returns
    -------
    ReviewResult
        The result of the review interaction
    """
    if input_func is None:
        input_func = input
    
    print_review_prompt(script_path, run_command, warnings, object_name, version)
    
    while True:
        try:
            response = input_func("Action [run/done/cancel/view/edit]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled by user.")
            return ReviewResult(
                action=ReviewAction.CANCEL,
                should_execute=False,
                should_continue=False,
                message="Cancelled by user interrupt",
            )
        
        if response in ("run", "r", "yes", "y"):
            return ReviewResult(
                action=ReviewAction.RUN,
                should_execute=True,
                should_continue=True,
            )
        
        elif response in ("done", "d", "skip"):
            if expected_outputs:
                missing_files = [f for f in expected_outputs if not os.path.exists(f)]
                if missing_files:
                    print()
                    print("WARNING: The following expected output files were not found:")
                    for f in missing_files:
                        print(f"  - {f}")
                    print()
                    print("This suggests the script may not have been run yet.")
                    print("Choose 'run' to execute the script, or 'done' again to continue anyway.")
                    print()
                    continue
            return ReviewResult(
                action=ReviewAction.DONE,
                should_execute=False,
                should_continue=True,
                message="User indicated script was run manually",
            )
        
        elif response in ("cancel", "c", "quit", "exit", "q", "no", "n"):
            return ReviewResult(
                action=ReviewAction.CANCEL,
                should_execute=False,
                should_continue=False,
                message="Cancelled by user",
            )
        
        elif response in ("view", "v"):
            print()
            print("-" * 40)
            print(read_script_contents(script_path))
            print("-" * 40)
            print()
        
        elif response in ("edit", "e"):
            if open_in_editor(script_path):
                print(f"Opened {script_path} in editor.")
                print("Make your changes, save, and then choose an action.")
            else:
                print("No editor available. Please edit the file manually:")
                print(f"  {script_path}")
            print()
        
        elif response == "":
            print("Please enter an action: run, done, cancel, view, or edit")
        
        else:
            print(f"Unknown action: '{response}'")
            print("Valid actions: run (yes/y), done, cancel (no/n), view, edit")


def auto_review(skip_execution: bool = False) -> ReviewResult:
    """
    Automatic review (no human interaction).
    
    Parameters
    ----------
    skip_execution : bool
        If True, return DONE (skip execution)
        If False, return RUN (execute automatically)
        
    Returns
    -------
    ReviewResult
        Result indicating automatic action
    """
    if skip_execution:
        return ReviewResult(
            action=ReviewAction.DONE,
            should_execute=False,
            should_continue=True,
            message="Auto-review: skipped execution",
        )
    else:
        return ReviewResult(
            action=ReviewAction.RUN,
            should_execute=True,
            should_continue=True,
            message="Auto-review: executing automatically",
        )


def run_review_gate(
    script_path: str,
    run_command: str,
    warnings: Optional[List[str]] = None,
    object_name: str = "",
    version: int = 1,
    interactive: bool = True,
    auto_run: bool = False,
    input_func: Optional[Callable[[str], str]] = None,
    expected_outputs: Optional[List[str]] = None,
) -> ReviewResult:
    """
    Run the review gate with appropriate mode.
    
    Parameters
    ----------
    script_path : str
        Path to the generated script
    run_command : str
        Command to run the script
    warnings : List[str], optional
        List of warnings about the script
    object_name : str
        Name of the object being generated
    version : int
        Version number of the script
    interactive : bool
        Whether to run in interactive mode
    auto_run : bool
        If not interactive, whether to auto-run or skip
    input_func : Callable, optional
        Custom input function (for notebook compatibility)
    expected_outputs : List[str], optional
        List of expected output file paths. If provided and user chooses 'done',
        will warn if these files don't exist.
        
    Returns
    -------
    ReviewResult
        The result of the review gate
    """
    if not interactive:
        return auto_review(skip_execution=not auto_run)
    
    return interactive_review(
        script_path=script_path,
        run_command=run_command,
        warnings=warnings,
        object_name=object_name,
        version=version,
        input_func=input_func,
        expected_outputs=expected_outputs,
    )
