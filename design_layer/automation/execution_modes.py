"""
Execution Modes

Defines execution modes for the automation pipeline, controlling how LLM-generated
code is handled: write-only, review-then-run, or auto-run.
"""

from enum import Enum
from typing import Optional


class ExecutionMode(Enum):
    """
    Execution mode for LLM-generated scripts.
    
    Attributes
    ----------
    WRITE_ONLY : str
        Generate script and write to disk, but don't execute.
        Useful for manual review and offline execution.
    REVIEW_THEN_RUN : str
        Generate script, pause for human review, then run after confirmation.
        Default mode for interactive workflows.
    AUTO_RUN : str
        Generate script and execute automatically without human review.
        Use with caution - recommended only for trusted, well-tested prompts.
    """
    WRITE_ONLY = "write_only"
    REVIEW_THEN_RUN = "review_then_run"
    AUTO_RUN = "auto_run"


DEFAULT_EXECUTION_MODE = ExecutionMode.REVIEW_THEN_RUN


def parse_execution_mode(mode_str: Optional[str]) -> ExecutionMode:
    """
    Parse a string into an ExecutionMode enum value.
    
    Parameters
    ----------
    mode_str : str or None
        String representation of execution mode.
        Valid values: "write_only", "review_then_run", "auto_run"
        
    Returns
    -------
    ExecutionMode
        The corresponding ExecutionMode enum value.
        Returns DEFAULT_EXECUTION_MODE if mode_str is None or empty.
        
    Raises
    ------
    ValueError
        If mode_str is not a valid execution mode.
        
    Examples
    --------
    >>> parse_execution_mode("write_only")
    <ExecutionMode.WRITE_ONLY: 'write_only'>
    >>> parse_execution_mode(None)
    <ExecutionMode.REVIEW_THEN_RUN: 'review_then_run'>
    """
    if not mode_str:
        return DEFAULT_EXECUTION_MODE
    
    mode_str = mode_str.lower().strip()
    
    mode_map = {
        "write_only": ExecutionMode.WRITE_ONLY,
        "write-only": ExecutionMode.WRITE_ONLY,
        "writeonly": ExecutionMode.WRITE_ONLY,
        "review_then_run": ExecutionMode.REVIEW_THEN_RUN,
        "review-then-run": ExecutionMode.REVIEW_THEN_RUN,
        "reviewthenrun": ExecutionMode.REVIEW_THEN_RUN,
        "review": ExecutionMode.REVIEW_THEN_RUN,
        "auto_run": ExecutionMode.AUTO_RUN,
        "auto-run": ExecutionMode.AUTO_RUN,
        "autorun": ExecutionMode.AUTO_RUN,
        "auto": ExecutionMode.AUTO_RUN,
    }
    
    if mode_str in mode_map:
        return mode_map[mode_str]
    
    valid_modes = ["write_only", "review_then_run", "auto_run"]
    raise ValueError(
        f"Invalid execution mode: '{mode_str}'. "
        f"Valid modes are: {', '.join(valid_modes)}"
    )


def should_write_script(mode: ExecutionMode) -> bool:
    """
    Check if the execution mode requires writing a script to disk.
    
    Parameters
    ----------
    mode : ExecutionMode
        The execution mode to check.
        
    Returns
    -------
    bool
        True if scripts should be written to disk.
    """
    return mode in (
        ExecutionMode.WRITE_ONLY,
        ExecutionMode.REVIEW_THEN_RUN,
        ExecutionMode.AUTO_RUN,
    )


def should_pause_for_review(mode: ExecutionMode) -> bool:
    """
    Check if the execution mode requires pausing for human review.
    
    Parameters
    ----------
    mode : ExecutionMode
        The execution mode to check.
        
    Returns
    -------
    bool
        True if execution should pause for human review before running.
    """
    return mode == ExecutionMode.REVIEW_THEN_RUN


def should_execute(mode: ExecutionMode) -> bool:
    """
    Check if the execution mode allows script execution.
    
    Parameters
    ----------
    mode : ExecutionMode
        The execution mode to check.
        
    Returns
    -------
    bool
        True if scripts should be executed (after review if applicable).
    """
    return mode in (ExecutionMode.REVIEW_THEN_RUN, ExecutionMode.AUTO_RUN)


def get_mode_description(mode: ExecutionMode) -> str:
    """
    Get a human-readable description of an execution mode.
    
    Parameters
    ----------
    mode : ExecutionMode
        The execution mode to describe.
        
    Returns
    -------
    str
        Human-readable description of the mode.
    """
    descriptions = {
        ExecutionMode.WRITE_ONLY: (
            "Write-only mode: Scripts are generated and saved to disk but not executed. "
            "Review and run manually."
        ),
        ExecutionMode.REVIEW_THEN_RUN: (
            "Review-then-run mode: Scripts are generated and saved, then you can review "
            "before deciding to run, skip, or cancel."
        ),
        ExecutionMode.AUTO_RUN: (
            "Auto-run mode: Scripts are generated and executed automatically without "
            "human review. Use with caution."
        ),
    }
    return descriptions.get(mode, f"Unknown mode: {mode}")
