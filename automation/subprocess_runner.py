"""
Subprocess Runner

Safe subprocess execution for LLM-generated scripts with timeout enforcement,
output capture, and environment isolation.
"""

import os
import sys
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path


DEFAULT_TIMEOUT_SECONDS = 300  # 5 minutes


@dataclass
class RunResult:
    """
    Result of running a script via subprocess.
    
    Attributes
    ----------
    success : bool
        Whether the script executed successfully (exit code 0)
    exit_code : int
        Process exit code
    elapsed_seconds : float
        Total execution time in seconds
    stdout : str
        Captured stdout output
    stderr : str
        Captured stderr output
    log_path : str
        Path to the log file
    timed_out : bool
        Whether the process was killed due to timeout
    error : str or None
        Error message if execution failed
    last_lines : List[str]
        Last N lines of output for quick inspection
    """
    success: bool
    exit_code: int
    elapsed_seconds: float
    stdout: str
    stderr: str
    log_path: str
    timed_out: bool = False
    error: Optional[str] = None
    last_lines: List[str] = field(default_factory=list)


def build_environment(
    object_dir: str,
    extra_env: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    Build the environment for subprocess execution.
    
    Parameters
    ----------
    object_dir : str
        Path to the object directory (set as ORGAN_AGENT_OUTPUT_DIR)
    extra_env : Dict[str, str], optional
        Additional environment variables to set
        
    Returns
    -------
    Dict[str, str]
        Environment dictionary for subprocess
    """
    env = os.environ.copy()
    
    # Set the output directory environment variable
    env['ORGAN_AGENT_OUTPUT_DIR'] = os.path.abspath(object_dir)
    
    # Ensure Python can find the generation and validity modules
    # by adding the repo root to PYTHONPATH
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    existing_pythonpath = env.get('PYTHONPATH', '')
    if existing_pythonpath:
        env['PYTHONPATH'] = f"{repo_root}:{existing_pythonpath}"
    else:
        env['PYTHONPATH'] = repo_root
    
    # Add any extra environment variables
    if extra_env:
        env.update(extra_env)
    
    return env


def get_last_lines(text: str, n: int = 20) -> List[str]:
    """
    Get the last N lines of text.
    
    Parameters
    ----------
    text : str
        Text to extract lines from
    n : int
        Number of lines to return
        
    Returns
    -------
    List[str]
        Last N lines
    """
    lines = text.strip().split('\n')
    return lines[-n:] if len(lines) > n else lines


def run_script(
    script_path: str,
    object_dir: str,
    version: int,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    python_executable: Optional[str] = None,
    extra_env: Optional[Dict[str, str]] = None,
    capture_output: bool = True,
) -> RunResult:
    """
    Run a Python script via subprocess with timeout and output capture.
    
    Parameters
    ----------
    script_path : str
        Path to the Python script to execute
    object_dir : str
        Working directory and output directory for the script
    version : int
        Version number for log file naming
    timeout_seconds : float
        Maximum execution time in seconds (default: 300)
    python_executable : str, optional
        Python executable to use (default: sys.executable)
    extra_env : Dict[str, str], optional
        Additional environment variables
    capture_output : bool
        Whether to capture stdout/stderr (default: True)
        
    Returns
    -------
    RunResult
        Result containing exit code, output, and execution details
    """
    if python_executable is None:
        python_executable = sys.executable
    
    # Ensure paths are absolute
    script_path = os.path.abspath(script_path)
    object_dir = os.path.abspath(object_dir)
    
    # Verify script exists
    if not os.path.exists(script_path):
        return RunResult(
            success=False,
            exit_code=-1,
            elapsed_seconds=0.0,
            stdout="",
            stderr="",
            log_path="",
            error=f"Script not found: {script_path}",
        )
    
    # Ensure object directory exists
    os.makedirs(object_dir, exist_ok=True)
    
    # Define log path
    log_filename = f"run_v{version:03d}.log"
    log_path = os.path.join(object_dir, log_filename)
    
    # Build environment
    env = build_environment(object_dir, extra_env)
    
    # Build command
    cmd = [python_executable, script_path]
    
    # Track execution time
    start_time = time.time()
    
    stdout_content = ""
    stderr_content = ""
    timed_out = False
    exit_code = -1
    error_msg = None
    
    try:
        if capture_output:
            # Run with output capture
            result = subprocess.run(
                cmd,
                cwd=object_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            
            stdout_content = result.stdout
            stderr_content = result.stderr
            exit_code = result.returncode
            
        else:
            # Run without capture (output goes to terminal)
            result = subprocess.run(
                cmd,
                cwd=object_dir,
                env=env,
                timeout=timeout_seconds,
            )
            exit_code = result.returncode
            
    except subprocess.TimeoutExpired as e:
        timed_out = True
        exit_code = -1
        error_msg = f"Script timed out after {timeout_seconds} seconds"
        
        # Try to capture partial output
        if hasattr(e, 'stdout') and e.stdout:
            stdout_content = e.stdout if isinstance(e.stdout, str) else e.stdout.decode('utf-8', errors='replace')
        if hasattr(e, 'stderr') and e.stderr:
            stderr_content = e.stderr if isinstance(e.stderr, str) else e.stderr.decode('utf-8', errors='replace')
            
    except Exception as e:
        exit_code = -1
        error_msg = f"Failed to execute script: {e}"
    
    elapsed_seconds = time.time() - start_time
    
    # Write log file
    try:
        with open(log_path, 'w') as f:
            f.write(f"Script: {script_path}\n")
            f.write(f"Working directory: {object_dir}\n")
            f.write(f"Python: {python_executable}\n")
            f.write(f"Timeout: {timeout_seconds}s\n")
            f.write(f"Exit code: {exit_code}\n")
            f.write(f"Elapsed: {elapsed_seconds:.2f}s\n")
            f.write(f"Timed out: {timed_out}\n")
            f.write("\n" + "=" * 40 + " STDOUT " + "=" * 40 + "\n")
            f.write(stdout_content)
            f.write("\n" + "=" * 40 + " STDERR " + "=" * 40 + "\n")
            f.write(stderr_content)
            if error_msg:
                f.write("\n" + "=" * 40 + " ERROR " + "=" * 40 + "\n")
                f.write(error_msg)
    except Exception as e:
        if error_msg:
            error_msg += f"; Failed to write log: {e}"
        else:
            error_msg = f"Failed to write log: {e}"
    
    # Determine success
    success = exit_code == 0 and not timed_out
    
    # Get last lines for quick inspection
    combined_output = stdout_content + "\n" + stderr_content
    last_lines = get_last_lines(combined_output)
    
    return RunResult(
        success=success,
        exit_code=exit_code,
        elapsed_seconds=elapsed_seconds,
        stdout=stdout_content,
        stderr=stderr_content,
        log_path=log_path,
        timed_out=timed_out,
        error=error_msg,
        last_lines=last_lines,
    )


def print_run_summary(result: RunResult, verbose: bool = True) -> None:
    """
    Print a summary of the run result.
    
    Parameters
    ----------
    result : RunResult
        The run result to summarize
    verbose : bool
        Whether to print detailed output
    """
    print()
    print("-" * 40)
    print("Script Execution Summary")
    print("-" * 40)
    
    status = "SUCCESS" if result.success else "FAILED"
    if result.timed_out:
        status = "TIMEOUT"
    
    print(f"Status: {status}")
    print(f"Exit code: {result.exit_code}")
    print(f"Elapsed: {result.elapsed_seconds:.2f}s")
    print(f"Log file: {result.log_path}")
    
    if result.error:
        print(f"Error: {result.error}")
    
    if verbose and result.last_lines:
        print()
        print("Last lines of output:")
        for line in result.last_lines[-10:]:
            print(f"  {line}")
    
    print("-" * 40)
