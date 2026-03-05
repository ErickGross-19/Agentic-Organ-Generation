"""
Script Contract Prompt Templates

Templates for generating LLM prompts that enforce the script contract requirements
for the execution mode workflow.
"""

from typing import Dict, Any, Optional, List


SCRIPT_CONTRACT_REQUIREMENTS = """
IMPORTANT: The generated script MUST follow these requirements:

1. OUTPUT DIRECTORY:
   - Read the output directory from environment variable ORGAN_AGENT_OUTPUT_DIR
   - Example: OUTPUT_DIR = os.environ.get("ORGAN_AGENT_OUTPUT_DIR", ".")
   - All output files must be written relative to OUTPUT_DIR

2. ENTRY POINT:
   - Define a main() function as the entry point
   - The script should be runnable via: python script.py

3. REQUIRED ARTIFACTS:
   - Save the network to: network.json (relative to OUTPUT_DIR)
   - Export STL mesh to: mesh_network.stl (relative to OUTPUT_DIR)

4. ARTIFACTS FOOTER:
   - Print ARTIFACTS_JSON at the end of execution
   - Format: ARTIFACTS_JSON: {"files": [...], "metrics": {...}, "status": "success"}
   - Include all created files in the "files" list
   - Include relevant metrics (node_count, segment_count, etc.)

5. BANNED OPERATIONS:
   - Do NOT use subprocess, os.system, or similar shell execution
   - Do NOT use eval() or exec() on untrusted input
   - Do NOT use pip install or modify the environment
   - Do NOT delete files or write outside OUTPUT_DIR
   - Do NOT use network requests or external APIs

6. ERROR HANDLING:
   - Wrap main logic in try/except
   - On error, print ARTIFACTS_JSON with status="failed" and error message
   - Exit with non-zero code on failure
"""


def script_contract_prompt(
    spec_path: str,
    output_units: str = "mm",
    additional_requirements: Optional[str] = None,
    feedback_context: Optional[str] = None,
    previous_context: Optional[str] = None,
) -> str:
    """
    Generate a prompt for creating a generation script that follows the contract.
    
    Parameters
    ----------
    spec_path : str
        Path to the spec JSON file
    output_units : str
        Units for export (default: mm)
    additional_requirements : str, optional
        Additional requirements to include
    feedback_context : str, optional
        Previous feedback from user
    previous_context : str, optional
        Context from previous attempts
        
    Returns
    -------
    str
        Formatted prompt for the LLM
    """
    prompt = f"""Generate a Python script for vascular network generation based on the spec.

Spec file: {spec_path}

{SCRIPT_CONTRACT_REQUIREMENTS}

GENERATION REQUIREMENTS:
- Use generation.api.design_from_spec() or appropriate generation functions
- Use output_units="{output_units}" for exports
- Follow Murray's law for bifurcation radii
- Ensure mesh is watertight
"""

    if additional_requirements:
        prompt += f"\nADDITIONAL REQUIREMENTS:\n{additional_requirements}\n"

    if feedback_context:
        prompt += f"\nPREVIOUS FEEDBACK FROM USER:\n{feedback_context}\n"

    if previous_context:
        prompt += f"\nCONTEXT FROM PREVIOUS ATTEMPTS (use workable parts):\n{previous_context}\n"

    prompt += """
If generation fails, report the error and suggest parameter adjustments.

Provide complete, runnable Python code in a ```python code block.
"""

    return prompt


def script_contract_header() -> str:
    """
    Return the standard header to prepend to generated scripts.
    
    Returns
    -------
    str
        Header code block
    """
    return '''"""
Auto-generated vascular network generation script.

This script follows the execution mode contract:
- Reads OUTPUT_DIR from ORGAN_AGENT_OUTPUT_DIR environment variable
- Defines main() as entry point
- Prints ARTIFACTS_JSON footer on completion
"""

import os
import sys
import json

OUTPUT_DIR = os.environ.get("ORGAN_AGENT_OUTPUT_DIR", ".")
os.makedirs(OUTPUT_DIR, exist_ok=True)

'''


def script_contract_footer() -> str:
    """
    Return the standard footer to append to generated scripts.
    
    Returns
    -------
    str
        Footer code block
    """
    return '''

def _print_artifacts_json(files: list, metrics: dict, status: str = "success", error: str = None):
    """Print the ARTIFACTS_JSON footer."""
    result = {
        "files": files,
        "metrics": metrics,
        "status": status,
    }
    if error:
        result["error"] = error
    print(f"ARTIFACTS_JSON: {json.dumps(result)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        _print_artifacts_json([], {}, "failed", str(e))
        sys.exit(1)
'''


def get_banned_patterns() -> List[str]:
    """
    Return list of patterns that should be flagged in generated scripts.
    
    Returns
    -------
    List[str]
        List of suspicious patterns
    """
    return [
        "subprocess",
        "os.system",
        "os.popen",
        "eval(",
        "exec(",
        "compile(",
        "__import__",
        "pip install",
        "pip3 install",
        "shutil.rmtree",
        "os.remove",
        "os.unlink",
        "os.rmdir",
        "requests.",
        "urllib.",
        "http.client",
        "socket.",
    ]


def validate_script_contract(code: str) -> Dict[str, Any]:
    """
    Validate that a script follows the contract requirements.
    
    Parameters
    ----------
    code : str
        The script code to validate
        
    Returns
    -------
    Dict[str, Any]
        Validation result with:
        - valid: bool
        - warnings: List[str]
        - errors: List[str]
    """
    warnings = []
    errors = []
    
    if "ORGAN_AGENT_OUTPUT_DIR" not in code:
        warnings.append("Script does not read ORGAN_AGENT_OUTPUT_DIR environment variable")
    
    if "def main(" not in code and "def main():" not in code:
        warnings.append("Script does not define a main() function")
    
    if "ARTIFACTS_JSON" not in code:
        warnings.append("Script does not print ARTIFACTS_JSON footer")
    
    banned = get_banned_patterns()
    for pattern in banned:
        if pattern in code:
            errors.append(f"Script contains banned pattern: {pattern}")
    
    return {
        "valid": len(errors) == 0,
        "warnings": warnings,
        "errors": errors,
    }
