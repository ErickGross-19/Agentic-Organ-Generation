"""
Design Iteration Task Templates

Pre-built prompts for iterative design improvement tasks.
"""

from typing import Dict, Any, Optional, List


def iterate_design_prompt(
    current_design_path: str,
    validation_report_path: Optional[str] = None,
    improvement_goals: Optional[List[str]] = None,
    max_iterations: int = 5,
) -> str:
    """
    Generate a prompt for iterative design improvement.
    
    Parameters
    ----------
    current_design_path : str
        Path to the current design (mesh or spec JSON)
    validation_report_path : str, optional
        Path to validation report from previous iteration
    improvement_goals : list, optional
        Specific goals for improvement
    max_iterations : int
        Maximum iterations to attempt
        
    Returns
    -------
    str
        Formatted prompt for design iteration
    """
    goals = improvement_goals or [
        "Fix any validation failures",
        "Improve manufacturability",
        "Optimize flow distribution",
    ]
    
    prompt = f"""Iteratively improve the organ structure design.

**Current Design:** {current_design_path}
**Validation Report:** {validation_report_path if validation_report_path else "Not provided"}
**Maximum Iterations:** {max_iterations}

**Improvement Goals:**
"""
    
    for i, goal in enumerate(goals, 1):
        prompt += f"{i}. {goal}\n"
    
    prompt += """
**Iteration Process:**
1. Load the current design and validation report (if available)
2. Identify the most critical issues to address
3. Modify the design to address the issues
4. Re-run validation checks
5. Compare metrics before and after
6. Repeat until all goals are met or max iterations reached

**For Each Iteration, Report:**
- What issue was addressed
- What modification was made
- Validation results after modification
- Whether the goal was achieved

**Output:**
- Final improved design (mesh files)
- Comparison report showing improvement metrics
- List of remaining issues (if any)

Use both generation and validity modules as needed.
"""
    
    return prompt


def fix_validation_issues_prompt(
    mesh_path: str,
    validation_report: Dict[str, Any],
) -> str:
    """
    Generate a prompt to fix specific validation issues.
    
    Parameters
    ----------
    mesh_path : str
        Path to the mesh with issues
    validation_report : dict
        Validation report containing failed checks
        
    Returns
    -------
    str
        Formatted prompt for fixing issues
    """
    # Extract failed checks from report
    failed_checks = []
    if "reports" in validation_report:
        for category, report in validation_report["reports"].items():
            if isinstance(report, dict) and "checks" in report:
                for check in report["checks"]:
                    if not check.get("passed", True):
                        failed_checks.append({
                            "category": category,
                            "name": check.get("check_name", "unknown"),
                            "message": check.get("message", ""),
                            "details": check.get("details", {}),
                        })
    
    prompt = f"""Fix the validation issues in the organ structure.

**Mesh Path:** {mesh_path}

**Failed Validation Checks:**
"""
    
    if failed_checks:
        for i, check in enumerate(failed_checks, 1):
            prompt += f"""
{i}. **{check['category']}/{check['name']}**
   - Message: {check['message']}
   - Details: {check['details']}
"""
    else:
        prompt += "No specific failed checks provided. Please run validation first.\n"
    
    prompt += """
**Instructions:**
1. Analyze each failed check to understand the root cause
2. Determine the best approach to fix each issue:
   - Mesh issues: Use mesh repair operations
   - Graph issues: Modify network topology
   - Flow issues: Adjust radii or add/remove segments
   - Printability issues: Scale features or modify geometry
3. Apply fixes in order of priority (most critical first)
4. Re-validate after each fix to confirm resolution
5. Document all changes made

**Common Fixes:**
- Non-watertight mesh: Fill holes, merge close vertices
- Murray's law violation: Adjust child radii at bifurcations
- Collisions: Move segments apart or reduce radii
- Thin walls: Increase wall thickness or reduce channel size
- Unsupported overhangs: Add support structures or reorient

Provide complete Python code for each fix.
"""
    
    return prompt


def optimize_structure_prompt(
    mesh_path: str,
    optimization_target: str = "flow_uniformity",
    constraints: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate a prompt for structure optimization.
    
    Parameters
    ----------
    mesh_path : str
        Path to the mesh to optimize
    optimization_target : str
        What to optimize: "flow_uniformity", "coverage", "printability", "all"
    constraints : dict, optional
        Constraints to maintain during optimization
        
    Returns
    -------
    str
        Formatted prompt for optimization
    """
    constraints = constraints or {}
    
    prompt = f"""Optimize the organ structure for: {optimization_target}

**Input Mesh:** {mesh_path}

**Optimization Target:** {optimization_target}
"""
    
    if optimization_target == "flow_uniformity":
        prompt += """
**Flow Uniformity Optimization:**
- Goal: Ensure uniform flow distribution to all terminal branches
- Metrics: Flow variance, pressure drop uniformity
- Methods: Adjust radii, add/remove branches, modify bifurcation angles
"""
    elif optimization_target == "coverage":
        prompt += """
**Coverage Optimization:**
- Goal: Maximize perfusion coverage of the domain
- Metrics: Void fraction, distance to nearest vessel
- Methods: Add branches to underserved regions, adjust growth parameters
"""
    elif optimization_target == "printability":
        prompt += """
**Printability Optimization:**
- Goal: Maximize manufacturability while maintaining function
- Metrics: Min channel diameter, wall thickness, overhang angles
- Methods: Scale features, add supports, reorient structure
"""
    else:  # "all"
        prompt += """
**Multi-Objective Optimization:**
- Balance flow uniformity, coverage, and printability
- Use Pareto optimization to find best trade-offs
- Report trade-off curves for different objectives
"""
    
    prompt += f"""
**Constraints to Maintain:**
- Minimum channel diameter: {constraints.get('min_channel_diameter', 0.5)} mm
- Minimum wall thickness: {constraints.get('min_wall_thickness', 0.3)} mm
- Maximum segments: {constraints.get('max_segments', 10000)}
- Domain bounds: Must stay within original domain

**Instructions:**
1. Load the current structure
2. Analyze current performance on the optimization target
3. Apply optimization algorithm (gradient-based or evolutionary)
4. Validate that constraints are still satisfied
5. Compare before/after metrics
6. Export optimized structure

Provide complete Python code and report optimization results.
"""
    
    return prompt
