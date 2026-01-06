"""
Validation Task Templates

Pre-built prompts for structure validation tasks.
"""

from typing import Dict, Any, Optional


def validate_structure_prompt(
    mesh_path: str,
    validation_stage: str = "both",
    manufacturing_constraints: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate a prompt for validating an organ structure.
    
    Parameters
    ----------
    mesh_path : str
        Path to the mesh file to validate
    validation_stage : str
        Validation stage: "pre_embedding", "post_embedding", or "both"
    manufacturing_constraints : dict, optional
        Manufacturing constraints for printability checks
        
    Returns
    -------
    str
        Formatted prompt for validation
    """
    constraints = manufacturing_constraints or {}
    
    prompt = f"""Validate the organ structure at: {mesh_path}

**Validation Stage:** {validation_stage}

**Manufacturing Constraints:**
- Minimum channel diameter: {constraints.get('min_channel_diameter', 0.5)} mm
- Minimum wall thickness: {constraints.get('min_wall_thickness', 0.3)} mm
- Maximum overhang angle: {constraints.get('max_overhang_angle', 45)} degrees
- Build plate size: {constraints.get('plate_size', (200, 200, 200))} mm

**Instructions:**
1. Load the mesh from the specified path
2. Run the appropriate validation checks based on the stage
3. Generate a comprehensive validation report
4. Identify any issues or warnings
5. Suggest fixes for any failed checks
6. Save the validation report to JSON

Please provide complete Python code and interpret the results.
"""
    
    return prompt


def validate_pre_embedding_prompt(
    mesh_path: str,
    network_path: Optional[str] = None,
) -> str:
    """
    Generate a prompt for pre-embedding validation.
    
    Parameters
    ----------
    mesh_path : str
        Path to the mesh file
    network_path : str, optional
        Path to network JSON file for graph/flow checks
        
    Returns
    -------
    str
        Formatted prompt for pre-embedding validation
    """
    prompt = f"""Run pre-embedding validation on the structure.

**Input Files:**
- Mesh: {mesh_path}
- Network: {network_path if network_path else "Not provided (skip graph/flow checks)"}

**Pre-Embedding Checks to Run:**

1. **Mesh Checks:**
   - Watertightness: Is the mesh closed with no holes?
   - Manifoldness: Are all edges shared by exactly 2 faces?
   - Surface quality: Are face aspect ratios acceptable?
   - Degenerate faces: Are there any zero-area faces?

2. **Graph Checks** (if network provided):
   - Murray's law: Do bifurcation radii follow r_parent^3 = sum(r_child^3)?
   - Branch order: Is the maximum branch order reasonable?
   - Collisions: Are there any segment collisions?
   - Self-intersections: Are there any zero-length segments?

3. **Flow Checks** (if network has flow solution):
   - Flow plausibility: Is mass conserved at junctions?
   - Reynolds number: Is flow laminar (Re < 2300)?
   - Pressure monotonicity: Does pressure decrease along flow?

**Instructions:**
1. Load the mesh and optionally the network
2. Run all applicable pre-embedding checks
3. Generate a detailed report with pass/fail status for each check
4. List all warnings and their severity
5. Provide recommendations for fixing any issues

Use the validity.pre_embedding module for checks.
"""
    
    return prompt


def validate_post_embedding_prompt(
    embedded_mesh_path: str,
    manufacturing_constraints: Optional[Dict[str, Any]] = None,
    expected_outlets: int = 2,
) -> str:
    """
    Generate a prompt for post-embedding validation.
    
    Parameters
    ----------
    embedded_mesh_path : str
        Path to the embedded mesh (domain with void)
    manufacturing_constraints : dict, optional
        Manufacturing constraints
    expected_outlets : int
        Expected number of outlet openings
        
    Returns
    -------
    str
        Formatted prompt for post-embedding validation
    """
    constraints = manufacturing_constraints or {}
    
    prompt = f"""Run post-embedding validation on the embedded structure.

**Input File:**
- Embedded mesh (domain with void): {embedded_mesh_path}

**Manufacturing Constraints:**
- Minimum channel diameter: {constraints.get('min_channel_diameter', 0.5)} mm
- Minimum wall thickness: {constraints.get('min_wall_thickness', 0.3)} mm
- Maximum overhang angle: {constraints.get('max_overhang_angle', 45)} degrees
- Expected outlets: {expected_outlets}

**Post-Embedding Checks to Run:**

1. **Connectivity Checks:**
   - Port accessibility: Are fluid ports accessible from the exterior?
   - Trapped fluid: Is there any fluid that cannot be reached?
   - Channel continuity: Are all channels connected?

2. **Printability Checks:**
   - Minimum channel diameter: Are all channels wide enough to print?
   - Wall thickness: Are all walls thick enough to print?
   - Unsupported features: Are there excessive overhangs?

3. **Domain Checks:**
   - Outlets open: Are the expected number of outlets present?
   - Domain coverage: Is the void fraction appropriate?

**Instructions:**
1. Load the embedded mesh
2. Run all post-embedding checks with the specified constraints
3. Generate a detailed report
4. Identify any manufacturability issues
5. Suggest modifications to fix any problems

Use the validity.post_embedding module for checks.
"""
    
    return prompt
