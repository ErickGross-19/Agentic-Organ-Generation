"""
Generation Task Templates

Pre-built prompts for organ structure generation tasks.
"""

from typing import Dict, Any, Optional, Tuple


def generate_structure_prompt(
    organ_type: str,
    constraints: Optional[Dict[str, Any]] = None,
    output_format: str = "domain_with_void",
) -> str:
    """
    Generate a prompt for creating an organ structure.
    
    Parameters
    ----------
    organ_type : str
        Type of organ (e.g., "liver", "kidney", "custom")
    constraints : dict, optional
        Manufacturing and design constraints:
        - plate_size: (width, height, depth) in mm
        - min_channel_diameter: minimum channel diameter in mm
        - min_wall_thickness: minimum wall thickness in mm
        - num_segments: target number of vascular segments
        - seed: random seed for reproducibility
    output_format : str
        Output format: "domain_with_void", "surface_mesh", or "both"
        
    Returns
    -------
    str
        Formatted prompt for the LLM
    """
    constraints = constraints or {}
    
    prompt = f"""Generate a vascular network structure for a {organ_type} organ.

**Requirements:**
- Organ type: {organ_type}
- Output format: {output_format} (primary: domain-with-void scaffold, supplementary: surface mesh)

**Manufacturing Constraints:**
"""
    
    if "plate_size" in constraints:
        prompt += f"- Build plate size: {constraints['plate_size']} mm\n"
    else:
        prompt += "- Build plate size: (200, 200, 200) mm (default)\n"
    
    if "min_channel_diameter" in constraints:
        prompt += f"- Minimum channel diameter: {constraints['min_channel_diameter']} mm\n"
    else:
        prompt += "- Minimum channel diameter: 0.5 mm (default)\n"
    
    if "min_wall_thickness" in constraints:
        prompt += f"- Minimum wall thickness: {constraints['min_wall_thickness']} mm\n"
    else:
        prompt += "- Minimum wall thickness: 0.3 mm (default)\n"
    
    if "num_segments" in constraints:
        prompt += f"- Target segments: {constraints['num_segments']}\n"
    
    if "seed" in constraints:
        prompt += f"- Random seed: {constraints['seed']}\n"
    
    prompt += """
**Instructions:**
1. Create a design specification for the vascular network
2. Generate the network using appropriate generation method
3. Run pre-embedding validation checks
4. Embed the structure into a domain
5. Run post-embedding validation checks
6. Export both domain-with-void scaffold and surface mesh STL files
7. Report any validation warnings or issues

Please provide complete, runnable Python code using the generation and validity modules.
"""
    
    return prompt


def generate_liver_prompt(
    arterial_segments: int = 500,
    venous_segments: int = 500,
    plate_size: Tuple[float, float, float] = (200.0, 200.0, 200.0),
    seed: Optional[int] = None,
) -> str:
    """
    Generate a prompt specifically for liver vascular network generation.
    
    Parameters
    ----------
    arterial_segments : int
        Number of arterial tree segments
    venous_segments : int
        Number of venous tree segments
    plate_size : tuple
        Build plate dimensions (width, height, depth) in mm
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    str
        Formatted prompt for liver generation
    """
    prompt = f"""Generate a liver vascular network with dual arterial and venous trees.

**Liver-Specific Requirements:**
- Arterial tree segments: {arterial_segments}
- Venous tree segments: {venous_segments}
- Build plate size: {plate_size} mm
- Random seed: {seed if seed is not None else "auto"}

**Anatomical Considerations:**
- Liver has a dual blood supply (hepatic artery + portal vein)
- Arterial and venous trees should meet at a "meeting shell" (capillary bed)
- Use Murray's law (gamma=3.0) for bifurcation radii
- Typical inlet radius: 2-5 mm for main vessels

**Instructions:**
1. Use the liver generator: `from generation.organ_generators.liver import generate_liver_vasculature`
2. Configure with LiverVascularConfig for anatomically realistic parameters
3. Generate both arterial and venous trees
4. Create anastomoses at the meeting shell
5. Validate the combined network
6. Embed into an ellipsoid domain matching liver shape
7. Export domain-with-void scaffold and surface mesh

Please provide complete Python code with all imports and configuration.
"""
    
    return prompt


def generate_custom_organ_prompt(
    organ_name: str,
    domain_shape: str = "ellipsoid",
    domain_dimensions: Tuple[float, float, float] = (100.0, 80.0, 60.0),
    inlet_positions: Optional[list] = None,
    outlet_positions: Optional[list] = None,
    constraints: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate a prompt for custom organ structure generation.
    
    Parameters
    ----------
    organ_name : str
        Name of the custom organ
    domain_shape : str
        Domain shape: "ellipsoid" or "box"
    domain_dimensions : tuple
        Domain dimensions (a, b, c) for ellipsoid or (w, h, d) for box in mm
    inlet_positions : list, optional
        List of inlet positions [(x, y, z), ...]
    outlet_positions : list, optional
        List of outlet positions [(x, y, z), ...]
    constraints : dict, optional
        Additional constraints
        
    Returns
    -------
    str
        Formatted prompt for custom organ generation
    """
    constraints = constraints or {}
    
    prompt = f"""Generate a custom vascular network for: {organ_name}

**Domain Configuration:**
- Shape: {domain_shape}
- Dimensions: {domain_dimensions} mm
"""
    
    if inlet_positions:
        prompt += f"- Inlet positions: {inlet_positions}\n"
    else:
        prompt += "- Inlet positions: auto (based on domain shape)\n"
    
    if outlet_positions:
        prompt += f"- Outlet positions: {outlet_positions}\n"
    else:
        prompt += "- Outlet positions: auto (based on domain shape)\n"
    
    prompt += f"""
**Constraints:**
- Minimum channel diameter: {constraints.get('min_channel_diameter', 0.5)} mm
- Minimum wall thickness: {constraints.get('min_wall_thickness', 0.3)} mm
- Target coverage: {constraints.get('target_coverage', 0.8)} (fraction of domain)

**Instructions:**
1. Create a DesignSpec with the specified domain and inlet/outlet configuration
2. Use space colonization algorithm for organic growth
3. Apply appropriate growth parameters for the organ type
4. Validate the network structure
5. Embed into the domain
6. Run all validation checks
7. Export both domain-with-void scaffold and surface mesh

Provide complete Python code using the generation library's high-level API.
"""
    
    return prompt
