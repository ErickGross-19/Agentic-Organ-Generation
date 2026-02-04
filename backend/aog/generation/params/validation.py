"""Parameter validation with bounds checking.

This module provides validation for SpaceColonizationParams to ensure
parameters are within reasonable ranges and catch common mistakes.

Units: All spatial parameters are in millimeters.
"""

from typing import List, Tuple
from ..ops.space_colonization import SpaceColonizationParams


PARAM_BOUNDS = {
    "influence_radius": (1.0, 100.0, "mm"),  # 1mm to 100mm
    "kill_radius": (0.1, 50.0, "mm"),  # 0.1mm to 50mm
    "step_size": (0.1, 50.0, "mm"),  # 0.1mm to 50mm
    "min_radius": (0.01, 10.0, "mm"),  # 0.01mm to 10mm
    "taper_factor": (0.5, 1.0, "ratio"),  # 50% to 100%
    "max_steps": (1, 1000, "steps"),
    "directional_bias": (0.0, 1.0, "ratio"),
    "max_deviation_deg": (0.0, 180.0, "degrees"),
    "smoothing_weight": (0.0, 1.0, "ratio"),
    "min_attractions_for_bifurcation": (1, 20, "count"),
    "max_children_per_node": (1, 5, "count"),
    "bifurcation_angle_threshold_deg": (0.0, 180.0, "degrees"),
    "bifurcation_probability": (0.0, 1.0, "probability"),
    "max_curvature_deg": (0.0, 180.0, "degrees"),
    "min_clearance": (0.0, 100.0, "mm"),  # 0 to 100mm
}


def validate_params(params: SpaceColonizationParams) -> Tuple[bool, List[str]]:
    """
    Validate SpaceColonizationParams against bounds.
    
    Parameters
    ----------
    params : SpaceColonizationParams
        Parameters to validate
        
    Returns
    -------
    is_valid : bool
        True if all parameters are valid
    warnings : list of str
        List of validation warnings/errors
    """
    warnings = []
    
    for param_name, (min_val, max_val, unit) in PARAM_BOUNDS.items():
        value = getattr(params, param_name, None)
        
        if value is None:
            continue  # Optional parameter
        
        if value < min_val:
            warnings.append(
                f"{param_name} = {value} {unit} is below minimum {min_val} {unit}"
            )
        elif value > max_val:
            warnings.append(
                f"{param_name} = {value} {unit} exceeds maximum {max_val} {unit}"
            )
    
    if params.kill_radius >= params.influence_radius:
        warnings.append(
            f"kill_radius ({params.kill_radius}mm) should be < influence_radius ({params.influence_radius}mm)"
        )
    
    if params.step_size > params.influence_radius:
        warnings.append(
            f"step_size ({params.step_size}mm) is larger than influence_radius ({params.influence_radius}mm), "
            "which may cause poor coverage"
        )
    
    if params.min_radius > params.step_size:
        warnings.append(
            f"min_radius ({params.min_radius}mm) is larger than step_size ({params.step_size}mm), "
            "which may prevent growth"
        )
    
    if params.taper_factor < 0.7:
        warnings.append(
            f"taper_factor ({params.taper_factor}) is very aggressive (< 0.7), "
            "network may reach min_radius quickly"
        )
    
    if params.encourage_bifurcation:
        if params.min_attractions_for_bifurcation < 2:
            warnings.append(
                f"min_attractions_for_bifurcation ({params.min_attractions_for_bifurcation}) "
                "should be >= 2 for meaningful bifurcation"
            )
        
        if params.max_children_per_node < 2:
            warnings.append(
                f"max_children_per_node ({params.max_children_per_node}) should be >= 2 "
                "when encourage_bifurcation is True"
            )
    
    if params.directional_bias > 0 and params.preferred_direction is None:
        warnings.append(
            f"directional_bias = {params.directional_bias} but preferred_direction is None"
        )
    
    if params.max_curvature_deg is not None and params.max_curvature_deg < 20.0:
        warnings.append(
            f"max_curvature_deg = {params.max_curvature_deg}° is very restrictive (< 20°), "
            "may prevent growth"
        )
    
    if params.min_clearance is not None and params.min_clearance > params.influence_radius:
        warnings.append(
            f"min_clearance ({params.min_clearance}mm) is larger than influence_radius "
            f"({params.influence_radius}mm), which may prevent all growth"
        )
    
    is_valid = len(warnings) == 0
    return is_valid, warnings


def validate_and_warn(params: SpaceColonizationParams) -> SpaceColonizationParams:
    """
    Validate parameters and print warnings.
    
    Parameters
    ----------
    params : SpaceColonizationParams
        Parameters to validate
        
    Returns
    -------
    params : SpaceColonizationParams
        Same parameters (for chaining)
    """
    is_valid, warnings = validate_params(params)
    
    if not is_valid:
        print(f"⚠️  Parameter validation warnings ({len(warnings)}):")
        for warning in warnings:
            print(f"  - {warning}")
    
    return params
