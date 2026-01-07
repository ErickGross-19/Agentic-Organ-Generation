"""
High-level API for building vascular networks from specifications.

UNIT CONVENTIONS
----------------
This module uses the compile_domain() function to convert user-facing spec classes
into runtime domain objects. See generation/specs/compile.py for details on:
- Spec units (meters)
- Runtime units (meters)
- Output units (configurable, default mm)
- Coordinate frame conventions
"""

from typing import Optional
import logging
import numpy as np

from ..specs.design_spec import DesignSpec
from ..specs.compile import compile_domain
from ..core.network import VascularNetwork
from ..core.types import Point3D
from ..core.result import OperationStatus
from ..ops import create_network, add_inlet, add_outlet, space_colonization_step
from ..ops.space_colonization import SpaceColonizationParams

logger = logging.getLogger(__name__)


class DesignGenerationError(Exception):
    """Raised when design generation fails to produce a valid network."""
    pass


def design_from_spec(
    spec: DesignSpec,
    max_retries: int = 3,
    fail_on_empty: bool = True,
) -> VascularNetwork:
    """
    Build a vascular network from a design specification.
    
    This is the main entry point for LLM-driven vascular network design.
    
    The domain specification is compiled into a runtime domain object using
    compile_domain(), which centralizes unit handling, defaults, and transforms.
    
    If space colonization fails to produce any growth, the function will retry
    with adaptive parameters (increased influence radius, step size) up to
    max_retries times. If all retries fail and fail_on_empty=True, raises
    DesignGenerationError.
    
    Parameters
    ----------
    spec : DesignSpec
        Design specification containing domain, tree/dual_tree config, and metadata.
        All geometric values (positions, radii, sizes) are in METERS internally.
    max_retries : int
        Maximum number of retry attempts with adaptive parameters (default: 3)
    fail_on_empty : bool
        If True (default), raise DesignGenerationError when no growth occurs.
        If False, return the network with only inlet/outlet nodes.
        
    Returns
    -------
    VascularNetwork
        Generated vascular network with nodes and segments.
        Geometry is in METERS internally; use UnitContext for export conversion.
        
    Raises
    ------
    DesignGenerationError
        If fail_on_empty=True and no growth occurs after all retries.
    """
    domain = compile_domain(spec.domain)
    
    # Create network
    network = create_network(domain=domain, seed=spec.seed)
    
    # Handle single tree or dual tree
    if spec.tree is not None:
        tree = spec.tree
        
        # Compute inlet/outlet directions relative to domain center
        domain_center = np.array([domain.center.x, domain.center.y, domain.center.z])
        
        for inlet_spec in tree.inlets:
            inlet_pos = np.array(inlet_spec.position)
            # Direction points inward from inlet toward domain center
            direction_vec = domain_center - inlet_pos
            direction_norm = np.linalg.norm(direction_vec)
            if direction_norm > 1e-9:
                direction = tuple(direction_vec / direction_norm)
            else:
                direction = (0.0, 0.0, -1.0)  # Default downward
            add_inlet(network, position=Point3D(*inlet_spec.position), 
                     radius=inlet_spec.radius, vessel_type=inlet_spec.vessel_type, direction=direction)
        for outlet_spec in tree.outlets:
            outlet_pos = np.array(outlet_spec.position)
            # Direction points outward from domain center toward outlet
            direction_vec = outlet_pos - domain_center
            direction_norm = np.linalg.norm(direction_vec)
            if direction_norm > 1e-9:
                direction = tuple(direction_vec / direction_norm)
            else:
                direction = (0.0, 0.0, 1.0)  # Default upward
            add_outlet(network, position=Point3D(*outlet_spec.position),
                      radius=outlet_spec.radius, vessel_type=outlet_spec.vessel_type, direction=direction)
        
        tissue_points = domain.sample_points(n_points=1000, seed=spec.seed)
        col_spec = tree.colonization
        
        # Build base params from spec
        base_params = {
            "influence_radius": col_spec.influence_radius,
            "kill_radius": col_spec.kill_radius,
            "step_size": col_spec.step_size,
            "max_steps": col_spec.max_steps,
            "min_radius": col_spec.min_radius,
            "taper_factor": col_spec.radius_decay if hasattr(col_spec, 'radius_decay') else 0.95,
            "vessel_type": tree.inlets[0].vessel_type if tree.inlets else "arterial",
            "preferred_direction": col_spec.preferred_direction,
            "directional_bias": col_spec.directional_bias,
            "max_deviation_deg": col_spec.max_deviation_deg,
            "smoothing_weight": col_spec.smoothing_weight,
            "encourage_bifurcation": col_spec.encourage_bifurcation,
            "min_attractions_for_bifurcation": col_spec.min_attractions_for_bifurcation,
            "max_children_per_node": col_spec.max_children_per_node,
            "bifurcation_angle_threshold_deg": col_spec.bifurcation_angle_threshold_deg,
            "bifurcation_probability": col_spec.bifurcation_probability,
        }
        
        # Try colonization with adaptive retries
        result = _run_colonization_with_retry(
            network, tissue_points, base_params, spec.seed, max_retries
        )
        
        if result.status == OperationStatus.WARNING and fail_on_empty:
            raise DesignGenerationError(
                f"Space colonization failed to produce any growth after {max_retries} retries. "
                f"Result: {result.message}. "
                f"Suggestions: increase influence_radius, reduce kill_radius, or check inlet placement."
            )
    
    elif spec.dual_tree is not None:
        dual = spec.dual_tree
        
        # Compute inlet/outlet directions relative to domain center
        domain_center = np.array([domain.center.x, domain.center.y, domain.center.z])
        
        for inlet_spec in dual.arterial_inlets:
            inlet_pos = np.array(inlet_spec.position)
            # Direction points inward from inlet toward domain center
            direction_vec = domain_center - inlet_pos
            direction_norm = np.linalg.norm(direction_vec)
            if direction_norm > 1e-9:
                direction = tuple(direction_vec / direction_norm)
            else:
                direction = (0.0, 0.0, -1.0)  # Default downward
            add_inlet(network, position=Point3D(*inlet_spec.position),
                     radius=inlet_spec.radius, vessel_type="arterial", direction=direction)
        for outlet_spec in dual.venous_outlets:
            outlet_pos = np.array(outlet_spec.position)
            # Direction points outward from domain center toward outlet
            direction_vec = outlet_pos - domain_center
            direction_norm = np.linalg.norm(direction_vec)
            if direction_norm > 1e-9:
                direction = tuple(direction_vec / direction_norm)
            else:
                direction = (0.0, 0.0, 1.0)  # Default upward
            add_outlet(network, position=Point3D(*outlet_spec.position),
                      radius=outlet_spec.radius, vessel_type="venous", direction=direction)
        
        tissue_points = domain.sample_points(n_points=1000, seed=spec.seed)
        
        # Arterial tree
        art_spec = dual.arterial_colonization
        art_params = {
            "influence_radius": art_spec.influence_radius,
            "kill_radius": art_spec.kill_radius,
            "step_size": art_spec.step_size,
            "max_steps": art_spec.max_steps,
            "min_radius": art_spec.min_radius,
            "taper_factor": art_spec.radius_decay,
            "vessel_type": "arterial",
            "preferred_direction": art_spec.preferred_direction,
            "directional_bias": art_spec.directional_bias,
            "max_deviation_deg": art_spec.max_deviation_deg,
            "smoothing_weight": art_spec.smoothing_weight,
            "encourage_bifurcation": art_spec.encourage_bifurcation,
            "min_attractions_for_bifurcation": art_spec.min_attractions_for_bifurcation,
            "max_children_per_node": art_spec.max_children_per_node,
            "bifurcation_angle_threshold_deg": art_spec.bifurcation_angle_threshold_deg,
            "bifurcation_probability": art_spec.bifurcation_probability,
        }
        art_result = _run_colonization_with_retry(
            network, tissue_points, art_params, spec.seed, max_retries
        )
        
        if art_result.status == OperationStatus.WARNING and fail_on_empty:
            raise DesignGenerationError(
                f"Arterial tree colonization failed to produce any growth after {max_retries} retries. "
                f"Result: {art_result.message}. "
                f"Suggestions: increase influence_radius, reduce kill_radius, or check inlet placement."
            )
        
        # Venous tree
        ven_spec = dual.venous_colonization
        ven_params = {
            "influence_radius": ven_spec.influence_radius,
            "kill_radius": ven_spec.kill_radius,
            "step_size": ven_spec.step_size,
            "max_steps": ven_spec.max_steps,
            "min_radius": ven_spec.min_radius,
            "taper_factor": ven_spec.radius_decay,
            "vessel_type": "venous",
            "preferred_direction": ven_spec.preferred_direction,
            "directional_bias": ven_spec.directional_bias,
            "max_deviation_deg": ven_spec.max_deviation_deg,
            "smoothing_weight": ven_spec.smoothing_weight,
            "encourage_bifurcation": ven_spec.encourage_bifurcation,
            "min_attractions_for_bifurcation": ven_spec.min_attractions_for_bifurcation,
            "max_children_per_node": ven_spec.max_children_per_node,
            "bifurcation_angle_threshold_deg": ven_spec.bifurcation_angle_threshold_deg,
            "bifurcation_probability": ven_spec.bifurcation_probability,
        }
        ven_result = _run_colonization_with_retry(
            network, tissue_points, ven_params, spec.seed, max_retries
        )
        
        if ven_result.status == OperationStatus.WARNING and fail_on_empty:
            raise DesignGenerationError(
                f"Venous tree colonization failed to produce any growth after {max_retries} retries. "
                f"Result: {ven_result.message}. "
                f"Suggestions: increase influence_radius, reduce kill_radius, or check outlet placement."
            )
    
    return network


def _run_colonization_with_retry(
    network: VascularNetwork,
    tissue_points: np.ndarray,
    base_params: dict,
    seed: Optional[int],
    max_retries: int,
) -> "OperationResult":
    """
    Run space colonization with adaptive retries on failure.
    
    If colonization produces no growth, retries with progressively larger
    influence_radius and step_size to help seeds reach tissue points.
    
    Parameters
    ----------
    network : VascularNetwork
        Network to grow
    tissue_points : np.ndarray
        Array of tissue points (N, 3)
    base_params : dict
        Base parameters for SpaceColonizationParams
    seed : int, optional
        Random seed
    max_retries : int
        Maximum retry attempts
        
    Returns
    -------
    OperationResult
        Result from the last colonization attempt
    """
    params_dict = base_params.copy()
    
    for attempt in range(max_retries + 1):
        if attempt > 0:
            # Adaptive parameter adjustment for retries
            # Increase influence_radius by 50% each retry
            params_dict["influence_radius"] = base_params["influence_radius"] * (1.5 ** attempt)
            # Increase step_size by 25% each retry
            params_dict["step_size"] = base_params["step_size"] * (1.25 ** attempt)
            # Decrease kill_radius by 20% each retry (makes it easier to grow)
            params_dict["kill_radius"] = base_params["kill_radius"] * (0.8 ** attempt)
            
            logger.warning(
                f"Colonization retry {attempt}/{max_retries}: "
                f"influence_radius={params_dict['influence_radius']:.6f}, "
                f"step_size={params_dict['step_size']:.6f}, "
                f"kill_radius={params_dict['kill_radius']:.6f}"
            )
        
        params = SpaceColonizationParams(**params_dict)
        result = space_colonization_step(
            network, tissue_points=tissue_points, params=params, seed=seed
        )
        
        # Check if growth occurred
        if result.status != OperationStatus.WARNING:
            # Success or partial success - growth occurred
            if attempt > 0:
                logger.info(f"Colonization succeeded on retry {attempt} with adaptive parameters")
            return result
        
        # No growth - will retry if attempts remain
        if attempt < max_retries:
            logger.warning(f"Colonization attempt {attempt + 1} produced no growth, retrying...")
    
    # All retries exhausted
    logger.error(f"Colonization failed after {max_retries + 1} attempts")
    return result
