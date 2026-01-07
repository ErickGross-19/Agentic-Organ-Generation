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
import numpy as np

from ..specs.design_spec import DesignSpec
from ..specs.compile import compile_domain
from ..core.network import VascularNetwork
from ..core.types import Point3D
from ..ops import create_network, add_inlet, add_outlet, space_colonization_step
from ..ops.space_colonization import SpaceColonizationParams


def design_from_spec(spec: DesignSpec) -> VascularNetwork:
    """
    Build a vascular network from a design specification.
    
    This is the main entry point for LLM-driven vascular network design.
    
    The domain specification is compiled into a runtime domain object using
    compile_domain(), which centralizes unit handling, defaults, and transforms.
    
    Parameters
    ----------
    spec : DesignSpec
        Design specification containing domain, tree/dual_tree config, and metadata.
        All geometric values (positions, radii, sizes) are in METERS internally.
        
    Returns
    -------
    VascularNetwork
        Generated vascular network with nodes and segments.
        Geometry is in METERS internally; use UnitContext for export conversion.
    """
    domain = compile_domain(spec.domain)
    
    # Create network
    network = create_network(domain=domain, seed=spec.seed)
    
    # Handle single tree or dual tree
    if spec.tree is not None:
        tree = spec.tree
        for inlet_spec in tree.inlets:
            direction = tuple(-np.array(inlet_spec.position) / (np.linalg.norm(inlet_spec.position) + 1e-9))
            add_inlet(network, position=Point3D(*inlet_spec.position), 
                     radius=inlet_spec.radius, vessel_type=inlet_spec.vessel_type, direction=direction)
        for outlet_spec in tree.outlets:
            direction = tuple(np.array(outlet_spec.position) / (np.linalg.norm(outlet_spec.position) + 1e-9))
            add_outlet(network, position=Point3D(*outlet_spec.position),
                      radius=outlet_spec.radius, vessel_type=outlet_spec.vessel_type, direction=direction)
        
        tissue_points = domain.sample_points(n_points=1000, seed=spec.seed)
        col_spec = tree.colonization
        params = SpaceColonizationParams(
            influence_radius=col_spec.influence_radius,
            kill_radius=col_spec.kill_radius,
            step_size=col_spec.step_size,
            max_steps=col_spec.max_steps,
            min_radius=col_spec.min_radius,
            taper_factor=col_spec.radius_decay if hasattr(col_spec, 'radius_decay') else 0.95,
            vessel_type=tree.inlets[0].vessel_type if tree.inlets else "arterial",
            preferred_direction=col_spec.preferred_direction,
            directional_bias=col_spec.directional_bias,
            max_deviation_deg=col_spec.max_deviation_deg,
            smoothing_weight=col_spec.smoothing_weight,
            encourage_bifurcation=col_spec.encourage_bifurcation,
            min_attractions_for_bifurcation=col_spec.min_attractions_for_bifurcation,
            max_children_per_node=col_spec.max_children_per_node,
            bifurcation_angle_threshold_deg=col_spec.bifurcation_angle_threshold_deg,
            bifurcation_probability=col_spec.bifurcation_probability,
        )
        space_colonization_step(network, tissue_points=tissue_points, params=params, seed=spec.seed)
    
    elif spec.dual_tree is not None:
        dual = spec.dual_tree
        for inlet_spec in dual.arterial_inlets:
            direction = tuple(-np.array(inlet_spec.position) / (np.linalg.norm(inlet_spec.position) + 1e-9))
            add_inlet(network, position=Point3D(*inlet_spec.position),
                     radius=inlet_spec.radius, vessel_type="arterial", direction=direction)
        for outlet_spec in dual.venous_outlets:
            direction = tuple(np.array(outlet_spec.position) / (np.linalg.norm(outlet_spec.position) + 1e-9))
            add_outlet(network, position=Point3D(*outlet_spec.position),
                      radius=outlet_spec.radius, vessel_type="venous", direction=direction)
        
        tissue_points = domain.sample_points(n_points=1000, seed=spec.seed)
        
        # Arterial tree
        art_spec = dual.arterial_colonization
        art_params = SpaceColonizationParams(
            influence_radius=art_spec.influence_radius, kill_radius=art_spec.kill_radius,
            step_size=art_spec.step_size, max_steps=art_spec.max_steps,
            min_radius=art_spec.min_radius, taper_factor=art_spec.radius_decay,
            vessel_type="arterial", preferred_direction=art_spec.preferred_direction,
            directional_bias=art_spec.directional_bias, max_deviation_deg=art_spec.max_deviation_deg,
            smoothing_weight=art_spec.smoothing_weight, encourage_bifurcation=art_spec.encourage_bifurcation,
            min_attractions_for_bifurcation=art_spec.min_attractions_for_bifurcation,
            max_children_per_node=art_spec.max_children_per_node,
            bifurcation_angle_threshold_deg=art_spec.bifurcation_angle_threshold_deg,
            bifurcation_probability=art_spec.bifurcation_probability,
        )
        space_colonization_step(network, tissue_points=tissue_points, params=art_params, seed=spec.seed)
        
        # Venous tree
        ven_spec = dual.venous_colonization
        ven_params = SpaceColonizationParams(
            influence_radius=ven_spec.influence_radius, kill_radius=ven_spec.kill_radius,
            step_size=ven_spec.step_size, max_steps=ven_spec.max_steps,
            min_radius=ven_spec.min_radius, taper_factor=ven_spec.radius_decay,
            vessel_type="venous", preferred_direction=ven_spec.preferred_direction,
            directional_bias=ven_spec.directional_bias, max_deviation_deg=ven_spec.max_deviation_deg,
            smoothing_weight=ven_spec.smoothing_weight, encourage_bifurcation=ven_spec.encourage_bifurcation,
            min_attractions_for_bifurcation=ven_spec.min_attractions_for_bifurcation,
            max_children_per_node=ven_spec.max_children_per_node,
            bifurcation_angle_threshold_deg=ven_spec.bifurcation_angle_threshold_deg,
            bifurcation_probability=ven_spec.bifurcation_probability,
        )
        space_colonization_step(network, tissue_points=tissue_points, params=ven_params, seed=spec.seed)
    
    return network
