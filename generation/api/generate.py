"""
Unified generation API for vascular networks and void meshes.

This module provides the main entry points for generating vascular networks
using different backends, all parameterized with policy objects.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from typing import Optional, Tuple, Dict, Any, Union, TYPE_CHECKING
import logging
import hashlib
import json
import warnings
import numpy as np

# Import policies from aog_policies (canonical source for runner contract)
from aog_policies import (
    GrowthPolicy,
    CollisionPolicy,
    TissueSamplingPolicy,
    OperationReport,
)
from aog_policies.collision import UnifiedCollisionPolicy
from ..specs.design_spec import DesignSpec, DomainSpec
from ..specs.compile import compile_domain
from ..core.network import VascularNetwork
from ..core.types import Point3D
from ..core.domain import DomainSpec as RuntimeDomainSpec, domain_from_dict

if TYPE_CHECKING:
    import trimesh

logger = logging.getLogger(__name__)


def _postpass_dict_to_unified_policy(postpass_dict: Dict[str, Any]) -> UnifiedCollisionPolicy:
    """
    Convert collision_postpass dict from backend_params to UnifiedCollisionPolicy.
    
    This adapter maps the JSON keys (with _m suffix for meters) to the
    UnifiedCollisionPolicy field names.
    
    Parameters
    ----------
    postpass_dict : dict
        Dictionary from backend_params["collision_postpass"] with keys like:
        - enabled: bool
        - min_clearance_m: float (meters)
        - min_radius_m: float (meters)
        - strategy_order: list of str
        - shrink_factor: float
        - shrink_max_iterations: int
        
    Returns
    -------
    UnifiedCollisionPolicy
        Policy object for use with detect_collisions/resolve_collisions
    """
    return UnifiedCollisionPolicy(
        enabled=postpass_dict.get("enabled", False),
        min_clearance=postpass_dict.get("min_clearance_m", postpass_dict.get("min_clearance", 0.00002)),
        min_radius=postpass_dict.get("min_radius_m", postpass_dict.get("min_radius", 0.00005)),
        strategy_order=postpass_dict.get("strategy_order", ["shrink", "terminate"]),
        shrink_factor=postpass_dict.get("shrink_factor", 0.9),
        shrink_max_iterations=postpass_dict.get("shrink_max_iterations", 6),
        # Enable segment-segment checks for postpass
        check_segment_segment=True,
        check_segment_mesh=False,
        check_segment_boundary=True,
        check_node_boundary=True,
    )


def _validate_and_fix_inlet_direction(
    inlet: Dict[str, Any],
    domain: "RuntimeDomainSpec",
) -> Dict[str, Any]:
    """
    Validate inlet direction and auto-flip if pointing outward from top face.
    
    This guard prevents a common misconfiguration where inlet directions point
    outward (+Z) instead of inward (-Z) for top-face inlets, which causes
    immediate growth termination.
    
    Parameters
    ----------
    inlet : dict
        Inlet specification with position and direction
    domain : RuntimeDomainSpec
        Domain object to determine face positions
        
    Returns
    -------
    dict
        Inlet spec with potentially corrected direction
    """
    position = inlet.get("position")
    direction = inlet.get("direction") or inlet.get("growth_inward_direction")
    
    if position is None or direction is None:
        return inlet
    
    # Get domain bounds to determine if inlet is on top face
    bounds = domain.get_bounds()  # (x_min, x_max, y_min, y_max, z_min, z_max)
    z_max = bounds[5]
    z_min = bounds[4]
    domain_height = z_max - z_min
    
    # Check if inlet is on top face (within 10% of domain height from top)
    inlet_z = position[2] if len(position) > 2 else 0
    is_on_top_face = abs(inlet_z - z_max) < 0.1 * domain_height
    
    # Check if direction points outward (+Z)
    direction_z = direction[2] if len(direction) > 2 else 0
    points_outward = direction_z > 0
    
    if is_on_top_face and points_outward:
        # Auto-flip direction to point inward
        corrected_direction = [-direction[0], -direction[1], -direction[2]]
        inlet_name = inlet.get("name", "unnamed")
        logger.warning(
            f"Inlet '{inlet_name}' direction points out of domain (direction[2]={direction_z:.3f} > 0 "
            f"for top-face inlet at z={inlet_z:.6f}). Auto-flipping to inward direction "
            f"{corrected_direction} for robustness."
        )
        
        # Return a copy with corrected direction
        corrected_inlet = inlet.copy()
        if "direction" in corrected_inlet:
            corrected_inlet["direction"] = corrected_direction
        if "growth_inward_direction" in corrected_inlet:
            corrected_inlet["growth_inward_direction"] = corrected_direction
        return corrected_inlet
    
    return inlet


def _coerce_domain(domain: Union[Dict[str, Any], DomainSpec, "RuntimeDomainSpec"]) -> "RuntimeDomainSpec":
    """
    Coerce domain input to a runtime Domain object.
    
    This helper supports three input types:
    1. Dict domain spec (runner/JSON): {"type": "box", "x_min": ..., ...}
    2. Runtime Domain objects (already compiled): BoxDomain, CylinderDomain, etc.
    3. Legacy spec dataclasses: DomainSpec from specs.design_spec
    
    Parameters
    ----------
    domain : dict, DomainSpec, or RuntimeDomainSpec
        Domain specification in any supported format
        
    Returns
    -------
    RuntimeDomainSpec
        Compiled runtime domain object
        
    Raises
    ------
    ValueError
        If domain type is not recognized
    """
    if isinstance(domain, dict):
        return domain_from_dict(domain)
    
    if isinstance(domain, RuntimeDomainSpec):
        return domain
    
    if hasattr(domain, 'type') and hasattr(domain, 'to_dict'):
        return compile_domain(domain)
    
    raise ValueError(
        f"Cannot coerce domain of type {type(domain).__name__}. "
        "Expected dict, RuntimeDomainSpec, or legacy DomainSpec."
    )


def _compute_cache_key(obj: Any) -> str:
    """
    Compute a stable hash key for caching.
    
    Parameters
    ----------
    obj : Any
        Object to hash (must be JSON-serializable or have to_dict method)
        
    Returns
    -------
    str
        Hex digest of the object's hash
    """
    if hasattr(obj, 'to_dict'):
        data = obj.to_dict()
    elif isinstance(obj, dict):
        data = obj
    else:
        data = str(obj)
    
    json_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


class GenerationContext:
    """
    Context for caching expensive computations during generation.
    
    This class provides a cache contract for storing and retrieving
    intermediate results keyed by stable hashes.
    
    Cached items include:
    - compiled_domain: Compiled domain object
    - domain_mesh: Domain mesh (for embedding)
    - face_frames: MeshDomain PCA/OBB results
    - effective_pitches: Derived pitches/tolerances per operation
    - pathfinding_coarse: Coarse pathfinding solution + corridor info
    - void_mesh: Unioned void mesh before embedding
    """
    
    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._hits: int = 0
        self._misses: int = 0
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a cached value by key."""
        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return default
    
    def set(self, key: str, value: Any) -> None:
        """Set a cached value by key."""
        self._cache[key] = value
    
    def has(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        return key in self._cache
    
    def get_or_compute(self, key: str, compute_fn: callable) -> Any:
        """
        Get a cached value or compute and cache it.
        
        Parameters
        ----------
        key : str
            Cache key
        compute_fn : callable
            Function to compute the value if not cached
            
        Returns
        -------
        Any
            Cached or computed value
        """
        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        
        self._misses += 1
        value = compute_fn()
        self._cache[key] = value
        return value
    
    def stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._cache),
        }
    
    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0


def generate_network(
    generator_kind: str,
    domain: Union[DomainSpec, "Domain"],
    ports: Dict[str, Any],
    growth_policy: Optional[GrowthPolicy] = None,
    collision_policy: Optional[CollisionPolicy] = None,
    seed: Optional[int] = None,
) -> Tuple[VascularNetwork, OperationReport]:
    """
    Generate a vascular network within the given domain.
    
    This is the unified entry point for network generation, supporting
    multiple backend algorithms.
    
    Parameters
    ----------
    generator_kind : str
        Generation algorithm to use:
        - "space_colonization": Attractor-based organic growth
        - "kary_tree": Recursive bifurcation to target terminal count
        - "cco_hybrid": CCO with Murray's Law optimization
        - "programmatic": DSL-based programmatic generation with pathfinding
    domain : DomainSpec or Domain
        Domain specification or compiled domain object
    ports : dict
        Port configuration with keys:
        - "inlets": List of inlet specs (position, radius, vessel_type)
        - "outlets": List of outlet specs (position, radius, vessel_type)
    growth_policy : GrowthPolicy, optional
        Policy controlling growth parameters
    collision_policy : CollisionPolicy, optional
        Policy controlling collision detection
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    network : VascularNetwork
        Generated vascular network
    report : OperationReport
        Report with requested/effective policy and metadata
    """
    if growth_policy is None:
        growth_policy = GrowthPolicy()
    if collision_policy is None:
        collision_policy = CollisionPolicy()
    
    # Coerce domain to runtime Domain object (supports dict, runtime, and legacy specs)
    compiled_domain = _coerce_domain(domain)
    
    warnings = []
    metadata = {
        "generator_kind": generator_kind,
        "seed": seed,
    }
    
    # Select backend based on generator_kind
    if generator_kind == "space_colonization":
        network, gen_meta = _generate_space_colonization(
            compiled_domain, ports, growth_policy, collision_policy, seed
        )
        metadata.update(gen_meta)
        
    elif generator_kind == "kary_tree":
        warnings.warn(
            "kary_tree is deprecated and will be removed in a future release; "
            "use scaffold_topdown instead.",
            DeprecationWarning,
            stacklevel=2
        )
        network, gen_meta = _generate_kary_tree(
            compiled_domain, ports, growth_policy, collision_policy, seed
        )
        metadata.update(gen_meta)
        
        # Check terminal tolerance
        if growth_policy.target_terminals is not None:
            achieved = gen_meta.get("terminal_count", 0)
            target = growth_policy.target_terminals
            tolerance = growth_policy.terminal_tolerance
            if abs(achieved - target) / max(target, 1) > tolerance:
                warnings.append(
                    f"Terminal count {achieved} outside tolerance "
                    f"({tolerance*100:.0f}%) of target {target}"
                )
        
    elif generator_kind == "cco_hybrid":
        network, gen_meta = _generate_cco_hybrid(
            compiled_domain, ports, growth_policy, collision_policy, seed
        )
        metadata.update(gen_meta)
        
    elif generator_kind == "programmatic":
        network, gen_meta = _generate_programmatic(
            compiled_domain, ports, growth_policy, collision_policy, seed
        )
        metadata.update(gen_meta)
    
    elif generator_kind == "scaffold_topdown":
        network, gen_meta = _generate_scaffold_topdown(
            compiled_domain, ports, growth_policy, collision_policy, seed
        )
        metadata.update(gen_meta)
        
    else:
        raise ValueError(f"Unknown generator_kind: {generator_kind}")
    
    # Run collision postpass if enabled in backend_params
    # This is backend-agnostic and works for any generator
    postpass_config = growth_policy.backend_params.get("collision_postpass", {})
    if postpass_config.get("enabled", False):
        from ..ops.collision.unified import detect_collisions, resolve_collisions
        
        postpass_policy = _postpass_dict_to_unified_policy(postpass_config)
        
        # Get runtime domain for boundary checks
        runtime_domain = None
        if hasattr(compiled_domain, 'spec') and compiled_domain.spec is not None:
            runtime_domain = domain_from_dict(compiled_domain.spec)
        
        # Detect collisions
        collision_result = detect_collisions(
            network=network,
            domain=runtime_domain,
            policy=postpass_policy,
        )
        
        postpass_stats = {
            "postpass_enabled": True,
            "collisions_detected": collision_result.collision_count,
            "collision_types": {},
        }
        
        # Count collisions by type
        for collision in collision_result.collisions:
            ctype = collision.type.value
            postpass_stats["collision_types"][ctype] = postpass_stats["collision_types"].get(ctype, 0) + 1
        
        # Resolve collisions if any found
        if collision_result.has_collisions:
            resolution_result = resolve_collisions(
                network=network,
                collision_result=collision_result,
                domain=runtime_domain,
                policy=postpass_policy,
            )
            
            collisions_resolved = resolution_result.metadata.get(
                "resolved_count", 
                collision_result.collision_count - resolution_result.remaining_collisions
            )
            postpass_stats["collisions_resolved"] = collisions_resolved
            postpass_stats["collisions_unresolved"] = resolution_result.remaining_collisions
            postpass_stats["resolution_strategies_used"] = list(set(
                attempt.strategy for attempt in resolution_result.attempts if attempt.success
            ))
            
            if resolution_result.remaining_collisions > 0:
                warnings.append(
                    f"Collision postpass: {resolution_result.remaining_collisions} collisions "
                    f"could not be resolved"
                )
        
        metadata["collision_postpass"] = postpass_stats
        logger.info(
            f"Collision postpass: detected {collision_result.collision_count}, "
            f"resolved {postpass_stats.get('collisions_resolved', 0)}"
        )
    
    report = OperationReport(
        operation="generate_network",
        success=True,
        requested_policy=growth_policy.to_dict(),
        effective_policy=growth_policy.to_dict(),
        warnings=warnings,
        metadata=metadata,
    )
    
    return network, report


def _generate_space_colonization(
    domain: "Domain",
    ports: Dict[str, Any],
    growth_policy: GrowthPolicy,
    collision_policy: CollisionPolicy,
    seed: Optional[int],
    tissue_sampling_policy: Optional[TissueSamplingPolicy] = None,
) -> Tuple[VascularNetwork, Dict[str, Any]]:
    """Generate network using space colonization algorithm.
    
    This function supports the new policy-driven space colonization with:
    - Trunk-first growth (prevents inlet starburst)
    - Apical dominance (reduces parallel linear growth)
    - Angular clustering-based splitting (enables proper branching)
    
    The SpaceColonizationPolicy is extracted from growth_policy.backend_params
    if present. Otherwise, defaults are used.
    
    Multi-inlet support:
    - If multiple inlets are defined and multi_inlet_mode is set, uses
      SpaceColonizationBackend.generate_multi_inlet() method
    - Supported modes: "blended" (recommended), "partitioned_xy", "forest"
    - "forest_with_merge" is deprecated and now aliases to "blended"
    """
    from ..ops import create_network, add_inlet, add_outlet
    from ..ops.space_colonization import SpaceColonizationParams, space_colonization_step_v2
    from ..utils.tissue_sampling import sample_tissue_points
    from ..rules.constraints import BranchingConstraints
    from aog_policies.space_colonization import SpaceColonizationPolicy
    
    inlets = ports.get("inlets", [])
    backend_params = getattr(growth_policy, 'backend_params', None) or {}
    multi_inlet_mode = backend_params.get('multi_inlet_mode')
    
    # Deprecation: forest_with_merge is now an alias for blended
    if multi_inlet_mode == "forest_with_merge":
        warnings.warn(
            "multi_inlet_mode='forest_with_merge' is deprecated and will be removed in a future release. "
            "Use 'blended' instead. The 'forest_with_merge' mode now behaves identically to 'blended'.",
            DeprecationWarning,
            stacklevel=2,
        )
        multi_inlet_mode = "blended"
    
    # Validate and fix inlet directions (auto-flip if pointing outward from top face)
    validated_inlets = [_validate_and_fix_inlet_direction(inlet, domain) for inlet in inlets]
    
    # Use SpaceColonizationBackend.generate_multi_inlet() for multiple inlets
    if len(validated_inlets) > 1 and multi_inlet_mode:
        from ..backends.space_colonization_backend import SpaceColonizationBackend, SpaceColonizationConfig
        from aog_policies import TissueSamplingPolicy
        
        # If tissue_sampling_policy is None, check backend_params for tissue_sampling config
        effective_tissue_sampling_policy = tissue_sampling_policy
        if effective_tissue_sampling_policy is None:
            tissue_sampling_config = backend_params.get("tissue_sampling")
            if tissue_sampling_config and isinstance(tissue_sampling_config, dict):
                if tissue_sampling_config.get("enabled", True):
                    effective_tissue_sampling_policy = TissueSamplingPolicy.from_dict(tissue_sampling_config)
        
        config = SpaceColonizationConfig(
            attraction_distance=backend_params.get('influence_radius', 0.010),
            kill_distance=backend_params.get('kill_radius', 0.002),
            step_size=growth_policy.step_size or 0.002,
            num_attractors=backend_params.get('num_attraction_points', 1000),
            max_iterations=growth_policy.max_iterations or 500,
            multi_inlet_mode=multi_inlet_mode,
            collision_merge_distance=backend_params.get('collision_merge_distance', 0.0003),
            max_inlets=backend_params.get('max_inlets', 10),
            multi_inlet_blend_sigma=backend_params.get('multi_inlet_blend_sigma', 0.0),
            directional_bias=backend_params.get('directional_bias', 0.0),
            max_deviation_deg=backend_params.get('max_deviation_deg', 180.0),
            seed=seed,
            # Radius and taper control
            min_radius=backend_params.get('min_radius', growth_policy.min_radius),
            taper_factor=backend_params.get('taper_factor', 0.95),
            # Bifurcation control
            encourage_bifurcation=backend_params.get('encourage_bifurcation', False),
            max_children_per_node=backend_params.get('max_children_per_node', 2),
            bifurcation_probability=backend_params.get('bifurcation_probability', 0.7),
            min_attractions_for_bifurcation=backend_params.get('min_attractions_for_bifurcation', 3),
            bifurcation_angle_threshold_deg=backend_params.get('bifurcation_angle_threshold_deg', 40.0),
            # Step control
            max_steps=backend_params.get('max_steps', 100),
            # Single-step refactor parameters
            progress=backend_params.get('progress', False),
            kdtree_rebuild_tip_every=backend_params.get('kdtree_rebuild_tip_every', 1),
            kdtree_rebuild_all_nodes_every=backend_params.get('kdtree_rebuild_all_nodes_every', 10),
            kdtree_rebuild_all_nodes_min_new_nodes=backend_params.get('kdtree_rebuild_all_nodes_min_new_nodes', 5),
            stall_steps_per_inlet=backend_params.get('stall_steps_per_inlet', 10),
            interleaving_strategy=backend_params.get('interleaving_strategy', 'round_robin'),
            # Partitioned mode parameters (for partitioned_xy and forest modes)
            partitioned_directional_bias=backend_params.get('partitioned_directional_bias', 1.0),
            partitioned_max_deviation_deg=backend_params.get('partitioned_max_deviation_deg', 30.0),
            partitioned_cone_angle_deg=backend_params.get('partitioned_cone_angle_deg', 30.0),
            partitioned_cylinder_radius=backend_params.get('partitioned_cylinder_radius', 0.001),
        )
        
        inlet_specs = []
        for inlet in validated_inlets:
            spec = {
                "position": inlet.get("position", (0, 0, 0)),
                "radius": inlet.get("radius", 0.001),
            }
            direction = inlet.get("direction") or inlet.get("growth_inward_direction")
            if direction:
                spec["direction"] = direction
            inlet_specs.append(spec)
        
        vessel_type = validated_inlets[0].get("vessel_type", "arterial")
        target = growth_policy.target_terminals or 128
        
        backend = SpaceColonizationBackend()
        network = backend.generate_multi_inlet(
            domain=domain,
            num_outlets=target,
            inlets=inlet_specs,
            vessel_type=vessel_type,
            config=config,
            rng_seed=seed,
            tissue_sampling_policy=effective_tissue_sampling_policy,
        )
        
        terminal_count = sum(1 for n in network.nodes.values() if n.node_type == "terminal")
        inlet_count = sum(1 for n in network.nodes.values() if n.node_type == "inlet")
        
        return network, {
            "terminal_count": terminal_count,
            "inlet_count": inlet_count,
            "node_count": len(network.nodes),
            "segment_count": len(network.segments),
            "multi_inlet_mode": multi_inlet_mode,
        }
    
    # Single inlet or no multi_inlet_mode: use original space_colonization_step_v2 path
    network = create_network(domain=domain, seed=seed)
    
    for inlet in validated_inlets:
        pos = inlet.get("position")
        if pos is None:
            raise ValueError("Inlet missing required 'position' field")
        direction = inlet.get("direction")
        if direction is None:
            raise ValueError("Inlet missing required 'direction' field")
        radius = inlet.get("radius")
        if radius is None:
            raise ValueError("Inlet missing required 'radius' field")
        vessel_type = inlet.get("vessel_type", "arterial")
        add_inlet(
            network,
            position=Point3D(*pos),
            direction=tuple(direction),
            radius=radius,
            vessel_type=vessel_type,
        )
    
    for outlet in ports.get("outlets", []):
        pos = outlet.get("position")
        if pos is None:
            raise ValueError("Outlet missing required 'position' field")
        direction = outlet.get("direction")
        if direction is None:
            raise ValueError("Outlet missing required 'direction' field")
        radius = outlet.get("radius")
        if radius is None:
            raise ValueError("Outlet missing required 'radius' field")
        vessel_type = outlet.get("vessel_type", "venous")
        add_outlet(
            network,
            position=Point3D(*pos),
            direction=tuple(direction),
            radius=radius,
            vessel_type=vessel_type,
        )
    
    if tissue_sampling_policy is None:
        tissue_sampling_policy = TissueSamplingPolicy()
    
    tissue_points, sampling_report = sample_tissue_points(
        domain=domain,
        ports=ports,
        policy=tissue_sampling_policy,
        seed=seed,
    )
    
    min_radius = growth_policy.min_radius
    
    sc_policy = None
    backend_params = getattr(growth_policy, 'backend_params', None) or {}
    if backend_params:
        sc_policy_dict = backend_params.get('space_colonization_policy', {})
        if sc_policy_dict:
            sc_policy = SpaceColonizationPolicy.from_dict(sc_policy_dict)
        else:
            sc_policy = SpaceColonizationPolicy.from_dict(backend_params)
    
    if sc_policy is None:
        sc_policy = SpaceColonizationPolicy()
    
    validation_errors = sc_policy.validate()
    if validation_errors:
        raise ValueError(f"Invalid SpaceColonizationPolicy: {validation_errors}")
    
    params = SpaceColonizationParams(
        max_steps=growth_policy.max_iterations,
        step_size=growth_policy.step_size,
        min_radius=min_radius,
        influence_radius=backend_params.get('influence_radius', 0.015),
        kill_radius=backend_params.get('kill_radius', 0.003),
    )
    
    constraints = BranchingConstraints(
        min_segment_length=max(growth_policy.min_segment_length, sc_policy.min_branch_segment_length),
        min_radius=min_radius,
    )
    
    result = space_colonization_step_v2(
        network,
        tissue_points=tissue_points,
        params=params,
        constraints=constraints,
        seed=seed,
        sc_policy=sc_policy,
    )
    
    terminal_count = sum(1 for n in network.nodes.values() if n.node_type == "terminal")
    
    report = {
        "terminal_count": terminal_count,
        "node_count": len(network.nodes),
        "segment_count": len(network.segments),
    }
    
    if result.metadata and "tree_metrics" in result.metadata:
        report["tree_metrics"] = result.metadata["tree_metrics"]
    
    return network, report


def _generate_kary_tree(
    domain: "Domain",
    ports: Dict[str, Any],
    growth_policy: GrowthPolicy,
    collision_policy: CollisionPolicy,
    seed: Optional[int],
) -> Tuple[VascularNetwork, Dict[str, Any]]:
    """Generate network using k-ary tree recursive bifurcation via KaryTreeBackend.
    
    The tree scale is determined by:
    1. If growth_policy.step_size is explicitly set (> 0), use it as branch_length
    2. Otherwise, let KaryTreeBackend compute branch_length from domain size
       using tree_extent_fraction (default 0.4 = tree fills ~40% of domain)
    
    This ensures the tree scales appropriately with the domain size.
    
    Multi-inlet support:
    - If multiple inlets are defined, uses generate_multi_inlet method
    - Supports "forest" mode (separate trees) and "merge_to_trunk" mode (default)
    - Growth direction can be specified per-inlet via "direction" or "growth_inward_direction"
    
    Downward-biased growth:
    - primary_axis: Primary growth direction (default: inferred from inlet direction or -Z)
    - max_deviation_deg: Maximum angle from primary axis
    - upward_forbidden: Forbid growth with positive Z component
    - wall_margin: Minimum distance from domain boundary
    """
    from ..backends.kary_tree_backend import KaryTreeBackend, KaryTreeConfig
    
    inlets = ports.get("inlets", [])
    if not inlets:
        raise ValueError("K-ary tree requires at least one inlet")
    
    # Validate and fix inlet directions (auto-flip if pointing outward from top face)
    validated_inlets = [_validate_and_fix_inlet_direction(inlet, domain) for inlet in inlets]
    
    target = growth_policy.target_terminals or 128
    
    explicit_branch_length = None
    if growth_policy.step_size is not None and growth_policy.step_size > 0:
        explicit_branch_length = growth_policy.step_size
    
    backend_params = getattr(growth_policy, 'backend_params', {}) or {}
    tree_extent_fraction = backend_params.get('tree_extent_fraction', 0.4)
    branch_length_decay = backend_params.get('branch_length_decay', 0.8)
    angle_deg = backend_params.get('angle_deg', 30.0)
    angle_variation_deg = backend_params.get('angle_variation_deg', 5.0)
    
    primary_axis = backend_params.get('primary_axis')
    max_deviation_deg = backend_params.get('max_deviation_deg', 90.0)
    upward_forbidden = backend_params.get('upward_forbidden', False)
    azimuth_jitter_deg = backend_params.get('azimuth_jitter_deg', 180.0)
    elevation_jitter_deg = backend_params.get('elevation_jitter_deg')
    wall_margin = backend_params.get('wall_margin', backend_params.get('wall_margin_m', 0.0))
    
    multi_inlet_mode = backend_params.get('multi_inlet_mode', 'merge_to_trunk')
    trunk_depth_fraction = backend_params.get('trunk_depth_fraction', 0.2)
    trunk_merge_radius = backend_params.get('trunk_merge_radius')
    max_inlets = backend_params.get('max_inlets', 10)
    
    config = KaryTreeConfig(
        k=backend_params.get('k', 2),
        target_terminals=target,
        terminal_tolerance=growth_policy.terminal_tolerance,
        branch_length=explicit_branch_length,
        branch_length_decay=branch_length_decay,
        angle_deg=angle_deg,
        angle_variation_deg=angle_variation_deg,
        min_radius=growth_policy.min_segment_length / 10,
        tree_extent_fraction=tree_extent_fraction,
        use_domain_scaling=(explicit_branch_length is None),
        primary_axis=tuple(primary_axis) if primary_axis else None,
        max_deviation_deg=max_deviation_deg,
        upward_forbidden=upward_forbidden,
        azimuth_jitter_deg=azimuth_jitter_deg,
        elevation_jitter_deg=elevation_jitter_deg,
        wall_margin=wall_margin,
        multi_inlet_mode=multi_inlet_mode,
        trunk_depth_fraction=trunk_depth_fraction,
        trunk_merge_radius=trunk_merge_radius,
        max_inlets=max_inlets,
        seed=seed,
    )
    
    backend = KaryTreeBackend()
    
    if len(validated_inlets) > 1:
        inlet_specs = []
        for inlet in validated_inlets:
            spec = {
                "position": inlet.get("position", (0, 0, 0)),
                "radius": inlet.get("radius", 0.001),
                "is_surface_opening": inlet.get("is_surface_opening", False),
            }
            direction = inlet.get("direction") or inlet.get("growth_inward_direction")
            if direction:
                spec["direction"] = direction
            inlet_specs.append(spec)
        
        vessel_type = validated_inlets[0].get("vessel_type", "arterial")
        network = backend.generate_multi_inlet(
            domain=domain,
            num_outlets=target,
            inlets=inlet_specs,
            vessel_type=vessel_type,
            config=config,
            rng_seed=seed,
        )
    else:
        inlet = validated_inlets[0]
        inlet_position = np.array(inlet.get("position", (0, 0, 0)))
        inlet_radius = inlet.get("radius", 0.001)
        vessel_type = inlet.get("vessel_type", "arterial")
        
        inlet_direction = inlet.get("direction") or inlet.get("growth_inward_direction")
        if inlet_direction and config.primary_axis is None:
            config = KaryTreeConfig(
                **{k: getattr(config, k) for k in config.__dataclass_fields__},
                primary_axis=tuple(inlet_direction),
            )
        
        network = backend.generate(
            domain=domain,
            num_outlets=target,
            inlet_position=inlet_position,
            inlet_radius=inlet_radius,
            vessel_type=vessel_type,
            config=config,
            rng_seed=seed,
        )
    
    terminal_count = sum(1 for n in network.nodes.values() if n.node_type == "terminal")
    inlet_count = sum(1 for n in network.nodes.values() if n.node_type == "inlet")
    
    return network, {
        "terminal_count": terminal_count,
        "inlet_count": inlet_count,
        "node_count": len(network.nodes),
        "segment_count": len(network.segments),
        "multi_inlet_mode": multi_inlet_mode if len(validated_inlets) > 1 else "single",
    }


def _generate_scaffold_topdown(
    domain: "Domain",
    ports: Dict[str, Any],
    growth_policy: GrowthPolicy,
    collision_policy: CollisionPolicy,
    seed: Optional[int],
) -> Tuple[VascularNetwork, Dict[str, Any]]:
    """Generate network using scaffold top-down recursive branching with collision avoidance.
    
    This backend creates tree structures by recursively splitting branches with
    configurable branching factor (splits), depth (levels), and online collision
    avoidance. The algorithm is inspired by scaffold_web_collision.py but operates
    entirely in meters and produces VascularNetwork output.
    
    Key features:
    - Recursive branching with configurable splits and levels
    - Tapering radius via ratio parameter
    - Lateral spread in plane orthogonal to primary axis
    - Curved branches represented as polylines for collision detection
    - Online collision avoidance with rotation attempts and spread reduction
    - Optional post-pass collision cleanup
    
    Multi-inlet support:
    - If multiple inlets are defined, each inlet grows its own tree
    - Trees share a common spatial index for inter-tree collision avoidance
    """
    from ..backends.scaffold_topdown_backend import (
        ScaffoldTopDownBackend,
        ScaffoldTopDownConfig,
        CollisionOnlineConfig,
        CollisionPostpassConfig,
    )
    
    inlets = ports.get("inlets", [])
    if not inlets:
        raise ValueError("Scaffold top-down requires at least one inlet")
    
    validated_inlets = [_validate_and_fix_inlet_direction(inlet, domain) for inlet in inlets]
    
    backend_params = getattr(growth_policy, 'backend_params', {}) or {}
    
    collision_online_params = backend_params.get('collision_online', {})
    collision_online = CollisionOnlineConfig(
        enabled=collision_online_params.get('enabled', True),
        buffer_abs_m=collision_online_params.get('buffer_abs_m', 0.00002),
        buffer_rel=collision_online_params.get('buffer_rel', 0.05),
        cell_size_m=collision_online_params.get('cell_size_m', 0.0005),
        rotation_attempts=collision_online_params.get('rotation_attempts', 14),
        reduction_factors=collision_online_params.get('reduction_factors', [0.6, 0.35]),
        max_attempts_per_child=collision_online_params.get('max_attempts_per_child', 18),
        on_fail=collision_online_params.get('on_fail', 'terminate_branch'),
    )
    
    collision_postpass_params = backend_params.get('collision_postpass', {})
    collision_postpass = CollisionPostpassConfig(
        enabled=collision_postpass_params.get('enabled', True),
        min_clearance_m=collision_postpass_params.get('min_clearance_m', 0.00002),
        strategy_order=collision_postpass_params.get('strategy_order', ['shrink', 'terminate']),
        shrink_factor=collision_postpass_params.get('shrink_factor', 0.9),
        shrink_max_iterations=collision_postpass_params.get('shrink_max_iterations', 6),
    )
    
    primary_axis = backend_params.get('primary_axis', (0.0, 0.0, -1.0))
    if isinstance(primary_axis, list):
        primary_axis = tuple(primary_axis)
    
    config = ScaffoldTopDownConfig(
        primary_axis=primary_axis,
        splits=backend_params.get('splits', 3),
        levels=backend_params.get('levels', 6),
        ratio=backend_params.get('ratio', 0.78),
        step_length=backend_params.get('step_length', 0.002),
        step_decay=backend_params.get('step_decay', 0.92),
        spread=backend_params.get('spread', 0.0015),
        spread_decay=backend_params.get('spread_decay', 0.90),
        cone_angle_deg=backend_params.get('cone_angle_deg', 70.0),
        jitter_deg=backend_params.get('jitter_deg', 12.0),
        curvature=backend_params.get('curvature', 0.35),
        curve_samples=backend_params.get('curve_samples', 7),
        wall_margin_m=backend_params.get('wall_margin_m', 0.0001),
        min_radius=backend_params.get('min_radius', growth_policy.min_radius or 0.00005),
        min_segment_length=growth_policy.min_segment_length,
        max_segment_length=growth_policy.max_segment_length,
        seed=seed,
        collision_online=collision_online,
        collision_postpass=collision_postpass,
    )
    
    backend = ScaffoldTopDownBackend()
    
    if len(validated_inlets) > 1:
        inlet_specs = []
        for inlet in validated_inlets:
            spec = {
                "position": inlet.get("position", (0, 0, 0)),
                "radius": inlet.get("radius", 0.001),
            }
            inlet_specs.append(spec)
        
        vessel_type = validated_inlets[0].get("vessel_type", "arterial")
        network = backend.generate_multi_inlet(
            domain=domain,
            num_outlets=growth_policy.target_terminals or 128,
            inlets=inlet_specs,
            vessel_type=vessel_type,
            config=config,
            rng_seed=seed,
        )
    else:
        inlet = validated_inlets[0]
        inlet_position = np.array(inlet.get("position", (0, 0, 0)))
        inlet_radius = inlet.get("radius", 0.001)
        vessel_type = inlet.get("vessel_type", "arterial")
        
        network = backend.generate(
            domain=domain,
            num_outlets=growth_policy.target_terminals or 128,
            inlet_position=inlet_position,
            inlet_radius=inlet_radius,
            vessel_type=vessel_type,
            config=config,
            rng_seed=seed,
        )
    
    terminal_count = sum(1 for n in network.nodes.values() if n.node_type == "terminal")
    inlet_count = sum(1 for n in network.nodes.values() if n.node_type == "inlet")
    
    generation_stats = backend.get_generation_stats()
    
    return network, {
        "backend": "scaffold_topdown",
        "terminal_count": terminal_count,
        "inlet_count": inlet_count,
        "node_count": len(network.nodes),
        "segment_count": len(network.segments),
        "levels": config.levels,
        "splits": config.splits,
        "collision_online_stats": {
            "segments_proposed": generation_stats.get("segments_proposed", 0),
            "segments_created": generation_stats.get("segments_created", 0),
            "collisions_detected": generation_stats.get("collisions_detected", 0),
            "rotations_successful": generation_stats.get("rotations_successful", 0),
            "branches_terminated": generation_stats.get("branches_terminated", 0),
            "branches_skipped": generation_stats.get("branches_skipped", 0),
        },
    }


def _generate_cco_hybrid(
    domain: "Domain",
    ports: Dict[str, Any],
    growth_policy: GrowthPolicy,
    collision_policy: CollisionPolicy,
    seed: Optional[int],
) -> Tuple[VascularNetwork, Dict[str, Any]]:
    """Generate network using CCO hybrid algorithm."""
    from ..backends.cco_hybrid_backend import CCOHybridBackend, CCOConfig
    
    backend = CCOHybridBackend()
    
    # Get first inlet
    inlets = ports.get("inlets", [])
    if not inlets:
        raise ValueError("CCO hybrid requires at least one inlet")
    
    inlet = inlets[0]
    inlet_position = np.array(inlet.get("position", (0, 0, 0)))
    inlet_radius = inlet.get("radius", 0.001)
    
    # Target outlets
    target = growth_policy.target_terminals or 128
    
    config = CCOConfig(
        seed=seed,
        check_collisions=collision_policy.check_collisions,
        collision_clearance=collision_policy.collision_clearance,
    )
    
    network = backend.generate(
        domain=domain,
        num_outlets=target,
        inlet_position=inlet_position,
        inlet_radius=inlet_radius,
        config=config,
        rng_seed=seed,
    )
    
    # Count terminals using string node_type
    terminal_count = sum(1 for n in network.nodes.values() if n.node_type == "terminal")
    
    return network, {
        "terminal_count": terminal_count,
        "node_count": len(network.nodes),
        "segment_count": len(network.segments),
    }


def _generate_programmatic(
    domain: "Domain",
    ports: Dict[str, Any],
    growth_policy: GrowthPolicy,
    collision_policy: CollisionPolicy,
    seed: Optional[int],
) -> Tuple[VascularNetwork, Dict[str, Any]]:
    """Generate network using programmatic backend with DSL-based generation.
    
    The programmatic backend enables orchestrated generation where paths/graphs
    are built from a declarative DSL with pathfinding as one algorithm option
    and consistent collision-aware execution.
    
    Features:
    - Multiple path algorithms (A*, straight, bezier, hybrid)
    - Waypoint-based routing with skip-on-failure
    - Collision detection and resolution (reroute/shrink/terminate)
    - Obstacle inflation = clearance + local radius
    
    Backend Params (via GrowthPolicy.backend_params):
    - mode: "network" | "mesh"
    - path_algorithm: "astar_voxel" | "straight" | "bezier" | "hybrid"
    - waypoint_policy: {allow_skip: bool, ...}
    - pathfinding_policy: {voxel_pitch, clearance, max_nodes, timeout_s, ...}
    - radius_policy: {...}
    - steps: [{op: ..., ...}, ...]
    """
    from ..backends.programmatic_backend import (
        ProgrammaticBackend,
        ProgramCollisionPolicy,
        StepSpec,
    )
    from aog_policies.generation import ProgramPolicy
    from aog_policies.pathfinding import WaypointPolicy
    
    backend = ProgrammaticBackend()
    
    # Get first inlet
    inlets = ports.get("inlets", [])
    if not inlets:
        raise ValueError("Programmatic backend requires at least one inlet")
    
    inlet = inlets[0]
    inlet_position = tuple(inlet.get("position", (0, 0, 0)))
    
    # Use ProgramPolicy's default_radius as fallback instead of hardcoded 1mm
    policy_default_radius = ProgramPolicy().default_radius
    inlet_radius = inlet.get("radius", policy_default_radius)
    
    # Check if backend_params is provided in growth_policy
    backend_params = growth_policy.backend_params or {}
    
    if backend_params:
        # Use backend_params to configure the programmatic backend
        # This enables unified API: generate_network(generator_kind="programmatic", 
        #                                            growth_policy=GrowthPolicy(backend_params={...}))
        network, report = backend.generate_from_backend_params(
            domain=domain,
            ports=ports,
            backend_params=backend_params,
            collision_policy=collision_policy,
            seed=seed,
        )
    else:
        # Build default configuration from collision_policy and ports
        # Build collision policy from CollisionPolicy
        prog_collision_policy = ProgramCollisionPolicy(
            enabled=collision_policy.check_collisions,
            min_clearance=collision_policy.collision_clearance,
            inflate_by_radius=True,
            check_after_each_step=True,
        )
        
        # Build waypoint policy with skip-on-failure and warnings
        waypoint_policy = WaypointPolicy(
            skip_unreachable=True,
            max_skip_count=3,
            emit_warnings=True,
            fallback_direct=True,
        )
        
        # Build default steps: inlet -> outlets
        steps = [
            StepSpec.add_node(
                "inlet",
                position=inlet_position,
                node_type="inlet",
                radius=inlet_radius,
            )
        ]
        
        # Add routes to outlets
        outlets = ports.get("outlets", [])
        for i, outlet in enumerate(outlets):
            outlet_pos = tuple(outlet.get("position", (0, 0, 0)))
            outlet_radius = outlet.get("radius", inlet_radius * 0.5)
            steps.append(
                StepSpec.route(
                    "inlet",
                    to=outlet_pos,
                    algorithm="astar_voxel",
                    radius=outlet_radius,
                    clearance=collision_policy.collision_clearance,
                )
            )
        
        # Build program policy
        policy = ProgramPolicy(
            mode="network",
            steps=steps,
            path_algorithm="astar_voxel",
            collision_policy=prog_collision_policy,
            waypoint_policy=waypoint_policy,
            default_radius=inlet_radius,
            default_clearance=collision_policy.collision_clearance,
        )
        
        # Generate network
        network, report = backend.generate_from_program(domain, ports, policy)
    
    # Count terminals using string node_type
    terminal_count = sum(1 for n in network.nodes.values() if n.node_type == "terminal")
    
    return network, {
        "terminal_count": terminal_count,
        "node_count": len(network.nodes),
        "segment_count": len(network.segments),
        "steps_executed": report.steps_executed,
        "steps_total": report.steps_total,
        "warnings": report.warnings,
    }


def generate_void_mesh(
    kind: str,
    domain: Union[DomainSpec, "Domain"],
    ports: Dict[str, Any],
    channel_policy: Optional["ChannelPolicy"] = None,
    growth_policy: Optional[GrowthPolicy] = None,
    synthesis_policy: Optional["MeshSynthesisPolicy"] = None,
    merge_policy: Optional["MeshMergePolicy"] = None,
) -> Tuple["trimesh.Trimesh", OperationReport]:
    """
    Generate a void mesh for embedding into a domain.
    
    Parameters
    ----------
    kind : str
        Type of void to generate:
        - "primitive_channels": Simple channel primitives
        - "network_synthesis": Generate network then synthesize mesh
    domain : DomainSpec or Domain
        Domain specification
    ports : dict
        Port configuration
    channel_policy : ChannelPolicy, optional
        Policy for channel primitives
    growth_policy : GrowthPolicy, optional
        Policy for network growth (if kind="network_synthesis")
    synthesis_policy : MeshSynthesisPolicy, optional
        Policy for mesh synthesis
    merge_policy : MeshMergePolicy, optional
        Policy for merging channel meshes. If provided, controls how disconnected
        channel meshes are merged. Set keep_largest_component=False to preserve
        all disconnected tubes.
        
    Returns
    -------
    mesh : trimesh.Trimesh
        Generated void mesh
    report : OperationReport
        Report with metadata
    """
    from ..policies import ChannelPolicy, MeshSynthesisPolicy
    
    if channel_policy is None:
        channel_policy = ChannelPolicy()
    if synthesis_policy is None:
        synthesis_policy = MeshSynthesisPolicy()
    
    warnings = []
    metadata = {"kind": kind}
    
    if kind == "primitive_channels":
        # B3 FIX: Use create_channel_from_policy instead of legacy create_channels_from_ports
        from ..ops.primitives.channels import create_channel_from_policy
        from ..utils.port_placement import place_ports_on_domain
        from ..utils.faces import face_frame
        import trimesh
        import numpy as np
        
        channel_meshes = []
        channel_reports = []
        
        # Get domain depth for length calculations
        if hasattr(domain, 'height'):
            domain_depth = domain.height
        elif hasattr(domain, 'z_max') and hasattr(domain, 'z_min'):
            domain_depth = domain.z_max - domain.z_min
        elif hasattr(domain, 'semi_axis_c'):
            domain_depth = 2 * domain.semi_axis_c
        else:
            domain_depth = 0.01  # Default 10mm
        
        # Get face center for fang-hook radial direction
        face = channel_policy.to_dict().get('face', 'top') if hasattr(channel_policy, 'to_dict') else 'top'
        try:
            _, _, _, _, face_center = face_frame(face, domain)
        except Exception:
            face_center = None
        
        # Get effective radius from port placement policy if available
        effective_radius = None
        
        # Create channels from inlets
        for inlet in ports.get("inlets", []):
            pos = inlet.get("position", (0, 0, 0))
            direction = inlet.get("direction", (0, 0, -1))
            radius = inlet.get("radius", 0.001)
            
            channel_mesh, channel_report = create_channel_from_policy(
                start=pos,
                direction=direction,
                radius=radius,
                policy=channel_policy,
                domain_depth=domain_depth,
                face_center=face_center,
                effective_radius=effective_radius,
            )
            channel_meshes.append(channel_mesh)
            channel_reports.append(channel_report)
            warnings.extend(channel_report.warnings)
        
        # Merge all channel meshes using voxel-first union for watertight result
        if channel_meshes:
            if len(channel_meshes) == 1:
                mesh = channel_meshes[0]
            else:
                # Use voxel-based merge for proper union (not just concatenation)
                # This produces a single watertight void component
                from ..ops.mesh.merge import merge_meshes
                from ..policies import MeshMergePolicy
                try:
                    # Use provided merge_policy if available, otherwise create safe default
                    # that does NOT drop disconnected components
                    effective_merge_policy = merge_policy
                    if effective_merge_policy is None:
                        # Safe default if caller didn't provide one
                        # keep_largest_component=False ensures all disconnected tubes are preserved
                        effective_merge_policy = MeshMergePolicy(
                            mode="voxel",
                            voxel_pitch=5e-5,
                            auto_adjust_pitch=True,
                            keep_largest_component=False,
                            fill_voxels=False,
                        )
                    merged_mesh, merge_report = merge_meshes(channel_meshes, effective_merge_policy)
                    if merged_mesh is None or len(merged_mesh.vertices) == 0:
                        # E1 FIX: Try coarser voxel merge before falling back to concatenation
                        coarse_policy = MeshMergePolicy(
                            mode="voxel",
                            voxel_pitch=1e-4,  # 100 microns - coarser
                            auto_adjust_pitch=True,
                            keep_largest_component=False,
                            fill_voxels=False,
                        )
                        merged_mesh, merge_report = merge_meshes(channel_meshes, coarse_policy)
                        if merged_mesh is not None and len(merged_mesh.vertices) > 0:
                            mesh = merged_mesh
                            warnings.append("Used coarser voxel merge (100µm pitch)")
                            warnings.extend(merge_report.warnings)
                        else:
                            mesh = trimesh.util.concatenate(channel_meshes)
                            warnings.append(
                                "Voxel merge failed, using concatenation. "
                                "Result may have self-intersections."
                            )
                    else:
                        mesh = merged_mesh
                        warnings.extend(merge_report.warnings)
                except Exception as e:
                    # E1 FIX: Try coarser voxel merge before falling back to concatenation
                    try:
                        coarse_policy = MeshMergePolicy(
                            mode="voxel",
                            voxel_pitch=1e-4,  # 100 microns - coarser
                            auto_adjust_pitch=True,
                            keep_largest_component=False,
                            fill_voxels=False,
                        )
                        merged_mesh, merge_report = merge_meshes(channel_meshes, coarse_policy)
                        if merged_mesh is not None and len(merged_mesh.vertices) > 0:
                            mesh = merged_mesh
                            warnings.append(f"Primary merge failed ({e}), used coarser voxel merge")
                            warnings.extend(merge_report.warnings)
                        else:
                            mesh = trimesh.util.concatenate(channel_meshes)
                            warnings.append(
                                f"Voxel merge failed ({e}), using concatenation. "
                                "Result may have self-intersections."
                            )
                    except Exception:
                        mesh = trimesh.util.concatenate(channel_meshes)
                        warnings.append(
                            f"Voxel merge failed ({e}), using concatenation. "
                            "Result may have self-intersections."
                        )
        else:
            mesh = trimesh.Trimesh()
        
        metadata["channel_count"] = len(channel_meshes)
        metadata["channel_reports"] = [r.metadata for r in channel_reports]
        
    elif kind == "network_synthesis":
        if growth_policy is None:
            growth_policy = GrowthPolicy()
        
        # Generate network first
        network, net_report = generate_network(
            generator_kind=growth_policy.backend,
            domain=domain,
            ports=ports,
            growth_policy=growth_policy,
        )
        warnings.extend(net_report.warnings)
        metadata["network"] = net_report.metadata
        
        # Synthesize mesh
        from ..ops.mesh.synthesis import synthesize_mesh
        mesh, synth_report = synthesize_mesh(network, synthesis_policy)
        warnings.extend(synth_report.warnings)
        metadata["synthesis"] = synth_report.metadata
        
    else:
        raise ValueError(f"Unknown void mesh kind: {kind}")
    
    report = OperationReport(
        operation="generate_void_mesh",
        success=True,
        requested_policy=channel_policy.to_dict(),
        effective_policy=channel_policy.to_dict(),
        warnings=warnings,
        metadata=metadata,
    )
    
    return mesh, report


def build_component(
    component_spec: Optional[Dict[str, Any]] = None,
    ctx: Optional[Union[Dict[str, Any], GenerationContext]] = None,
    *,
    domain: Optional[Any] = None,
    ports: Optional[Dict[str, Any]] = None,
    growth_policy: Optional[GrowthPolicy] = None,
    collision_policy: Optional[CollisionPolicy] = None,
    generator_kind: Optional[str] = None,
) -> Tuple[Any, OperationReport]:
    """
    Build a component from a specification.
    
    This is a high-level function that dispatches to the appropriate
    generator based on the component type.
    
    Supports two calling conventions:
    A) New style: build_component(component_spec: dict, ctx=...)
    B) Legacy style: build_component(domain=..., ports=..., growth_policy=..., ...)
    
    Parameters
    ----------
    component_spec : dict, optional
        Component specification with keys:
        - "type": "network" | "mesh" | "void"
        - "generator": generator kind
        - "domain": domain spec
        - "ports": port configuration
        - "policies": dict of policy overrides
    ctx : dict or GenerationContext, optional
        Context for caching expensive computations. If a dict is provided,
        it will be wrapped in a GenerationContext. Cached items include:
        - compiled_domain: Compiled domain object
        - domain_mesh: Domain mesh (for embedding)
        - face_frames: MeshDomain PCA/OBB results
        - effective_pitches: Derived pitches/tolerances per operation
        - pathfinding_coarse: Coarse pathfinding solution + corridor info
        - void_mesh: Unioned void mesh before embedding
    domain : Domain, optional
        (Legacy) Domain object for network generation
    ports : dict, optional
        (Legacy) Port configuration
    growth_policy : GrowthPolicy, optional
        (Legacy) Growth policy for network generation
    collision_policy : CollisionPolicy, optional
        (Legacy) Collision policy for network generation
    generator_kind : str, optional
        (Legacy) Generator kind (default: "space_colonization")
        
    Returns
    -------
    component : VascularNetwork or trimesh.Trimesh
        Generated component
    report : OperationReport
        Report with metadata
    """
    # Handle legacy calling convention
    if domain is not None:
        # Legacy style: build_component(domain=..., ports=..., growth_policy=...)
        component_spec = {
            "type": "network",
            "generator": generator_kind or "cco_hybrid",
            "domain": domain,
            "ports": ports or {},
            "policies": {
                "growth": growth_policy.to_dict() if growth_policy else {},
                "collision": collision_policy.to_dict() if collision_policy else {},
            },
        }
    elif component_spec is None:
        raise ValueError(
            "Must provide either component_spec dict or legacy kwargs (domain=, ports=, growth_policy=)"
        )
    
    if ctx is None:
        ctx = GenerationContext()
    elif isinstance(ctx, dict):
        gen_ctx = GenerationContext()
        for k, v in ctx.items():
            gen_ctx.set(k, v)
        ctx = gen_ctx
    
    component_type = component_spec.get("type", "network")
    generator = component_spec.get("generator", "space_colonization")
    domain_spec = component_spec.get("domain")
    ports_spec = component_spec.get("ports", {})
    policies = component_spec.get("policies", {})
    
    domain_cache_key = f"compiled_domain:{_compute_cache_key(domain_spec)}"
    compiled_domain = ctx.get_or_compute(
        domain_cache_key,
        lambda: _coerce_domain(domain_spec)
    )
    
    if component_type == "network":
        growth_policy_obj = GrowthPolicy.from_dict(policies.get("growth", {}))
        collision_policy_obj = CollisionPolicy.from_dict(policies.get("collision", {}))
        
        result, report = generate_network(
            generator_kind=generator,
            domain=compiled_domain,
            ports=ports_spec,
            growth_policy=growth_policy_obj,
            collision_policy=collision_policy_obj,
        )
        
        report.metadata["cache_stats"] = ctx.stats()
        return result, report
        
    elif component_type in ("mesh", "void"):
        from ..policies import ChannelPolicy, MeshSynthesisPolicy
        
        channel_policy = ChannelPolicy.from_dict(policies.get("channel", {}))
        growth_policy_obj = GrowthPolicy.from_dict(policies.get("growth", {}))
        synthesis_policy = MeshSynthesisPolicy.from_dict(policies.get("synthesis", {}))
        
        result, report = generate_void_mesh(
            kind=generator,
            domain=compiled_domain,
            ports=ports_spec,
            channel_policy=channel_policy,
            growth_policy=growth_policy_obj,
            synthesis_policy=synthesis_policy,
        )
        
        report.metadata["cache_stats"] = ctx.stats()
        return result, report
        
    else:
        raise ValueError(f"Unknown component type: {component_type}")


__all__ = [
    "generate_network",
    "generate_void_mesh",
    "build_component",
    "GenerationContext",
    "_coerce_domain",
]
