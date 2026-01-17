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
import numpy as np

# Import policies from aog_policies (canonical source for runner contract)
from aog_policies import (
    GrowthPolicy,
    CollisionPolicy,
    TissueSamplingPolicy,
    OperationReport,
)
from ..specs.design_spec import DesignSpec, DomainSpec
from ..specs.compile import compile_domain
from ..core.network import VascularNetwork
from ..core.types import Point3D
from ..core.domain import DomainSpec as RuntimeDomainSpec, domain_from_dict

if TYPE_CHECKING:
    import trimesh

logger = logging.getLogger(__name__)


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
        
    else:
        raise ValueError(f"Unknown generator_kind: {generator_kind}")
    
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
    """Generate network using space colonization algorithm."""
    from ..ops import create_network, add_inlet, add_outlet, space_colonization_step
    from ..ops.space_colonization import SpaceColonizationParams
    from ..utils.tissue_sampling import sample_tissue_points
    from ..rules.constraints import BranchingConstraints
    
    network = create_network(domain=domain, seed=seed)
    
    # Add inlets
    for inlet in ports.get("inlets", []):
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
    
    # Add outlets
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
    
    # Sample tissue points using policy-driven sampling
    if tissue_sampling_policy is None:
        tissue_sampling_policy = TissueSamplingPolicy()
    
    tissue_points, sampling_report = sample_tissue_points(
        domain=domain,
        ports=ports,
        policy=tissue_sampling_policy,
        seed=seed,
    )
    
    # Run colonization with policy-driven parameters
    # Derive min_radius from growth_policy.min_segment_length (radius is half of segment length)
    # GrowthPolicy.min_segment_length always has a default value (0.0002m), so no fallback needed
    min_radius = growth_policy.min_segment_length / 2
    
    params = SpaceColonizationParams(
        max_steps=growth_policy.max_iterations,
        step_size=growth_policy.step_size,
        min_radius=min_radius,
    )
    
    # Create constraints with policy-driven min_segment_length
    # This ensures the growth respects the policy's min_segment_length constraint
    constraints = BranchingConstraints(
        min_segment_length=growth_policy.min_segment_length,
        min_radius=min_radius,
    )
    
    space_colonization_step(network, tissue_points=tissue_points, params=params, constraints=constraints, seed=seed)
    
    # Count terminals using string node_type
    terminal_count = sum(1 for n in network.nodes.values() if n.node_type == "terminal")
    
    return network, {
        "terminal_count": terminal_count,
        "node_count": len(network.nodes),
        "segment_count": len(network.segments),
    }


def _generate_kary_tree(
    domain: "Domain",
    ports: Dict[str, Any],
    growth_policy: GrowthPolicy,
    collision_policy: CollisionPolicy,
    seed: Optional[int],
) -> Tuple[VascularNetwork, Dict[str, Any]]:
    """Generate network using k-ary tree recursive bifurcation via KaryTreeBackend."""
    from ..backends.kary_tree_backend import KaryTreeBackend, KaryTreeConfig
    
    # Get first inlet for the backend
    inlets = ports.get("inlets", [])
    if not inlets:
        raise ValueError("K-ary tree requires at least one inlet")
    
    inlet = inlets[0]
    inlet_position = np.array(inlet.get("position", (0, 0, 0)))
    inlet_radius = inlet.get("radius", 0.001)
    vessel_type = inlet.get("vessel_type", "arterial")
    
    # Target terminals
    target = growth_policy.target_terminals or 128
    
    # Create backend config from growth policy
    config = KaryTreeConfig(
        k=2,
        target_terminals=target,
        terminal_tolerance=growth_policy.terminal_tolerance,
        branch_length=growth_policy.step_size,
        min_radius=growth_policy.min_segment_length / 10,  # Approximate
        seed=seed,
    )
    
    # Use backend to generate network
    backend = KaryTreeBackend()
    network = backend.generate(
        domain=domain,
        num_outlets=target,
        inlet_position=inlet_position,
        inlet_radius=inlet_radius,
        vessel_type=vessel_type,
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
                    # Use a reasonable default pitch for voxel merge
                    # 5e-5 (50 microns) is suitable for fine vascular structures
                    merge_policy = MeshMergePolicy(
                        mode="voxel",
                        voxel_pitch=5e-5,
                        auto_adjust_pitch=True,
                    )
                    merged_mesh, merge_report = merge_meshes(channel_meshes, merge_policy)
                    if merged_mesh is None or len(merged_mesh.vertices) == 0:
                        # E1 FIX: Try coarser voxel merge before falling back to concatenation
                        coarse_policy = MeshMergePolicy(
                            mode="voxel",
                            voxel_pitch=1e-4,  # 100 microns - coarser
                            auto_adjust_pitch=True,
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
