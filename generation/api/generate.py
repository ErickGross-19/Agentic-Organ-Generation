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
import numpy as np

from ..policies import (
    GrowthPolicy,
    CollisionPolicy,
    TissueSamplingPolicy,
    OperationReport,
)
from ..specs.design_spec import DesignSpec, DomainSpec
from ..specs.compile import compile_domain
from ..core.network import VascularNetwork
from ..core.types import Point3D

if TYPE_CHECKING:
    import trimesh

logger = logging.getLogger(__name__)


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
    
    # Compile domain if needed
    if hasattr(domain, 'type'):
        compiled_domain = compile_domain(domain)
    else:
        compiled_domain = domain
    
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
    
    network = create_network(domain=domain, seed=seed)
    
    # Add inlets
    for inlet in ports.get("inlets", []):
        pos = inlet.get("position", (0, 0, 0))
        radius = inlet.get("radius", 0.001)
        vessel_type = inlet.get("vessel_type", "arterial")
        add_inlet(
            network,
            position=Point3D(*pos),
            radius=radius,
            vessel_type=vessel_type,
        )
    
    # Add outlets
    for outlet in ports.get("outlets", []):
        pos = outlet.get("position", (0, 0, 0))
        radius = outlet.get("radius", 0.001)
        vessel_type = outlet.get("vessel_type", "venous")
        add_outlet(
            network,
            position=Point3D(*pos),
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
    
    # Run colonization
    params = SpaceColonizationParams(
        max_steps=growth_policy.max_iterations,
    )
    space_colonization_step(network, tissue_points=tissue_points, params=params, seed=seed)
    
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
    """
    from ..backends.programmatic_backend import (
        ProgrammaticBackend,
        ProgramPolicy,
        ProgramCollisionPolicy,
        WaypointPolicy,
        StepSpec,
    )
    
    backend = ProgrammaticBackend()
    
    # Get first inlet
    inlets = ports.get("inlets", [])
    if not inlets:
        raise ValueError("Programmatic backend requires at least one inlet")
    
    inlet = inlets[0]
    inlet_position = tuple(inlet.get("position", (0, 0, 0)))
    inlet_radius = inlet.get("radius", 0.001)
    
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
        from ..ops.primitives.channels import create_channels_from_ports
        
        mesh, channel_meta = create_channels_from_ports(
            domain, ports, channel_policy
        )
        metadata.update(channel_meta)
        
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
    component_spec: Dict[str, Any],
    ctx: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, OperationReport]:
    """
    Build a component from a specification.
    
    This is a high-level function that dispatches to the appropriate
    generator based on the component type.
    
    Parameters
    ----------
    component_spec : dict
        Component specification with keys:
        - "type": "network" | "mesh" | "void"
        - "generator": generator kind
        - "domain": domain spec
        - "ports": port configuration
        - "policies": dict of policy overrides
    ctx : dict, optional
        Context with shared state
        
    Returns
    -------
    component : VascularNetwork or trimesh.Trimesh
        Generated component
    report : OperationReport
        Report with metadata
    """
    component_type = component_spec.get("type", "network")
    generator = component_spec.get("generator", "space_colonization")
    domain = component_spec.get("domain")
    ports = component_spec.get("ports", {})
    policies = component_spec.get("policies", {})
    
    if component_type == "network":
        growth_policy = GrowthPolicy.from_dict(policies.get("growth", {}))
        collision_policy = CollisionPolicy.from_dict(policies.get("collision", {}))
        
        return generate_network(
            generator_kind=generator,
            domain=domain,
            ports=ports,
            growth_policy=growth_policy,
            collision_policy=collision_policy,
        )
        
    elif component_type in ("mesh", "void"):
        from ..policies import ChannelPolicy, MeshSynthesisPolicy
        
        channel_policy = ChannelPolicy.from_dict(policies.get("channel", {}))
        growth_policy = GrowthPolicy.from_dict(policies.get("growth", {}))
        synthesis_policy = MeshSynthesisPolicy.from_dict(policies.get("synthesis", {}))
        
        return generate_void_mesh(
            kind=generator,
            domain=domain,
            ports=ports,
            channel_policy=channel_policy,
            growth_policy=growth_policy,
            synthesis_policy=synthesis_policy,
        )
        
    else:
        raise ValueError(f"Unknown component type: {component_type}")


__all__ = [
    "generate_network",
    "generate_void_mesh",
    "build_component",
]
