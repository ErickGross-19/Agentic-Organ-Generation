"""
Programmatic backend for building vascular networks from JSON-friendly plans.

This backend enables orchestrated generation where paths/graphs are built from
a declarative DSL with pathfinding as one algorithm option and consistent
collision-aware execution.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    TYPE_CHECKING,
)
import numpy as np
import logging

from .base import GenerationBackend, BackendConfig
from ..core.network import VascularNetwork, Node, VesselSegment
from ..core.domain import DomainSpec
from ..core.types import Point3D, TubeGeometry
from ..core.result import OperationResult, OperationStatus, Delta
from aog_policies.pathfinding import WaypointPolicy
from aog_policies.collision import RadiusPolicy, RetryPolicy, UnifiedCollisionPolicy
from aog_policies.generation import ProgramPolicy

if TYPE_CHECKING:
    import trimesh
    from aog_policies.resolution import ResolutionPolicy

logger = logging.getLogger(__name__)


class PathAlgorithm(str, Enum):
    """Available path algorithms for routing."""
    ASTAR_VOXEL = "astar_voxel"
    STRAIGHT = "straight"
    BEZIER = "bezier"
    HYBRID = "hybrid"


class StepOp(str, Enum):
    """Available step operations in the DSL."""
    ADD_NODE = "add_node"
    ROUTE = "route"
    TURN = "turn"
    BIFURCATE = "bifurcate"
    MERGE = "merge"
    CONNECT_TO_OUTLET = "connect_to_outlet"


class CollisionStrategy(str, Enum):
    """Collision resolution strategies."""
    REROUTE = "reroute"
    SHRINK = "shrink"
    TERMINATE = "terminate"
    VOXEL_MERGE_FALLBACK = "voxel_merge_fallback"


ProgramCollisionPolicy = UnifiedCollisionPolicy


@dataclass
class StepSpec:
    """
    Specification for a single step in the programmatic DSL.
    
    Each step represents an operation like adding a node, routing a path,
    creating a bifurcation, etc.
    """
    op: str
    params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {"op": self.op, **self.params}
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StepSpec":
        op = d.get("op", "")
        params = {k: v for k, v in d.items() if k != "op"}
        return cls(op=op, params=params)
    
    @classmethod
    def add_node(
        cls,
        node_id: str,
        source: Optional[str] = None,
        position: Optional[Tuple[float, float, float]] = None,
        node_type: str = "junction",
        radius: Optional[float] = None,
    ) -> "StepSpec":
        """Create an add_node step."""
        params = {"id": node_id, "node_type": node_type}
        if source is not None:
            params["source"] = source
        if position is not None:
            params["position"] = position
        if radius is not None:
            params["radius"] = radius
        return cls(op=StepOp.ADD_NODE.value, params=params)
    
    @classmethod
    def route(
        cls,
        from_node: str,
        to: Union[str, Tuple[float, float, float]],
        via: Optional[List[Tuple[float, float, float]]] = None,
        algorithm: str = "astar_voxel",
        radius: Optional[float] = None,
        clearance: Optional[float] = None,
    ) -> "StepSpec":
        """Create a route step."""
        params = {
            "from": from_node,
            "to": to,
            "algorithm": algorithm,
        }
        if via is not None:
            params["via"] = via
        if radius is not None:
            params["radius"] = radius
        if clearance is not None:
            params["clearance"] = clearance
        return cls(op=StepOp.ROUTE.value, params=params)
    
    @classmethod
    def turn(
        cls,
        at: str,
        angle_deg: float,
        axis: str = "z",
    ) -> "StepSpec":
        """Create a turn step."""
        return cls(
            op=StepOp.TURN.value,
            params={"at": at, "angle_deg": angle_deg, "axis": axis},
        )
    
    @classmethod
    def bifurcate(
        cls,
        at: str,
        k: int = 2,
        branch_angles_deg: Optional[List[float]] = None,
        radius_split: str = "murray",
    ) -> "StepSpec":
        """Create a bifurcate step."""
        params = {"at": at, "k": k, "radius_split": radius_split}
        if branch_angles_deg is not None:
            params["branch_angles_deg"] = branch_angles_deg
        return cls(op=StepOp.BIFURCATE.value, params=params)
    
    @classmethod
    def merge(
        cls,
        nodes: List[str],
        into: str,
    ) -> "StepSpec":
        """Create a merge step."""
        return cls(
            op=StepOp.MERGE.value,
            params={"nodes": nodes, "into": into},
        )
    
    @classmethod
    def connect_to_outlet(
        cls,
        from_node: str,
        outlet: str,
        algorithm: str = "astar_voxel",
    ) -> "StepSpec":
        """Create a connect_to_outlet step."""
        return cls(
            op=StepOp.CONNECT_TO_OUTLET.value,
            params={"from": from_node, "outlet": outlet, "algorithm": algorithm},
        )


@dataclass
class StepResult:
    """Result of executing a single step."""
    success: bool
    step_index: int
    op: str
    created_node_ids: List[int] = field(default_factory=list)
    created_segment_ids: List[int] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationReport:
    """Report from programmatic generation."""
    success: bool
    mode: str
    steps_executed: int
    steps_total: int
    step_results: List[StepResult] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    collision_resolutions: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "mode": self.mode,
            "steps_executed": self.steps_executed,
            "steps_total": self.steps_total,
            "step_results": [
                {
                    "success": r.success,
                    "step_index": r.step_index,
                    "op": r.op,
                    "created_node_ids": r.created_node_ids,
                    "created_segment_ids": r.created_segment_ids,
                    "warnings": r.warnings,
                    "errors": r.errors,
                    "metadata": r.metadata,
                }
                for r in self.step_results
            ],
            "warnings": self.warnings,
            "errors": self.errors,
            "collision_resolutions": self.collision_resolutions,
            "metadata": self.metadata,
        }


class ProgrammaticBackend(GenerationBackend):
    """
    Backend for building vascular networks from JSON-friendly programmatic plans.
    
    This backend enables orchestrated generation where paths/graphs are built
    from a declarative DSL with pathfinding as one algorithm option and
    consistent collision-aware execution.
    
    Features:
    - Human-readable node IDs with internal numeric mapping
    - Multiple path algorithms (A*, straight, bezier, hybrid)
    - Waypoint-based routing with skip-on-failure
    - Collision detection and resolution (reroute/shrink/terminate)
    - Murray's Law bifurcation support
    - Merge operations for complex topologies
    
    Example usage:
        policy = ProgramPolicy(
            steps=[
                StepSpec.add_node("A", source="inlet[0]"),
                StepSpec.route("A", (0.01, 0.01, 0.01), algorithm="astar_voxel"),
                StepSpec.bifurcate("A", k=2, branch_angles_deg=[30, -30]),
            ]
        )
        backend = ProgrammaticBackend()
        network, report = backend.generate_from_program(domain, ports, policy)
    """
    
    def __init__(self):
        """Initialize the programmatic backend."""
        self._node_id_map: Dict[str, int] = {}
        self._reverse_id_map: Dict[int, str] = {}
        self._obstacles: List[Dict[str, Any]] = []
        self._current_network: Optional[VascularNetwork] = None
    
    @property
    def supports_dual_tree(self) -> bool:
        """Whether this backend can generate dual arterial-venous trees."""
        return False
    
    @property
    def supports_closed_loop(self) -> bool:
        """Whether this backend can generate closed-loop networks."""
        return True
    
    def generate(
        self,
        domain: DomainSpec,
        num_outlets: int,
        inlet_position: np.ndarray,
        inlet_radius: float,
        vessel_type: str = "arterial",
        config: Optional[BackendConfig] = None,
        rng_seed: Optional[int] = None,
    ) -> VascularNetwork:
        """
        Generate a vascular network using the standard backend interface.
        
        This method creates a simple network with the inlet and routes to
        randomly sampled outlet positions. For more control, use
        generate_from_program() instead.
        """
        if config is None:
            config = BackendConfig(seed=rng_seed)
        
        rng = np.random.default_rng(rng_seed)
        
        # Sample outlet positions
        outlet_positions = domain.sample_points(num_outlets, seed=rng_seed)
        
        # Build a simple program
        steps = [
            StepSpec.add_node(
                "inlet",
                position=tuple(inlet_position),
                node_type="inlet",
                radius=inlet_radius,
            )
        ]
        
        for i, pos in enumerate(outlet_positions):
            outlet_id = f"outlet_{i}"
            steps.append(
                StepSpec.route(
                    "inlet",
                    to=tuple(pos),
                    algorithm="straight",
                    radius=inlet_radius * 0.5,
                )
            )
        
        policy = ProgramPolicy(
            mode="network",
            steps=steps,
            path_algorithm="straight",
            default_radius=inlet_radius,
        )
        
        network, report = self.generate_from_program(domain, {}, policy)
        return network
    
    def generate_from_program(
        self,
        domain: DomainSpec,
        ports: Dict[str, Any],
        policy: ProgramPolicy,
    ) -> Tuple[Union[VascularNetwork, "trimesh.Trimesh"], GenerationReport]:
        """
        Generate a vascular network or mesh from a programmatic plan.
        
        Parameters
        ----------
        domain : DomainSpec
            Geometric domain for the network
        ports : dict
            Port configuration with "inlets" and "outlets"
        policy : ProgramPolicy
            Program policy containing steps and sub-policies
            
        Returns
        -------
        component : VascularNetwork or trimesh.Trimesh
            Generated component (network or mesh based on policy.mode)
        report : GenerationReport
            Detailed report of the generation process
        """
        # Reset state
        self._node_id_map.clear()
        self._reverse_id_map.clear()
        self._obstacles.clear()
        
        # Initialize network
        self._current_network = VascularNetwork(
            domain=domain,
            metadata={"generator": "programmatic_backend"},
        )
        
        # Store ports for reference
        self._ports = ports
        self._policy = policy
        
        # Execute steps
        step_results = []
        warnings = []
        errors = []
        collision_resolutions = []
        
        for i, step in enumerate(policy.steps):
            result = self._execute_step(i, step, policy)
            step_results.append(result)
            
            if not result.success:
                errors.extend(result.errors)
                if policy.retry_policy.max_retries > 0:
                    # Attempt retry
                    for retry in range(policy.retry_policy.max_retries):
                        logger.info(f"Retrying step {i} (attempt {retry + 1})")
                        result = self._execute_step(i, step, policy)
                        if result.success:
                            break
                
                if not result.success:
                    logger.error(f"Step {i} failed after retries: {result.errors}")
                    break
            
            warnings.extend(result.warnings)
            
            # Check collisions after step if enabled
            if policy.collision_policy.enabled and policy.collision_policy.check_after_each_step:
                collision_result = self._check_and_resolve_collisions(policy)
                if collision_result:
                    collision_resolutions.append(collision_result)
        
        # Build report
        success = all(r.success for r in step_results)
        report = GenerationReport(
            success=success,
            mode=policy.mode,
            steps_executed=len([r for r in step_results if r.success]),
            steps_total=len(policy.steps),
            step_results=step_results,
            warnings=warnings,
            errors=errors,
            collision_resolutions=collision_resolutions,
            metadata={
                "node_id_map": dict(self._node_id_map),
                "total_nodes": len(self._current_network.nodes),
                "total_segments": len(self._current_network.segments),
            },
        )
        
        # Return network or convert to mesh
        if policy.mode == "mesh":
            mesh = self._network_to_mesh()
            return mesh, report
        else:
            return self._current_network, report
    
    def generate_from_backend_params(
        self,
        domain: DomainSpec,
        ports: Dict[str, Any],
        backend_params: Dict[str, Any],
        collision_policy: Optional["CollisionPolicy"] = None,
        seed: Optional[int] = None,
    ) -> Tuple[Union[VascularNetwork, "trimesh.Trimesh"], GenerationReport]:
        """
        Generate a vascular network from backend_params dict.
        
        This method enables unified API usage where all backends can be
        configured through GrowthPolicy.backend_params. It validates required
        keys, uses defaults for missing keys, and emits effective_policy
        in the report.
        
        Parameters
        ----------
        domain : DomainSpec
            Geometric domain for the network
        ports : dict
            Port configuration with "inlets" and "outlets"
        backend_params : dict
            JSON-serializable backend configuration with keys:
            - mode: "network" | "mesh" (default: "network")
            - path_algorithm: "astar_voxel" | "straight" | "bezier" | "hybrid"
            - waypoint_policy: {allow_skip: bool, ...}
            - pathfinding_policy: {voxel_pitch, clearance, max_nodes, ...}
            - radius_policy: {...}
            - steps: [{op: ..., ...}, ...]
        collision_policy : CollisionPolicy, optional
            Collision policy from the unified API
        seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        component : VascularNetwork or trimesh.Trimesh
            Generated component (network or mesh based on mode)
        report : GenerationReport
            Detailed report with effective_policy reflecting defaults
        """
        # Validate and extract backend_params with defaults
        mode = backend_params.get("mode", "network")
        path_algorithm = backend_params.get("path_algorithm", "astar_voxel")
        
        # Build waypoint policy from backend_params or use defaults
        waypoint_params = backend_params.get("waypoint_policy", {})
        waypoint_policy = WaypointPolicy(
            skip_unreachable=waypoint_params.get("allow_skip", True),
            max_skip_count=waypoint_params.get("max_skip_count", 3),
            emit_warnings=waypoint_params.get("emit_warnings", True),
            fallback_direct=waypoint_params.get("fallback_direct", True),
        )
        
        # Build pathfinding/collision policy from backend_params
        pathfinding_params = backend_params.get("pathfinding_policy", {})
        default_clearance = pathfinding_params.get("clearance", 0.0002)
        
        # Check for collision_policy in backend_params (takes precedence for strategy_order)
        collision_policy_params = backend_params.get("collision_policy", {})
        
        # Use collision_policy from unified API if provided, but merge with backend_params
        if collision_policy is not None:
            prog_collision_policy = ProgramCollisionPolicy(
                enabled=collision_policy_params.get("enabled", collision_policy.check_collisions),
                min_clearance=collision_policy_params.get("min_clearance", collision_policy.collision_clearance),
                inflate_by_radius=collision_policy_params.get("inflate_by_radius", True),
                check_after_each_step=collision_policy_params.get("check_after_each_step", True),
                strategy_order=collision_policy_params.get("strategy_order", ["reroute", "shrink", "terminate"]),
                min_radius=collision_policy_params.get("min_radius", 0.0001),
                check_segment_segment=collision_policy_params.get("check_segment_segment", True),
                check_segment_boundary=collision_policy_params.get("check_segment_boundary", True),
                check_segment_mesh=collision_policy_params.get("check_segment_mesh", False),
            )
        else:
            prog_collision_policy = ProgramCollisionPolicy(
                enabled=collision_policy_params.get("enabled", pathfinding_params.get("check_collisions", True)),
                min_clearance=collision_policy_params.get("min_clearance", default_clearance),
                inflate_by_radius=collision_policy_params.get("inflate_by_radius", True),
                check_after_each_step=collision_policy_params.get("check_after_each_step", True),
                strategy_order=collision_policy_params.get("strategy_order", ["reroute", "shrink", "terminate"]),
                min_radius=collision_policy_params.get("min_radius", 0.0001),
                check_segment_segment=collision_policy_params.get("check_segment_segment", True),
                check_segment_boundary=collision_policy_params.get("check_segment_boundary", True),
                check_segment_mesh=collision_policy_params.get("check_segment_mesh", False),
            )
        
        # Build radius policy from backend_params
        radius_params = backend_params.get("radius_policy", {})
        radius_policy = RadiusPolicy(
            mode=radius_params.get("mode", "murray"),
            murray_exponent=radius_params.get("murray_exponent", 3.0),
            taper_factor=radius_params.get("taper_factor", 0.8),
            min_radius=radius_params.get("min_radius", 0.0001),
            max_radius=radius_params.get("max_radius", 0.005),
        )
        
        # Build retry policy
        retry_params = backend_params.get("retry_policy", {})
        retry_policy = RetryPolicy(
            max_retries=retry_params.get("max_retries", 3),
            backoff_factor=retry_params.get("backoff_factor", 1.5),
            retry_with_larger_clearance=retry_params.get("retry_with_larger_clearance", True),
            clearance_increase_factor=retry_params.get("clearance_increase_factor", 1.2),
        )
        
        # Get default radius from first inlet or use policy default
        inlets = ports.get("inlets", [])
        # Use ProgramPolicy's default_radius as the base default
        policy_default_radius = ProgramPolicy().default_radius
        default_radius = policy_default_radius
        if inlets:
            default_radius = inlets[0].get("radius", policy_default_radius)
        default_radius = backend_params.get("default_radius", default_radius)
        
        # Build steps from backend_params or generate default steps
        steps_data = backend_params.get("steps", [])
        if steps_data:
            steps = [StepSpec.from_dict(s) for s in steps_data]
        else:
            # Generate default steps: inlet -> outlets
            steps = []
            if inlets:
                inlet = inlets[0]
                inlet_position = tuple(inlet.get("position", (0, 0, 0)))
                inlet_radius = inlet.get("radius", default_radius)
                
                steps.append(
                    StepSpec.add_node(
                        "inlet",
                        position=inlet_position,
                        node_type="inlet",
                        radius=inlet_radius,
                    )
                )
                
                # Add routes to outlets
                outlets = ports.get("outlets", [])
                for i, outlet in enumerate(outlets):
                    outlet_pos = tuple(outlet.get("position", (0, 0, 0)))
                    outlet_radius = outlet.get("radius", inlet_radius * 0.5)
                    steps.append(
                        StepSpec.route(
                            "inlet",
                            to=outlet_pos,
                            algorithm=path_algorithm,
                            radius=outlet_radius,
                            clearance=default_clearance,
                        )
                    )
        
        # Build program policy
        policy = ProgramPolicy(
            mode=mode,
            steps=steps,
            path_algorithm=path_algorithm,
            collision_policy=prog_collision_policy,
            retry_policy=retry_policy,
            waypoint_policy=waypoint_policy,
            radius_policy=radius_policy,
            default_radius=default_radius,
            default_clearance=default_clearance,
        )
        
        # Generate using the standard method
        result, report = self.generate_from_program(domain, ports, policy)
        
        # Add effective_policy to report metadata
        report.metadata["effective_policy"] = policy.to_dict()
        report.metadata["backend_params_provided"] = backend_params
        
        return result, report
    
    def _execute_step(
        self,
        index: int,
        step: StepSpec,
        policy: ProgramPolicy,
    ) -> StepResult:
        """Execute a single step from the program."""
        op = step.op
        params = step.params
        
        try:
            if op == StepOp.ADD_NODE.value:
                return self._execute_add_node(index, params, policy)
            elif op == StepOp.ROUTE.value:
                return self._execute_route(index, params, policy)
            elif op == StepOp.TURN.value:
                return self._execute_turn(index, params, policy)
            elif op == StepOp.BIFURCATE.value:
                return self._execute_bifurcate(index, params, policy)
            elif op == StepOp.MERGE.value:
                return self._execute_merge(index, params, policy)
            elif op == StepOp.CONNECT_TO_OUTLET.value:
                return self._execute_connect_to_outlet(index, params, policy)
            else:
                return StepResult(
                    success=False,
                    step_index=index,
                    op=op,
                    errors=[f"Unknown operation: {op}"],
                )
        except Exception as e:
            logger.exception(f"Error executing step {index}")
            return StepResult(
                success=False,
                step_index=index,
                op=op,
                errors=[str(e)],
            )
    
    def _execute_add_node(
        self,
        index: int,
        params: Dict[str, Any],
        policy: ProgramPolicy,
    ) -> StepResult:
        """Execute an add_node step."""
        node_id_str = params.get("id", f"node_{index}")
        source = params.get("source")
        position = params.get("position")
        node_type = params.get("node_type", "junction")
        radius = params.get("radius", policy.default_radius)
        
        # Determine position
        if position is not None:
            pos = Point3D.from_tuple(position)
        elif source is not None:
            pos = self._resolve_source_position(source)
            if pos is None:
                return StepResult(
                    success=False,
                    step_index=index,
                    op=StepOp.ADD_NODE.value,
                    errors=[f"Could not resolve source: {source}"],
                )
        else:
            return StepResult(
                success=False,
                step_index=index,
                op=StepOp.ADD_NODE.value,
                errors=["Either position or source must be specified"],
            )
        
        # Create node
        numeric_id = self._current_network.id_gen.next_node_id()
        node = Node(
            id=numeric_id,
            position=pos,
            node_type=node_type,
            vessel_type="arterial",
            attributes={"radius": radius, "human_id": node_id_str},
        )
        
        self._current_network.add_node(node)
        self._node_id_map[node_id_str] = numeric_id
        self._reverse_id_map[numeric_id] = node_id_str
        
        return StepResult(
            success=True,
            step_index=index,
            op=StepOp.ADD_NODE.value,
            created_node_ids=[numeric_id],
            metadata={"human_id": node_id_str, "position": pos.to_tuple()},
        )
    
    def _execute_route(
        self,
        index: int,
        params: Dict[str, Any],
        policy: ProgramPolicy,
    ) -> StepResult:
        """Execute a route step."""
        from_node = params.get("from")
        to = params.get("to")
        via = params.get("via", [])
        algorithm = params.get("algorithm", policy.path_algorithm)
        radius = params.get("radius", policy.default_radius)
        clearance = params.get("clearance", policy.default_clearance)
        
        # Resolve from node
        from_numeric_id = self._node_id_map.get(from_node)
        if from_numeric_id is None:
            return StepResult(
                success=False,
                step_index=index,
                op=StepOp.ROUTE.value,
                errors=[f"Unknown from node: {from_node}"],
            )
        
        from_node_obj = self._current_network.get_node(from_numeric_id)
        start_pos = from_node_obj.position.to_array()
        
        # Resolve destination
        if isinstance(to, str):
            to_numeric_id = self._node_id_map.get(to)
            if to_numeric_id is None:
                return StepResult(
                    success=False,
                    step_index=index,
                    op=StepOp.ROUTE.value,
                    errors=[f"Unknown to node: {to}"],
                )
            to_node_obj = self._current_network.get_node(to_numeric_id)
            end_pos = to_node_obj.position.to_array()
        else:
            end_pos = np.array(to)
        
        # Build waypoints list
        waypoints = [np.array(wp) for wp in via]
        
        # Find path
        path_pts, path_warnings = self._find_path(
            start_pos,
            end_pos,
            waypoints,
            algorithm,
            clearance,
            radius,
            policy,
        )
        
        if path_pts is None or len(path_pts) < 2:
            return StepResult(
                success=False,
                step_index=index,
                op=StepOp.ROUTE.value,
                errors=["Failed to find path"],
                warnings=path_warnings,
            )
        
        # Create segments along path
        created_node_ids = []
        created_segment_ids = []
        current_node_id = from_numeric_id
        
        for i in range(1, len(path_pts)):
            pt = path_pts[i]
            
            # Create new node (except if it's the destination node)
            if isinstance(to, str) and i == len(path_pts) - 1:
                new_node_id = self._node_id_map.get(to)
            else:
                new_node_id = self._current_network.id_gen.next_node_id()
                new_node = Node(
                    id=new_node_id,
                    position=Point3D.from_array(pt),
                    node_type="junction" if i < len(path_pts) - 1 else "terminal",
                    vessel_type="arterial",
                    attributes={"radius": radius},
                )
                self._current_network.add_node(new_node)
                created_node_ids.append(new_node_id)
            
            # Create segment
            prev_node = self._current_network.get_node(current_node_id)
            curr_node = self._current_network.get_node(new_node_id)
            
            segment_id = self._current_network.id_gen.next_segment_id()
            segment = VesselSegment(
                id=segment_id,
                start_node_id=current_node_id,
                end_node_id=new_node_id,
                geometry=TubeGeometry(
                    start=prev_node.position,
                    end=curr_node.position,
                    radius_start=radius,
                    radius_end=radius,
                ),
                vessel_type="arterial",
            )
            self._current_network.add_segment(segment)
            created_segment_ids.append(segment_id)
            
            # Add to obstacles
            self._add_segment_obstacle(segment, clearance)
            
            current_node_id = new_node_id
        
        return StepResult(
            success=True,
            step_index=index,
            op=StepOp.ROUTE.value,
            created_node_ids=created_node_ids,
            created_segment_ids=created_segment_ids,
            warnings=path_warnings,
            metadata={
                "algorithm": algorithm,
                "path_length": len(path_pts),
                "waypoints_used": len(via),
            },
        )
    
    def _execute_turn(
        self,
        index: int,
        params: Dict[str, Any],
        policy: ProgramPolicy,
    ) -> StepResult:
        """Execute a turn step."""
        at = params.get("at")
        angle_deg = params.get("angle_deg", 0)
        axis = params.get("axis", "z")
        
        # Resolve node
        at_numeric_id = self._node_id_map.get(at)
        if at_numeric_id is None:
            return StepResult(
                success=False,
                step_index=index,
                op=StepOp.TURN.value,
                errors=[f"Unknown node: {at}"],
            )
        
        # Store turn information in node attributes
        node = self._current_network.get_node(at_numeric_id)
        if node is None:
            return StepResult(
                success=False,
                step_index=index,
                op=StepOp.TURN.value,
                errors=[f"Node not found: {at}"],
            )
        
        node.attributes["turn_angle_deg"] = angle_deg
        node.attributes["turn_axis"] = axis
        
        return StepResult(
            success=True,
            step_index=index,
            op=StepOp.TURN.value,
            metadata={"at": at, "angle_deg": angle_deg, "axis": axis},
        )
    
    def _execute_bifurcate(
        self,
        index: int,
        params: Dict[str, Any],
        policy: ProgramPolicy,
    ) -> StepResult:
        """Execute a bifurcate step."""
        at = params.get("at")
        k = params.get("k", 2)
        branch_angles_deg = params.get("branch_angles_deg")
        radius_split = params.get("radius_split", "murray")
        
        # Resolve node
        at_numeric_id = self._node_id_map.get(at)
        if at_numeric_id is None:
            return StepResult(
                success=False,
                step_index=index,
                op=StepOp.BIFURCATE.value,
                errors=[f"Unknown node: {at}"],
            )
        
        parent_node = self._current_network.get_node(at_numeric_id)
        if parent_node is None:
            return StepResult(
                success=False,
                step_index=index,
                op=StepOp.BIFURCATE.value,
                errors=[f"Node not found: {at}"],
            )
        
        # Get parent radius
        parent_radius = parent_node.attributes.get("radius", policy.default_radius)
        
        # Compute child radii using Murray's Law or other method
        if radius_split == "murray":
            child_radius = self._compute_murray_child_radius(
                parent_radius, k, policy.radius_policy.murray_exponent
            )
        else:
            child_radius = parent_radius * policy.radius_policy.taper_factor
        
        # Default branch angles if not specified
        if branch_angles_deg is None:
            branch_angles_deg = self._compute_default_branch_angles(k)
        
        # Get parent direction (from incoming segment)
        parent_direction = self._get_node_direction(at_numeric_id)
        if parent_direction is None:
            parent_direction = np.array([0, 0, -1])  # Default downward
        
        # Create branch nodes
        created_node_ids = []
        created_segment_ids = []
        branch_length = 0.002  # 2mm default branch length
        
        for i, angle_deg in enumerate(branch_angles_deg[:k]):
            # Rotate direction by angle
            angle_rad = np.radians(angle_deg)
            branch_direction = self._rotate_vector(parent_direction, angle_rad)
            
            # Compute branch endpoint
            branch_end = parent_node.position.to_array() + branch_direction * branch_length
            
            # Create branch node
            branch_id_str = f"{at}_branch_{i}"
            branch_numeric_id = self._current_network.id_gen.next_node_id()
            branch_node = Node(
                id=branch_numeric_id,
                position=Point3D.from_array(branch_end),
                node_type="junction",
                vessel_type="arterial",
                attributes={"radius": child_radius, "human_id": branch_id_str},
            )
            self._current_network.add_node(branch_node)
            self._node_id_map[branch_id_str] = branch_numeric_id
            self._reverse_id_map[branch_numeric_id] = branch_id_str
            created_node_ids.append(branch_numeric_id)
            
            # Create segment
            segment_id = self._current_network.id_gen.next_segment_id()
            segment = VesselSegment(
                id=segment_id,
                start_node_id=at_numeric_id,
                end_node_id=branch_numeric_id,
                geometry=TubeGeometry(
                    start=parent_node.position,
                    end=branch_node.position,
                    radius_start=parent_radius,
                    radius_end=child_radius,
                ),
                vessel_type="arterial",
            )
            self._current_network.add_segment(segment)
            created_segment_ids.append(segment_id)
            
            # Add to obstacles
            self._add_segment_obstacle(segment, policy.default_clearance)
        
        # Update parent node type
        parent_node.node_type = "junction"
        
        return StepResult(
            success=True,
            step_index=index,
            op=StepOp.BIFURCATE.value,
            created_node_ids=created_node_ids,
            created_segment_ids=created_segment_ids,
            metadata={
                "k": k,
                "branch_angles_deg": branch_angles_deg[:k],
                "parent_radius": parent_radius,
                "child_radius": child_radius,
            },
        )
    
    def _execute_merge(
        self,
        index: int,
        params: Dict[str, Any],
        policy: ProgramPolicy,
    ) -> StepResult:
        """Execute a merge step."""
        nodes = params.get("nodes", [])
        into = params.get("into")
        
        if len(nodes) < 2:
            return StepResult(
                success=False,
                step_index=index,
                op=StepOp.MERGE.value,
                errors=["Merge requires at least 2 nodes"],
            )
        
        # Resolve node IDs
        numeric_ids = []
        for node_id in nodes:
            numeric_id = self._node_id_map.get(node_id)
            if numeric_id is None:
                return StepResult(
                    success=False,
                    step_index=index,
                    op=StepOp.MERGE.value,
                    errors=[f"Unknown node: {node_id}"],
                )
            numeric_ids.append(numeric_id)
        
        # Compute merge position (centroid)
        positions = []
        radii = []
        for nid in numeric_ids:
            node = self._current_network.get_node(nid)
            positions.append(node.position.to_array())
            radii.append(node.attributes.get("radius", policy.default_radius))
        
        merge_pos = np.mean(positions, axis=0)
        merge_radius = max(radii)  # Use largest radius
        
        # Create merge node
        merge_numeric_id = self._current_network.id_gen.next_node_id()
        merge_node = Node(
            id=merge_numeric_id,
            position=Point3D.from_array(merge_pos),
            node_type="junction",
            vessel_type="arterial",
            attributes={"radius": merge_radius, "human_id": into},
        )
        self._current_network.add_node(merge_node)
        self._node_id_map[into] = merge_numeric_id
        self._reverse_id_map[merge_numeric_id] = into
        
        # Create segments from each source node to merge node
        created_segment_ids = []
        for nid in numeric_ids:
            source_node = self._current_network.get_node(nid)
            source_radius = source_node.attributes.get("radius", policy.default_radius)
            
            segment_id = self._current_network.id_gen.next_segment_id()
            segment = VesselSegment(
                id=segment_id,
                start_node_id=nid,
                end_node_id=merge_numeric_id,
                geometry=TubeGeometry(
                    start=source_node.position,
                    end=merge_node.position,
                    radius_start=source_radius,
                    radius_end=merge_radius,
                ),
                vessel_type="arterial",
            )
            self._current_network.add_segment(segment)
            created_segment_ids.append(segment_id)
            
            # Add to obstacles
            self._add_segment_obstacle(segment, policy.default_clearance)
        
        return StepResult(
            success=True,
            step_index=index,
            op=StepOp.MERGE.value,
            created_node_ids=[merge_numeric_id],
            created_segment_ids=created_segment_ids,
            metadata={
                "merged_nodes": nodes,
                "into": into,
                "merge_position": merge_pos.tolist(),
            },
        )
    
    def _execute_connect_to_outlet(
        self,
        index: int,
        params: Dict[str, Any],
        policy: ProgramPolicy,
    ) -> StepResult:
        """Execute a connect_to_outlet step."""
        from_node = params.get("from")
        outlet = params.get("outlet")
        algorithm = params.get("algorithm", policy.path_algorithm)
        
        # Resolve from node
        from_numeric_id = self._node_id_map.get(from_node)
        if from_numeric_id is None:
            return StepResult(
                success=False,
                step_index=index,
                op=StepOp.CONNECT_TO_OUTLET.value,
                errors=[f"Unknown from node: {from_node}"],
            )
        
        # Resolve outlet position
        outlet_pos = self._resolve_outlet_position(outlet)
        if outlet_pos is None:
            return StepResult(
                success=False,
                step_index=index,
                op=StepOp.CONNECT_TO_OUTLET.value,
                errors=[f"Could not resolve outlet: {outlet}"],
            )
        
        # Route to outlet
        route_params = {
            "from": from_node,
            "to": outlet_pos.to_tuple(),
            "algorithm": algorithm,
        }
        
        return self._execute_route(index, route_params, policy)
    
    def _resolve_source_position(self, source: str) -> Optional[Point3D]:
        """Resolve a source reference to a position."""
        # Handle inlet[n] format
        if source.startswith("inlet["):
            try:
                idx = int(source[6:-1])
                inlets = self._ports.get("inlets", [])
                if idx < len(inlets):
                    pos = inlets[idx].get("position")
                    if pos is not None:
                        return Point3D.from_tuple(pos)
            except (ValueError, IndexError):
                pass
        
        # Handle outlet[n] format
        if source.startswith("outlet["):
            try:
                idx = int(source[7:-1])
                outlets = self._ports.get("outlets", [])
                if idx < len(outlets):
                    pos = outlets[idx].get("position")
                    if pos is not None:
                        return Point3D.from_tuple(pos)
            except (ValueError, IndexError):
                pass
        
        # Handle node reference
        if source in self._node_id_map:
            numeric_id = self._node_id_map[source]
            node = self._current_network.get_node(numeric_id)
            if node is not None:
                return node.position
        
        return None
    
    def _resolve_outlet_position(self, outlet: str) -> Optional[Point3D]:
        """Resolve an outlet reference to a position."""
        # Handle outlet[n] format
        if outlet.startswith("outlet["):
            try:
                idx = int(outlet[7:-1])
                outlets = self._ports.get("outlets", [])
                if idx < len(outlets):
                    pos = outlets[idx].get("position")
                    if pos is not None:
                        return Point3D.from_tuple(pos)
            except (ValueError, IndexError):
                pass
        
        return None
    
    def _find_path(
        self,
        start: np.ndarray,
        end: np.ndarray,
        waypoints: List[np.ndarray],
        algorithm: str,
        clearance: float,
        radius: float,
        policy: ProgramPolicy,
    ) -> Tuple[Optional[List[np.ndarray]], List[str]]:
        """
        Find a path from start to end, optionally through waypoints.
        
        C1 FIX: All A* pathfinding requests are routed through hierarchical
        pathfinding module. Even if caller requests algorithm="astar_voxel",
        it will use hierarchical A* internally for scale-aware routing.
        
        Returns (path_points, warnings).
        """
        warnings = []
        
        if algorithm == "straight":
            # Simple straight-line path
            path_pts = [start]
            for wp in waypoints:
                path_pts.append(wp)
            path_pts.append(end)
            return path_pts, warnings
        
        elif algorithm == "astar_voxel":
            # C1 FIX: Route through hierarchical pathfinding (mandatory)
            # Even single-grid A* requests go through hierarchical module
            warnings.append("Using hierarchical pathfinding (C1: mandatory for all A* requests)")
            return self._find_path_astar(
                start, end, waypoints, clearance, radius, policy, warnings
            )
        
        elif algorithm == "bezier":
            # Bezier curve through waypoints
            return self._find_path_bezier(start, end, waypoints, warnings)
        
        elif algorithm == "hybrid":
            # C1 FIX: Try hierarchical A* first, fall back to straight
            path, astar_warnings = self._find_path_astar(
                start, end, waypoints, clearance, radius, policy, []
            )
            if path is not None:
                astar_warnings.append("Using hierarchical pathfinding (C1: mandatory)")
                return path, astar_warnings
            warnings.append("Hierarchical A* failed, falling back to straight path")
            path_pts = [start]
            for wp in waypoints:
                path_pts.append(wp)
            path_pts.append(end)
            return path_pts, warnings
        
        else:
            warnings.append(f"Unknown algorithm: {algorithm}, using straight")
            path_pts = [start]
            for wp in waypoints:
                path_pts.append(wp)
            path_pts.append(end)
            return path_pts, warnings
    
    def _find_path_astar(
        self,
        start: np.ndarray,
        end: np.ndarray,
        waypoints: List[np.ndarray],
        clearance: float,
        radius: float,
        policy: ProgramPolicy,
        warnings: List[str],
    ) -> Tuple[Optional[List[np.ndarray]], List[str]]:
        """
        Find path using A* algorithm.
        
        C4 FIX: Uses hierarchical pathfinding with policy-driven pitch selection
        and emits warnings/metrics for pitch relaxation.
        """
        # Build full waypoint list
        all_points = [start] + waypoints + [end]
        
        full_path = [start]
        skipped_waypoints = []
        pathfinding_metadata: List[Dict[str, Any]] = []
        
        for i in range(len(all_points) - 1):
            segment_start = all_points[i]
            segment_end = all_points[i + 1]
            
            # C4 FIX: _astar_segment now returns (path, metadata) tuple
            segment_path, segment_metadata = self._astar_segment(
                segment_start, segment_end, clearance, radius
            )
            pathfinding_metadata.append(segment_metadata)
            
            # C4 FIX: Collect warnings from segment pathfinding
            if segment_metadata.get("warnings"):
                warnings.extend(segment_metadata["warnings"])
            
            if segment_path is None:
                # Segment failed
                if i > 0 and i < len(all_points) - 1:
                    # This is a waypoint, try to skip it
                    if policy.waypoint_policy.skip_unreachable:
                        skipped_waypoints.append(i)
                        if policy.waypoint_policy.emit_warnings:
                            warnings.append(
                                f"Skipped waypoint {i}: unreachable"
                            )
                        continue
                
                # Can't skip start or end
                if policy.waypoint_policy.fallback_direct:
                    # Try direct path
                    direct_path, direct_metadata = self._astar_segment(start, end, clearance, radius)
                    if direct_metadata.get("warnings"):
                        warnings.extend(direct_metadata["warnings"])
                    if direct_path is not None:
                        warnings.append("Using direct path (waypoints skipped)")
                        return direct_path, warnings
                
                return None, warnings
            
            # Add segment path (skip first point to avoid duplicates)
            full_path.extend(segment_path[1:])
        
        # C4 FIX: Emit summary warnings for pitch relaxation
        any_pitch_relaxed = any(m.get("pitch_relaxed", False) for m in pathfinding_metadata)
        if any_pitch_relaxed:
            warnings.append("Pitch was relaxed during pathfinding due to voxel budget constraints")
        
        return full_path, warnings
    
    def _astar_segment(
        self,
        start: np.ndarray,
        end: np.ndarray,
        clearance: float,
        radius: float,
        resolution_policy: Optional["ResolutionPolicy"] = None,
    ) -> Tuple[Optional[List[np.ndarray]], Dict[str, Any]]:
        """
        Find path for a single segment using hierarchical A*.
        
        C4 FIX: Uses policy-driven hierarchical pathfinding with resolution
        resolver integration. Emits warnings and metrics for pitch relaxation.
        
        Uses the hierarchical pathfinding implementation for nontrivial
        obstacle avoidance. Falls back to direct path if collision-free.
        
        Returns
        -------
        tuple
            (path_points, metadata) where metadata includes warnings and metrics
        """
        metadata: Dict[str, Any] = {
            "warnings": [],
            "hierarchical_used": False,
            "pitch_relaxed": False,
        }
        
        # Check if direct path is collision-free
        if self._is_segment_collision_free(start, end, radius, clearance):
            return [start, end], metadata
        
        # Use hierarchical pathfinding for nontrivial cases
        from ..ops.pathfinding.hierarchical_astar import find_path_hierarchical
        from aog_policies.pathfinding import HierarchicalPathfindingPolicy as PolicyClass
        
        # Build obstacles list from current network
        obstacles = []
        for obs in self._obstacles:
            obstacles.append({
                "start": np.array(obs["start"]),
                "end": np.array(obs["end"]),
                "radius": obs["radius"],
            })
        
        # C4 FIX: Create policy from resolution policy if available
        if resolution_policy is not None:
            policy = PolicyClass.from_resolution_policy(
                resolution_policy,
                clearance=clearance,
                use_resolution_policy=True,
            )
        else:
            # Fallback to clearance-derived settings
            policy = PolicyClass(
                clearance=clearance,
                pitch_coarse=max(clearance * 2, 0.0001),  # At least 100µm
                pitch_fine=max(clearance / 2, 0.000005),  # At least 5µm
            )
        
        # Run hierarchical pathfinding
        result = find_path_hierarchical(
            domain=self._current_network.domain,
            start=start,
            goal=end,
            local_radius=radius,
            obstacles=obstacles,
            network=self._current_network,
            policy=policy,
            resolution_policy=resolution_policy,
        )
        
        # C4 FIX: Capture metadata from hierarchical pathfinding
        metadata["hierarchical_used"] = True
        metadata["coarse_nodes_explored"] = result.coarse_nodes_explored
        metadata["fine_nodes_explored"] = result.fine_nodes_explored
        metadata["effective_fine_pitch"] = result.effective_fine_pitch
        metadata["pitch_relaxed"] = result.pitch_was_relaxed
        metadata["corridor_voxels"] = result.corridor_voxels
        
        if result.warnings:
            metadata["warnings"].extend(result.warnings)
        
        if result.metadata:
            metadata["coarse_pitch"] = result.metadata.get("coarse_pitch")
            metadata["coarse_pitch_relaxed"] = result.metadata.get("coarse_pitch_relaxed", False)
        
        if result.success and result.path_pts:
            return result.path_pts, metadata
        
        # Log warning if pathfinding failed
        if result.errors:
            logger.warning(f"Hierarchical pathfinding failed: {result.errors}")
            metadata["errors"] = result.errors
        
        return None, metadata
    
    def _find_path_bezier(
        self,
        start: np.ndarray,
        end: np.ndarray,
        waypoints: List[np.ndarray],
        warnings: List[str],
    ) -> Tuple[Optional[List[np.ndarray]], List[str]]:
        """Find path using Bezier curve."""
        control_points = [start] + waypoints + [end]
        
        # Generate Bezier curve points
        n_samples = max(10, len(control_points) * 5)
        path_pts = []
        
        for t in np.linspace(0, 1, n_samples):
            pt = self._bezier_point(control_points, t)
            path_pts.append(pt)
        
        return path_pts, warnings
    
    def _bezier_point(
        self,
        control_points: List[np.ndarray],
        t: float,
    ) -> np.ndarray:
        """Compute point on Bezier curve at parameter t."""
        points = [np.array(p) for p in control_points]
        n = len(points) - 1
        
        # De Casteljau's algorithm
        while len(points) > 1:
            new_points = []
            for i in range(len(points) - 1):
                new_points.append((1 - t) * points[i] + t * points[i + 1])
            points = new_points
        
        return points[0]
    
    def _is_segment_collision_free(
        self,
        start: np.ndarray,
        end: np.ndarray,
        radius: float,
        clearance: float,
    ) -> bool:
        """Check if a segment is collision-free with existing obstacles."""
        for obstacle in self._obstacles:
            obs_start = np.array(obstacle["start"])
            obs_end = np.array(obstacle["end"])
            obs_radius = obstacle["radius"]
            
            # Check capsule-capsule collision
            dist = self._segment_to_segment_distance(start, end, obs_start, obs_end)
            min_dist = radius + obs_radius + clearance
            
            if dist < min_dist:
                return False
        
        return True
    
    def _segment_to_segment_distance(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        q1: np.ndarray,
        q2: np.ndarray,
    ) -> float:
        """Compute minimum distance between two line segments."""
        d1 = p2 - p1
        d2 = q2 - q1
        r = p1 - q1
        
        a = np.dot(d1, d1)
        b = np.dot(d1, d2)
        c = np.dot(d2, d2)
        d = np.dot(d1, r)
        e = np.dot(d2, r)
        
        denom = a * c - b * b
        
        if abs(denom) < 1e-10:
            s = 0.0
            t = d / a if abs(a) > 1e-10 else 0.0
        else:
            s = (b * d - a * e) / denom
            t = (c * d - b * e) / denom
        
        s = np.clip(s, 0, 1)
        t = np.clip(t, 0, 1)
        
        closest_p = p1 + t * d1
        closest_q = q1 + s * d2
        
        return float(np.linalg.norm(closest_p - closest_q))
    
    def _add_segment_obstacle(
        self,
        segment: VesselSegment,
        clearance: float,
    ) -> None:
        """Add a segment to the obstacle list."""
        self._obstacles.append({
            "start": segment.geometry.start.to_tuple(),
            "end": segment.geometry.end.to_tuple(),
            "radius": segment.geometry.mean_radius() + clearance,
            "segment_id": segment.id,
        })
    
    def _compute_murray_child_radius(
        self,
        parent_radius: float,
        k: int,
        exponent: float = 3.0,
    ) -> float:
        """Compute child radius using Murray's Law."""
        # r_parent^n = k * r_child^n
        # r_child = r_parent / k^(1/n)
        return parent_radius / (k ** (1.0 / exponent))
    
    def _compute_default_branch_angles(self, k: int) -> List[float]:
        """Compute default branch angles for k-way bifurcation."""
        if k == 2:
            return [30.0, -30.0]
        else:
            # Distribute evenly
            angles = []
            for i in range(k):
                angle = (i - (k - 1) / 2) * (60.0 / (k - 1)) if k > 1 else 0
                angles.append(angle)
            return angles
    
    def _get_node_direction(self, node_id: int) -> Optional[np.ndarray]:
        """Get the direction vector at a node from incoming segments."""
        connected_segments = self._current_network.get_connected_segment_ids(node_id)
        
        if not connected_segments:
            return None
        
        # Use the first incoming segment's direction
        for seg_id in connected_segments:
            segment = self._current_network.get_segment(seg_id)
            if segment.end_node_id == node_id:
                # This segment ends at our node
                start = segment.geometry.start.to_array()
                end = segment.geometry.end.to_array()
                direction = end - start
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    return direction / norm
        
        return None
    
    def _rotate_vector(
        self,
        vector: np.ndarray,
        angle: float,
        axis: str = "z",
    ) -> np.ndarray:
        """Rotate a vector around an axis."""
        c = np.cos(angle)
        s = np.sin(angle)
        
        if axis == "z":
            R = np.array([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1],
            ])
        elif axis == "y":
            R = np.array([
                [c, 0, s],
                [0, 1, 0],
                [-s, 0, c],
            ])
        elif axis == "x":
            R = np.array([
                [1, 0, 0],
                [0, c, -s],
                [0, s, c],
            ])
        else:
            return vector
        
        return R @ vector
    
    def _check_and_resolve_collisions(
        self,
        policy: ProgramPolicy,
    ) -> Optional[Dict[str, Any]]:
        """Check for collisions and resolve using policy strategies."""
        if not policy.collision_policy.enabled:
            return None
        
        from ..ops.collision import get_collisions
        
        result = get_collisions(
            self._current_network,
            min_clearance=policy.collision_policy.min_clearance,
        )
        
        collisions = result.metadata.get("collisions", [])
        if not collisions:
            return None
        
        resolution_result = {
            "collisions_found": len(collisions),
            "strategies_tried": [],
            "resolved": False,
        }
        
        for strategy in policy.collision_policy.strategy_order:
            if strategy == "reroute":
                # Attempt reroute (would use pathfinding)
                resolution_result["strategies_tried"].append("reroute")
                # Placeholder - full implementation would reroute
                
            elif strategy == "shrink":
                # Attempt shrink
                resolution_result["strategies_tried"].append("shrink")
                # Shrink radii of colliding segments
                for seg_id_a, seg_id_b, distance in collisions:
                    for seg_id in [seg_id_a, seg_id_b]:
                        segment = self._current_network.get_segment(seg_id)
                        if segment:
                            new_radius = max(
                                segment.geometry.radius_start * 0.9,
                                policy.collision_policy.min_radius,
                            )
                            segment.geometry.radius_start = new_radius
                            segment.geometry.radius_end = new_radius
                
            elif strategy == "terminate":
                # Mark as terminated
                resolution_result["strategies_tried"].append("terminate")
                resolution_result["resolved"] = True
                break
        
        return resolution_result
    
    def _network_to_mesh(self) -> "trimesh.Trimesh":
        """Convert the current network to a mesh."""
        from ..ops.mesh.synthesis import synthesize_mesh
        from ...aog_policies import MeshSynthesisPolicy
        
        policy = MeshSynthesisPolicy()
        mesh, report = synthesize_mesh(self._current_network, policy)
        return mesh


__all__ = [
    "ProgrammaticBackend",
    "ProgramPolicy",
    "StepSpec",
    "StepOp",
    "PathAlgorithm",
    "CollisionStrategy",
    "WaypointPolicy",
    "RadiusPolicy",
    "RetryPolicy",
    "ProgramCollisionPolicy",
    "StepResult",
    "GenerationReport",
]
