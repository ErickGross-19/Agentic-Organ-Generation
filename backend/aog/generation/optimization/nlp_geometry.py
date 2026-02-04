"""
NLP-based global geometry optimization for vascular networks.

Implements Jessen-style optimization where node positions, radii,
and pressures are optimized to minimize total vessel volume while
satisfying physiological constraints.

Decision variables:
- Node coordinates (x_v in R^3) for non-fixed nodes
- Segment radii (r_e)
- Segment lengths (l_e)
- Node pressures (p_v)

Constraints:
- Geometry consistency: l_e = |x_u - x_v|
- Murray's law at bifurcations
- Poiseuille pressure drop per segment
- Kirchhoff flow conservation
- Homogeneous terminal flow distribution
- Total pressure drop constraint
- Bounds on radii and lengths

Objective: Minimize total vessel volume = sum(pi * r_e^2 * l_e)

Note: The library uses METERS internally for all geometry.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
import numpy as np
import math

from ..core.network import VascularNetwork, Node, VesselSegment
from ..core.types import Point3D


@dataclass
class NLPConfig:
    """Configuration for NLP geometry optimization."""
    
    min_radius: float = 1e-4
    max_radius: float = 0.01
    min_length: float = 1e-4
    max_length: float = 0.1
    
    min_pressure: float = 0.0
    max_pressure: float = 20000.0
    
    target_pressure_drop: float = 13332.0
    
    murray_exponent: float = 3.0
    murray_tolerance: float = 0.1
    
    viscosity: float = 0.0035
    
    fix_terminal_positions: bool = True
    fix_root_position: bool = True
    fix_root_subtree_depth: int = 0
    
    solver_tolerance: float = 1e-6
    max_iterations: int = 1000
    
    cleanup_degenerate_segments: bool = True
    degenerate_length_factor: float = 2.0
    
    # Convergence improvement options
    scale_constraints: bool = True  # Scale constraints to improve conditioning
    position_margin_factor: float = 0.5  # Margin factor for position bounds (0.5 = 50% of network size)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "min_radius": self.min_radius,
            "max_radius": self.max_radius,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "min_pressure": self.min_pressure,
            "max_pressure": self.max_pressure,
            "target_pressure_drop": self.target_pressure_drop,
            "murray_exponent": self.murray_exponent,
            "murray_tolerance": self.murray_tolerance,
            "viscosity": self.viscosity,
            "fix_terminal_positions": self.fix_terminal_positions,
            "fix_root_position": self.fix_root_position,
            "solver_tolerance": self.solver_tolerance,
            "max_iterations": self.max_iterations,
        }


@dataclass
class NLPResult:
    """Result of NLP geometry optimization."""
    
    success: bool = False
    
    initial_volume: float = 0.0
    final_volume: float = 0.0
    volume_reduction: float = 0.0
    
    iterations: int = 0
    final_objective: float = 0.0
    constraint_violation: float = 0.0
    
    optimized_positions: Optional[Dict[int, np.ndarray]] = None
    optimized_radii: Optional[Dict[int, float]] = None
    optimized_pressures: Optional[Dict[int, float]] = None
    
    trifurcations_created: int = 0
    segments_removed: int = 0
    
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "initial_volume": self.initial_volume,
            "final_volume": self.final_volume,
            "volume_reduction": self.volume_reduction,
            "iterations": self.iterations,
            "final_objective": self.final_objective,
            "constraint_violation": self.constraint_violation,
            "trifurcations_created": self.trifurcations_created,
            "segments_removed": self.segments_removed,
            "warnings": self.warnings,
            "errors": self.errors,
        }


class NLPProblem:
    """
    NLP problem formulation for vascular network optimization.
    
    Builds the optimization problem with decision variables, constraints,
    and objective function.
    """
    
    def __init__(self, network: VascularNetwork, config: NLPConfig):
        self.network = network
        self.config = config
        
        self._fixed_node_ids: Set[int] = set()
        self._variable_node_ids: List[int] = []
        self._segment_ids: List[int] = []
        
        self._n_position_vars = 0
        self._n_radius_vars = 0
        self._n_pressure_vars = 0
        self._n_vars = 0
        
        self._node_to_pos_idx: Dict[int, int] = {}
        self._seg_to_radius_idx: Dict[int, int] = {}
        self._node_to_pressure_idx: Dict[int, int] = {}
        
        self._inlet_node_ids: List[int] = []
        self._outlet_node_ids: List[int] = []
        self._junction_node_ids: List[int] = []
        
        self._build_variable_mapping()
    
    def _build_variable_mapping(self) -> None:
        """Build mapping from network elements to optimization variables."""
        for node in self.network.nodes.values():
            if node.node_type == "inlet":
                self._inlet_node_ids.append(node.id)
                if self.config.fix_root_position:
                    self._fixed_node_ids.add(node.id)
            elif node.node_type in ["outlet", "terminal"]:
                self._outlet_node_ids.append(node.id)
                if self.config.fix_terminal_positions:
                    self._fixed_node_ids.add(node.id)
            elif node.node_type == "junction":
                self._junction_node_ids.append(node.id)
        
        idx = 0
        for node_id in self.network.nodes.keys():
            if node_id not in self._fixed_node_ids:
                self._variable_node_ids.append(node_id)
                self._node_to_pos_idx[node_id] = idx
                idx += 3
        self._n_position_vars = idx
        
        for seg_id in self.network.segments.keys():
            self._segment_ids.append(seg_id)
            self._seg_to_radius_idx[seg_id] = idx
            idx += 1
        self._n_radius_vars = len(self._segment_ids)
        
        for node_id in self.network.nodes.keys():
            self._node_to_pressure_idx[node_id] = idx
            idx += 1
        self._n_pressure_vars = len(self.network.nodes)
        
        self._n_vars = idx
    
    def get_initial_x(self) -> np.ndarray:
        """Get initial variable values from current network state."""
        x = np.zeros(self._n_vars)
        
        for node_id in self._variable_node_ids:
            node = self.network.nodes[node_id]
            idx = self._node_to_pos_idx[node_id]
            x[idx:idx + 3] = node.position.to_array()
        
        for seg_id in self._segment_ids:
            segment = self.network.segments[seg_id]
            idx = self._seg_to_radius_idx[seg_id]
            x[idx] = segment.geometry.mean_radius()
        
        pressure_offset = self._n_position_vars + self._n_radius_vars
        inlet_pressure = self.config.max_pressure * 0.8
        outlet_pressure = inlet_pressure - self.config.target_pressure_drop
        
        for node_id, node in self.network.nodes.items():
            idx = self._node_to_pressure_idx[node_id]
            if node.node_type == "inlet":
                x[idx] = inlet_pressure
            elif node.node_type in ["outlet", "terminal"]:
                x[idx] = max(outlet_pressure, self.config.min_pressure)
            else:
                x[idx] = (inlet_pressure + outlet_pressure) / 2
        
        return x
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get variable bounds."""
        lb = np.full(self._n_vars, -np.inf)
        ub = np.full(self._n_vars, np.inf)
        
        # Compute network bounding box for position bounds
        all_positions = np.array([
            node.position.to_array() for node in self.network.nodes.values()
        ])
        network_min = all_positions.min(axis=0)
        network_max = all_positions.max(axis=0)
        network_size = network_max - network_min
        
        # Add margin to allow node movement during optimization
        # Use config.position_margin_factor (default 0.5 = 50% of network size)
        margin = np.maximum(network_size * self.config.position_margin_factor, 0.01)  # At least 1cm margin
        pos_lb = network_min - margin
        pos_ub = network_max + margin
        
        for node_id in self._variable_node_ids:
            idx = self._node_to_pos_idx[node_id]
            lb[idx:idx + 3] = pos_lb
            ub[idx:idx + 3] = pos_ub
        
        for seg_id in self._segment_ids:
            idx = self._seg_to_radius_idx[seg_id]
            lb[idx] = self.config.min_radius
            ub[idx] = self.config.max_radius
        
        for node_id in self.network.nodes.keys():
            idx = self._node_to_pressure_idx[node_id]
            lb[idx] = self.config.min_pressure
            ub[idx] = self.config.max_pressure
        
        return lb, ub
    
    def objective(self, x: np.ndarray) -> float:
        """
        Compute objective function: total vessel volume.
        
        Volume = sum(pi * r_e^2 * l_e)
        """
        total_volume = 0.0
        
        for seg_id in self._segment_ids:
            segment = self.network.segments[seg_id]
            
            r_idx = self._seg_to_radius_idx[seg_id]
            radius = x[r_idx]
            
            length = self._compute_segment_length(x, segment)
            
            total_volume += math.pi * radius ** 2 * length
        
        return total_volume
    
    def objective_gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute gradient of objective function."""
        grad = np.zeros(self._n_vars)
        
        for seg_id in self._segment_ids:
            segment = self.network.segments[seg_id]
            
            r_idx = self._seg_to_radius_idx[seg_id]
            radius = x[r_idx]
            
            length = self._compute_segment_length(x, segment)
            
            grad[r_idx] = 2 * math.pi * radius * length
            
            start_pos = self._get_node_position(x, segment.start_node_id)
            end_pos = self._get_node_position(x, segment.end_node_id)
            
            if length > 1e-10:
                direction = (end_pos - start_pos) / length
                
                if segment.start_node_id in self._node_to_pos_idx:
                    idx = self._node_to_pos_idx[segment.start_node_id]
                    grad[idx:idx + 3] -= math.pi * radius ** 2 * direction
                
                if segment.end_node_id in self._node_to_pos_idx:
                    idx = self._node_to_pos_idx[segment.end_node_id]
                    grad[idx:idx + 3] += math.pi * radius ** 2 * direction
        
        return grad
    
    def constraints(self, x: np.ndarray) -> np.ndarray:
        """
        Compute constraint violations.
        
        Returns array of constraint values (should be zero for equality,
        non-negative for inequality constraints).
        """
        constraints = []
        
        for node_id in self._junction_node_ids:
            connected_segs = self.network.get_connected_segment_ids(node_id)
            if len(connected_segs) >= 2:
                radii = []
                for seg_id in connected_segs:
                    r_idx = self._seg_to_radius_idx[seg_id]
                    radii.append(x[r_idx])
                
                radii = sorted(radii, reverse=True)
                parent_r = radii[0]
                child_radii = radii[1:]
                
                gamma = self.config.murray_exponent
                lhs = parent_r ** gamma
                rhs = sum(r ** gamma for r in child_radii)
                
                constraints.append(lhs - rhs)
        
        for seg_id in self._segment_ids:
            segment = self.network.segments[seg_id]
            
            r_idx = self._seg_to_radius_idx[seg_id]
            radius = x[r_idx]
            length = self._compute_segment_length(x, segment)
            
            p_start_idx = self._node_to_pressure_idx[segment.start_node_id]
            p_end_idx = self._node_to_pressure_idx[segment.end_node_id]
            p_start = x[p_start_idx]
            p_end = x[p_end_idx]
            
            if radius > 0 and length > 0:
                R = (8 * self.config.viscosity * length) / (math.pi * radius ** 4)
                flow = (p_start - p_end) / R if R > 0 else 0.0
            else:
                flow = 0.0
            
            constraints.append(flow)
        
        for node_id in self._junction_node_ids:
            connected_segs = self.network.get_connected_segment_ids(node_id)
            
            inflow = 0.0
            outflow = 0.0
            
            for seg_id in connected_segs:
                segment = self.network.segments[seg_id]
                r_idx = self._seg_to_radius_idx[seg_id]
                radius = x[r_idx]
                length = self._compute_segment_length(x, segment)
                
                p_start_idx = self._node_to_pressure_idx[segment.start_node_id]
                p_end_idx = self._node_to_pressure_idx[segment.end_node_id]
                p_start = x[p_start_idx]
                p_end = x[p_end_idx]
                
                if radius > 0 and length > 0:
                    R = (8 * self.config.viscosity * length) / (math.pi * radius ** 4)
                    flow = (p_start - p_end) / R if R > 0 else 0.0
                else:
                    flow = 0.0
                
                if segment.start_node_id == node_id:
                    outflow += abs(flow)
                else:
                    inflow += abs(flow)
            
            constraints.append(inflow - outflow)
        
        return np.array(constraints)
    
    def _compute_segment_length(self, x: np.ndarray, segment: VesselSegment) -> float:
        """Compute segment length from current variable values."""
        start_pos = self._get_node_position(x, segment.start_node_id)
        end_pos = self._get_node_position(x, segment.end_node_id)
        return float(np.linalg.norm(end_pos - start_pos))
    
    def _get_node_position(self, x: np.ndarray, node_id: int) -> np.ndarray:
        """Get node position from variables or fixed position."""
        if node_id in self._node_to_pos_idx:
            idx = self._node_to_pos_idx[node_id]
            return x[idx:idx + 3]
        else:
            return self.network.nodes[node_id].position.to_array()
    
    def apply_solution(self, x: np.ndarray) -> None:
        """Apply optimized solution to network."""
        for node_id in self._variable_node_ids:
            idx = self._node_to_pos_idx[node_id]
            pos = x[idx:idx + 3]
            self.network.nodes[node_id].position = Point3D(pos[0], pos[1], pos[2])
        
        for seg_id in self._segment_ids:
            idx = self._seg_to_radius_idx[seg_id]
            radius = x[idx]
            segment = self.network.segments[seg_id]
            segment.geometry.radius_start = radius
            segment.geometry.radius_end = radius
            segment.attributes["radius"] = radius
        
        for node_id in self.network.nodes.keys():
            idx = self._node_to_pressure_idx[node_id]
            pressure = x[idx]
            self.network.nodes[node_id].attributes["pressure"] = pressure


def build_nlp_problem(
    network: VascularNetwork,
    config: Optional[NLPConfig] = None,
) -> NLPProblem:
    """
    Build NLP problem for network optimization.
    
    Parameters
    ----------
    network : VascularNetwork
        Network to optimize
    config : NLPConfig, optional
        Optimization configuration
        
    Returns
    -------
    NLPProblem
        Configured NLP problem
    """
    if config is None:
        config = NLPConfig()
    
    return NLPProblem(network, config)


def optimize_geometry(
    network: VascularNetwork,
    config: Optional[NLPConfig] = None,
) -> NLPResult:
    """
    Optimize network geometry using NLP.
    
    Minimizes total vessel volume while satisfying physiological
    constraints (Murray's law, Poiseuille flow, Kirchhoff conservation).
    
    Parameters
    ----------
    network : VascularNetwork
        Network to optimize (modified in place)
    config : NLPConfig, optional
        Optimization configuration
        
    Returns
    -------
    NLPResult
        Optimization result
    """
    if config is None:
        config = NLPConfig()
    
    result = NLPResult()
    
    result.initial_volume = _compute_total_volume(network)
    
    problem = NLPProblem(network, config)
    
    x0 = problem.get_initial_x()
    lb, ub = problem.get_bounds()
    
    from .solvers import solve_nlp, SolverConfig, is_ipopt_available
    
    solver_config = SolverConfig(
        solver="ipopt" if is_ipopt_available() else "scipy",
        method="SLSQP",
        tolerance=config.solver_tolerance,
        max_iterations=config.max_iterations,
    )
    
    if not is_ipopt_available():
        result.warnings.append("IPOPT not available, using scipy fallback")
    
    try:
        solver_result = solve_nlp(
            objective=problem.objective,
            x0=x0,
            bounds=(lb, ub),
            gradient=problem.objective_gradient,
            constraints=problem.constraints,
            config=solver_config,
        )
        x_opt = solver_result.x
        success = solver_result.success
        iterations = solver_result.iterations
    except Exception as e:
        result.errors.append(f"Optimization failed: {e}")
        return result
    
    if success:
        problem.apply_solution(x_opt)
        
        if config.cleanup_degenerate_segments:
            removed, trifurcations = _cleanup_degenerate_segments(network, config)
            result.segments_removed = removed
            result.trifurcations_created = trifurcations
        
        result.final_volume = _compute_total_volume(network)
        result.volume_reduction = (result.initial_volume - result.final_volume) / result.initial_volume
        result.final_objective = problem.objective(x_opt)
        result.constraint_violation = float(np.max(np.abs(problem.constraints(x_opt))))
        result.iterations = iterations
        result.success = True
    else:
        result.errors.append("Optimization did not converge")
    
    return result


def _compute_total_volume(network: VascularNetwork) -> float:
    """Compute total vessel volume."""
    total = 0.0
    for segment in network.segments.values():
        r = segment.geometry.mean_radius()
        L = segment.geometry.length()
        total += math.pi * r ** 2 * L
    return total


def _cleanup_degenerate_segments(
    network: VascularNetwork,
    config: NLPConfig,
) -> Tuple[int, int]:
    """
    Remove degenerate segments (length < diameter).
    
    When a segment becomes shorter than its diameter, delete it
    and merge its endpoints. This may create trifurcations.
    
    Returns
    -------
    tuple
        (segments_removed, trifurcations_created)
    """
    segments_removed = 0
    trifurcations_created = 0
    
    segments_to_remove = []
    
    for seg_id, segment in network.segments.items():
        length = segment.geometry.length()
        diameter = 2 * segment.geometry.mean_radius()
        
        if length < diameter * config.degenerate_length_factor:
            if network.nodes[segment.end_node_id].node_type not in ["inlet", "outlet", "terminal"]:
                segments_to_remove.append(seg_id)
    
    for seg_id in segments_to_remove:
        if seg_id not in network.segments:
            continue
        
        segment = network.segments[seg_id]
        start_node_id = segment.start_node_id
        end_node_id = segment.end_node_id
        
        end_connected = network.get_connected_segment_ids(end_node_id)
        end_connected = [sid for sid in end_connected if sid != seg_id]
        
        for other_seg_id in end_connected:
            other_seg = network.segments[other_seg_id]
            if other_seg.start_node_id == end_node_id:
                other_seg.start_node_id = start_node_id
            elif other_seg.end_node_id == end_node_id:
                other_seg.end_node_id = start_node_id
        
        del network.segments[seg_id]
        segments_removed += 1
        
        if end_node_id in network.nodes:
            del network.nodes[end_node_id]
        
        start_connected = network.get_connected_segment_ids(start_node_id)
        if len(start_connected) > 2:
            trifurcations_created += 1
            network.nodes[start_node_id].node_type = "junction"
    
    return segments_removed, trifurcations_created
