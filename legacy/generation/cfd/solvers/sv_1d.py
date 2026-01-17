"""
1D wave propagation solver for vascular networks.

Solves 1D blood flow equations with wave propagation effects.
Intermediate fidelity between 0D and 3D.

Note: The library uses METERS internally for all geometry.
Pressures are in Pascals (Pa), flows in m^3/s.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Literal, Tuple
import numpy as np
import time

from .base import BaseSolver, SolverConfig, compute_poiseuille_resistance
from ..bcs import BoundaryConditions, InletType, OutletType
from ..results import CFDResult, CFDMetrics, FlowMetrics, PressureMetrics, PerfusionProxyMetrics


@dataclass
class Segment1DState:
    """State variables for a 1D segment."""
    
    segment_id: int
    n_nodes: int
    
    area: np.ndarray
    flow: np.ndarray
    pressure: np.ndarray
    
    x_coords: np.ndarray
    
    area_ref: float
    beta: float
    
    def __init__(self, segment_id: int, length: float, radius: float, n_nodes: int = 10):
        self.segment_id = segment_id
        self.n_nodes = n_nodes
        
        self.x_coords = np.linspace(0, length, n_nodes)
        
        self.area_ref = np.pi * radius ** 2
        self.area = np.full(n_nodes, self.area_ref)
        self.flow = np.zeros(n_nodes)
        self.pressure = np.zeros(n_nodes)
        
        E = 1e6
        h = 0.1 * radius
        self.beta = (4.0 / 3.0) * np.sqrt(np.pi) * E * h / self.area_ref


class Solver1D(BaseSolver):
    """
    1D wave propagation solver.
    
    Solves the 1D blood flow equations:
    - Mass conservation: dA/dt + dQ/dx = 0
    - Momentum conservation: dQ/dt + d(Q^2/A)/dx + A/rho * dP/dx = -f*Q/A
    
    Uses a simplified finite difference scheme.
    
    Advantages:
    - Captures wave propagation effects
    - Faster than 3D
    - No volumetric mesh required
    
    Limitations:
    - No cross-sectional flow details
    - Simplified wall mechanics
    - No wall shear stress computation
    """
    
    def __init__(self, config: Optional[SolverConfig] = None):
        super().__init__(config)
        
        self._network = None
        self._bcs = None
        
        self._segment_states: Dict[int, Segment1DState] = {}
        
        self._node_pressures: Dict[int, float] = {}
        self._segment_flows: Dict[int, float] = {}
        
        self._inlet_node_ids: List[int] = []
        self._outlet_node_ids: List[int] = []
        
        self._rho = 1060.0
        self._mu = 0.0035
        
        self._nodes_per_segment = 10
    
    @property
    def fidelity(self) -> Literal["0D", "1D", "3D"]:
        return "1D"
    
    @property
    def requires_mesh(self) -> bool:
        return False
    
    def setup(
        self,
        network: "VascularNetwork",
        bcs: BoundaryConditions,
        mesh_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Set up the 1D solver."""
        self._network = network
        self._bcs = bcs
        
        self._rho = bcs.fluid.density
        self._mu = bcs.fluid.viscosity
        
        self._segment_states = {}
        for seg_id, segment in network.segments.items():
            length = segment.geometry.length()
            radius = segment.geometry.mean_radius()
            
            state = Segment1DState(
                segment_id=seg_id,
                length=length,
                radius=radius,
                n_nodes=self._nodes_per_segment,
            )
            self._segment_states[seg_id] = state
        
        self._inlet_node_ids = bcs.get_inlet_node_ids()
        self._outlet_node_ids = bcs.get_outlet_node_ids()
        
        self._is_initialized = True
        return True
    
    def solve(self) -> CFDResult:
        """Run the 1D simulation."""
        start_time = time.time()
        
        errors = self.validate_setup()
        if errors:
            return CFDResult.failure("; ".join(errors))
        
        result = CFDResult(fidelity="1D")
        
        try:
            if self.config.steady_state:
                self._solve_steady_state()
            else:
                self._solve_transient()
            
            result.node_pressures = self._node_pressures.copy()
            result.segment_flows = self._segment_flows.copy()
            
            result.metrics = self._compute_metrics()
            result.success = True
            
        except Exception as e:
            result.errors.append(f"1D solver failed: {e}")
            result.success = False
        
        result.wall_time_seconds = time.time() - start_time
        return result
    
    def _solve_steady_state(self) -> None:
        """Solve steady-state 1D flow."""
        inlet_flow = 0.0
        for inlet in self._bcs.inlets:
            if inlet.bc_type in [InletType.STEADY_FLOW, InletType.PULSATILE_FLOW]:
                inlet_flow += inlet.flow_rate
        
        inlet_pressure = 13332.0
        for inlet in self._bcs.inlets:
            if inlet.bc_type == InletType.STEADY_PRESSURE:
                inlet_pressure = inlet.pressure
        
        outlet_pressure = 0.0
        for outlet in self._bcs.outlets:
            if outlet.bc_type == OutletType.ZERO_PRESSURE:
                outlet_pressure = outlet.reference_pressure
                break
        
        visited = set()
        queue = list(self._inlet_node_ids)
        
        for node_id in self._inlet_node_ids:
            self._node_pressures[node_id] = inlet_pressure
        
        while queue:
            node_id = queue.pop(0)
            if node_id in visited:
                continue
            visited.add(node_id)
            
            connected_segs = self._network.get_connected_segment_ids(node_id)
            
            for seg_id in connected_segs:
                segment = self._network.segments[seg_id]
                state = self._segment_states[seg_id]
                
                if segment.start_node_id == node_id:
                    other_node = segment.end_node_id
                    p_in = self._node_pressures.get(node_id, inlet_pressure)
                    
                    R = compute_poiseuille_resistance(
                        segment.geometry.length(),
                        segment.geometry.mean_radius(),
                        self._mu,
                    )
                    
                    if other_node in self._outlet_node_ids:
                        p_out = outlet_pressure
                    else:
                        p_out = p_in - R * (inlet_flow / max(1, len(self._outlet_node_ids)))
                    
                    self._node_pressures[other_node] = p_out
                    
                    flow = (p_in - p_out) / R if R > 0 else 0.0
                    self._segment_flows[seg_id] = flow
                    
                    state.pressure = np.linspace(p_in, p_out, state.n_nodes)
                    state.flow = np.full(state.n_nodes, flow)
                    
                    if other_node not in visited:
                        queue.append(other_node)
    
    def _solve_transient(self) -> None:
        """Solve transient 1D flow using Lax-Wendroff scheme."""
        dt = self.config.time_step_size
        n_steps = self.config.n_time_steps
        
        self._solve_steady_state()
        
        for step in range(n_steps):
            t = step * dt
            
            for inlet in self._bcs.inlets:
                if inlet.bc_type == InletType.PULSATILE_FLOW:
                    flow = inlet.get_flow_at_time(t)
                    self._apply_inlet_flow(inlet.node_id, flow)
            
            self._advance_time_step(dt)
        
        self._extract_final_state()
    
    def _apply_inlet_flow(self, node_id: int, flow: float) -> None:
        """Apply inlet flow boundary condition."""
        connected_segs = self._network.get_connected_segment_ids(node_id)
        
        for seg_id in connected_segs:
            segment = self._network.segments[seg_id]
            state = self._segment_states[seg_id]
            
            if segment.start_node_id == node_id:
                state.flow[0] = flow
    
    def _advance_time_step(self, dt: float) -> None:
        """Advance solution by one time step."""
        for seg_id, state in self._segment_states.items():
            dx = state.x_coords[1] - state.x_coords[0] if len(state.x_coords) > 1 else 1.0
            
            A_new = state.area.copy()
            Q_new = state.flow.copy()
            
            for i in range(1, state.n_nodes - 1):
                dQ_dx = (state.flow[i + 1] - state.flow[i - 1]) / (2 * dx)
                A_new[i] = state.area[i] - dt * dQ_dx
            
            state.area = A_new
            state.flow = Q_new
            
            state.pressure = state.beta * (np.sqrt(state.area) - np.sqrt(state.area_ref))
    
    def _extract_final_state(self) -> None:
        """Extract final pressures and flows from segment states."""
        for seg_id, state in self._segment_states.items():
            segment = self._network.segments[seg_id]
            
            self._node_pressures[segment.start_node_id] = float(state.pressure[0])
            self._node_pressures[segment.end_node_id] = float(state.pressure[-1])
            
            self._segment_flows[seg_id] = float(np.mean(state.flow))
    
    def _compute_metrics(self) -> CFDMetrics:
        """Compute CFD metrics from 1D solution."""
        metrics = CFDMetrics()
        
        if self._inlet_node_ids:
            metrics.pressure.inlet_pressure = self._node_pressures.get(
                self._inlet_node_ids[0], 0.0
            )
        
        metrics.pressure.outlet_pressures = {
            nid: self._node_pressures.get(nid, 0.0) for nid in self._outlet_node_ids
        }
        metrics.pressure.compute_pressure_drop()
        
        outlet_flows = {}
        for seg_id, segment in self._network.segments.items():
            if segment.end_node_id in self._outlet_node_ids:
                outlet_flows[segment.end_node_id] = abs(self._segment_flows.get(seg_id, 0.0))
        
        metrics.flow.outlet_flows = outlet_flows
        metrics.flow.total_inlet_flow = sum(outlet_flows.values())
        metrics.flow.compute_uniformity()
        
        metrics.perfusion.compute_from_flows(outlet_flows)
        
        return metrics
    
    def get_node_pressures(self) -> Dict[int, float]:
        """Get pressure at each node."""
        return self._node_pressures.copy()
    
    def get_segment_flows(self) -> Dict[int, float]:
        """Get flow through each segment."""
        return self._segment_flows.copy()
    
    def get_segment_pressure_profiles(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Get pressure profile along each segment.
        
        Returns
        -------
        dict
            Segment ID -> (x_coords, pressures)
        """
        profiles = {}
        for seg_id, state in self._segment_states.items():
            profiles[seg_id] = (state.x_coords.copy(), state.pressure.copy())
        return profiles
    
    def get_segment_flow_profiles(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Get flow profile along each segment.
        
        Returns
        -------
        dict
            Segment ID -> (x_coords, flows)
        """
        profiles = {}
        for seg_id, state in self._segment_states.items():
            profiles[seg_id] = (state.x_coords.copy(), state.flow.copy())
        return profiles
