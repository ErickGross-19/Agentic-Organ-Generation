"""
0D lumped parameter solver for vascular networks.

Solves steady-state or transient flow using a resistor network model.
Fast solver suitable for quick sanity checks and parameter sweeps.

Note: The library uses METERS internally for all geometry.
Pressures are in Pascals (Pa), flows in m^3/s.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Literal, Tuple
import numpy as np
import time

from .base import BaseSolver, SolverConfig, build_resistance_network
from ..bcs import BoundaryConditions, InletType, OutletType
from ..results import CFDResult, CFDMetrics, compute_metrics_from_0d_solution


class Solver0D(BaseSolver):
    """
    0D lumped parameter solver.
    
    Models the vascular network as a resistor network and solves
    for pressures and flows using linear algebra.
    
    Advantages:
    - Very fast (milliseconds for 1000+ segments)
    - No mesh required
    - Good for parameter sweeps and quick validation
    
    Limitations:
    - No spatial resolution within vessels
    - No wave propagation effects
    - No wall shear stress computation
    """
    
    def __init__(self, config: Optional[SolverConfig] = None):
        super().__init__(config)
        
        self._network = None
        self._bcs = None
        self._resistances: Dict[int, float] = {}
        
        self._node_pressures: Dict[int, float] = {}
        self._segment_flows: Dict[int, float] = {}
        
        self._node_id_to_idx: Dict[int, int] = {}
        self._idx_to_node_id: Dict[int, int] = {}
        self._seg_id_to_idx: Dict[int, int] = {}
        
        self._inlet_node_ids: List[int] = []
        self._outlet_node_ids: List[int] = []
    
    @property
    def fidelity(self) -> Literal["0D", "1D", "3D"]:
        return "0D"
    
    @property
    def requires_mesh(self) -> bool:
        return False
    
    def setup(
        self,
        network: "VascularNetwork",
        bcs: BoundaryConditions,
        mesh_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Set up the 0D solver."""
        self._network = network
        self._bcs = bcs
        
        self._resistances = build_resistance_network(
            network, bcs.fluid.viscosity
        )
        
        self._node_id_to_idx = {}
        self._idx_to_node_id = {}
        for idx, node_id in enumerate(network.nodes.keys()):
            self._node_id_to_idx[node_id] = idx
            self._idx_to_node_id[idx] = node_id
        
        self._seg_id_to_idx = {
            seg_id: idx for idx, seg_id in enumerate(network.segments.keys())
        }
        
        self._inlet_node_ids = bcs.get_inlet_node_ids()
        self._outlet_node_ids = bcs.get_outlet_node_ids()
        
        self._is_initialized = True
        return True
    
    def solve(self) -> CFDResult:
        """Run the 0D simulation."""
        start_time = time.time()
        
        errors = self.validate_setup()
        if errors:
            result = CFDResult.failure("; ".join(errors))
            return result
        
        result = CFDResult(fidelity="0D")
        
        try:
            if self.config.steady_state:
                self._solve_steady_state()
            else:
                self._solve_transient()
            
            result.node_pressures = self._node_pressures.copy()
            result.segment_flows = self._segment_flows.copy()
            result.segment_resistances = self._resistances.copy()
            
            result.metrics = compute_metrics_from_0d_solution(
                self._node_pressures,
                self._segment_flows,
                self._inlet_node_ids[0] if self._inlet_node_ids else 0,
                self._outlet_node_ids,
                self._network,
            )
            
            result.success = True
            
        except Exception as e:
            result.errors.append(f"0D solver failed: {e}")
            result.success = False
        
        result.wall_time_seconds = time.time() - start_time
        return result
    
    def _solve_steady_state(self) -> None:
        """Solve steady-state flow problem."""
        n_nodes = len(self._network.nodes)
        n_segments = len(self._network.segments)
        
        if n_nodes == 0:
            return
        
        A = np.zeros((n_nodes, n_nodes))
        b = np.zeros(n_nodes)
        
        for seg_id, segment in self._network.segments.items():
            start_idx = self._node_id_to_idx[segment.start_node_id]
            end_idx = self._node_id_to_idx[segment.end_node_id]
            
            R = self._resistances[seg_id]
            if R <= 0:
                R = 1e-10
            
            conductance = 1.0 / R
            
            A[start_idx, start_idx] += conductance
            A[start_idx, end_idx] -= conductance
            A[end_idx, start_idx] -= conductance
            A[end_idx, end_idx] += conductance
        
        for inlet in self._bcs.inlets:
            idx = self._node_id_to_idx.get(inlet.node_id)
            if idx is None:
                continue
            
            if inlet.bc_type == InletType.STEADY_PRESSURE:
                A[idx, :] = 0
                A[idx, idx] = 1.0
                b[idx] = inlet.pressure
            elif inlet.bc_type == InletType.STEADY_FLOW:
                b[idx] += inlet.flow_rate
        
        for outlet in self._bcs.outlets:
            idx = self._node_id_to_idx.get(outlet.node_id)
            if idx is None:
                continue
            
            if outlet.bc_type == OutletType.ZERO_PRESSURE:
                A[idx, :] = 0
                A[idx, idx] = 1.0
                b[idx] = outlet.reference_pressure
            elif outlet.bc_type == OutletType.RESISTANCE:
                A[idx, idx] += 1.0 / outlet.resistance
                b[idx] += outlet.reference_pressure / outlet.resistance
        
        try:
            pressures = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            pressures = np.linalg.lstsq(A, b, rcond=None)[0]
        
        for idx, p in enumerate(pressures):
            node_id = self._idx_to_node_id[idx]
            self._node_pressures[node_id] = float(p)
        
        for seg_id, segment in self._network.segments.items():
            p_start = self._node_pressures[segment.start_node_id]
            p_end = self._node_pressures[segment.end_node_id]
            R = self._resistances[seg_id]
            
            if R > 0:
                flow = (p_start - p_end) / R
            else:
                flow = 0.0
            
            self._segment_flows[seg_id] = float(flow)
    
    def _solve_transient(self) -> None:
        """Solve transient flow problem (simplified)."""
        self._solve_steady_state()
    
    def get_node_pressures(self) -> Dict[int, float]:
        """Get pressure at each node."""
        return self._node_pressures.copy()
    
    def get_segment_flows(self) -> Dict[int, float]:
        """Get flow through each segment."""
        return self._segment_flows.copy()
    
    def get_segment_resistances(self) -> Dict[int, float]:
        """Get resistance of each segment."""
        return self._resistances.copy()
    
    def compute_total_resistance(self) -> float:
        """
        Compute total resistance from inlet to outlets.
        
        Returns
        -------
        float
            Total network resistance (Pa.s/m^3)
        """
        if not self._inlet_node_ids or not self._outlet_node_ids:
            return 0.0
        
        inlet_pressure = self._node_pressures.get(self._inlet_node_ids[0], 0.0)
        
        outlet_pressures = [
            self._node_pressures.get(nid, 0.0) for nid in self._outlet_node_ids
        ]
        mean_outlet_pressure = np.mean(outlet_pressures) if outlet_pressures else 0.0
        
        total_flow = sum(
            abs(self._segment_flows.get(seg_id, 0.0))
            for seg_id, seg in self._network.segments.items()
            if seg.start_node_id in self._inlet_node_ids
        )
        
        if total_flow > 0:
            return (inlet_pressure - mean_outlet_pressure) / total_flow
        return float('inf')
