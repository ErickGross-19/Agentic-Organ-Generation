"""
CFD result schema and metric computations.

Provides standardized result structures for all fidelity levels (0D/1D/3D).

Note: The library uses METERS internally for all geometry.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class FlowMetrics:
    """Flow-related metrics from CFD simulation."""
    
    total_inlet_flow: float = 0.0
    outlet_flows: Dict[int, float] = field(default_factory=dict)
    flow_uniformity: float = 0.0
    min_outlet_flow: float = 0.0
    max_outlet_flow: float = 0.0
    mean_outlet_flow: float = 0.0
    std_outlet_flow: float = 0.0
    
    def compute_uniformity(self) -> None:
        """Compute flow uniformity from outlet flows."""
        if not self.outlet_flows:
            self.flow_uniformity = 0.0
            return
        
        flows = list(self.outlet_flows.values())
        self.min_outlet_flow = min(flows)
        self.max_outlet_flow = max(flows)
        self.mean_outlet_flow = np.mean(flows)
        self.std_outlet_flow = np.std(flows)
        
        if self.mean_outlet_flow > 0:
            self.flow_uniformity = 1.0 - (self.std_outlet_flow / self.mean_outlet_flow)
        else:
            self.flow_uniformity = 0.0


@dataclass
class PressureMetrics:
    """Pressure-related metrics from CFD simulation."""
    
    inlet_pressure: float = 0.0
    outlet_pressures: Dict[int, float] = field(default_factory=dict)
    pressure_drop_root_to_terminals: float = 0.0
    min_pressure: float = 0.0
    max_pressure: float = 0.0
    mean_pressure: float = 0.0
    
    def compute_pressure_drop(self) -> None:
        """Compute pressure drop from inlet to outlets."""
        if not self.outlet_pressures:
            self.pressure_drop_root_to_terminals = 0.0
            return
        
        pressures = list(self.outlet_pressures.values())
        self.min_pressure = min(pressures)
        self.max_pressure = max(pressures)
        self.mean_pressure = np.mean(pressures)
        self.pressure_drop_root_to_terminals = self.inlet_pressure - self.mean_pressure


@dataclass
class WallShearStressMetrics:
    """Wall shear stress metrics (3D only)."""
    
    min_wss: float = 0.0
    max_wss: float = 0.0
    mean_wss: float = 0.0
    std_wss: float = 0.0
    low_wss_area_fraction: float = 0.0
    high_wss_area_fraction: float = 0.0
    wss_field: Optional[np.ndarray] = None
    
    low_wss_threshold: float = 0.4
    high_wss_threshold: float = 4.0


@dataclass
class PerfusionProxyMetrics:
    """Perfusion proxy metrics derived from CFD results."""
    
    perfusable_fraction: float = 0.0
    min_flow_threshold: float = 1e-9
    outlets_meeting_threshold: int = 0
    total_outlets: int = 0
    
    def compute_from_flows(self, outlet_flows: Dict[int, float]) -> None:
        """Compute perfusion proxy from outlet flows."""
        self.total_outlets = len(outlet_flows)
        self.outlets_meeting_threshold = sum(
            1 for flow in outlet_flows.values() if flow >= self.min_flow_threshold
        )
        if self.total_outlets > 0:
            self.perfusable_fraction = self.outlets_meeting_threshold / self.total_outlets
        else:
            self.perfusable_fraction = 0.0


@dataclass
class CFDMetrics:
    """Combined CFD metrics from simulation."""
    
    flow: FlowMetrics = field(default_factory=FlowMetrics)
    pressure: PressureMetrics = field(default_factory=PressureMetrics)
    wss: Optional[WallShearStressMetrics] = None
    perfusion: PerfusionProxyMetrics = field(default_factory=PerfusionProxyMetrics)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "flow": {
                "total_inlet_flow": self.flow.total_inlet_flow,
                "outlet_flows": self.flow.outlet_flows,
                "flow_uniformity": self.flow.flow_uniformity,
                "min_outlet_flow": self.flow.min_outlet_flow,
                "max_outlet_flow": self.flow.max_outlet_flow,
                "mean_outlet_flow": self.flow.mean_outlet_flow,
                "std_outlet_flow": self.flow.std_outlet_flow,
            },
            "pressure": {
                "inlet_pressure": self.pressure.inlet_pressure,
                "outlet_pressures": self.pressure.outlet_pressures,
                "pressure_drop_root_to_terminals": self.pressure.pressure_drop_root_to_terminals,
                "min_pressure": self.pressure.min_pressure,
                "max_pressure": self.pressure.max_pressure,
                "mean_pressure": self.pressure.mean_pressure,
            },
            "perfusion": {
                "perfusable_fraction": self.perfusion.perfusable_fraction,
                "outlets_meeting_threshold": self.perfusion.outlets_meeting_threshold,
                "total_outlets": self.perfusion.total_outlets,
            },
        }
        
        if self.wss is not None:
            result["wss"] = {
                "min_wss": self.wss.min_wss,
                "max_wss": self.wss.max_wss,
                "mean_wss": self.wss.mean_wss,
                "std_wss": self.wss.std_wss,
                "low_wss_area_fraction": self.wss.low_wss_area_fraction,
                "high_wss_area_fraction": self.wss.high_wss_area_fraction,
            }
        
        return result


@dataclass
class CFDResult:
    """Complete CFD simulation result."""
    
    success: bool = False
    fidelity: str = "0D"
    metrics: CFDMetrics = field(default_factory=CFDMetrics)
    
    geometry_file: Optional[str] = None
    mesh_file: Optional[str] = None
    vtk_output_file: Optional[str] = None
    csv_output_file: Optional[str] = None
    
    node_pressures: Optional[Dict[int, float]] = None
    node_flows: Optional[Dict[int, float]] = None
    segment_flows: Optional[Dict[int, float]] = None
    segment_resistances: Optional[Dict[int, float]] = None
    
    velocity_field: Optional[np.ndarray] = None
    pressure_field: Optional[np.ndarray] = None
    wss_field: Optional[np.ndarray] = None
    
    solver_iterations: int = 0
    solver_residual: float = 0.0
    wall_time_seconds: float = 0.0
    
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_success(self) -> bool:
        """Check if simulation was successful."""
        return self.success and len(self.errors) == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "fidelity": self.fidelity,
            "metrics": self.metrics.to_dict(),
            "geometry_file": self.geometry_file,
            "mesh_file": self.mesh_file,
            "vtk_output_file": self.vtk_output_file,
            "csv_output_file": self.csv_output_file,
            "solver_iterations": self.solver_iterations,
            "solver_residual": self.solver_residual,
            "wall_time_seconds": self.wall_time_seconds,
            "warnings": self.warnings,
            "errors": self.errors,
            "metadata": self.metadata,
        }
    
    @classmethod
    def failure(cls, error: str) -> "CFDResult":
        """Create a failure result."""
        result = cls(success=False)
        result.errors.append(error)
        return result


def compute_metrics_from_0d_solution(
    node_pressures: Dict[int, float],
    segment_flows: Dict[int, float],
    inlet_node_id: int,
    outlet_node_ids: List[int],
    network: Optional["VascularNetwork"] = None,
) -> CFDMetrics:
    """
    Compute CFD metrics from 0D lumped parameter solution.
    
    Parameters
    ----------
    node_pressures : dict
        Pressure at each node (Pa)
    segment_flows : dict
        Flow through each segment (m^3/s)
    inlet_node_id : int
        ID of inlet node
    outlet_node_ids : list
        IDs of outlet nodes
    network : VascularNetwork, optional
        Network for finding segments connected to nodes
        
    Returns
    -------
    CFDMetrics
        Computed metrics
    """
    metrics = CFDMetrics()
    
    metrics.pressure.inlet_pressure = node_pressures.get(inlet_node_id, 0.0)
    metrics.pressure.outlet_pressures = {
        nid: node_pressures.get(nid, 0.0) for nid in outlet_node_ids
    }
    metrics.pressure.compute_pressure_drop()
    
    total_inlet_flow = 0.0
    outlet_flows = {}
    
    if network is not None:
        for seg_id, segment in network.segments.items():
            flow = abs(segment_flows.get(seg_id, 0.0))
            if segment.start_node_id == inlet_node_id or segment.end_node_id == inlet_node_id:
                total_inlet_flow += flow
            if segment.end_node_id in outlet_node_ids:
                outlet_flows[segment.end_node_id] = flow
            elif segment.start_node_id in outlet_node_ids:
                outlet_flows[segment.start_node_id] = flow
    else:
        for seg_id, flow in segment_flows.items():
            total_inlet_flow = max(total_inlet_flow, abs(flow))
        outlet_flows = {nid: 0.0 for nid in outlet_node_ids}
    
    metrics.flow.total_inlet_flow = total_inlet_flow
    metrics.flow.outlet_flows = outlet_flows
    metrics.flow.compute_uniformity()
    
    metrics.perfusion.compute_from_flows(metrics.flow.outlet_flows)
    
    return metrics


def compute_metrics_from_3d_solution(
    pressure_field: np.ndarray,
    velocity_field: np.ndarray,
    wss_field: np.ndarray,
    inlet_pressure: float,
    outlet_pressures: Dict[int, float],
    outlet_flows: Dict[int, float],
) -> CFDMetrics:
    """
    Compute CFD metrics from 3D FE CFD solution.
    
    Parameters
    ----------
    pressure_field : np.ndarray
        Pressure field on mesh nodes
    velocity_field : np.ndarray
        Velocity field on mesh nodes
    wss_field : np.ndarray
        Wall shear stress on surface mesh
    inlet_pressure : float
        Inlet pressure (Pa)
    outlet_pressures : dict
        Pressure at each outlet (Pa)
    outlet_flows : dict
        Flow at each outlet (m^3/s)
        
    Returns
    -------
    CFDMetrics
        Computed metrics
    """
    metrics = CFDMetrics()
    
    metrics.pressure.inlet_pressure = inlet_pressure
    metrics.pressure.outlet_pressures = outlet_pressures
    metrics.pressure.compute_pressure_drop()
    
    metrics.flow.outlet_flows = outlet_flows
    metrics.flow.total_inlet_flow = sum(outlet_flows.values())
    metrics.flow.compute_uniformity()
    
    metrics.wss = WallShearStressMetrics()
    if len(wss_field) > 0:
        metrics.wss.min_wss = float(np.min(wss_field))
        metrics.wss.max_wss = float(np.max(wss_field))
        metrics.wss.mean_wss = float(np.mean(wss_field))
        metrics.wss.std_wss = float(np.std(wss_field))
        
        total_area = len(wss_field)
        low_wss_count = np.sum(wss_field < metrics.wss.low_wss_threshold)
        high_wss_count = np.sum(wss_field > metrics.wss.high_wss_threshold)
        metrics.wss.low_wss_area_fraction = low_wss_count / total_area
        metrics.wss.high_wss_area_fraction = high_wss_count / total_area
        metrics.wss.wss_field = wss_field
    
    metrics.perfusion.compute_from_flows(outlet_flows)
    
    return metrics
