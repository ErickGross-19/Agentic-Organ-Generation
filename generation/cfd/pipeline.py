"""
CFD pipeline orchestrator.

Coordinates the full CFD workflow:
1. Build watertight geometry from VascularNetwork
2. Generate volumetric mesh
3. Assign boundary conditions
4. Run solver (0D/1D/3D)
5. Postprocess and compute metrics

Note: The library uses METERS internally for all geometry.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Literal, Union
import time
import os

from ..core.network import VascularNetwork
from .geometry import build_watertight_geometry, GeometryConfig
from .meshing import generate_mesh, MeshConfig, MeshResult
from .bcs import (
    BoundaryConditions, FluidProperties, InletBC, OutletBC,
    InletType, OutletType, create_default_bcs_from_network,
)
from .results import CFDResult, CFDMetrics
from .solvers import BaseSolver, SolverConfig, Solver0D, Solver1D, Solver3D


@dataclass
class CFDConfig:
    """
    Configuration for CFD pipeline.
    
    Specifies fidelity level, fluid properties, boundary conditions,
    mesh settings, solver settings, and output options.
    """
    
    fidelity: Literal["0D", "1D", "3D"] = "0D"
    
    fluid: FluidProperties = field(default_factory=FluidProperties)
    
    inlet_type: InletType = InletType.STEADY_FLOW
    inlet_flow_rate: float = 1e-6
    inlet_pressure: float = 13332.0
    
    outlet_type: OutletType = OutletType.ZERO_PRESSURE
    outlet_resistance: Optional[float] = None
    
    wall_type: Literal["rigid", "fsi"] = "rigid"
    
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    
    output_dir: Optional[str] = None
    output_vtk: bool = True
    output_csv: bool = True
    output_summary_json: bool = True
    
    skip_geometry: bool = False
    skip_meshing: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "fidelity": self.fidelity,
            "fluid": self.fluid.to_dict(),
            "inlet_type": self.inlet_type.value,
            "inlet_flow_rate": self.inlet_flow_rate,
            "inlet_pressure": self.inlet_pressure,
            "outlet_type": self.outlet_type.value,
            "outlet_resistance": self.outlet_resistance,
            "wall_type": self.wall_type,
            "geometry": self.geometry.to_dict(),
            "mesh": self.mesh.to_dict(),
            "solver": self.solver.to_dict(),
            "output_dir": self.output_dir,
            "output_vtk": self.output_vtk,
            "output_csv": self.output_csv,
            "output_summary_json": self.output_summary_json,
        }


def run_cfd_pipeline(
    network: VascularNetwork,
    config: Optional[CFDConfig] = None,
) -> CFDResult:
    """
    Run the complete CFD pipeline on a vascular network.
    
    Pipeline stages:
    1. Build watertight 3D geometry (if needed for fidelity)
    2. Generate volumetric mesh (if needed for 3D)
    3. Create boundary conditions
    4. Run solver
    5. Postprocess and compute metrics
    
    Parameters
    ----------
    network : VascularNetwork
        Vascular network to simulate
    config : CFDConfig, optional
        Pipeline configuration
        
    Returns
    -------
    CFDResult
        Simulation results including metrics and output files
    """
    if config is None:
        config = CFDConfig()
    
    start_time = time.time()
    result = CFDResult(fidelity=config.fidelity)
    
    if config.output_dir:
        os.makedirs(config.output_dir, exist_ok=True)
    
    geometry_data = None
    mesh_result = None
    
    if config.fidelity == "3D" and not config.skip_geometry:
        geometry_path = None
        if config.output_dir:
            geometry_path = os.path.join(config.output_dir, "geometry.stl")
        
        try:
            geometry_data = build_watertight_geometry(
                network, config.geometry, geometry_path
            )
            result.geometry_file = geometry_data.get("file_path")
        except Exception as e:
            result.warnings.append(f"Geometry generation warning: {e}")
    
    if config.fidelity == "3D" and not config.skip_meshing:
        if geometry_data is not None:
            mesh_path = None
            if config.output_dir:
                mesh_path = os.path.join(config.output_dir, "mesh.vtk")
            
            try:
                mesh_result = generate_mesh(
                    geometry_data["vertices"],
                    geometry_data["faces"],
                    config.mesh,
                    mesh_path,
                )
                result.mesh_file = mesh_result.file_path
                
                if mesh_result.warnings:
                    result.warnings.extend(mesh_result.warnings)
                if mesh_result.errors:
                    result.errors.extend(mesh_result.errors)
                    
            except Exception as e:
                result.warnings.append(f"Mesh generation warning: {e}")
    
    bcs = _create_boundary_conditions(network, config)
    
    solver = _create_solver(config)
    
    mesh_data = None
    if mesh_result is not None and mesh_result.success:
        mesh_data = {
            "vertices": mesh_result.vertices,
            "cells": mesh_result.cells,
        }
    elif config.fidelity == "3D":
        if geometry_data is not None:
            mesh_data = {
                "vertices": geometry_data["vertices"],
                "cells": _create_simple_cells(geometry_data["vertices"]),
            }
        else:
            result.errors.append("3D solver requires mesh but mesh generation failed")
            result.wall_time_seconds = time.time() - start_time
            return result
    
    if not solver.setup(network, bcs, mesh_data):
        result.errors.append("Solver setup failed")
        result.wall_time_seconds = time.time() - start_time
        return result
    
    solver_result = solver.solve()
    
    result.success = solver_result.success
    result.metrics = solver_result.metrics
    result.node_pressures = solver_result.node_pressures
    result.segment_flows = solver_result.segment_flows
    result.segment_resistances = solver_result.segment_resistances
    result.velocity_field = solver_result.velocity_field
    result.pressure_field = solver_result.pressure_field
    result.wss_field = solver_result.wss_field
    result.solver_iterations = solver_result.solver_iterations
    result.solver_residual = solver_result.solver_residual
    result.warnings.extend(solver_result.warnings)
    result.errors.extend(solver_result.errors)
    
    if config.output_dir and result.success:
        _write_outputs(result, config)
    
    result.wall_time_seconds = time.time() - start_time
    result.metadata["config"] = config.to_dict()
    
    return result


def _create_boundary_conditions(
    network: VascularNetwork,
    config: CFDConfig,
) -> BoundaryConditions:
    """Create boundary conditions from config."""
    bcs = BoundaryConditions(fluid=config.fluid, wall_type=config.wall_type)
    
    inlet_nodes = [n for n in network.nodes.values() if n.node_type == "inlet"]
    outlet_nodes = [n for n in network.nodes.values() if n.node_type in ["outlet", "terminal"]]
    
    if not inlet_nodes:
        for node in network.nodes.values():
            connected = network.get_connected_segment_ids(node.id)
            if len(connected) == 1:
                seg = network.segments[connected[0]]
                if seg.start_node_id == node.id:
                    inlet_nodes = [node]
                    break
    
    for inlet_node in inlet_nodes:
        if config.inlet_type == InletType.STEADY_FLOW:
            inlet = InletBC.steady_flow(
                inlet_node.id,
                config.inlet_flow_rate / max(1, len(inlet_nodes)),
            )
        elif config.inlet_type == InletType.STEADY_PRESSURE:
            inlet = InletBC.steady_pressure(inlet_node.id, config.inlet_pressure)
        else:
            inlet = InletBC.steady_flow(inlet_node.id, config.inlet_flow_rate)
        
        bcs.add_inlet(inlet)
    
    for outlet_node in outlet_nodes:
        if config.outlet_type == OutletType.ZERO_PRESSURE:
            outlet = OutletBC.zero_pressure(outlet_node.id)
        elif config.outlet_type == OutletType.RESISTANCE:
            r = config.outlet_resistance or 1e10
            outlet = OutletBC.resistance_bc(outlet_node.id, r)
        else:
            outlet = OutletBC.zero_pressure(outlet_node.id)
        
        bcs.add_outlet(outlet)
    
    return bcs


def _create_solver(config: CFDConfig) -> BaseSolver:
    """Create solver based on fidelity level."""
    if config.fidelity == "0D":
        return Solver0D(config.solver)
    elif config.fidelity == "1D":
        return Solver1D(config.solver)
    elif config.fidelity == "3D":
        return Solver3D(config.solver)
    else:
        return Solver0D(config.solver)


def _create_simple_cells(vertices: "np.ndarray") -> "np.ndarray":
    """Create simple tetrahedral cells from vertices."""
    import numpy as np
    
    n_verts = len(vertices)
    if n_verts < 4:
        return np.array([])
    
    cells = []
    for i in range(0, n_verts - 3, 4):
        cells.append([i, i + 1, i + 2, i + 3])
    
    return np.array(cells) if cells else np.array([])


def _write_outputs(result: CFDResult, config: CFDConfig) -> None:
    """Write output files."""
    import json
    
    if config.output_csv and result.node_pressures:
        csv_path = os.path.join(config.output_dir, "pressures.csv")
        with open(csv_path, 'w') as f:
            f.write("node_id,pressure_pa\n")
            for node_id, pressure in result.node_pressures.items():
                f.write(f"{node_id},{pressure}\n")
        result.csv_output_file = csv_path
    
    if config.output_csv and result.segment_flows:
        flow_csv_path = os.path.join(config.output_dir, "flows.csv")
        with open(flow_csv_path, 'w') as f:
            f.write("segment_id,flow_m3s\n")
            for seg_id, flow in result.segment_flows.items():
                f.write(f"{seg_id},{flow}\n")
    
    if config.output_summary_json:
        summary_path = os.path.join(config.output_dir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)


def run_multifidelity_pipeline(
    network: VascularNetwork,
    fidelities: List[Literal["0D", "1D", "3D"]] = ["0D", "1D"],
    base_config: Optional[CFDConfig] = None,
) -> Dict[str, CFDResult]:
    """
    Run CFD pipeline at multiple fidelity levels.
    
    Useful for validation and comparison between fidelity levels.
    
    Parameters
    ----------
    network : VascularNetwork
        Vascular network to simulate
    fidelities : list
        List of fidelity levels to run
    base_config : CFDConfig, optional
        Base configuration (fidelity will be overridden)
        
    Returns
    -------
    dict
        Fidelity level -> CFDResult
    """
    if base_config is None:
        base_config = CFDConfig()
    
    results = {}
    
    for fidelity in fidelities:
        config = CFDConfig(
            fidelity=fidelity,
            fluid=base_config.fluid,
            inlet_type=base_config.inlet_type,
            inlet_flow_rate=base_config.inlet_flow_rate,
            inlet_pressure=base_config.inlet_pressure,
            outlet_type=base_config.outlet_type,
            outlet_resistance=base_config.outlet_resistance,
            wall_type=base_config.wall_type,
            geometry=base_config.geometry,
            mesh=base_config.mesh,
            solver=base_config.solver,
            output_dir=base_config.output_dir,
        )
        
        if config.output_dir:
            config.output_dir = os.path.join(base_config.output_dir, fidelity)
        
        results[fidelity] = run_cfd_pipeline(network, config)
    
    return results


def compare_fidelity_results(
    results: Dict[str, CFDResult],
) -> Dict[str, Any]:
    """
    Compare results across fidelity levels.
    
    Parameters
    ----------
    results : dict
        Fidelity level -> CFDResult
        
    Returns
    -------
    dict
        Comparison metrics
    """
    comparison = {
        "fidelities": list(results.keys()),
        "pressure_drops": {},
        "flow_uniformities": {},
        "wall_times": {},
    }
    
    for fidelity, result in results.items():
        if result.success:
            comparison["pressure_drops"][fidelity] = result.metrics.pressure.pressure_drop_root_to_terminals
            comparison["flow_uniformities"][fidelity] = result.metrics.flow.flow_uniformity
            comparison["wall_times"][fidelity] = result.wall_time_seconds
    
    if "0D" in comparison["pressure_drops"] and "3D" in comparison["pressure_drops"]:
        dp_0d = comparison["pressure_drops"]["0D"]
        dp_3d = comparison["pressure_drops"]["3D"]
        if dp_3d != 0:
            comparison["pressure_drop_0d_vs_3d_error"] = abs(dp_0d - dp_3d) / abs(dp_3d)
    
    if "1D" in comparison["pressure_drops"] and "3D" in comparison["pressure_drops"]:
        dp_1d = comparison["pressure_drops"]["1D"]
        dp_3d = comparison["pressure_drops"]["3D"]
        if dp_3d != 0:
            comparison["pressure_drop_1d_vs_3d_error"] = abs(dp_1d - dp_3d) / abs(dp_3d)
    
    return comparison
