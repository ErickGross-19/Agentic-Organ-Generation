"""
Export utilities for vascular network outputs.

This module provides standardized functions for saving meshes, reports,
and other artifacts with consistent naming conventions.

UNIT CONVENTIONS
----------------
Internal units are METERS. Output units are configurable (default: mm).
"""

from typing import Optional, Dict, Any, Union, TYPE_CHECKING
from pathlib import Path
import json
import time
import logging

from ..policies import OutputPolicy, OperationReport

if TYPE_CHECKING:
    import trimesh

logger = logging.getLogger(__name__)


def make_run_dir(
    output_policy: Optional[OutputPolicy] = None,
    run_name: Optional[str] = None,
) -> Path:
    """
    Create a run directory for output artifacts.
    
    Parameters
    ----------
    output_policy : OutputPolicy, optional
        Policy controlling output location and naming
    run_name : str, optional
        Custom run name (overrides naming convention)
        
    Returns
    -------
    Path
        Path to the created run directory
    """
    if output_policy is None:
        output_policy = OutputPolicy()
    
    base_dir = Path(output_policy.output_dir)
    
    if run_name:
        run_dir = base_dir / run_name
    elif output_policy.naming_convention == "timestamped":
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = base_dir / f"run_{timestamp}"
    else:
        run_dir = base_dir / "run"
    
    run_dir.mkdir(parents=True, exist_ok=True)
    
    return run_dir


def save_mesh(
    mesh: "trimesh.Trimesh",
    rel_path: str,
    output_policy: Optional[OutputPolicy] = None,
    run_dir: Optional[Path] = None,
) -> Path:
    """
    Save a mesh to file with unit conversion.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to save
    rel_path : str
        Relative path within run directory (e.g., "domain_with_void.stl")
    output_policy : OutputPolicy, optional
        Policy controlling output units and location
    run_dir : Path, optional
        Run directory (created if not provided)
        
    Returns
    -------
    Path
        Path to the saved file
    """
    import trimesh
    
    if output_policy is None:
        output_policy = OutputPolicy()
    
    if run_dir is None:
        run_dir = make_run_dir(output_policy)
    
    output_path = run_dir / rel_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Apply unit conversion
    if output_policy.output_units == "mm":
        # Internal units are meters, convert to mm
        mesh_export = mesh.copy()
        mesh_export.apply_scale(1000.0)
    else:
        mesh_export = mesh
    
    mesh_export.export(str(output_path))
    logger.info(f"Saved mesh to {output_path}")
    
    return output_path


def write_json(
    data: Union[Dict[str, Any], OperationReport],
    rel_path: str,
    output_policy: Optional[OutputPolicy] = None,
    run_dir: Optional[Path] = None,
) -> Path:
    """
    Write JSON data to file.
    
    Parameters
    ----------
    data : dict or OperationReport
        Data to write (converted to dict if OperationReport)
    rel_path : str
        Relative path within run directory (e.g., "report.json")
    output_policy : OutputPolicy, optional
        Policy controlling output location
    run_dir : Path, optional
        Run directory (created if not provided)
        
    Returns
    -------
    Path
        Path to the saved file
    """
    if output_policy is None:
        output_policy = OutputPolicy()
    
    if run_dir is None:
        run_dir = make_run_dir(output_policy)
    
    output_path = run_dir / rel_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert OperationReport to dict
    if hasattr(data, 'to_dict'):
        data = data.to_dict()
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    logger.info(f"Saved JSON to {output_path}")
    
    return output_path


def save_network(
    network: "VascularNetwork",
    rel_path: str,
    output_policy: Optional[OutputPolicy] = None,
    run_dir: Optional[Path] = None,
) -> Path:
    """
    Save a vascular network to JSON file.
    
    Parameters
    ----------
    network : VascularNetwork
        Network to save
    rel_path : str
        Relative path within run directory (e.g., "network.json")
    output_policy : OutputPolicy, optional
        Policy controlling output units and location
    run_dir : Path, optional
        Run directory (created if not provided)
        
    Returns
    -------
    Path
        Path to the saved file
    """
    if output_policy is None:
        output_policy = OutputPolicy()
    
    if run_dir is None:
        run_dir = make_run_dir(output_policy)
    
    output_path = run_dir / rel_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert network to serializable dict
    scale = 1000.0 if output_policy.output_units == "mm" else 1.0
    
    network_dict = {
        "nodes": {},
        "segments": {},
        "metadata": {
            "output_units": output_policy.output_units,
            "scale_from_meters": scale,
        }
    }
    
    for node_id, node in network.nodes.items():
        network_dict["nodes"][str(node_id)] = {
            "position": [node.position.x * scale, node.position.y * scale, node.position.z * scale],
            "radius": node.radius * scale if hasattr(node, 'radius') else None,
            "node_type": node.node_type.value if hasattr(node.node_type, 'value') else str(node.node_type),
            "vessel_type": node.vessel_type if hasattr(node, 'vessel_type') else None,
        }
    
    for seg_id, segment in network.segments.items():
        network_dict["segments"][str(seg_id)] = {
            "start_node": segment.start_node_id,
            "end_node": segment.end_node_id,
            "start_radius": segment.start_radius * scale if hasattr(segment, 'start_radius') else None,
            "end_radius": segment.end_radius * scale if hasattr(segment, 'end_radius') else None,
            "vessel_type": segment.vessel_type if hasattr(segment, 'vessel_type') else None,
        }
    
    with open(output_path, 'w') as f:
        json.dump(network_dict, f, indent=2)
    
    logger.info(f"Saved network to {output_path}")
    
    return output_path


def export_all(
    run_dir: Path,
    solid_mesh: Optional["trimesh.Trimesh"] = None,
    void_mesh: Optional["trimesh.Trimesh"] = None,
    network: Optional["VascularNetwork"] = None,
    report: Optional[OperationReport] = None,
    output_policy: Optional[OutputPolicy] = None,
) -> Dict[str, Path]:
    """
    Export all artifacts for a generation run.
    
    Parameters
    ----------
    run_dir : Path
        Run directory for outputs
    solid_mesh : trimesh.Trimesh, optional
        Domain with void (saved as domain_with_void.stl)
    void_mesh : trimesh.Trimesh, optional
        Void mesh (saved as structure.stl)
    network : VascularNetwork, optional
        Network (saved as network.json)
    report : OperationReport, optional
        Report (saved as report.json)
    output_policy : OutputPolicy, optional
        Policy controlling output format
        
    Returns
    -------
    dict
        Mapping of artifact names to saved paths
    """
    if output_policy is None:
        output_policy = OutputPolicy()
    
    paths = {}
    
    if solid_mesh is not None:
        paths["solid"] = save_mesh(solid_mesh, "domain_with_void.stl", output_policy, run_dir)
    
    if void_mesh is not None:
        paths["void"] = save_mesh(void_mesh, "structure.stl", output_policy, run_dir)
    
    if network is not None:
        paths["network"] = save_network(network, "network.json", output_policy, run_dir)
    
    if report is not None and output_policy.save_reports:
        paths["report"] = write_json(report, "report.json", output_policy, run_dir)
    
    return paths


__all__ = [
    "make_run_dir",
    "save_mesh",
    "write_json",
    "save_network",
    "export_all",
]
