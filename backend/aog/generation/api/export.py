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

from aog_policies import OutputPolicy, OperationReport

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
    
    Uses the network's to_dict() method for proper serialization,
    then applies unit scaling if needed.
    
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
    
    # Use network's to_dict() method for proper serialization
    network_dict = network.to_dict()
    
    # Apply unit scaling if needed
    scale = 1000.0 if output_policy.output_units == "mm" else 1.0
    
    if scale != 1.0:
        network_dict = _scale_network_dict(network_dict, scale)
    
    # Add output metadata
    network_dict["output_metadata"] = {
        "output_units": output_policy.output_units,
        "scale_from_meters": scale,
    }
    
    with open(output_path, 'w') as f:
        json.dump(network_dict, f, indent=2)
    
    logger.info(f"Saved network to {output_path}")
    
    return output_path


def _scale_network_dict(network_dict: Dict[str, Any], scale: float) -> Dict[str, Any]:
    """
    Apply unit scaling to a network dictionary.
    
    Scales all position and radius values by the given factor.
    
    Parameters
    ----------
    network_dict : dict
        Network dictionary from network.to_dict()
    scale : float
        Scale factor to apply
        
    Returns
    -------
    dict
        Scaled network dictionary
    """
    import copy
    scaled = copy.deepcopy(network_dict)
    
    # Scale node positions
    for node_id, node_data in scaled.get("nodes", {}).items():
        if "position" in node_data:
            pos = node_data["position"]
            if isinstance(pos, dict):
                # Point3D format: {"x": ..., "y": ..., "z": ...}
                pos["x"] = pos.get("x", 0) * scale
                pos["y"] = pos.get("y", 0) * scale
                pos["z"] = pos.get("z", 0) * scale
            elif isinstance(pos, (list, tuple)):
                node_data["position"] = [p * scale for p in pos]
        
        # Scale radius in attributes if present and not None
        if "attributes" in node_data and "radius" in node_data["attributes"]:
            if node_data["attributes"]["radius"] is not None:
                node_data["attributes"]["radius"] *= scale
    
    # Scale segment geometry
    for seg_id, seg_data in scaled.get("segments", {}).items():
        if "geometry" in seg_data:
            geom = seg_data["geometry"]
            
            # Scale start position
            if "start" in geom:
                start = geom["start"]
                if isinstance(start, dict):
                    start["x"] = start.get("x", 0) * scale
                    start["y"] = start.get("y", 0) * scale
                    start["z"] = start.get("z", 0) * scale
                elif isinstance(start, (list, tuple)):
                    geom["start"] = [p * scale for p in start]
            
            # Scale end position
            if "end" in geom:
                end = geom["end"]
                if isinstance(end, dict):
                    end["x"] = end.get("x", 0) * scale
                    end["y"] = end.get("y", 0) * scale
                    end["z"] = end.get("z", 0) * scale
                elif isinstance(end, (list, tuple)):
                    geom["end"] = [p * scale for p in end]
            
            # Scale radii (handle None values)
            if "radius_start" in geom and geom["radius_start"] is not None:
                geom["radius_start"] *= scale
            if "radius_end" in geom and geom["radius_end"] is not None:
                geom["radius_end"] *= scale
            
            # Scale centerline points if present
            if "centerline_points" in geom and geom["centerline_points"]:
                scaled_points = []
                for pt in geom["centerline_points"]:
                    if isinstance(pt, dict):
                        scaled_points.append({
                            "x": pt.get("x", 0) * scale,
                            "y": pt.get("y", 0) * scale,
                            "z": pt.get("z", 0) * scale,
                        })
                    elif isinstance(pt, (list, tuple)):
                        scaled_points.append([p * scale for p in pt])
                geom["centerline_points"] = scaled_points
    
    # Scale domain bounds if present (handle None values)
    if "domain" in scaled:
        domain = scaled["domain"]
        for key in ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max", 
                    "radius", "height", "semi_a", "semi_b", "semi_c"]:
            if key in domain and domain[key] is not None:
                domain[key] *= scale
        if "center" in domain:
            center = domain["center"]
            if isinstance(center, dict):
                center["x"] = center.get("x", 0) * scale
                center["y"] = center.get("y", 0) * scale
                center["z"] = center.get("z", 0) * scale
    
    return scaled


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
