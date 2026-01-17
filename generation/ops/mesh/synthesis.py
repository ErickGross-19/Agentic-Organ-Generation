"""
Mesh synthesis from vascular networks.

This module provides functions for converting vascular networks to
triangle meshes with policy-controlled options.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, TYPE_CHECKING
import numpy as np
import logging

from ...policies import MeshSynthesisPolicy, OperationReport

if TYPE_CHECKING:
    import trimesh
    from ...core.network import VascularNetwork

logger = logging.getLogger(__name__)


def synthesize_mesh(
    network: "VascularNetwork",
    policy: Optional[MeshSynthesisPolicy] = None,
) -> Tuple["trimesh.Trimesh", OperationReport]:
    """
    Synthesize a triangle mesh from a vascular network.
    
    This wraps the mesh_adapter.to_trimesh function with policy controls.
    
    Parameters
    ----------
    network : VascularNetwork
        Network to convert to mesh
    policy : MeshSynthesisPolicy, optional
        Policy controlling synthesis options
        
    Returns
    -------
    mesh : trimesh.Trimesh
        Synthesized mesh
    report : OperationReport
        Report with synthesis statistics
    """
    import trimesh
    from ...adapters.mesh_adapter import to_trimesh
    
    if policy is None:
        policy = MeshSynthesisPolicy()
    
    warnings = []
    metadata = {
        "add_node_spheres": policy.add_node_spheres,
        "cap_ends": policy.cap_ends,
        "segments_per_circle": policy.segments_per_circle,
        "mutate_network_in_place": policy.mutate_network_in_place,
        "radius_clamp_mode": policy.radius_clamp_mode,
    }
    
    # Build effective policy
    # PATCH 5: Include voxel repair policy fields
    effective_policy = MeshSynthesisPolicy(
        add_node_spheres=policy.add_node_spheres,
        cap_ends=policy.cap_ends,
        radius_clamp_min=policy.radius_clamp_min,
        radius_clamp_max=policy.radius_clamp_max,
        voxel_repair_synthesis=policy.voxel_repair_synthesis,
        voxel_repair_pitch=policy.voxel_repair_pitch,
        voxel_repair_auto_adjust=policy.voxel_repair_auto_adjust,
        voxel_repair_max_steps=policy.voxel_repair_max_steps,
        voxel_repair_step_factor=policy.voxel_repair_step_factor,
        segments_per_circle=policy.segments_per_circle,
        mutate_network_in_place=policy.mutate_network_in_place,
        radius_clamp_mode=policy.radius_clamp_mode,
    )
    
    # Apply radius clamping if specified
    # Clamp TubeGeometry.radius_start and radius_end (the correct fields)
    clamped_count = 0
    work_network = network
    
    if policy.radius_clamp_min is not None or policy.radius_clamp_max is not None:
        # Determine if we should mutate in place or work on a copy
        should_mutate = policy.mutate_network_in_place or policy.radius_clamp_mode == "mutate"
        
        if not should_mutate:
            # Create a deep copy of the network to avoid mutating the original
            import copy
            work_network = copy.deepcopy(network)
            metadata["network_copied"] = True
        
        for segment in work_network.segments.values():
            # Access radius through geometry attribute (TubeGeometry)
            if hasattr(segment, 'geometry') and segment.geometry is not None:
                geom = segment.geometry
                
                # Clamp radius_start
                if hasattr(geom, 'radius_start') and geom.radius_start is not None:
                    if policy.radius_clamp_min and geom.radius_start < policy.radius_clamp_min:
                        geom.radius_start = policy.radius_clamp_min
                        clamped_count += 1
                    if policy.radius_clamp_max and geom.radius_start > policy.radius_clamp_max:
                        geom.radius_start = policy.radius_clamp_max
                        clamped_count += 1
                
                # Clamp radius_end
                if hasattr(geom, 'radius_end') and geom.radius_end is not None:
                    if policy.radius_clamp_min and geom.radius_end < policy.radius_clamp_min:
                        geom.radius_end = policy.radius_clamp_min
                        clamped_count += 1
                    if policy.radius_clamp_max and geom.radius_end > policy.radius_clamp_max:
                        geom.radius_end = policy.radius_clamp_max
                        clamped_count += 1
            
            # Also check for direct radius attributes (legacy support)
            elif hasattr(segment, 'radius') and segment.radius is not None:
                if policy.radius_clamp_min and segment.radius < policy.radius_clamp_min:
                    segment.radius = policy.radius_clamp_min
                    clamped_count += 1
                if policy.radius_clamp_max and segment.radius > policy.radius_clamp_max:
                    segment.radius = policy.radius_clamp_max
                    clamped_count += 1
        
        if clamped_count > 0:
            warnings.append(f"Clamped {clamped_count} radii to policy limits")
            metadata["radii_clamped"] = clamped_count
    
    # Synthesize mesh using adapter (use work_network which may be clamped)
    try:
        # Note: to_trimesh uses different parameter names than policy fields
        # include_node_spheres (not add_node_spheres)
        # include_caps (not cap_ends)
        # radial_resolution (not segments_per_circle)
        result = to_trimesh(
            work_network,
            include_node_spheres=policy.add_node_spheres,
            include_caps=policy.cap_ends,
            radial_resolution=policy.segments_per_circle,
        )
        # to_trimesh returns OperationResult with mesh in metadata['mesh']
        # Check if operation failed first using is_success() method
        if hasattr(result, 'is_success') and not result.is_success():
            raise ValueError(f"to_trimesh failed: {getattr(result, 'message', 'unknown error')}")
        
        # Extract mesh from OperationResult.metadata['mesh']
        mesh = None
        if hasattr(result, 'metadata') and isinstance(result.metadata, dict):
            mesh = result.metadata.get('mesh')
        
        if mesh is None:
            # Fallback: check if result has mesh attribute directly
            if hasattr(result, 'mesh'):
                mesh = result.mesh
            elif hasattr(result, 'vertices'):
                # Result is the mesh itself
                mesh = result
            else:
                raise ValueError(f"to_trimesh returned no mesh. Result type: {type(result)}, metadata keys: {list(result.metadata.keys()) if hasattr(result, 'metadata') else 'N/A'}")
        
        # PATCH 5: Apply voxel repair if requested, using policy-driven parameters
        if policy.voxel_repair_synthesis:
            mesh, repair_metadata = _voxel_repair(
                mesh,
                pitch=policy.voxel_repair_pitch,
                auto_adjust=policy.voxel_repair_auto_adjust,
                max_steps=policy.voxel_repair_max_steps,
                step_factor=policy.voxel_repair_step_factor,
            )
            metadata["voxel_repair_applied"] = True
            metadata["voxel_repair_metadata"] = repair_metadata
        
        metadata["vertex_count"] = len(mesh.vertices)
        metadata["face_count"] = len(mesh.faces)
        metadata["is_watertight"] = mesh.is_watertight
        
        success = True
        
    except Exception as e:
        logger.error(f"Mesh synthesis failed: {e}")
        mesh = trimesh.Trimesh()
        success = False
        metadata["error"] = str(e)
    
    report = OperationReport(
        operation="synthesize_mesh",
        success=success,
        requested_policy=policy.to_dict(),
        effective_policy=effective_policy.to_dict(),
        warnings=warnings,
        metadata=metadata,
    )
    
    return mesh, report


def _voxel_repair(
    mesh: "trimesh.Trimesh",
    pitch: float = 1e-4,
    auto_adjust: bool = True,
    max_steps: int = 4,
    step_factor: float = 1.5,
) -> Tuple["trimesh.Trimesh", Dict[str, Any]]:
    """
    PATCH 5: Apply voxel-based repair to a mesh with policy-driven parameters.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to repair
    pitch : float
        Initial voxel pitch in meters
    auto_adjust : bool
        If True, automatically increase pitch on failure
    max_steps : int
        Maximum number of pitch adjustment steps
    step_factor : float
        Factor to multiply pitch by on each adjustment step
        
    Returns
    -------
    repaired : trimesh.Trimesh
        Repaired mesh (or original if repair failed)
    metadata : dict
        Repair metadata including effective_pitch, steps_taken, success
    """
    import trimesh
    
    metadata = {
        "requested_pitch": pitch,
        "effective_pitch": pitch,
        "steps_taken": 0,
        "success": False,
        "auto_adjusted": False,
    }
    
    current_pitch = pitch
    
    for step in range(max_steps):
        metadata["steps_taken"] = step + 1
        metadata["effective_pitch"] = current_pitch
        
        try:
            voxels = mesh.voxelized(current_pitch)
            voxels = voxels.fill()
            repaired = voxels.marching_cubes
            
            # Check for coordinate system issues
            in_extent = float(np.max(mesh.extents))
            out_extent = float(np.max(repaired.extents))
            
            if in_extent > 0 and out_extent / in_extent > 50:
                repaired.apply_transform(voxels.transform)
            
            repaired.merge_vertices()
            repaired.remove_unreferenced_vertices()
            
            if repaired.volume < 0:
                repaired.invert()
            trimesh.repair.fix_normals(repaired)
            
            metadata["success"] = True
            metadata["output_vertex_count"] = len(repaired.vertices)
            metadata["output_face_count"] = len(repaired.faces)
            
            return repaired, metadata
            
        except Exception as e:
            logger.warning(f"Voxel repair failed at pitch {current_pitch}: {e}")
            
            if not auto_adjust or step >= max_steps - 1:
                # No more adjustments allowed, return original
                metadata["error"] = str(e)
                return mesh, metadata
            
            # Increase pitch and try again
            current_pitch *= step_factor
            metadata["auto_adjusted"] = True
            logger.info(f"Auto-adjusting voxel repair pitch to {current_pitch}")
    
    # Should not reach here, but return original mesh if we do
    return mesh, metadata


__all__ = [
    "synthesize_mesh",
    "MeshSynthesisPolicy",
]
