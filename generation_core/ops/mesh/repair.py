"""
Mesh repair operations.

This module provides a thin wrapper around validity repair functions
for use in the generation pipeline.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, TYPE_CHECKING
import logging

from ...policies import RepairPolicy, OperationReport

if TYPE_CHECKING:
    import trimesh

logger = logging.getLogger(__name__)


@dataclass
class MeshRepairPolicy:
    """
    Policy for mesh repair operations (generation-side wrapper).
    
    This is a simplified policy for use in the generation pipeline.
    For full repair capabilities, use validity.api.repair.
    """
    voxel_repair: bool = True
    voxel_pitch: float = 1e-4  # 0.1mm
    remove_small_components: bool = True
    min_component_faces: int = 500
    fill_holes: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "voxel_repair": self.voxel_repair,
            "voxel_pitch": self.voxel_pitch,
            "remove_small_components": self.remove_small_components,
            "min_component_faces": self.min_component_faces,
            "fill_holes": self.fill_holes,
        }
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "MeshRepairPolicy":
        return MeshRepairPolicy(**{k: v for k, v in d.items() if k in MeshRepairPolicy.__dataclass_fields__})


def repair_mesh(
    mesh: "trimesh.Trimesh",
    policy: Optional[MeshRepairPolicy] = None,
) -> Tuple["trimesh.Trimesh", OperationReport]:
    """
    Repair a mesh using the specified policy.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to repair
    policy : MeshRepairPolicy, optional
        Policy controlling repair operations
        
    Returns
    -------
    repaired : trimesh.Trimesh
        Repaired mesh
    report : OperationReport
        Report with repair statistics
    """
    import trimesh
    import numpy as np
    
    if policy is None:
        policy = MeshRepairPolicy()
    
    warnings = []
    metadata = {
        "operations_applied": [],
        "original_vertex_count": len(mesh.vertices),
        "original_face_count": len(mesh.faces),
        "original_watertight": mesh.is_watertight,
    }
    
    repaired = mesh.copy()
    
    # Apply voxel repair
    if policy.voxel_repair:
        try:
            voxels = repaired.voxelized(policy.voxel_pitch)
            voxels = voxels.fill()
            repaired = voxels.marching_cubes
            
            # Check for coordinate system issues
            in_extent = float(np.max(mesh.extents))
            out_extent = float(np.max(repaired.extents))
            
            if in_extent > 0 and out_extent / in_extent > 50:
                repaired.apply_transform(voxels.transform)
            
            metadata["operations_applied"].append("voxel_repair")
            
        except Exception as e:
            warnings.append(f"Voxel repair failed: {e}")
            logger.warning(f"Voxel repair failed: {e}")
    
    # Remove small components
    if policy.remove_small_components:
        try:
            from validity.mesh.voxel_utils import remove_small_components
            
            original_parts = len(repaired.split(only_watertight=False))
            repaired = remove_small_components(
                repaired,
                min_faces=policy.min_component_faces,
            )
            final_parts = len(repaired.split(only_watertight=False))
            
            if original_parts != final_parts:
                metadata["components_removed"] = original_parts - final_parts
                metadata["operations_applied"].append("remove_small_components")
                
        except Exception as e:
            warnings.append(f"Remove small components failed: {e}")
            logger.warning(f"Remove small components failed: {e}")
    
    # Fill holes
    if policy.fill_holes:
        try:
            trimesh.repair.fill_holes(repaired)
            metadata["operations_applied"].append("fill_holes")
        except Exception as e:
            warnings.append(f"Fill holes failed: {e}")
            logger.warning(f"Fill holes failed: {e}")
    
    # Final cleanup
    repaired.merge_vertices()
    repaired.remove_unreferenced_vertices()
    
    if repaired.volume < 0:
        repaired.invert()
    trimesh.repair.fix_normals(repaired)
    
    # Final statistics
    metadata["final_vertex_count"] = len(repaired.vertices)
    metadata["final_face_count"] = len(repaired.faces)
    metadata["final_watertight"] = repaired.is_watertight
    
    report = OperationReport(
        operation="repair_mesh",
        success=True,
        requested_policy=policy.to_dict(),
        effective_policy=policy.to_dict(),
        warnings=warnings,
        metadata=metadata,
    )
    
    return repaired, report


__all__ = [
    "repair_mesh",
    "MeshRepairPolicy",
]
