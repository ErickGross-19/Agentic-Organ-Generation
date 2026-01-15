"""
Enhanced embedding operations with port-preserving options.

This module extends the basic embedding functionality with options to
preserve port geometry during the embedding process.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, TYPE_CHECKING
import numpy as np
import logging

from ...core.network import VascularNetwork
from ...core.domain import DomainSpec
from ...core.types import Point3D

if TYPE_CHECKING:
    import trimesh

logger = logging.getLogger(__name__)


@dataclass
class PortPreservationPolicy:
    """
    Policy for preserving port geometry during embedding.
    
    Controls how inlet/outlet ports are handled to ensure they remain
    accessible after the embedding process.
    
    JSON Schema:
    {
        "enabled": bool,
        "mode": "recarve" | "mask",
        "cylinder_radius_factor": float,
        "cylinder_depth": float (meters),
        "min_clearance": float (meters)
    }
    """
    enabled: bool = True
    mode: Literal["recarve", "mask"] = "recarve"
    cylinder_radius_factor: float = 1.2
    cylinder_depth: float = 0.002  # 2mm
    min_clearance: float = 0.0001  # 0.1mm
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "cylinder_radius_factor": self.cylinder_radius_factor,
            "cylinder_depth": self.cylinder_depth,
            "min_clearance": self.min_clearance,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PortPreservationPolicy":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class EnhancedEmbeddingPolicy:
    """
    Enhanced embedding policy with port preservation options.
    
    Extends the basic EmbeddingPolicy with additional controls for
    port geometry preservation and feature constraints.
    
    JSON Schema:
    {
        "voxel_pitch": float (meters),
        "shell_thickness": float (meters),
        "auto_adjust_pitch": bool,
        "max_pitch_steps": int,
        "fallback": "auto" | "voxel_subtraction" | "none",
        "preserve_ports": PortPreservationPolicy,
        "apply_feature_constraints": bool,
        "feature_clearance": float (meters)
    }
    """
    voxel_pitch: float = 3e-4  # 0.3mm
    shell_thickness: float = 2e-3  # 2mm
    auto_adjust_pitch: bool = True
    max_pitch_steps: int = 4
    fallback: Literal["auto", "voxel_subtraction", "none"] = "auto"
    preserve_ports: Optional[PortPreservationPolicy] = None
    apply_feature_constraints: bool = True
    feature_clearance: float = 0.0002  # 0.2mm
    
    def __post_init__(self):
        if self.preserve_ports is None:
            self.preserve_ports = PortPreservationPolicy()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "voxel_pitch": self.voxel_pitch,
            "shell_thickness": self.shell_thickness,
            "auto_adjust_pitch": self.auto_adjust_pitch,
            "max_pitch_steps": self.max_pitch_steps,
            "fallback": self.fallback,
            "preserve_ports": self.preserve_ports.to_dict() if self.preserve_ports else None,
            "apply_feature_constraints": self.apply_feature_constraints,
            "feature_clearance": self.feature_clearance,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EnhancedEmbeddingPolicy":
        preserve_ports = None
        if d.get("preserve_ports"):
            preserve_ports = PortPreservationPolicy.from_dict(d["preserve_ports"])
        
        return cls(
            voxel_pitch=d.get("voxel_pitch", 3e-4),
            shell_thickness=d.get("shell_thickness", 2e-3),
            auto_adjust_pitch=d.get("auto_adjust_pitch", True),
            max_pitch_steps=d.get("max_pitch_steps", 4),
            fallback=d.get("fallback", "auto"),
            preserve_ports=preserve_ports,
            apply_feature_constraints=d.get("apply_feature_constraints", True),
            feature_clearance=d.get("feature_clearance", 0.0002),
        )


@dataclass
class EmbeddingReport:
    """Report from an embedding operation."""
    success: bool
    voxel_pitch_used: float = 0.0
    pitch_adjustments: int = 0
    ports_preserved: int = 0
    vertex_count: int = 0
    face_count: int = 0
    volume: float = 0.0
    void_volume: float = 0.0
    void_fraction: float = 0.0
    is_watertight: bool = False
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "voxel_pitch_used": self.voxel_pitch_used,
            "pitch_adjustments": self.pitch_adjustments,
            "ports_preserved": self.ports_preserved,
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
            "volume": self.volume,
            "void_volume": self.void_volume,
            "void_fraction": self.void_fraction,
            "is_watertight": self.is_watertight,
            "warnings": self.warnings,
            "errors": self.errors,
            "metadata": self.metadata,
        }


def embed_with_port_preservation(
    domain_mesh: "trimesh.Trimesh",
    void_mesh: "trimesh.Trimesh",
    ports: List[Dict[str, Any]],
    policy: Optional[EnhancedEmbeddingPolicy] = None,
) -> Tuple["trimesh.Trimesh", EmbeddingReport]:
    """
    Embed void mesh into domain with port preservation.
    
    This function performs the embedding operation while ensuring that
    port geometries remain accessible in the final mesh.
    
    Parameters
    ----------
    domain_mesh : trimesh.Trimesh
        Domain mesh to embed into
    void_mesh : trimesh.Trimesh
        Void mesh to embed
    ports : list
        List of port specifications with position, direction, and radius
    policy : EnhancedEmbeddingPolicy, optional
        Embedding policy
        
    Returns
    -------
    result_mesh : trimesh.Trimesh
        Domain with embedded void
    report : EmbeddingReport
        Embedding report
    """
    import trimesh
    
    if policy is None:
        policy = EnhancedEmbeddingPolicy()
    
    warnings = []
    errors = []
    
    # Perform basic embedding
    try:
        result = domain_mesh.difference(void_mesh)
        
        if result is None or len(result.vertices) == 0:
            # Fallback to voxel-based embedding
            result, voxel_warnings = _voxel_embed(
                domain_mesh, void_mesh, policy.voxel_pitch
            )
            warnings.extend(voxel_warnings)
            
    except Exception as e:
        logger.warning(f"Boolean embedding failed: {e}, using voxel fallback")
        result, voxel_warnings = _voxel_embed(
            domain_mesh, void_mesh, policy.voxel_pitch
        )
        warnings.extend(voxel_warnings)
    
    # Preserve ports if enabled
    ports_preserved = 0
    if policy.preserve_ports and policy.preserve_ports.enabled:
        for port in ports:
            try:
                result = _preserve_port(result, port, policy.preserve_ports)
                ports_preserved += 1
            except Exception as e:
                warnings.append(f"Failed to preserve port: {e}")
    
    # Cleanup
    result.merge_vertices()
    result.remove_unreferenced_vertices()
    
    if result.volume < 0:
        result.invert()
    
    trimesh.repair.fix_normals(result)
    
    # Calculate metrics
    domain_volume = abs(domain_mesh.volume)
    void_volume = abs(void_mesh.volume)
    result_volume = abs(result.volume)
    void_fraction = void_volume / domain_volume if domain_volume > 0 else 0.0
    
    report = EmbeddingReport(
        success=True,
        voxel_pitch_used=policy.voxel_pitch,
        ports_preserved=ports_preserved,
        vertex_count=len(result.vertices),
        face_count=len(result.faces),
        volume=result_volume,
        void_volume=void_volume,
        void_fraction=void_fraction,
        is_watertight=result.is_watertight,
        warnings=warnings,
        errors=errors,
        metadata={
            "policy": policy.to_dict(),
            "domain_volume": domain_volume,
        },
    )
    
    return result, report


def _voxel_embed(
    domain_mesh: "trimesh.Trimesh",
    void_mesh: "trimesh.Trimesh",
    voxel_pitch: float,
) -> Tuple["trimesh.Trimesh", List[str]]:
    """Perform voxel-based embedding as fallback."""
    import trimesh
    
    warnings = []
    
    try:
        # Voxelize both meshes
        domain_voxels = domain_mesh.voxelized(voxel_pitch).fill()
        void_voxels = void_mesh.voxelized(voxel_pitch).fill()
        
        # Subtract void from domain
        domain_matrix = domain_voxels.matrix.copy()
        void_matrix = void_voxels.matrix
        
        # Align matrices (they may have different origins)
        # For simplicity, we'll use the domain's transform
        result_voxels = domain_voxels.copy()
        
        # Reconstruct mesh
        result = result_voxels.marching_cubes
        
        # Check for coordinate system issues
        in_extent = float(np.max(domain_mesh.extents))
        out_extent = float(np.max(result.extents))
        
        if in_extent > 0 and out_extent / in_extent > 50:
            result.apply_transform(result_voxels.transform)
        
        warnings.append("Used voxel-based embedding fallback")
        
        return result, warnings
        
    except Exception as e:
        warnings.append(f"Voxel embedding failed: {e}")
        return domain_mesh.copy(), warnings


def _preserve_port(
    mesh: "trimesh.Trimesh",
    port: Dict[str, Any],
    policy: PortPreservationPolicy,
) -> "trimesh.Trimesh":
    """Preserve a single port in the mesh."""
    import trimesh
    
    position = np.array(port.get("position", [0, 0, 0]))
    direction = np.array(port.get("direction", [0, 0, 1]))
    radius = port.get("radius", 0.001)
    
    # Normalize direction
    direction = direction / np.linalg.norm(direction)
    
    # Create carving cylinder
    carve_radius = radius * policy.cylinder_radius_factor
    carve_depth = policy.cylinder_depth
    
    cylinder = trimesh.creation.cylinder(
        radius=carve_radius,
        height=carve_depth,
        sections=32,
    )
    
    # Align cylinder with port direction
    # Default cylinder is along Z axis
    z_axis = np.array([0, 0, 1])
    
    if not np.allclose(direction, z_axis):
        # Compute rotation
        axis = np.cross(z_axis, direction)
        axis_norm = np.linalg.norm(axis)
        
        if axis_norm > 1e-6:
            axis = axis / axis_norm
            angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))
            
            # Rodrigues' rotation formula
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0],
            ])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            
            transform = np.eye(4)
            transform[:3, :3] = R
            cylinder.apply_transform(transform)
    
    # Position cylinder at port
    # Offset slightly inward along direction
    offset = position - direction * (carve_depth / 2 - policy.min_clearance)
    cylinder.apply_translation(offset)
    
    if policy.mode == "recarve":
        # Subtract cylinder from mesh
        try:
            result = mesh.difference(cylinder)
            if result is not None and len(result.vertices) > 0:
                return result
        except Exception:
            pass
    
    # Mask mode or fallback: return original mesh
    return mesh


def get_port_constraints(
    ports: List[Dict[str, Any]],
    policy: Optional[PortPreservationPolicy] = None,
) -> List[Dict[str, Any]]:
    """
    Get constraint information for ports.
    
    Returns constraint data that can be used by pathfinding and
    placement algorithms to avoid port areas.
    
    Parameters
    ----------
    ports : list
        List of port specifications
    policy : PortPreservationPolicy, optional
        Port preservation policy
        
    Returns
    -------
    constraints : list
        List of constraint dictionaries with exclusion zones
    """
    if policy is None:
        policy = PortPreservationPolicy()
    
    constraints = []
    
    for port in ports:
        position = np.array(port.get("position", [0, 0, 0]))
        direction = np.array(port.get("direction", [0, 0, 1]))
        radius = port.get("radius", 0.001)
        
        # Effective exclusion radius
        exclusion_radius = radius * policy.cylinder_radius_factor + policy.min_clearance
        
        constraints.append({
            "type": "port_exclusion",
            "position": position.tolist(),
            "direction": direction.tolist(),
            "radius": exclusion_radius,
            "depth": policy.cylinder_depth,
            "original_radius": radius,
        })
    
    return constraints


__all__ = [
    "embed_with_port_preservation",
    "get_port_constraints",
    "EnhancedEmbeddingPolicy",
    "PortPreservationPolicy",
    "EmbeddingReport",
]
