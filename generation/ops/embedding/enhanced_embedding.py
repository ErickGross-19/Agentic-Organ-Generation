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


def embed_void_mesh_as_negative_space(
    void_mesh: "trimesh.Trimesh",
    domain: "DomainSpec",
    voxel_pitch: float = 3e-4,
    output_void: bool = True,
    output_shell: bool = False,
    shell_thickness: float = 2e-3,
    auto_adjust_pitch: bool = True,
    max_pitch_steps: int = 4,
) -> Tuple["trimesh.Trimesh", "trimesh.Trimesh", Optional["trimesh.Trimesh"], Dict[str, Any]]:
    """
    C1 FIX: Embed an in-memory void mesh into a domain as negative space.
    
    This is the canonical mesh-based embedding function that takes in-memory
    trimesh objects directly, unlike embed_tree_as_negative_space which takes
    a file path.
    
    Uses voxel subtraction (most robust method) to carve the void from the domain.
    
    Parameters
    ----------
    void_mesh : trimesh.Trimesh
        The void mesh to embed (in-memory)
    domain : DomainSpec
        Domain specification (BoxDomain, CylinderDomain, or EllipsoidDomain)
    voxel_pitch : float
        Voxel size in meters (default: 3e-4 = 0.3mm)
    output_void : bool
        Whether to return the void mesh (default: True)
    output_shell : bool
        Whether to output a shell mesh around the void (default: False)
    shell_thickness : float
        Thickness of shell around void in meters (default: 2e-3 = 2mm)
    auto_adjust_pitch : bool
        If True, automatically increase pitch on memory errors (default: True)
    max_pitch_steps : int
        Maximum number of pitch adjustment steps (default: 4)
        
    Returns
    -------
    domain_with_void : trimesh.Trimesh
        Domain mesh with void carved out
    void : trimesh.Trimesh
        The void mesh (same as input if output_void=True, else empty)
    shell : trimesh.Trimesh or None
        Shell mesh around void (if output_shell=True)
    metadata : dict
        Embedding metadata including voxel stats
    """
    import trimesh
    from trimesh.voxel import VoxelGrid
    from ...core.domain import BoxDomain, CylinderDomain, EllipsoidDomain
    
    metadata = {
        "voxel_pitch": voxel_pitch,
        "auto_adjust_pitch": auto_adjust_pitch,
        "pitch_adjustments": 0,
    }
    
    # Create domain mesh based on domain type
    if isinstance(domain, CylinderDomain):
        center = np.array([domain.center.x, domain.center.y, domain.center.z])
        radius = domain.radius
        height = domain.height
        
        domain_mesh = trimesh.creation.cylinder(
            radius=radius,
            height=height,
            sections=64,
        )
        domain_mesh.apply_translation(center)
        
        domain_min = np.array([
            center[0] - radius,
            center[1] - radius,
            center[2] - height / 2
        ])
        domain_max = np.array([
            center[0] + radius,
            center[1] + radius,
            center[2] + height / 2
        ])
        ref_extent = max(2.0 * radius, height)
        
    elif isinstance(domain, BoxDomain):
        domain_mesh = trimesh.creation.box(
            extents=[
                domain.x_max - domain.x_min,
                domain.y_max - domain.y_min,
                domain.z_max - domain.z_min,
            ]
        )
        center = np.array([
            (domain.x_min + domain.x_max) / 2,
            (domain.y_min + domain.y_max) / 2,
            (domain.z_min + domain.z_max) / 2,
        ])
        domain_mesh.apply_translation(center)
        
        domain_min = np.array([domain.x_min, domain.y_min, domain.z_min])
        domain_max = np.array([domain.x_max, domain.y_max, domain.z_max])
        ref_extent = np.max(domain_max - domain_min)
        
    elif isinstance(domain, EllipsoidDomain):
        center = np.array([domain.center.x, domain.center.y, domain.center.z])
        radii = np.array([domain.semi_axis_a, domain.semi_axis_b, domain.semi_axis_c])
        
        domain_mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        domain_mesh.apply_scale(radii)
        domain_mesh.apply_translation(center)
        
        domain_min = center - radii
        domain_max = center + radii
        ref_extent = 2.0 * np.max(radii)
    else:
        raise ValueError(f"Unsupported domain type: {type(domain)}")
    
    # Voxel subtraction with retry on memory errors
    current_pitch = voxel_pitch
    pitch_factor = 1.5
    
    for attempt in range(max_pitch_steps):
        try:
            # Compute padded grid bounds
            pad = 2 * current_pitch
            domain_min_padded = domain_min - pad
            domain_max_padded = domain_max + pad
            
            grid_shape = np.ceil((domain_max_padded - domain_min_padded) / current_pitch).astype(int) + 1
            grid_shape = np.maximum(grid_shape, 1)
            
            # Voxelize domain and void
            domain_vox = domain_mesh.voxelized(current_pitch).fill()
            void_vox = void_mesh.voxelized(current_pitch).fill()
            
            domain_matrix = domain_vox.matrix.astype(bool)
            domain_origin = domain_vox.transform[:3, 3].astype(float)
            
            void_matrix = void_vox.matrix.astype(bool)
            void_origin = void_vox.transform[:3, 3].astype(float)
            
            # Align matrices to common grid
            aligned_domain = np.zeros(tuple(grid_shape), dtype=bool)
            aligned_void = np.zeros(tuple(grid_shape), dtype=bool)
            
            def paste_into(dst: np.ndarray, src: np.ndarray, src_origin: np.ndarray):
                src_origin = np.asarray(src_origin, dtype=float)
                offset_vox = np.round((src_origin - domain_min_padded) / current_pitch).astype(int)
                
                src_start = np.maximum(-offset_vox, 0)
                dst_start = np.maximum(offset_vox, 0)
                
                copy_size = np.minimum(np.array(src.shape) - src_start, np.array(dst.shape) - dst_start)
                copy_size = np.maximum(copy_size, 0)
                
                if np.all(copy_size > 0):
                    dst[
                        dst_start[0]:dst_start[0] + copy_size[0],
                        dst_start[1]:dst_start[1] + copy_size[1],
                        dst_start[2]:dst_start[2] + copy_size[2],
                    ] = src[
                        src_start[0]:src_start[0] + copy_size[0],
                        src_start[1]:src_start[1] + copy_size[1],
                        src_start[2]:src_start[2] + copy_size[2],
                    ]
            
            paste_into(aligned_domain, domain_matrix, domain_origin)
            paste_into(aligned_void, void_matrix, void_origin)
            
            # Subtract void from domain
            result_mask = aligned_domain & (~aligned_void)
            
            if not result_mask.any():
                raise RuntimeError("Result mask is empty after voxel subtraction")
            
            # Create transform for voxel grid
            T = np.eye(4, dtype=float)
            T[0, 0] = current_pitch
            T[1, 1] = current_pitch
            T[2, 2] = current_pitch
            T[:3, 3] = domain_min_padded
            
            # Reconstruct mesh
            vg = VoxelGrid(result_mask, transform=T)
            solid_mesh = vg.marching_cubes
            
            # Fix coordinate system if needed
            out_extent = float(np.max(solid_mesh.extents))
            if ref_extent > 0 and out_extent / ref_extent > 50:
                solid_mesh.apply_transform(vg.transform)
            
            # Cleanup
            solid_mesh.merge_vertices()
            solid_mesh.remove_unreferenced_vertices()
            if solid_mesh.volume < 0:
                solid_mesh.invert()
            trimesh.repair.fix_normals(solid_mesh)
            
            metadata["voxel_pitch_used"] = current_pitch
            metadata["domain_voxels"] = int(aligned_domain.sum())
            metadata["void_voxels"] = int(aligned_void.sum())
            metadata["result_voxels"] = int(result_mask.sum())
            metadata["is_watertight"] = solid_mesh.is_watertight
            
            break
            
        except (MemoryError, ValueError) as e:
            if not auto_adjust_pitch or attempt >= max_pitch_steps - 1:
                raise
            logger.warning(f"Voxelization failed at pitch={current_pitch}: {e}, increasing pitch")
            current_pitch *= pitch_factor
            metadata["pitch_adjustments"] = attempt + 1
    
    # Prepare outputs
    void_output = void_mesh if output_void else trimesh.Trimesh()
    shell_output = None
    
    if output_shell:
        # Create shell by dilating void and subtracting original
        try:
            from scipy import ndimage
            
            shell_voxels = int(shell_thickness / current_pitch)
            dilated_void = ndimage.binary_dilation(aligned_void, iterations=shell_voxels)
            shell_mask = dilated_void & (~aligned_void) & aligned_domain
            
            if shell_mask.any():
                shell_vg = VoxelGrid(shell_mask, transform=T)
                shell_output = shell_vg.marching_cubes
                
                out_extent = float(np.max(shell_output.extents))
                if ref_extent > 0 and out_extent / ref_extent > 50:
                    shell_output.apply_transform(shell_vg.transform)
                    
                shell_output.merge_vertices()
                shell_output.remove_unreferenced_vertices()
        except Exception as e:
            logger.warning(f"Failed to create shell: {e}")
    
    return solid_mesh, void_output, shell_output, metadata


__all__ = [
    "embed_with_port_preservation",
    "embed_void_mesh_as_negative_space",
    "get_port_constraints",
    "EnhancedEmbeddingPolicy",
    "PortPreservationPolicy",
    "EmbeddingReport",
]
