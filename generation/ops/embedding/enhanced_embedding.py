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
    
    NOTE: Only "recarve" mode is supported. The "mask" mode has been deprecated
    as it does not properly preserve port geometry.
    
    JSON Schema:
    {
        "enabled": bool,
        "mode": "recarve",
        "cylinder_radius_factor": float,
        "cylinder_depth": float (meters),
        "min_clearance": float (meters)
    }
    """
    enabled: bool = True
    mode: Literal["recarve"] = "recarve"
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
class PortRecarveResult:
    """Result of recarving a single port."""
    port_index: int
    position: Tuple[float, float, float]
    direction: Tuple[float, float, float]
    original_radius: float
    carve_radius: float
    carve_depth: float
    voxels_carved: int = 0
    reached_boundary: bool = False
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "port_index": self.port_index,
            "position": self.position,
            "direction": self.direction,
            "original_radius": self.original_radius,
            "carve_radius": self.carve_radius,
            "carve_depth": self.carve_depth,
            "voxels_carved": self.voxels_carved,
            "reached_boundary": self.reached_boundary,
            "warnings": self.warnings,
        }


@dataclass
class RecarveReport:
    """
    Report from voxel-based port recarving operation.
    
    Contains per-port metrics and validation checks for debugging
    DesignSpec runs without needing to open meshes.
    """
    success: bool
    ports_carved: int = 0
    total_voxels_carved: int = 0
    voxel_pitch_used: float = 0.0
    port_results: List[PortRecarveResult] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    is_watertight_after: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "ports_carved": self.ports_carved,
            "total_voxels_carved": self.total_voxels_carved,
            "voxel_pitch_used": self.voxel_pitch_used,
            "port_results": [r.to_dict() for r in self.port_results],
            "warnings": self.warnings,
            "errors": self.errors,
            "is_watertight_after": self.is_watertight_after,
        }


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
    recarve_report: Optional[RecarveReport] = None
    
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
            "recarve_report": self.recarve_report.to_dict() if self.recarve_report else None,
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
    """
    Perform voxel-based embedding as fallback.
    
    DEPRECATED: This fallback should only be used when boolean operations fail.
    The preferred path is direct mesh boolean subtraction. This function is
    retained for robustness but emits a warning when used.
    
    The algorithm:
    1. Voxelize both domain and void meshes at the same pitch
    2. Transform void voxels to domain's coordinate frame
    3. Subtract void voxels from domain voxels
    4. Reconstruct mesh via marching cubes
    """
    import trimesh
    from trimesh.voxel import VoxelGrid
    
    warnings = []
    warnings.append(
        "DEPRECATED: Using voxel-based embedding fallback. "
        "This may produce lower quality results than direct boolean operations."
    )
    
    try:
        # Voxelize both meshes
        domain_voxels = domain_mesh.voxelized(voxel_pitch).fill()
        void_voxels = void_mesh.voxelized(voxel_pitch).fill()
        
        # Get the domain's transform and origin
        domain_transform = domain_voxels.transform
        domain_origin = domain_transform[:3, 3]
        domain_scale = domain_transform[0, 0]  # Assuming uniform scale
        
        # Get void's transform
        void_transform = void_voxels.transform
        void_origin = void_transform[:3, 3]
        void_scale = void_transform[0, 0]
        
        # Create a copy of domain matrix for modification
        result_matrix = domain_voxels.matrix.copy()
        
        # Transform void voxel coordinates to domain voxel coordinates
        # and subtract void from domain
        void_indices = np.argwhere(void_voxels.matrix)
        
        for idx in void_indices:
            # Convert void voxel index to world coordinates
            world_pos = void_origin + idx * void_scale
            
            # Convert world coordinates to domain voxel index
            domain_idx = ((world_pos - domain_origin) / domain_scale).astype(int)
            
            # Check bounds and subtract
            if (0 <= domain_idx[0] < result_matrix.shape[0] and
                0 <= domain_idx[1] < result_matrix.shape[1] and
                0 <= domain_idx[2] < result_matrix.shape[2]):
                result_matrix[domain_idx[0], domain_idx[1], domain_idx[2]] = False
        
        # Create new voxel grid with subtracted result
        result_voxels = VoxelGrid(
            trimesh.voxel.encoding.DenseEncoding(result_matrix),
            transform=domain_transform,
        )
        
        # Reconstruct mesh via marching cubes
        result = result_voxels.marching_cubes
        
        # Check for coordinate system issues
        in_extent = float(np.max(domain_mesh.extents))
        out_extent = float(np.max(result.extents))
        
        if in_extent > 0 and out_extent / in_extent > 50:
            result.apply_transform(domain_transform)
        
        return result, warnings
        
    except Exception as e:
        warnings.append(f"Voxel embedding failed: {e}")
        logger.error(f"Voxel embedding fallback failed: {e}")
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


def voxel_recarve_ports(
    mesh: "trimesh.Trimesh",
    ports: List[Dict[str, Any]],
    voxel_pitch: float = 3e-4,
    carve_radius_factor: float = 1.2,
    carve_depth: float = 0.002,
    carve_shape: Literal["cylinder", "frustum"] = "cylinder",
) -> Tuple["trimesh.Trimesh", RecarveReport]:
    """
    Voxel-based port preservation via recarving.
    
    This function carves port openings using voxel grid operations,
    eliminating dependency on boolean mesh backends (Blender/Cork/etc.).
    Works consistently across all environments.
    
    Algorithm:
    1. Voxelize the input mesh
    2. For each port, create a cylinder/frustum mask aligned to port direction
    3. Subtract the mask from the solid voxels
    4. Reconstruct mesh via marching cubes
    5. Validate watertightness and boundary reach
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        The embedded solid mesh to carve ports into
    ports : list of dict
        Port specifications with keys:
        - position: (x, y, z) tuple in meters
        - direction: (dx, dy, dz) tuple (outward normal)
        - radius: float in meters
    voxel_pitch : float
        Voxel size in meters (default: 0.3mm)
    carve_radius_factor : float
        Factor to multiply port radius for carving (default: 1.2)
    carve_depth : float
        Depth of carving in meters (default: 2mm)
    carve_shape : str
        Shape of carving: "cylinder" or "frustum" (default: cylinder)
        
    Returns
    -------
    result_mesh : trimesh.Trimesh
        Mesh with ports carved
    report : RecarveReport
        Detailed report with per-port metrics
    """
    import trimesh
    from trimesh.voxel import VoxelGrid
    
    warnings = []
    errors = []
    port_results = []
    total_voxels_carved = 0
    
    if not ports:
        return mesh, RecarveReport(
            success=True,
            ports_carved=0,
            voxel_pitch_used=voxel_pitch,
            is_watertight_after=mesh.is_watertight,
        )
    
    try:
        # Get mesh bounds
        mesh_min = mesh.bounds[0]
        mesh_max = mesh.bounds[1]
        ref_extent = float(np.max(mesh.extents))
        
        # Add padding for port carving
        pad = carve_depth + 2 * voxel_pitch
        grid_min = mesh_min - pad
        grid_max = mesh_max + pad
        
        grid_shape = np.ceil((grid_max - grid_min) / voxel_pitch).astype(int) + 1
        grid_shape = np.maximum(grid_shape, 1)
        
        # Voxelize the mesh
        mesh_vox = mesh.voxelized(voxel_pitch).fill()
        mesh_matrix = mesh_vox.matrix.astype(bool)
        mesh_origin = mesh_vox.transform[:3, 3].astype(float)
        
        # Create aligned grid
        aligned_solid = np.zeros(tuple(grid_shape), dtype=bool)
        
        # Paste mesh voxels into aligned grid
        offset_vox = np.round((mesh_origin - grid_min) / voxel_pitch).astype(int)
        src_start = np.maximum(-offset_vox, 0)
        dst_start = np.maximum(offset_vox, 0)
        copy_size = np.minimum(
            np.array(mesh_matrix.shape) - src_start,
            np.array(aligned_solid.shape) - dst_start
        )
        copy_size = np.maximum(copy_size, 0)
        
        if np.all(copy_size > 0):
            aligned_solid[
                dst_start[0]:dst_start[0] + copy_size[0],
                dst_start[1]:dst_start[1] + copy_size[1],
                dst_start[2]:dst_start[2] + copy_size[2],
            ] = mesh_matrix[
                src_start[0]:src_start[0] + copy_size[0],
                src_start[1]:src_start[1] + copy_size[1],
                src_start[2]:src_start[2] + copy_size[2],
            ]
        
        # Track original solid voxels for boundary check
        original_solid_count = int(aligned_solid.sum())
        
        # Process each port
        for i, port in enumerate(ports):
            position = np.array(port.get("position", [0, 0, 0]), dtype=float)
            direction = np.array(port.get("direction", [0, 0, 1]), dtype=float)
            radius = float(port.get("radius", 0.001))
            
            # Normalize direction
            dir_norm = np.linalg.norm(direction)
            if dir_norm > 1e-9:
                direction = direction / dir_norm
            else:
                direction = np.array([0, 0, 1])
            
            carve_radius = radius * carve_radius_factor
            port_warnings = []
            
            # Create cylinder mask in voxel space
            # Generate voxel coordinates
            voxel_coords = np.indices(tuple(grid_shape)).reshape(3, -1).T
            world_coords = voxel_coords * voxel_pitch + grid_min
            
            # Vector from port position to each voxel
            rel_pos = world_coords - position
            
            # Project onto port direction (along cylinder axis)
            along_axis = np.dot(rel_pos, direction)
            
            # Distance from axis (perpendicular)
            proj_on_axis = np.outer(along_axis, direction)
            perp_vec = rel_pos - proj_on_axis
            dist_from_axis = np.linalg.norm(perp_vec, axis=1)
            
            # Cylinder mask: within radius and within depth
            # Carve from port position inward (opposite to direction)
            # Port position is at the surface, carve inward
            if carve_shape == "frustum":
                # Frustum: radius decreases linearly with depth
                # At surface (along_axis=0): carve_radius
                # At depth (along_axis=-carve_depth): carve_radius * 0.5
                depth_fraction = np.clip(-along_axis / carve_depth, 0, 1)
                effective_radius = carve_radius * (1.0 - 0.5 * depth_fraction)
                in_shape = dist_from_axis <= effective_radius
            else:
                # Cylinder: constant radius
                in_shape = dist_from_axis <= carve_radius
            
            # Within depth range (carve inward from port position)
            in_depth = (along_axis >= -carve_depth) & (along_axis <= voxel_pitch)
            
            # Combined mask
            carve_mask_flat = in_shape & in_depth
            carve_mask = carve_mask_flat.reshape(tuple(grid_shape))
            
            # Count voxels to carve (only those that are currently solid)
            voxels_to_carve = aligned_solid & carve_mask
            voxels_carved = int(voxels_to_carve.sum())
            
            # Check if carve reached boundary (at least some voxels at edge of solid)
            # A port "reaches boundary" if the carved region includes voxels that
            # were at the edge of the solid (had non-solid neighbors)
            reached_boundary = False
            if voxels_carved > 0:
                # Check if any carved voxels were at the boundary
                # by checking if they had any non-solid neighbors before carving
                from scipy import ndimage
                
                # Dilate the inverse of solid to find boundary voxels
                non_solid = ~aligned_solid
                boundary_region = ndimage.binary_dilation(non_solid, iterations=1) & aligned_solid
                reached_boundary = bool(np.any(voxels_to_carve & boundary_region))
                
                if not reached_boundary:
                    port_warnings.append(
                        f"Port {i}: carve didn't reach outside boundary"
                    )
            else:
                port_warnings.append(f"Port {i}: no voxels carved (port may be outside mesh)")
            
            # Apply carve
            aligned_solid[carve_mask] = False
            total_voxels_carved += voxels_carved
            
            port_results.append(PortRecarveResult(
                port_index=i,
                position=tuple(position),
                direction=tuple(direction),
                original_radius=radius,
                carve_radius=carve_radius,
                carve_depth=carve_depth,
                voxels_carved=voxels_carved,
                reached_boundary=reached_boundary,
                warnings=port_warnings,
            ))
            warnings.extend(port_warnings)
        
        # Reconstruct mesh from voxels
        if not aligned_solid.any():
            errors.append("Result is empty after port carving")
            return mesh, RecarveReport(
                success=False,
                ports_carved=0,
                voxel_pitch_used=voxel_pitch,
                errors=errors,
            )
        
        # Create transform for voxel grid
        T = np.eye(4, dtype=float)
        T[0, 0] = voxel_pitch
        T[1, 1] = voxel_pitch
        T[2, 2] = voxel_pitch
        T[:3, 3] = grid_min
        
        vg = VoxelGrid(aligned_solid, transform=T)
        result_mesh = vg.marching_cubes
        
        # Fix coordinate system if needed
        out_extent = float(np.max(result_mesh.extents))
        if ref_extent > 0 and out_extent / ref_extent > 50:
            result_mesh.apply_transform(vg.transform)
        
        # Cleanup
        result_mesh.merge_vertices()
        result_mesh.remove_unreferenced_vertices()
        if result_mesh.volume < 0:
            result_mesh.invert()
        trimesh.repair.fix_normals(result_mesh)
        
        is_watertight = result_mesh.is_watertight
        if not is_watertight:
            warnings.append("Result mesh is not watertight after port carving")
        
        report = RecarveReport(
            success=True,
            ports_carved=len(ports),
            total_voxels_carved=total_voxels_carved,
            voxel_pitch_used=voxel_pitch,
            port_results=port_results,
            warnings=warnings,
            errors=errors,
            is_watertight_after=is_watertight,
        )
        
        return result_mesh, report
        
    except Exception as e:
        logger.error(f"Voxel port recarving failed: {e}")
        errors.append(f"Voxel recarving failed: {e}")
        return mesh, RecarveReport(
            success=False,
            ports_carved=0,
            voxel_pitch_used=voxel_pitch,
            port_results=port_results,
            warnings=warnings,
            errors=errors,
        )


def _compute_budget_first_pitch(
    domain: "DomainSpec",
    requested_pitch: float,
    max_voxels: int = 50_000_000,
    pitch_step_factor: float = 1.5,
) -> Tuple[float, bool]:
    """
    Compute the appropriate voxel pitch based on budget constraints.
    
    This implements budget-first pitch selection: instead of starting with
    the requested pitch and retrying on MemoryError, we compute the minimum
    pitch that fits within the voxel budget up front.
    
    Parameters
    ----------
    domain : DomainSpec
        Domain specification for computing bounds
    requested_pitch : float
        Requested voxel pitch in meters
    max_voxels : int
        Maximum number of voxels allowed (default: 50M)
    pitch_step_factor : float
        Factor to increase pitch by when relaxing (default: 1.5)
        
    Returns
    -------
    effective_pitch : float
        The pitch to use (may be larger than requested if budget exceeded)
    was_relaxed : bool
        True if pitch was relaxed due to budget constraints
    """
    from ...core.domain import BoxDomain, CylinderDomain, EllipsoidDomain
    
    # Get domain bounds
    bounds = domain.get_bounds()
    domain_min = np.array([bounds[0], bounds[2], bounds[4]])
    domain_max = np.array([bounds[1], bounds[3], bounds[5]])
    
    # Add padding
    pad = 2 * requested_pitch
    domain_min_padded = domain_min - pad
    domain_max_padded = domain_max + pad
    
    # Compute grid shape at requested pitch
    grid_shape = np.ceil((domain_max_padded - domain_min_padded) / requested_pitch).astype(int) + 1
    total_voxels = int(np.prod(grid_shape))
    
    if total_voxels <= max_voxels:
        return requested_pitch, False
    
    # Relax pitch until within budget
    current_pitch = requested_pitch
    while total_voxels > max_voxels:
        current_pitch *= pitch_step_factor
        pad = 2 * current_pitch
        domain_min_padded = domain_min - pad
        domain_max_padded = domain_max + pad
        grid_shape = np.ceil((domain_max_padded - domain_min_padded) / current_pitch).astype(int) + 1
        total_voxels = int(np.prod(grid_shape))
    
    logger.warning(
        f"Relaxed embedding pitch from {requested_pitch:.6f}m to {current_pitch:.6f}m "
        f"to fit within voxel budget ({max_voxels:,} voxels)"
    )
    
    return current_pitch, True


def embed_void_mesh_as_negative_space(
    void_mesh: "trimesh.Trimesh",
    domain: "DomainSpec",
    voxel_pitch: float = 3e-4,
    output_void: bool = True,
    output_shell: bool = False,
    shell_thickness: float = 2e-3,
    auto_adjust_pitch: bool = True,
    max_pitch_steps: int = 4,
    max_voxels: Optional[int] = None,
) -> Tuple["trimesh.Trimesh", "trimesh.Trimesh", Optional["trimesh.Trimesh"], Dict[str, Any]]:
    """
    C1 FIX: Embed an in-memory void mesh into a domain as negative space.
    
    This is the canonical mesh-based embedding function that takes in-memory
    trimesh objects directly, unlike embed_tree_as_negative_space which takes
    a file path.
    
    Uses voxel subtraction (most robust method) to carve the void from the domain.
    
    Budget-first pitch selection: If max_voxels is specified, the pitch will be
    relaxed up front if needed to fit within the voxel budget, rather than
    waiting for MemoryError.
    
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
    max_voxels : int, optional
        Maximum voxel budget. If specified, pitch will be relaxed up front
        to fit within budget (budget-first selection).
        
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
    
    # Budget-first pitch selection: compute appropriate pitch up front
    pitch_was_relaxed = False
    if max_voxels is not None:
        voxel_pitch, pitch_was_relaxed = _compute_budget_first_pitch(
            domain, voxel_pitch, max_voxels
        )
    
    metadata = {
        "voxel_pitch": voxel_pitch,
        "auto_adjust_pitch": auto_adjust_pitch,
        "pitch_adjustments": 0,
        "budget_first_relaxed": pitch_was_relaxed,
        "max_voxels_budget": max_voxels,
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
    "voxel_recarve_ports",
    "EnhancedEmbeddingPolicy",
    "PortPreservationPolicy",
    "EmbeddingReport",
    "RecarveReport",
    "PortRecarveResult",
]
