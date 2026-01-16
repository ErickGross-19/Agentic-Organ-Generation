"""
Canonical domain-to-mesh conversion.

This module provides a single policy-driven path from any runtime Domain
to a watertight mesh suitable for embedding and validity checks.

Ticket B1: Implement canonical domain_to_mesh() using DomainMeshingPolicy
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, Union
import logging
import hashlib
import json

import numpy as np

from aog_policies.domain import (
    DomainMeshingPolicy,
    PrimitiveMeshingPolicy,
    MeshDomainPolicy,
    ImplicitMeshingPolicy,
)
from aog_policies.resolution import ResolutionPolicy
from ..utils.resolution_resolver import resolve_pitch, ResolutionResult
from ..policies import OperationReport

if TYPE_CHECKING:
    import trimesh
    from ..core.domain import DomainSpec

logger = logging.getLogger(__name__)


@dataclass
class MeshingContext:
    """
    Context for domain meshing operations.
    
    Provides caching and tracking of meshing operations.
    """
    cache: Dict[str, "trimesh.Trimesh"] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def get_cached(self, key: str) -> Optional["trimesh.Trimesh"]:
        """Get a cached mesh by key."""
        return self.cache.get(key)
    
    def set_cached(self, key: str, mesh: "trimesh.Trimesh") -> None:
        """Cache a mesh by key."""
        self.cache[key] = mesh
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)
        logger.warning(warning)
    
    def set_metric(self, key: str, value: Any) -> None:
        """Set a metric value."""
        self.metrics[key] = value


def _compute_domain_hash(domain: "DomainSpec") -> str:
    """Compute a stable hash for a domain for caching purposes."""
    domain_dict = domain.to_dict()
    domain_json = json.dumps(domain_dict, sort_keys=True, default=str)
    return hashlib.md5(domain_json.encode()).hexdigest()


def _mesh_box(
    domain: "DomainSpec",
    policy: PrimitiveMeshingPolicy,
) -> "trimesh.Trimesh":
    """Create mesh for BoxDomain."""
    import trimesh
    
    bounds = domain.get_bounds()
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    
    extents = [x_max - x_min, y_max - y_min, z_max - z_min]
    center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
    
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    
    return mesh


def _mesh_cylinder(
    domain: "DomainSpec",
    policy: PrimitiveMeshingPolicy,
) -> "trimesh.Trimesh":
    """Create mesh for CylinderDomain."""
    import trimesh
    
    mesh = trimesh.creation.cylinder(
        radius=domain.radius,
        height=domain.height,
        sections=policy.sections_radial,
    )
    
    center = [domain.center.x, domain.center.y, domain.center.z]
    mesh.apply_translation(center)
    
    return mesh


def _mesh_ellipsoid(
    domain: "DomainSpec",
    policy: PrimitiveMeshingPolicy,
) -> "trimesh.Trimesh":
    """Create mesh for EllipsoidDomain."""
    import trimesh
    
    mesh = trimesh.creation.icosphere(subdivisions=policy.subdivisions)
    
    mesh.vertices[:, 0] *= domain.semi_axis_a
    mesh.vertices[:, 1] *= domain.semi_axis_b
    mesh.vertices[:, 2] *= domain.semi_axis_c
    
    center = [domain.center.x, domain.center.y, domain.center.z]
    mesh.apply_translation(center)
    
    return mesh


def _mesh_sphere(
    domain: "DomainSpec",
    policy: PrimitiveMeshingPolicy,
) -> "trimesh.Trimesh":
    """Create mesh for SphereDomain."""
    import trimesh
    
    mesh = trimesh.creation.icosphere(
        subdivisions=policy.subdivisions,
        radius=domain.radius,
    )
    
    center = [domain.center.x, domain.center.y, domain.center.z]
    mesh.apply_translation(center)
    
    return mesh


def _mesh_capsule(
    domain: "DomainSpec",
    policy: PrimitiveMeshingPolicy,
) -> "trimesh.Trimesh":
    """Create mesh for CapsuleDomain."""
    import trimesh
    
    mesh = trimesh.creation.capsule(
        radius=domain.radius,
        height=domain.length,
    )
    
    axis = np.array(domain.axis)
    z_axis = np.array([0, 0, 1])
    
    if not np.allclose(axis, z_axis):
        rotation_axis = np.cross(z_axis, axis)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        
        if rotation_axis_norm > 1e-10:
            rotation_axis = rotation_axis / rotation_axis_norm
            angle = np.arccos(np.clip(np.dot(z_axis, axis), -1, 1))
            
            c = np.cos(angle)
            s = np.sin(angle)
            t = 1 - c
            x, y, z = rotation_axis
            
            rotation_matrix = np.array([
                [t*x*x + c,   t*x*y - s*z, t*x*z + s*y, 0],
                [t*x*y + s*z, t*y*y + c,   t*y*z - s*x, 0],
                [t*x*z - s*y, t*y*z + s*x, t*z*z + c,   0],
                [0,           0,           0,           1],
            ])
            mesh.apply_transform(rotation_matrix)
    
    center = [domain.center.x, domain.center.y, domain.center.z]
    mesh.apply_translation(center)
    
    return mesh


def _mesh_frustum(
    domain: "DomainSpec",
    policy: PrimitiveMeshingPolicy,
) -> "trimesh.Trimesh":
    """Create mesh for FrustumDomain."""
    import trimesh
    
    sections = policy.sections_radial
    theta = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    
    half_height = domain.height / 2
    
    bottom_vertices = np.column_stack([
        domain.radius_bottom * np.cos(theta),
        domain.radius_bottom * np.sin(theta),
        np.full(sections, -half_height),
    ])
    
    top_vertices = np.column_stack([
        domain.radius_top * np.cos(theta),
        domain.radius_top * np.sin(theta),
        np.full(sections, half_height),
    ])
    
    bottom_center = np.array([[0, 0, -half_height]])
    top_center = np.array([[0, 0, half_height]])
    
    vertices = np.vstack([bottom_vertices, top_vertices, bottom_center, top_center])
    
    faces = []
    bottom_center_idx = 2 * sections
    top_center_idx = 2 * sections + 1
    
    for i in range(sections):
        next_i = (i + 1) % sections
        
        faces.append([i, next_i, bottom_center_idx])
        
        faces.append([sections + i, top_center_idx, sections + next_i])
        
        faces.append([i, sections + i, next_i])
        faces.append([next_i, sections + i, sections + next_i])
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces))
    
    axis = np.array(domain.axis)
    z_axis = np.array([0, 0, 1])
    
    if not np.allclose(axis, z_axis):
        rotation_axis = np.cross(z_axis, axis)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        
        if rotation_axis_norm > 1e-10:
            rotation_axis = rotation_axis / rotation_axis_norm
            angle = np.arccos(np.clip(np.dot(z_axis, axis), -1, 1))
            
            c = np.cos(angle)
            s = np.sin(angle)
            t = 1 - c
            x, y, z = rotation_axis
            
            rotation_matrix = np.array([
                [t*x*x + c,   t*x*y - s*z, t*x*z + s*y, 0],
                [t*x*y + s*z, t*y*y + c,   t*y*z - s*x, 0],
                [t*x*z - s*y, t*y*z + s*x, t*z*z + c,   0],
                [0,           0,           0,           1],
            ])
            mesh.apply_transform(rotation_matrix)
    
    center = [domain.center.x, domain.center.y, domain.center.z]
    mesh.apply_translation(center)
    
    return mesh


def _mesh_mesh_domain(
    domain: "DomainSpec",
    policy: MeshDomainPolicy,
    ctx: MeshingContext,
) -> "trimesh.Trimesh":
    """Create mesh for MeshDomain (load and optionally repair)."""
    import trimesh
    
    mesh = domain._mesh.copy()
    
    if policy.validate_watertight and not mesh.is_watertight:
        ctx.add_warning(f"MeshDomain mesh is not watertight: {domain.mesh_path}")
        
        if policy.repair_if_needed:
            try:
                mesh.fill_holes()
                mesh.fix_normals()
                
                if not mesh.is_watertight:
                    ctx.add_warning("Mesh repair did not achieve watertightness")
            except Exception as e:
                ctx.add_warning(f"Mesh repair failed: {e}")
    
    if policy.simplify_if_over_max and len(mesh.faces) > policy.max_faces:
        target_faces = int(policy.max_faces * policy.simplify_target_ratio)
        try:
            mesh = mesh.simplify_quadric_decimation(target_faces)
            ctx.add_warning(f"Simplified mesh from {len(domain._mesh.faces)} to {len(mesh.faces)} faces")
        except Exception as e:
            ctx.add_warning(f"Mesh simplification failed: {e}")
    
    return mesh


def _mesh_transform_domain(
    domain: "DomainSpec",
    meshing_policy: DomainMeshingPolicy,
    resolution_policy: Optional[ResolutionPolicy],
    ctx: MeshingContext,
) -> "trimesh.Trimesh":
    """Create mesh for TransformDomain."""
    base_mesh, _ = domain_to_mesh(
        domain.base_domain,
        meshing_policy,
        resolution_policy,
        ctx,
    )
    
    mesh = base_mesh.copy()
    
    transform = np.eye(4)
    transform[:3, :3] = domain.rotation * domain.scale
    transform[:3, 3] = domain.translation
    
    mesh.apply_transform(transform)
    
    return mesh


def _mesh_composite_domain(
    domain: "DomainSpec",
    meshing_policy: DomainMeshingPolicy,
    resolution_policy: Optional[ResolutionPolicy],
    ctx: MeshingContext,
) -> "trimesh.Trimesh":
    """Create mesh for CompositeDomain via marching cubes."""
    from skimage.measure import marching_cubes
    import trimesh
    
    implicit_policy = meshing_policy.implicit_policy
    
    bounds = domain.get_bounds()
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    
    extents = (x_max - x_min, y_max - y_min, z_max - z_min)
    bbox = (x_min, x_max, y_min, y_max, z_min, z_max)
    
    resolution_result: Optional[ResolutionResult] = None
    effective_pitch = implicit_policy.voxel_pitch
    
    if resolution_policy is not None and implicit_policy.auto_relax_pitch:
        resolution_result = resolve_pitch(
            op_name="domain_meshing_composite",
            requested_pitch=implicit_policy.voxel_pitch,
            bbox=bbox,
            resolution_policy=resolution_policy,
            max_voxels_override=implicit_policy.max_voxels,
        )
        effective_pitch = resolution_result.effective_pitch
        ctx.warnings.extend(resolution_result.warnings)
        
        if resolution_result.was_relaxed:
            ctx.set_metric("pitch_relaxed", True)
            ctx.set_metric("original_pitch", implicit_policy.voxel_pitch)
            ctx.set_metric("effective_pitch", effective_pitch)
    
    nx = max(2, int(np.ceil(extents[0] / effective_pitch)))
    ny = max(2, int(np.ceil(extents[1] / effective_pitch)))
    nz = max(2, int(np.ceil(extents[2] / effective_pitch)))
    
    ctx.set_metric("voxel_grid_size", (nx, ny, nz))
    ctx.set_metric("total_voxels", nx * ny * nz)
    
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    z = np.linspace(z_min, z_max, nz)
    
    from ..core.types import Point3D
    
    sdf_grid = np.zeros((nx, ny, nz))
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            for k, zk in enumerate(z):
                point = Point3D(xi, yj, zk)
                sdf_grid[i, j, k] = domain.signed_distance(point)
    
    try:
        verts, faces, normals, values = marching_cubes(
            sdf_grid,
            level=0.0,
            spacing=(extents[0] / (nx - 1), extents[1] / (ny - 1), extents[2] / (nz - 1)),
        )
        
        verts[:, 0] += x_min
        verts[:, 1] += y_min
        verts[:, 2] += z_min
        
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        
        if implicit_policy.smooth_iterations > 0:
            trimesh.smoothing.filter_laplacian(
                mesh,
                iterations=implicit_policy.smooth_iterations,
            )
        
        return mesh
        
    except Exception as e:
        ctx.add_warning(f"Marching cubes failed for composite domain: {e}")
        raise


def _mesh_implicit_domain(
    domain: "DomainSpec",
    meshing_policy: DomainMeshingPolicy,
    resolution_policy: Optional[ResolutionPolicy],
    ctx: MeshingContext,
) -> "trimesh.Trimesh":
    """Create mesh for ImplicitDomain via marching cubes."""
    from skimage.measure import marching_cubes
    import trimesh
    
    implicit_policy = meshing_policy.implicit_policy
    
    bounds = domain.get_bounds()
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    
    extents = (x_max - x_min, y_max - y_min, z_max - z_min)
    bbox = (x_min, x_max, y_min, y_max, z_min, z_max)
    
    resolution_result: Optional[ResolutionResult] = None
    effective_pitch = implicit_policy.voxel_pitch
    
    if resolution_policy is not None and implicit_policy.auto_relax_pitch:
        resolution_result = resolve_pitch(
            op_name="domain_meshing_implicit",
            requested_pitch=implicit_policy.voxel_pitch,
            bbox=bbox,
            resolution_policy=resolution_policy,
            max_voxels_override=implicit_policy.max_voxels,
        )
        effective_pitch = resolution_result.effective_pitch
        ctx.warnings.extend(resolution_result.warnings)
        
        if resolution_result.was_relaxed:
            ctx.set_metric("pitch_relaxed", True)
            ctx.set_metric("original_pitch", implicit_policy.voxel_pitch)
            ctx.set_metric("effective_pitch", effective_pitch)
    
    nx = max(2, int(np.ceil(extents[0] / effective_pitch)))
    ny = max(2, int(np.ceil(extents[1] / effective_pitch)))
    nz = max(2, int(np.ceil(extents[2] / effective_pitch)))
    
    ctx.set_metric("voxel_grid_size", (nx, ny, nz))
    ctx.set_metric("total_voxels", nx * ny * nz)
    
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    z = np.linspace(z_min, z_max, nz)
    
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    
    sdf_values = domain.sdf_array(points)
    sdf_grid = sdf_values.reshape((nx, ny, nz))
    
    try:
        verts, faces, normals, values = marching_cubes(
            sdf_grid,
            level=implicit_policy.iso_level,
            spacing=(extents[0] / (nx - 1), extents[1] / (ny - 1), extents[2] / (nz - 1)),
        )
        
        verts[:, 0] += x_min
        verts[:, 1] += y_min
        verts[:, 2] += z_min
        
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        
        if implicit_policy.smooth_iterations > 0:
            trimesh.smoothing.filter_laplacian(
                mesh,
                iterations=implicit_policy.smooth_iterations,
            )
        
        return mesh
        
    except Exception as e:
        ctx.add_warning(f"Marching cubes failed for implicit domain: {e}")
        raise


def domain_to_mesh(
    domain: "DomainSpec",
    meshing_policy: Optional[DomainMeshingPolicy] = None,
    resolution_policy: Optional[ResolutionPolicy] = None,
    ctx: Optional[MeshingContext] = None,
) -> Tuple["trimesh.Trimesh", OperationReport]:
    """
    Convert any domain to a watertight mesh.
    
    This is the canonical function for domain-to-mesh conversion, supporting
    all domain types with policy-driven behavior.
    
    Parameters
    ----------
    domain : DomainSpec
        The domain to convert to a mesh.
    meshing_policy : DomainMeshingPolicy, optional
        Policy controlling meshing behavior. If None, uses defaults.
    resolution_policy : ResolutionPolicy, optional
        Resolution policy for voxel budget control. If None, no budget enforcement.
    ctx : MeshingContext, optional
        Context for caching and tracking. If None, creates a new context.
    
    Returns
    -------
    mesh : trimesh.Trimesh
        The generated mesh.
    report : OperationReport
        Report with meshing statistics and warnings.
    
    Supported Domain Types
    ----------------------
    - BoxDomain: Analytic box mesh
    - CylinderDomain: Analytic cylinder mesh
    - EllipsoidDomain: Icosphere scaled to ellipsoid
    - SphereDomain: Icosphere at specified radius
    - CapsuleDomain: Capsule mesh with axis alignment
    - FrustumDomain: Truncated cone mesh
    - MeshDomain: Load and optionally repair mesh
    - TransformDomain: Mesh base domain and apply transform
    - CompositeDomain: Marching cubes on SDF
    - ImplicitDomain: Marching cubes on SDF
    """
    import trimesh
    
    if meshing_policy is None:
        meshing_policy = DomainMeshingPolicy()
    
    if ctx is None:
        ctx = MeshingContext()
    
    domain_hash = _compute_domain_hash(domain)
    
    if meshing_policy.cache_meshes:
        cached = ctx.get_cached(domain_hash)
        if cached is not None:
            report = OperationReport(
                operation="domain_to_mesh",
                success=True,
                requested_policy=meshing_policy.to_dict(),
                effective_policy=meshing_policy.to_dict(),
                warnings=[],
                metadata={"cached": True, "domain_type": domain.to_dict().get("type")},
            )
            return cached.copy(), report
    
    domain_type = domain.to_dict().get("type")
    mesh: Optional[trimesh.Trimesh] = None
    
    try:
        if domain_type == "box":
            mesh = _mesh_box(domain, meshing_policy.primitive_policy)
        
        elif domain_type == "cylinder":
            mesh = _mesh_cylinder(domain, meshing_policy.primitive_policy)
        
        elif domain_type == "ellipsoid":
            mesh = _mesh_ellipsoid(domain, meshing_policy.primitive_policy)
        
        elif domain_type == "sphere":
            mesh = _mesh_sphere(domain, meshing_policy.primitive_policy)
        
        elif domain_type == "capsule":
            mesh = _mesh_capsule(domain, meshing_policy.primitive_policy)
        
        elif domain_type == "frustum":
            mesh = _mesh_frustum(domain, meshing_policy.primitive_policy)
        
        elif domain_type == "mesh":
            mesh = _mesh_mesh_domain(domain, meshing_policy.mesh_policy, ctx)
        
        elif domain_type == "transform":
            mesh = _mesh_transform_domain(domain, meshing_policy, resolution_policy, ctx)
        
        elif domain_type == "composite":
            mesh = _mesh_composite_domain(domain, meshing_policy, resolution_policy, ctx)
        
        elif domain_type == "implicit":
            mesh = _mesh_implicit_domain(domain, meshing_policy, resolution_policy, ctx)
        
        else:
            raise ValueError(f"Unsupported domain type: {domain_type}")
        
        if not mesh.is_watertight and meshing_policy.emit_warnings:
            ctx.add_warning(f"Generated mesh for {domain_type} is not watertight")
        
        if meshing_policy.cache_meshes:
            ctx.set_cached(domain_hash, mesh.copy())
        
        report = OperationReport(
            operation="domain_to_mesh",
            success=True,
            requested_policy=meshing_policy.to_dict(),
            effective_policy=meshing_policy.to_dict(),
            warnings=ctx.warnings.copy(),
            metadata={
                "domain_type": domain_type,
                "vertices": len(mesh.vertices),
                "faces": len(mesh.faces),
                "is_watertight": mesh.is_watertight,
                **ctx.metrics,
            },
        )
        
        return mesh, report
        
    except Exception as e:
        logger.error(f"Failed to mesh domain of type {domain_type}: {e}")
        report = OperationReport(
            operation="domain_to_mesh",
            success=False,
            requested_policy=meshing_policy.to_dict(),
            effective_policy=meshing_policy.to_dict(),
            warnings=ctx.warnings.copy(),
            metadata={
                "domain_type": domain_type,
                "error": str(e),
                **ctx.metrics,
            },
        )
        raise


__all__ = [
    "domain_to_mesh",
    "MeshingContext",
]
