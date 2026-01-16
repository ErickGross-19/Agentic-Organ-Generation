"""
Domain meshing policies for AOG.

This module contains policy dataclasses for domain-to-mesh conversion.
All policies are JSON-serializable and support the "requested vs effective" pattern.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .resolution import ResolutionPolicy


@dataclass
class PrimitiveMeshingPolicy:
    """
    Policy for meshing primitive domains (Box, Cylinder, Ellipsoid, Sphere, Capsule, Frustum).
    
    Controls the number of sections/subdivisions for generating smooth meshes
    from analytic primitives.
    
    JSON Schema:
    {
        "sections_radial": int,
        "sections_axial": int,
        "sections_angular": int,
        "subdivisions": int
    }
    """
    sections_radial: int = 32
    sections_axial: int = 16
    sections_angular: int = 32
    subdivisions: int = 2
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sections_radial": self.sections_radial,
            "sections_axial": self.sections_axial,
            "sections_angular": self.sections_angular,
            "subdivisions": self.subdivisions,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PrimitiveMeshingPolicy":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class MeshDomainPolicy:
    """
    Policy for handling MeshDomain inputs.
    
    Controls loading, validation, and repair options for mesh-based domains.
    
    JSON Schema:
    {
        "validate_watertight": bool,
        "repair_if_needed": bool,
        "repair_voxel_pitch": float (meters),
        "max_faces": int,
        "simplify_if_over_max": bool,
        "simplify_target_ratio": float
    }
    """
    validate_watertight: bool = True
    repair_if_needed: bool = True
    repair_voxel_pitch: float = 5e-5  # 50µm
    max_faces: int = 500_000
    simplify_if_over_max: bool = True
    simplify_target_ratio: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "validate_watertight": self.validate_watertight,
            "repair_if_needed": self.repair_if_needed,
            "repair_voxel_pitch": self.repair_voxel_pitch,
            "max_faces": self.max_faces,
            "simplify_if_over_max": self.simplify_if_over_max,
            "simplify_target_ratio": self.simplify_target_ratio,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MeshDomainPolicy":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ImplicitMeshingPolicy:
    """
    Policy for meshing implicit/composite domains via marching cubes.
    
    Controls voxelization and marching cubes parameters for converting
    SDF-based domains to meshes.
    
    JSON Schema:
    {
        "voxel_pitch": float (meters),
        "max_voxels": int,
        "auto_relax_pitch": bool,
        "pitch_step_factor": float,
        "iso_level": float,
        "smooth_iterations": int
    }
    """
    voxel_pitch: float = 5e-5  # 50µm
    max_voxels: int = 50_000_000  # 50M voxels
    auto_relax_pitch: bool = True
    pitch_step_factor: float = 1.5
    iso_level: float = 0.0
    smooth_iterations: int = 2
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "voxel_pitch": self.voxel_pitch,
            "max_voxels": self.max_voxels,
            "auto_relax_pitch": self.auto_relax_pitch,
            "pitch_step_factor": self.pitch_step_factor,
            "iso_level": self.iso_level,
            "smooth_iterations": self.smooth_iterations,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ImplicitMeshingPolicy":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_resolution_policy(
        cls,
        resolution_policy: "ResolutionPolicy",
        **overrides,
    ) -> "ImplicitMeshingPolicy":
        """Create from ResolutionPolicy with optional overrides."""
        return cls(
            voxel_pitch=overrides.get("voxel_pitch", resolution_policy.embed_pitch),
            max_voxels=overrides.get("max_voxels", resolution_policy.get_max_voxels_for_operation("embed")),
            auto_relax_pitch=overrides.get("auto_relax_pitch", resolution_policy.auto_relax_pitch),
            pitch_step_factor=overrides.get("pitch_step_factor", resolution_policy.pitch_step_factor),
            **{k: v for k, v in overrides.items() if k not in [
                "voxel_pitch", "max_voxels", "auto_relax_pitch", "pitch_step_factor"
            ]},
        )


@dataclass
class DomainMeshingPolicy:
    """
    Unified policy for domain-to-mesh conversion.
    
    This policy controls how any domain type is converted to a mesh,
    with type-specific sub-policies for primitives, mesh domains, and
    implicit/composite domains.
    
    JSON Schema:
    {
        "primitive_policy": PrimitiveMeshingPolicy,
        "mesh_policy": MeshDomainPolicy,
        "implicit_policy": ImplicitMeshingPolicy,
        "cache_meshes": bool,
        "emit_warnings": bool,
        "target_face_count": int (optional),
        "min_face_count": int (optional),
        "max_face_count": int (optional),
        "voxel_pitch": float (optional, meters)
    }
    """
    primitive_policy: Optional[PrimitiveMeshingPolicy] = None
    mesh_policy: Optional[MeshDomainPolicy] = None
    implicit_policy: Optional[ImplicitMeshingPolicy] = None
    cache_meshes: bool = True
    emit_warnings: bool = True
    # Front-door config for face count control (maps to sub-policies)
    target_face_count: Optional[int] = None
    min_face_count: Optional[int] = None
    max_face_count: Optional[int] = None
    # Front-door config for voxel pitch (maps to implicit_policy)
    voxel_pitch: Optional[float] = None
    
    def __post_init__(self):
        if self.primitive_policy is None:
            self.primitive_policy = PrimitiveMeshingPolicy()
        if self.mesh_policy is None:
            self.mesh_policy = MeshDomainPolicy()
        if self.implicit_policy is None:
            self.implicit_policy = ImplicitMeshingPolicy()
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "primitive_policy": self.primitive_policy.to_dict() if self.primitive_policy else None,
            "mesh_policy": self.mesh_policy.to_dict() if self.mesh_policy else None,
            "implicit_policy": self.implicit_policy.to_dict() if self.implicit_policy else None,
            "cache_meshes": self.cache_meshes,
            "emit_warnings": self.emit_warnings,
        }
        # Include front-door config if set
        if self.target_face_count is not None:
            result["target_face_count"] = self.target_face_count
        if self.min_face_count is not None:
            result["min_face_count"] = self.min_face_count
        if self.max_face_count is not None:
            result["max_face_count"] = self.max_face_count
        if self.voxel_pitch is not None:
            result["voxel_pitch"] = self.voxel_pitch
        return result
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DomainMeshingPolicy":
        primitive_policy = None
        mesh_policy = None
        implicit_policy = None
        
        if d.get("primitive_policy"):
            primitive_policy = PrimitiveMeshingPolicy.from_dict(d["primitive_policy"])
        if d.get("mesh_policy"):
            mesh_policy = MeshDomainPolicy.from_dict(d["mesh_policy"])
        if d.get("implicit_policy"):
            implicit_policy = ImplicitMeshingPolicy.from_dict(d["implicit_policy"])
        
        return cls(
            primitive_policy=primitive_policy,
            mesh_policy=mesh_policy,
            implicit_policy=implicit_policy,
            cache_meshes=d.get("cache_meshes", True),
            emit_warnings=d.get("emit_warnings", True),
            target_face_count=d.get("target_face_count"),
            min_face_count=d.get("min_face_count"),
            max_face_count=d.get("max_face_count"),
            voxel_pitch=d.get("voxel_pitch"),
        )


__all__ = [
    "PrimitiveMeshingPolicy",
    "MeshDomainPolicy",
    "ImplicitMeshingPolicy",
    "DomainMeshingPolicy",
]
