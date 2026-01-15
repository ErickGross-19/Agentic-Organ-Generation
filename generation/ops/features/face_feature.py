"""
Face feature framework for domain mesh modifications.

This module provides a unified framework for creating features on domain faces,
such as ridges, grooves, and ports. Features return constraint information
that can be used by placement, pathfinding, and channel generation.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np
import logging

from ...core.domain import DomainSpec, CylinderDomain, BoxDomain

if TYPE_CHECKING:
    import trimesh

logger = logging.getLogger(__name__)


class FeatureType(str, Enum):
    """Types of face features."""
    RIDGE = "ridge"
    GROOVE = "groove"
    PORT = "port"
    BOSS = "boss"


class FaceId(str, Enum):
    """Identifier for faces of a domain."""
    X_MIN = "x_min"
    X_MAX = "x_max"
    Y_MIN = "y_min"
    Y_MAX = "y_max"
    Z_MIN = "z_min"
    Z_MAX = "z_max"
    TOP = "top"
    BOTTOM = "bottom"
    LATERAL = "lateral"


@dataclass
class FeatureConstraints:
    """
    Constraints returned by a face feature.
    
    These constraints inform placement, pathfinding, and channel generation
    about the effective usable area after the feature is applied.
    """
    effective_radius: Optional[float] = None
    effective_inner_radius: Optional[float] = None
    effective_outer_radius: Optional[float] = None
    exclusion_zones: List[Dict[str, Any]] = field(default_factory=list)
    min_clearance: float = 0.0
    max_depth: Optional[float] = None
    face_normal: Optional[np.ndarray] = None
    face_center: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "effective_radius": self.effective_radius,
            "effective_inner_radius": self.effective_inner_radius,
            "effective_outer_radius": self.effective_outer_radius,
            "exclusion_zones": self.exclusion_zones,
            "min_clearance": self.min_clearance,
            "max_depth": self.max_depth,
            "face_normal": self.face_normal.tolist() if self.face_normal is not None else None,
            "face_center": self.face_center.tolist() if self.face_center is not None else None,
        }


@dataclass
class FeatureResult:
    """Result of creating a face feature."""
    success: bool
    mesh: Optional["trimesh.Trimesh"] = None
    constraints: Optional[FeatureConstraints] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "constraints": self.constraints.to_dict() if self.constraints else None,
            "warnings": self.warnings,
            "errors": self.errors,
            "metadata": self.metadata,
        }


class FaceFeature(ABC):
    """
    Abstract base class for face features.
    
    Face features are geometric modifications applied to domain faces
    that return constraint information for downstream operations.
    """
    
    @property
    @abstractmethod
    def feature_type(self) -> FeatureType:
        """Return the type of this feature."""
        pass
    
    @abstractmethod
    def create(
        self,
        domain: DomainSpec,
        face: FaceId,
    ) -> FeatureResult:
        """
        Create the feature on the specified face.
        
        Parameters
        ----------
        domain : DomainSpec
            Domain to apply feature to
        face : FaceId
            Which face to apply the feature to
            
        Returns
        -------
        FeatureResult
            Result containing mesh and constraints
        """
        pass
    
    @abstractmethod
    def get_constraints(
        self,
        domain: DomainSpec,
        face: FaceId,
    ) -> FeatureConstraints:
        """
        Get constraints without creating the mesh.
        
        Useful for planning and validation before committing to mesh creation.
        """
        pass


@dataclass
class RidgeFeatureSpec:
    """
    Specification for a ridge feature.
    
    A ridge is a raised ring (for cylinders) or frame (for boxes) around
    the perimeter of a face.
    """
    height: float = 0.001  # 1mm
    thickness: float = 0.001  # 1mm
    inset: float = 0.0
    overlap: Optional[float] = None
    resolution: int = 64
    voxel_pitch: Optional[float] = None
    
    def __post_init__(self):
        if self.overlap is None:
            self.overlap = 0.5 * self.height
        if self.voxel_pitch is None:
            self.voxel_pitch = min(self.thickness / 4, self.height / 4)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "height": self.height,
            "thickness": self.thickness,
            "inset": self.inset,
            "overlap": self.overlap,
            "resolution": self.resolution,
            "voxel_pitch": self.voxel_pitch,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RidgeFeatureSpec":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class RidgeFeature(FaceFeature):
    """
    Ridge feature for creating raised rings or frames on domain faces.
    
    Returns effective_radius constraint that accounts for ridge geometry,
    enabling proper port placement and channel routing.
    """
    
    def __init__(self, spec: Optional[RidgeFeatureSpec] = None):
        self.spec = spec or RidgeFeatureSpec()
    
    @property
    def feature_type(self) -> FeatureType:
        return FeatureType.RIDGE
    
    def create(
        self,
        domain: DomainSpec,
        face: FaceId,
    ) -> FeatureResult:
        """Create a ridge feature on the specified face."""
        import trimesh
        from .ridge_helpers import create_annular_ridge, create_frame_ridge
        
        constraints = self.get_constraints(domain, face)
        
        try:
            if isinstance(domain, CylinderDomain):
                mesh = self._create_cylinder_ridge(domain, face)
            elif isinstance(domain, BoxDomain):
                mesh = self._create_box_ridge(domain, face)
            else:
                return FeatureResult(
                    success=False,
                    errors=[f"Unsupported domain type: {type(domain)}"],
                )
            
            return FeatureResult(
                success=True,
                mesh=mesh,
                constraints=constraints,
                metadata={
                    "spec": self.spec.to_dict(),
                    "domain_type": type(domain).__name__,
                    "face": face.value,
                },
            )
            
        except Exception as e:
            logger.exception("Failed to create ridge feature")
            return FeatureResult(
                success=False,
                constraints=constraints,
                errors=[str(e)],
            )
    
    def get_constraints(
        self,
        domain: DomainSpec,
        face: FaceId,
    ) -> FeatureConstraints:
        """Get constraints for the ridge feature."""
        if isinstance(domain, CylinderDomain):
            return self._get_cylinder_constraints(domain, face)
        elif isinstance(domain, BoxDomain):
            return self._get_box_constraints(domain, face)
        else:
            return FeatureConstraints()
    
    def _get_cylinder_constraints(
        self,
        domain: CylinderDomain,
        face: FaceId,
    ) -> FeatureConstraints:
        """Get constraints for cylinder domain."""
        radius = domain.radius
        center = np.array([domain.center.x, domain.center.y, domain.center.z])
        
        # Effective radius is inside the ridge
        outer_radius = radius - self.spec.inset
        inner_radius = outer_radius - self.spec.thickness
        effective_radius = inner_radius - self.spec.thickness * 0.1  # Small margin
        
        # Face center and normal
        if face in (FaceId.TOP, FaceId.Z_MAX):
            face_center = center + np.array([0, 0, domain.height / 2])
            face_normal = np.array([0, 0, 1])
        else:
            face_center = center - np.array([0, 0, domain.height / 2])
            face_normal = np.array([0, 0, -1])
        
        return FeatureConstraints(
            effective_radius=effective_radius,
            effective_inner_radius=inner_radius,
            effective_outer_radius=outer_radius,
            min_clearance=self.spec.thickness,
            max_depth=domain.height - self.spec.height,
            face_normal=face_normal,
            face_center=face_center,
            exclusion_zones=[{
                "type": "annular",
                "inner_radius": inner_radius,
                "outer_radius": outer_radius,
                "center": face_center.tolist(),
            }],
        )
    
    def _get_box_constraints(
        self,
        domain: BoxDomain,
        face: FaceId,
    ) -> FeatureConstraints:
        """Get constraints for box domain."""
        width = domain.x_max - domain.x_min
        depth = domain.y_max - domain.y_min
        height = domain.z_max - domain.z_min
        
        center_x = (domain.x_min + domain.x_max) / 2
        center_y = (domain.y_min + domain.y_max) / 2
        center_z = (domain.z_min + domain.z_max) / 2
        
        # Effective area is inside the ridge frame
        effective_width = width - 2 * (self.spec.inset + self.spec.thickness)
        effective_depth = depth - 2 * (self.spec.inset + self.spec.thickness)
        
        # Use effective radius as half the smaller dimension
        effective_radius = min(effective_width, effective_depth) / 2
        
        # Face center and normal
        if face in (FaceId.TOP, FaceId.Z_MAX):
            face_center = np.array([center_x, center_y, domain.z_max])
            face_normal = np.array([0, 0, 1])
        else:
            face_center = np.array([center_x, center_y, domain.z_min])
            face_normal = np.array([0, 0, -1])
        
        return FeatureConstraints(
            effective_radius=effective_radius,
            min_clearance=self.spec.thickness,
            max_depth=height - self.spec.height,
            face_normal=face_normal,
            face_center=face_center,
            exclusion_zones=[{
                "type": "frame",
                "width": width - 2 * self.spec.inset,
                "depth": depth - 2 * self.spec.inset,
                "thickness": self.spec.thickness,
                "center": face_center.tolist(),
            }],
        )
    
    def _create_cylinder_ridge(
        self,
        domain: CylinderDomain,
        face: FaceId,
    ) -> "trimesh.Trimesh":
        """Create ridge mesh for cylinder domain."""
        from .ridge_helpers import create_annular_ridge
        
        radius = domain.radius
        center_x = domain.center.x
        center_y = domain.center.y
        center_z = domain.center.z
        
        outer_radius = radius - self.spec.inset
        inner_radius = outer_radius - self.spec.thickness
        
        if face in (FaceId.TOP, FaceId.Z_MAX):
            z_base = center_z + domain.height / 2 - self.spec.overlap
        else:
            z_base = center_z - domain.height / 2 - self.spec.height + self.spec.overlap
        
        return create_annular_ridge(
            outer_radius=outer_radius,
            inner_radius=inner_radius,
            height=self.spec.height + self.spec.overlap,
            z_base=z_base,
            center_xy=(center_x, center_y),
            resolution=self.spec.resolution,
        )
    
    def _create_box_ridge(
        self,
        domain: BoxDomain,
        face: FaceId,
    ) -> "trimesh.Trimesh":
        """Create ridge mesh for box domain."""
        from .ridge_helpers import create_frame_ridge
        
        width = domain.x_max - domain.x_min
        depth = domain.y_max - domain.y_min
        center_x = (domain.x_min + domain.x_max) / 2
        center_y = (domain.y_min + domain.y_max) / 2
        
        if face in (FaceId.TOP, FaceId.Z_MAX):
            z_base = domain.z_max - self.spec.overlap
        else:
            z_base = domain.z_min - self.spec.height + self.spec.overlap
        
        return create_frame_ridge(
            width=width - 2 * self.spec.inset,
            depth=depth - 2 * self.spec.inset,
            height=self.spec.height + self.spec.overlap,
            thickness=self.spec.thickness,
            z_base=z_base,
            center_xy=(center_x, center_y),
        )


def create_ridge_with_constraints(
    domain: DomainSpec,
    face: FaceId = FaceId.TOP,
    height: float = 0.001,
    thickness: float = 0.001,
    inset: float = 0.0,
) -> Tuple[Optional["trimesh.Trimesh"], FeatureConstraints]:
    """
    Convenience function to create a ridge and get its constraints.
    
    Parameters
    ----------
    domain : DomainSpec
        Domain to apply ridge to
    face : FaceId
        Which face to apply the ridge to
    height : float
        Ridge height in meters
    thickness : float
        Ridge thickness in meters
    inset : float
        Inset from domain boundary
        
    Returns
    -------
    mesh : trimesh.Trimesh or None
        Ridge mesh (None if creation failed)
    constraints : FeatureConstraints
        Constraints for placement and routing
    """
    spec = RidgeFeatureSpec(
        height=height,
        thickness=thickness,
        inset=inset,
    )
    
    feature = RidgeFeature(spec)
    result = feature.create(domain, face)
    
    return result.mesh, result.constraints or FeatureConstraints()


__all__ = [
    "FaceFeature",
    "FeatureType",
    "FaceId",
    "FeatureConstraints",
    "FeatureResult",
    "RidgeFeature",
    "RidgeFeatureSpec",
    "create_ridge_with_constraints",
]
