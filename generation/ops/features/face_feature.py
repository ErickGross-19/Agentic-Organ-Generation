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


# PATCH 4: Canonical face string mapping (Option A)
CANONICAL_FACE_MAP: Dict[str, FaceId] = {
    # Option A canonical strings
    "top": FaceId.TOP,
    "bottom": FaceId.BOTTOM,
    "+x": FaceId.X_MAX,
    "-x": FaceId.X_MIN,
    "+y": FaceId.Y_MAX,
    "-y": FaceId.Y_MIN,
    "+z": FaceId.Z_MAX,
    "-z": FaceId.Z_MIN,
    # Also accept internal enum values for compatibility
    "x_min": FaceId.X_MIN,
    "x_max": FaceId.X_MAX,
    "y_min": FaceId.Y_MIN,
    "y_max": FaceId.Y_MAX,
    "z_min": FaceId.Z_MIN,
    "z_max": FaceId.Z_MAX,
    "lateral": FaceId.LATERAL,
}


def _parse_face_string(face: str) -> FaceId:
    """
    PATCH 4: Convert canonical face string to FaceId enum.
    
    Accepts Option A canonical strings:
    - "top", "bottom"
    - "+x", "-x", "+y", "-y", "+z", "-z"
    
    Also accepts internal enum values for compatibility.
    
    Parameters
    ----------
    face : str
        Canonical face string
        
    Returns
    -------
    FaceId
        Corresponding FaceId enum value
        
    Raises
    ------
    ValueError
        If face string is not recognized
    """
    face_lower = face.lower().strip()
    if face_lower in CANONICAL_FACE_MAP:
        return CANONICAL_FACE_MAP[face_lower]
    
    raise ValueError(
        f"Invalid face string: '{face}'. "
        f"Valid options are: {list(CANONICAL_FACE_MAP.keys())}"
    )


@dataclass
class RidgePolicy:
    """
    PATCH 4: Policy for ridge features, compatible with aog_policies pattern.
    
    JSON Schema:
    {
        "height": float (meters),
        "thickness": float (meters),
        "inset": float (meters),
        "overlap": float (meters) | null,
        "resolution": int
    }
    """
    height: float = 0.001  # 1mm
    thickness: float = 0.001  # 1mm
    inset: float = 0.0
    overlap: Optional[float] = None
    resolution: int = 64
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "height": self.height,
            "thickness": self.thickness,
            "inset": self.inset,
            "overlap": self.overlap,
            "resolution": self.resolution,
        }
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RidgePolicy":
        return RidgePolicy(**{k: v for k, v in d.items() if k in RidgePolicy.__dataclass_fields__})


@dataclass
class RidgeOperationReport:
    """
    PATCH 4: Operation report for ridge creation, following OperationReport pattern.
    """
    operation: str = "add_ridge"
    success: bool = True
    requested_policy: Dict[str, Any] = field(default_factory=dict)
    effective_policy: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "success": self.success,
            "requested_policy": self.requested_policy,
            "effective_policy": self.effective_policy,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }


def add_ridge(
    domain_mesh: "trimesh.Trimesh",
    face: str,
    ridge_policy: Optional[Union[RidgePolicy, Dict[str, Any]]] = None,
    domain_spec: Optional[DomainSpec] = None,
) -> Tuple["trimesh.Trimesh", FeatureConstraints, RidgeOperationReport]:
    """
    PATCH 4: Add a ridge to a domain mesh using canonical face strings.
    
    This is the public API for ridge creation, accepting Option A canonical
    face strings ("top", "bottom", "+x", "-x", "+y", "-y", "+z", "-z").
    
    Parameters
    ----------
    domain_mesh : trimesh.Trimesh
        Domain mesh to add ridge to
    face : str
        Canonical face string (e.g., "top", "bottom", "+z", "-z")
    ridge_policy : RidgePolicy or dict, optional
        Policy controlling ridge parameters
    domain_spec : DomainSpec, optional
        Domain specification for constraint calculation.
        If not provided, constraints will be estimated from mesh bounds.
        
    Returns
    -------
    mesh : trimesh.Trimesh
        Domain mesh with ridge added
    constraints : FeatureConstraints
        Constraints including effective_radius, face_center, face_normal
    report : RidgeOperationReport
        Report with requested/effective policy
    """
    import trimesh
    
    # Parse face string to FaceId
    try:
        face_id = _parse_face_string(face)
    except ValueError as e:
        return domain_mesh, FeatureConstraints(), RidgeOperationReport(
            success=False,
            warnings=[str(e)],
        )
    
    # Parse policy
    if ridge_policy is None:
        policy = RidgePolicy()
    elif isinstance(ridge_policy, dict):
        policy = RidgePolicy.from_dict(ridge_policy)
    else:
        policy = ridge_policy
    
    # Create ridge spec from policy
    spec = RidgeFeatureSpec(
        height=policy.height,
        thickness=policy.thickness,
        inset=policy.inset,
        overlap=policy.overlap,
        resolution=policy.resolution,
    )
    
    warnings = []
    
    # If domain_spec is provided, use RidgeFeature to create ridge
    if domain_spec is not None:
        feature = RidgeFeature(spec)
        result = feature.create(domain_spec, face_id)
        
        if result.success and result.mesh is not None:
            # Union ridge with domain mesh
            try:
                combined = domain_mesh.union(result.mesh)
                if combined is not None and len(combined.vertices) > 0:
                    domain_mesh = combined
                else:
                    warnings.append("Ridge union failed, returning original mesh")
            except Exception as e:
                warnings.append(f"Ridge union failed: {e}")
        
        constraints = result.constraints or FeatureConstraints()
        warnings.extend(result.warnings)
        
    else:
        # Estimate constraints from mesh bounds
        bounds = domain_mesh.bounds
        center = (bounds[0] + bounds[1]) / 2
        extents = bounds[1] - bounds[0]
        
        # Determine face center and normal based on face_id
        if face_id in (FaceId.TOP, FaceId.Z_MAX):
            face_center = np.array([center[0], center[1], bounds[1][2]])
            face_normal = np.array([0, 0, 1])
            effective_radius = min(extents[0], extents[1]) / 2 - policy.thickness - policy.inset
        elif face_id in (FaceId.BOTTOM, FaceId.Z_MIN):
            face_center = np.array([center[0], center[1], bounds[0][2]])
            face_normal = np.array([0, 0, -1])
            effective_radius = min(extents[0], extents[1]) / 2 - policy.thickness - policy.inset
        elif face_id == FaceId.X_MAX:
            face_center = np.array([bounds[1][0], center[1], center[2]])
            face_normal = np.array([1, 0, 0])
            effective_radius = min(extents[1], extents[2]) / 2 - policy.thickness - policy.inset
        elif face_id == FaceId.X_MIN:
            face_center = np.array([bounds[0][0], center[1], center[2]])
            face_normal = np.array([-1, 0, 0])
            effective_radius = min(extents[1], extents[2]) / 2 - policy.thickness - policy.inset
        elif face_id == FaceId.Y_MAX:
            face_center = np.array([center[0], bounds[1][1], center[2]])
            face_normal = np.array([0, 1, 0])
            effective_radius = min(extents[0], extents[2]) / 2 - policy.thickness - policy.inset
        elif face_id == FaceId.Y_MIN:
            face_center = np.array([center[0], bounds[0][1], center[2]])
            face_normal = np.array([0, -1, 0])
            effective_radius = min(extents[0], extents[2]) / 2 - policy.thickness - policy.inset
        else:
            face_center = center
            face_normal = np.array([0, 0, 1])
            effective_radius = min(extents) / 2 - policy.thickness - policy.inset
        
        constraints = FeatureConstraints(
            effective_radius=max(0.0, effective_radius),
            face_center=face_center,
            face_normal=face_normal,
            min_clearance=policy.thickness,
        )
        
        warnings.append("Ridge mesh not created (no domain_spec provided), constraints estimated from mesh bounds")
    
    # Build report
    report = RidgeOperationReport(
        success=True,
        requested_policy=policy.to_dict(),
        effective_policy=policy.to_dict(),
        warnings=warnings,
        metadata={
            "face": face,
            "face_id": face_id.value,
            "effective_radius": constraints.effective_radius,
            "face_center": constraints.face_center.tolist() if constraints.face_center is not None else None,
            "face_normal": constraints.face_normal.tolist() if constraints.face_normal is not None else None,
        },
    )
    
    return domain_mesh, constraints, report


__all__ = [
    "FaceFeature",
    "FeatureType",
    "FaceId",
    "FeatureConstraints",
    "FeatureResult",
    "RidgeFeature",
    "RidgeFeatureSpec",
    "create_ridge_with_constraints",
    # PATCH 4: New exports for canonical face string API
    "add_ridge",
    "RidgePolicy",
    "RidgeOperationReport",
    "CANONICAL_FACE_MAP",
    "_parse_face_string",
]
