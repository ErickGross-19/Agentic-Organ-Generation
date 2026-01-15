"""
Canonical face naming utilities for domain surfaces.

This module provides a unified face naming convention across the codebase.
All face references should use canonical string names:
    "top" | "bottom" | "+x" | "-x" | "+y" | "-y" | "+z" | "-z"

The "top" and "bottom" aliases map to "+z" and "-z" respectively for convenience.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from typing import Tuple, Optional, Union, TYPE_CHECKING
from enum import Enum
import numpy as np

if TYPE_CHECKING:
    from ..core.domain import DomainSpec


# Canonical face names
CANONICAL_FACES = frozenset([
    "top", "bottom",
    "+x", "-x",
    "+y", "-y",
    "+z", "-z",
])

# Aliases that map to canonical names
FACE_ALIASES = {
    "z_max": "+z",
    "z_min": "-z",
    "x_max": "+x",
    "x_min": "-x",
    "y_max": "+y",
    "y_min": "-y",
    "TOP": "top",
    "BOTTOM": "bottom",
    "Z_MAX": "+z",
    "Z_MIN": "-z",
    "X_MAX": "+x",
    "X_MIN": "-x",
    "Y_MAX": "+y",
    "Y_MIN": "-y",
}

# Mapping from canonical names to axis info (axis_index, sign)
FACE_AXIS_INFO = {
    "top": (2, 1),      # +Z
    "bottom": (2, -1),  # -Z
    "+x": (0, 1),
    "-x": (0, -1),
    "+y": (1, 1),
    "-y": (1, -1),
    "+z": (2, 1),
    "-z": (2, -1),
}


class FaceId(Enum):
    """
    Enumeration of domain face identifiers.
    
    This enum is provided for internal use only. External APIs should use
    canonical string names ("top", "bottom", "+x", etc.) and convert at
    module boundaries using validate_face() or face_to_enum().
    """
    X_MIN = "-x"
    X_MAX = "+x"
    Y_MIN = "-y"
    Y_MAX = "+y"
    Z_MIN = "-z"
    Z_MAX = "+z"
    TOP = "+z"      # Alias for Z_MAX
    BOTTOM = "-z"   # Alias for Z_MIN


def validate_face(face: Union[str, FaceId]) -> str:
    """
    Validate and normalize a face identifier to canonical string form.
    
    Accepts various input formats and returns the canonical string name.
    
    Parameters
    ----------
    face : str or FaceId
        Face identifier in any supported format:
        - Canonical strings: "top", "bottom", "+x", "-x", "+y", "-y", "+z", "-z"
        - Legacy strings: "z_max", "z_min", "x_max", etc.
        - FaceId enum values
        
    Returns
    -------
    str
        Canonical face name: "top", "bottom", "+x", "-x", "+y", "-y", "+z", "-z"
        
    Raises
    ------
    ValueError
        If face is not a recognized identifier
        
    Examples
    --------
    >>> validate_face("top")
    'top'
    >>> validate_face("z_max")
    '+z'
    >>> validate_face(FaceId.TOP)
    '+z'
    >>> validate_face("+x")
    '+x'
    """
    # Handle FaceId enum
    if isinstance(face, FaceId):
        return face.value
    
    # Handle string input
    if isinstance(face, str):
        # Check if already canonical
        if face in CANONICAL_FACES:
            return face
        
        # Check aliases
        if face in FACE_ALIASES:
            return FACE_ALIASES[face]
        
        # Try lowercase
        face_lower = face.lower()
        if face_lower in CANONICAL_FACES:
            return face_lower
        if face_lower in FACE_ALIASES:
            return FACE_ALIASES[face_lower]
    
    raise ValueError(
        f"Unknown face identifier: {face!r}. "
        f"Valid faces: {sorted(CANONICAL_FACES)}"
    )


def face_to_enum(face: Union[str, FaceId]) -> FaceId:
    """
    Convert a face identifier to FaceId enum (internal use only).
    
    Parameters
    ----------
    face : str or FaceId
        Face identifier in any supported format
        
    Returns
    -------
    FaceId
        Corresponding FaceId enum value
        
    Examples
    --------
    >>> face_to_enum("top")
    <FaceId.TOP: '+z'>
    >>> face_to_enum("+x")
    <FaceId.X_MAX: '+x'>
    """
    if isinstance(face, FaceId):
        return face
    
    canonical = validate_face(face)
    
    # Map canonical string to enum
    enum_map = {
        "top": FaceId.TOP,
        "bottom": FaceId.BOTTOM,
        "+x": FaceId.X_MAX,
        "-x": FaceId.X_MIN,
        "+y": FaceId.Y_MAX,
        "-y": FaceId.Y_MIN,
        "+z": FaceId.Z_MAX,
        "-z": FaceId.Z_MIN,
    }
    
    return enum_map[canonical]


def face_normal(face: Union[str, FaceId], domain: Optional["DomainSpec"] = None) -> Tuple[float, float, float]:
    """
    Get the outward-pointing normal vector for a face.
    
    Parameters
    ----------
    face : str or FaceId
        Face identifier
    domain : DomainSpec, optional
        Domain specification (currently unused, but reserved for future
        support of non-axis-aligned domains)
        
    Returns
    -------
    tuple of float
        Unit normal vector (nx, ny, nz) pointing outward from the face
        
    Examples
    --------
    >>> face_normal("top")
    (0.0, 0.0, 1.0)
    >>> face_normal("-x")
    (-1.0, 0.0, 0.0)
    >>> face_normal("bottom")
    (0.0, 0.0, -1.0)
    """
    canonical = validate_face(face)
    axis_idx, sign = FACE_AXIS_INFO[canonical]
    
    normal = [0.0, 0.0, 0.0]
    normal[axis_idx] = float(sign)
    
    return tuple(normal)


def face_center(face: Union[str, FaceId], domain: "DomainSpec") -> Tuple[float, float, float]:
    """
    Get the center point of a face on a domain.
    
    Parameters
    ----------
    face : str or FaceId
        Face identifier
    domain : DomainSpec
        Domain specification (BoxDomain, CylinderDomain, or EllipsoidDomain)
        
    Returns
    -------
    tuple of float
        Center point (x, y, z) of the face in meters
        
    Raises
    ------
    TypeError
        If domain type is not supported
        
    Examples
    --------
    >>> from generation.core.domain import CylinderDomain
    >>> from generation.core.types import Point3D
    >>> domain = CylinderDomain(center=Point3D(0, 0, 0), radius=0.004, height=0.003)
    >>> face_center("top", domain)
    (0.0, 0.0, 0.0015)
    """
    from ..core.domain import BoxDomain, CylinderDomain, EllipsoidDomain
    
    canonical = validate_face(face)
    
    if isinstance(domain, CylinderDomain):
        cx, cy, cz = domain.center.x, domain.center.y, domain.center.z
        
        if canonical in ("top", "+z"):
            return (cx, cy, cz + domain.height / 2)
        elif canonical in ("bottom", "-z"):
            return (cx, cy, cz - domain.height / 2)
        elif canonical == "+x":
            return (cx + domain.radius, cy, cz)
        elif canonical == "-x":
            return (cx - domain.radius, cy, cz)
        elif canonical == "+y":
            return (cx, cy + domain.radius, cz)
        elif canonical == "-y":
            return (cx, cy - domain.radius, cz)
        else:
            raise ValueError(f"Unsupported face for cylinder: {canonical}")
    
    elif isinstance(domain, BoxDomain):
        cx = (domain.x_min + domain.x_max) / 2
        cy = (domain.y_min + domain.y_max) / 2
        cz = (domain.z_min + domain.z_max) / 2
        
        if canonical in ("top", "+z"):
            return (cx, cy, domain.z_max)
        elif canonical in ("bottom", "-z"):
            return (cx, cy, domain.z_min)
        elif canonical == "+x":
            return (domain.x_max, cy, cz)
        elif canonical == "-x":
            return (domain.x_min, cy, cz)
        elif canonical == "+y":
            return (cx, domain.y_max, cz)
        elif canonical == "-y":
            return (cx, domain.y_min, cz)
        else:
            raise ValueError(f"Unsupported face for box: {canonical}")
    
    elif isinstance(domain, EllipsoidDomain):
        cx, cy, cz = domain.center.x, domain.center.y, domain.center.z
        
        if canonical in ("top", "+z"):
            return (cx, cy, cz + domain.semi_axis_c)
        elif canonical in ("bottom", "-z"):
            return (cx, cy, cz - domain.semi_axis_c)
        elif canonical == "+x":
            return (cx + domain.semi_axis_a, cy, cz)
        elif canonical == "-x":
            return (cx - domain.semi_axis_a, cy, cz)
        elif canonical == "+y":
            return (cx, cy + domain.semi_axis_b, cz)
        elif canonical == "-y":
            return (cx, cy - domain.semi_axis_b, cz)
        else:
            raise ValueError(f"Unsupported face for ellipsoid: {canonical}")
    
    else:
        raise TypeError(
            f"Unsupported domain type: {type(domain).__name__}. "
            f"Supported: BoxDomain, CylinderDomain, EllipsoidDomain"
        )


def face_frame(
    face: Union[str, FaceId],
    domain: "DomainSpec",
) -> Tuple[
    Tuple[float, float, float],  # origin
    Tuple[float, float, float],  # normal
    Tuple[float, float, float],  # u (tangent)
    Tuple[float, float, float],  # v (tangent)
    Tuple[float, float, float],  # center
]:
    """
    Build a reference frame for a face.
    
    Returns the origin, normal, and two tangent vectors (u, v) that define
    a local coordinate system on the face, plus the face center.
    
    Parameters
    ----------
    face : str or FaceId
        Face identifier
    domain : DomainSpec
        Domain specification
        
    Returns
    -------
    tuple
        (origin, normal, u, v, center) where:
        - origin: Point at the face center (same as center)
        - normal: Outward-pointing unit normal
        - u: First tangent vector (unit length)
        - v: Second tangent vector (unit length, perpendicular to u and normal)
        - center: Face center point
        
    Examples
    --------
    >>> from generation.core.domain import CylinderDomain
    >>> from generation.core.types import Point3D
    >>> domain = CylinderDomain(center=Point3D(0, 0, 0), radius=0.004, height=0.003)
    >>> origin, normal, u, v, center = face_frame("top", domain)
    >>> normal
    (0.0, 0.0, 1.0)
    """
    canonical = validate_face(face)
    center = face_center(canonical, domain)
    normal = face_normal(canonical, domain)
    
    # Build tangent vectors based on normal direction
    n = np.array(normal)
    
    # Choose a reference vector not parallel to normal
    if abs(n[2]) < 0.9:
        ref = np.array([0.0, 0.0, 1.0])
    else:
        ref = np.array([1.0, 0.0, 0.0])
    
    # u = ref - (ref . n) * n, then normalize
    u = ref - np.dot(ref, n) * n
    u = u / np.linalg.norm(u)
    
    # v = n x u
    v = np.cross(n, u)
    v = v / np.linalg.norm(v)
    
    return (
        center,  # origin
        normal,  # normal
        tuple(u.tolist()),  # u tangent
        tuple(v.tolist()),  # v tangent
        center,  # center (same as origin for faces)
    )


def opposite_face(face: Union[str, FaceId]) -> str:
    """
    Get the opposite face.
    
    Parameters
    ----------
    face : str or FaceId
        Face identifier
        
    Returns
    -------
    str
        Canonical name of the opposite face
        
    Examples
    --------
    >>> opposite_face("top")
    'bottom'
    >>> opposite_face("+x")
    '-x'
    """
    canonical = validate_face(face)
    
    opposites = {
        "top": "bottom",
        "bottom": "top",
        "+x": "-x",
        "-x": "+x",
        "+y": "-y",
        "-y": "+y",
        "+z": "-z",
        "-z": "+z",
    }
    
    return opposites[canonical]


__all__ = [
    "CANONICAL_FACES",
    "FACE_ALIASES",
    "FaceId",
    "validate_face",
    "face_to_enum",
    "face_normal",
    "face_center",
    "face_frame",
    "opposite_face",
]
