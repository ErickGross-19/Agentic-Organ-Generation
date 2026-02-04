"""
Compile design specifications into runtime domain objects.

This module provides the bridge between user-facing spec classes (EllipsoidSpec, BoxSpec)
and the runtime domain objects (EllipsoidDomain, BoxDomain) used internally.

UNIT CONVENTIONS
----------------
**Spec units**: Values in spec classes (EllipsoidSpec, BoxSpec, InletSpec, etc.) are in
METERS. This is the internal unit system used throughout the codebase. For example:
- EllipsoidSpec(semi_axes=(0.05, 0.045, 0.035)) represents 50mm x 45mm x 35mm
- InletSpec(radius=0.002) represents a 2mm radius inlet

**Runtime units**: The compiled Domain objects also use METERS internally. All geometric
operations (contains, sample_points, distance_to_boundary) operate in meters.

**Output units**: At export time (STL, JSON), values are converted from internal meters
to the user-specified output_units (default "mm") via UnitContext.

COORDINATE FRAME
----------------
The default coordinate frame is:
- Origin at domain center (0, 0, 0)
- X-axis: left-right (width)
- Y-axis: front-back (depth)  
- Z-axis: bottom-top (height)

For organ-specific coordinate frames (e.g., anatomical orientation), use the
transform parameter in compile_domain() to apply a rotation/translation.

USAGE
-----
>>> from generation.specs.design_spec import EllipsoidSpec
>>> from generation.specs.compile import compile_domain
>>>
>>> spec = EllipsoidSpec(center=(0, 0, 0), semi_axes=(0.05, 0.045, 0.035))
>>> domain = compile_domain(spec)
>>> # domain is now an EllipsoidDomain ready for use in generation
"""

from typing import Optional
import numpy as np

from .design_spec import DomainSpec as SpecDomainSpec, EllipsoidSpec, BoxSpec, CylinderSpec
from ..core.domain import DomainSpec as RuntimeDomainSpec, EllipsoidDomain, BoxDomain, CylinderDomain
from ..core.types import Point3D


def compile_domain(
    spec: SpecDomainSpec,
    transform: Optional[np.ndarray] = None,
) -> RuntimeDomainSpec:
    """
    Compile a domain specification into a runtime domain object.
    
    This function centralizes the conversion from user-facing spec classes
    to internal runtime domain objects, handling:
    - Unit validation (specs are in meters)
    - Default value application
    - Optional coordinate transforms
    
    Parameters
    ----------
    spec : DomainSpec (from design_spec)
        The domain specification (EllipsoidSpec or BoxSpec)
    transform : np.ndarray, optional
        4x4 homogeneous transformation matrix to apply to the domain.
        Use this for coordinate frame changes (rotation, translation).
        If None, no transform is applied.
        
    Returns
    -------
    DomainSpec (from core.domain)
        The compiled runtime domain (EllipsoidDomain or BoxDomain)
        
    Raises
    ------
    ValueError
        If the spec type is not supported
        
    Examples
    --------
    >>> from generation.specs.design_spec import EllipsoidSpec
    >>> spec = EllipsoidSpec(center=(0, 0, 0), semi_axes=(0.05, 0.045, 0.035))
    >>> domain = compile_domain(spec)
    >>> domain.contains(Point3D(0.01, 0.01, 0.01))
    True
    
    >>> # With a transform (90 degree rotation around Z)
    >>> import numpy as np
    >>> rot_z = np.array([
    ...     [0, -1, 0, 0],
    ...     [1,  0, 0, 0],
    ...     [0,  0, 1, 0],
    ...     [0,  0, 0, 1]
    ... ])
    >>> domain = compile_domain(spec, transform=rot_z)
    """
    if isinstance(spec, EllipsoidSpec):
        return _compile_ellipsoid(spec, transform)
    elif isinstance(spec, BoxSpec):
        return _compile_box(spec, transform)
    elif isinstance(spec, CylinderSpec):
        return _compile_cylinder(spec, transform)
    else:
        raise ValueError(f"Unsupported domain spec type: {type(spec).__name__}")


def _compile_ellipsoid(
    spec: EllipsoidSpec,
    transform: Optional[np.ndarray] = None,
) -> EllipsoidDomain:
    """
    Compile an EllipsoidSpec into an EllipsoidDomain.
    
    Parameters
    ----------
    spec : EllipsoidSpec
        Ellipsoid specification with center and semi_axes in meters
    transform : np.ndarray, optional
        4x4 transformation matrix
        
    Returns
    -------
    EllipsoidDomain
        Compiled ellipsoid domain
    """
    center = np.array(spec.center)
    
    if transform is not None:
        center_homogeneous = np.array([center[0], center[1], center[2], 1.0])
        transformed = transform @ center_homogeneous
        center = transformed[:3]
    
    return EllipsoidDomain(
        semi_axis_a=spec.semi_axes[0],
        semi_axis_b=spec.semi_axes[1],
        semi_axis_c=spec.semi_axes[2],
        center=Point3D(float(center[0]), float(center[1]), float(center[2])),
    )


def _compile_box(
    spec: BoxSpec,
    transform: Optional[np.ndarray] = None,
) -> BoxDomain:
    """
    Compile a BoxSpec into a BoxDomain.
    
    Parameters
    ----------
    spec : BoxSpec
        Box specification with center and size in meters
    transform : np.ndarray, optional
        4x4 transformation matrix (only translation is applied for axis-aligned boxes)
        
    Returns
    -------
    BoxDomain
        Compiled box domain
    """
    center = np.array(spec.center)
    
    if transform is not None:
        center_homogeneous = np.array([center[0], center[1], center[2], 1.0])
        transformed = transform @ center_homogeneous
        center = transformed[:3]
    
    return BoxDomain.from_center_and_size(
        center=Point3D(float(center[0]), float(center[1]), float(center[2])),
        width=spec.size[0],
        height=spec.size[1],
        depth=spec.size[2],
    )


def _compile_cylinder(
    spec: CylinderSpec,
    transform: Optional[np.ndarray] = None,
) -> CylinderDomain:
    """
    Compile a CylinderSpec into a CylinderDomain.
    
    Parameters
    ----------
    spec : CylinderSpec
        Cylinder specification with center, radius, and height in meters
    transform : np.ndarray, optional
        4x4 transformation matrix (only translation is applied)
        
    Returns
    -------
    CylinderDomain
        Compiled cylinder domain
    """
    center = np.array(spec.center)
    
    if transform is not None:
        center_homogeneous = np.array([center[0], center[1], center[2], 1.0])
        transformed = transform @ center_homogeneous
        center = transformed[:3]
    
    return CylinderDomain(
        radius=spec.radius,
        height=spec.height,
        center=Point3D(float(center[0]), float(center[1]), float(center[2])),
    )


def make_translation_transform(dx: float, dy: float, dz: float) -> np.ndarray:
    """
    Create a 4x4 translation transformation matrix.
    
    Parameters
    ----------
    dx, dy, dz : float
        Translation amounts in meters
        
    Returns
    -------
    np.ndarray
        4x4 homogeneous transformation matrix
    """
    return np.array([
        [1, 0, 0, dx],
        [0, 1, 0, dy],
        [0, 0, 1, dz],
        [0, 0, 0, 1],
    ], dtype=float)


def make_rotation_z_transform(angle_rad: float) -> np.ndarray:
    """
    Create a 4x4 rotation transformation matrix around the Z axis.
    
    Parameters
    ----------
    angle_rad : float
        Rotation angle in radians
        
    Returns
    -------
    np.ndarray
        4x4 homogeneous transformation matrix
    """
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [c, -s, 0, 0],
        [s,  c, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1],
    ], dtype=float)


def make_rotation_y_transform(angle_rad: float) -> np.ndarray:
    """
    Create a 4x4 rotation transformation matrix around the Y axis.
    
    Parameters
    ----------
    angle_rad : float
        Rotation angle in radians
        
    Returns
    -------
    np.ndarray
        4x4 homogeneous transformation matrix
    """
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [ c, 0, s, 0],
        [ 0, 1, 0, 0],
        [-s, 0, c, 0],
        [ 0, 0, 0, 1],
    ], dtype=float)


def make_rotation_x_transform(angle_rad: float) -> np.ndarray:
    """
    Create a 4x4 rotation transformation matrix around the X axis.
    
    Parameters
    ----------
    angle_rad : float
        Rotation angle in radians
        
    Returns
    -------
    np.ndarray
        4x4 homogeneous transformation matrix
    """
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [1, 0,  0, 0],
        [0, c, -s, 0],
        [0, s,  c, 0],
        [0, 0,  0, 1],
    ], dtype=float)
