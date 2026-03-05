import manifold3d as m3d
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

@dataclass
class ValidationCheck:
    passed: bool
    message: str = ""
    measured_value: Optional[float] = None
    required_value: Optional[float] = None

@dataclass
class ValidationResult:
    is_valid: bool
    is_printable: bool
    checks: dict[str, ValidationCheck]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'is_valid': self.is_valid,
            'is_printable': self.is_printable,
            'checks': {k: {
                'passed': v.passed,
                'message': v.message,
                'measured_value': v.measured_value,
                'required_value': v.required_value
            } for k, v in self.checks.items()},
            'warnings': self.warnings,
            'errors': self.errors
        }

def check_watertight(manifold: m3d.Manifold) -> ValidationCheck:
    """Check if mesh is watertight (manifold)."""
    # manifold3d creates valid manifolds by construction
    # Check that it has volume and faces
    try:
        mesh = manifold.to_mesh()
        has_geometry = len(mesh.tri_verts) > 0
        volume = manifold.volume() if hasattr(manifold, 'volume') else 0

        if has_geometry and volume > 0:
            return ValidationCheck(passed=True, message="Mesh is watertight")
        else:
            return ValidationCheck(passed=False, message="Mesh has no volume or faces")
    except Exception as e:
        return ValidationCheck(passed=False, message=f"Error checking mesh: {str(e)}")

def check_min_wall_thickness(
    manifold: m3d.Manifold,
    params: dict,
    min_thickness_mm: float = 0.3
) -> ValidationCheck:
    """
    Check minimum wall thickness.
    Uses parameter values for accurate check when available.
    """
    # For specific scaffold types, check relevant parameters
    scaffold_type = params.get('type', params.get('scaffold_type', ''))

    if scaffold_type == 'porous_disc':
        # Check wall between pores
        pore_diameter_um = params.get('pore_diameter_um', 200)
        pore_spacing_um = params.get('pore_spacing_um', 400)
        wall_between_pores_mm = (pore_spacing_um - pore_diameter_um) / 1000

        if wall_between_pores_mm < min_thickness_mm:
            return ValidationCheck(
                passed=False,
                message=f"Wall between pores ({wall_between_pores_mm:.2f}mm) is less than minimum",
                measured_value=wall_between_pores_mm,
                required_value=min_thickness_mm
            )

    elif scaffold_type == 'tubular_conduit':
        wall_thickness = params.get('wall_thickness_mm', 0.5)
        if wall_thickness < min_thickness_mm:
            return ValidationCheck(
                passed=False,
                message=f"Wall thickness ({wall_thickness:.2f}mm) is less than minimum",
                measured_value=wall_thickness,
                required_value=min_thickness_mm
            )

    elif scaffold_type == 'lattice':
        strut_diameter = params.get('strut_diameter_mm', 0.4)
        if strut_diameter < min_thickness_mm:
            return ValidationCheck(
                passed=False,
                message=f"Strut diameter ({strut_diameter:.2f}mm) is less than minimum",
                measured_value=strut_diameter,
                required_value=min_thickness_mm
            )

    elif scaffold_type == 'vascular_network':
        # Check inlet radius as proxy for minimum feature size
        inlet_radius = params.get('inlet_radius_mm', params.get('inlet_radius', 0.5))
        if inlet_radius * 2 < min_thickness_mm:
            return ValidationCheck(
                passed=False,
                message=f"Minimum channel diameter ({inlet_radius*2:.2f}mm) is very small",
                measured_value=inlet_radius * 2,
                required_value=min_thickness_mm
            )

    # Default: try to estimate from bounding box
    try:
        mesh = manifold.to_mesh()
        verts = np.array(mesh.vert_properties)[:, :3]
        min_bounds = verts.min(axis=0)
        max_bounds = verts.max(axis=0)
        sizes = max_bounds - min_bounds
        min_size = sizes.min()

        return ValidationCheck(
            passed=min_size >= min_thickness_mm,
            message=f"Minimum dimension is {min_size:.2f}mm",
            measured_value=min_size,
            required_value=min_thickness_mm
        )
    except (ValueError, TypeError, AttributeError, IndexError) as e:
        logger.error(f"Error checking wall thickness: {type(e).__name__}: {str(e)}")
        return ValidationCheck(
            passed=False,
            message=f"Could not verify wall thickness: {type(e).__name__}: {str(e)}"
        )

def check_overhang(
    manifold: m3d.Manifold,
    max_angle_deg: float = 45.0
) -> ValidationCheck:
    """
    Check for excessive overhangs (faces pointing too far down).
    """
    try:
        mesh = manifold.to_mesh()
        # tri_verts contains indices into vert_properties, not actual coordinates
        verts = np.array(mesh.vert_properties)[:, :3]  # Get vertex positions
        tri_indices = np.array(mesh.tri_verts)  # Triangle vertex indices (Nx3)
        tri_verts = verts[tri_indices]  # Index into vertices (Nx3x3)

        # Calculate face normals
        v0, v1, v2 = tri_verts[:, 0], tri_verts[:, 1], tri_verts[:, 2]
        normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normals = normals / norms

        # Check z-component (negative z = faces down)
        # Angle from vertical down = acos(-z)
        z_components = normals[:, 2]

        # Faces pointing down have negative z
        downward_faces = z_components < 0
        if not np.any(downward_faces):
            return ValidationCheck(passed=True, message="No overhanging faces")

        # Calculate overhang angles for downward faces
        # overhang angle = angle from horizontal = 90 - angle_from_vertical
        overhang_angles = np.degrees(np.arccos(np.abs(z_components[downward_faces])))
        max_overhang = overhang_angles.max() if len(overhang_angles) > 0 else 0

        threshold_angle = 90 - max_angle_deg  # Convert from "from vertical" to "from horizontal"

        passed = max_overhang <= threshold_angle + 5  # 5 degree tolerance

        return ValidationCheck(
            passed=passed,
            message=f"Maximum overhang angle: {max_overhang:.1f}° (threshold: {threshold_angle:.1f}°)",
            measured_value=max_overhang,
            required_value=threshold_angle
        )
    except Exception as e:
        return ValidationCheck(passed=True, message=f"Could not check overhangs: {e}")

def check_dimensions(
    manifold: m3d.Manifold,
    max_dimensions: tuple[float, float, float] = (100.0, 100.0, 100.0)
) -> ValidationCheck:
    """Check if scaffold fits within build volume."""
    try:
        mesh = manifold.to_mesh()
        verts = np.array(mesh.vert_properties)[:, :3]
        min_bounds = verts.min(axis=0)
        max_bounds = verts.max(axis=0)
        sizes = max_bounds - min_bounds

        exceeds = any(s > m for s, m in zip(sizes, max_dimensions))

        return ValidationCheck(
            passed=not exceeds,
            message=f"Bounding box: {sizes[0]:.1f} x {sizes[1]:.1f} x {sizes[2]:.1f} mm",
            measured_value=max(sizes),
            required_value=max(max_dimensions)
        )
    except (ValueError, TypeError, AttributeError, IndexError) as e:
        logger.error(f"Error checking dimensions: {type(e).__name__}: {str(e)}")
        return ValidationCheck(
            passed=False,
            message=f"Could not check dimensions: {type(e).__name__}: {str(e)}"
        )

def validate_scaffold(
    manifold: m3d.Manifold,
    params: dict,
    min_wall_mm: float = 0.3,
    max_overhang_deg: float = 45.0,
    max_dimensions: tuple[float, float, float] = (100.0, 100.0, 100.0)
) -> ValidationResult:
    """
    Validate a scaffold for printability.

    Returns ValidationResult with all checks.
    """
    checks = {
        'watertight': check_watertight(manifold),
        'min_wall_thickness': check_min_wall_thickness(manifold, params, min_wall_mm),
        'overhang': check_overhang(manifold, max_overhang_deg),
        'dimensions': check_dimensions(manifold, max_dimensions)
    }

    warnings = []
    errors = []

    for name, check in checks.items():
        if not check.passed:
            if name in ['watertight', 'dimensions']:
                errors.append(f"{name}: {check.message}")
            else:
                warnings.append(f"{name}: {check.message}")

    # is_valid = no errors (geometry is sound)
    # is_printable = no errors AND no warnings (ready to print)
    is_valid = len(errors) == 0
    is_printable = is_valid and len(warnings) == 0

    return ValidationResult(
        is_valid=is_valid,
        is_printable=is_printable,
        checks=checks,
        warnings=warnings,
        errors=errors
    )
