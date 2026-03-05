"""
Post-Embedding Printability Checks

Validates manufacturability constraints after embedding structure into a domain.
Checks include minimum channel diameter, wall thickness, and unsupported features.

These checks are parameterized by user-provided manufacturing constraints
(e.g., printer type, plate size, minimum feature size).
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from scipy import ndimage
import trimesh


@dataclass
class ManufacturingConfig:
    """
    Manufacturing constraints provided by user.
    
    Attributes
    ----------
    min_channel_diameter : float
        Minimum printable channel diameter in mm
    min_wall_thickness : float
        Minimum printable wall thickness in mm
    max_overhang_angle : float
        Maximum unsupported overhang angle in degrees (from vertical)
    plate_size : Tuple[float, float, float]
        Build plate dimensions (width, depth, height) in mm
    units : str
        Units used in the model ('mm', 'm', 'cm')
    printer_type : str
        Type of printer ('SLA', 'FDM', 'DLP', etc.)
    """
    min_channel_diameter: float = 0.5  # mm
    min_wall_thickness: float = 0.3  # mm
    max_overhang_angle: float = 45.0  # degrees
    plate_size: Tuple[float, float, float] = (200.0, 200.0, 200.0)  # mm
    units: str = "mm"
    printer_type: str = "SLA"


@dataclass
class PrintabilityCheckResult:
    """Result of a printability check."""
    passed: bool
    check_name: str
    message: str
    details: Dict[str, Any]
    warnings: List[str]


def check_min_channel_diameter(
    mesh: trimesh.Trimesh,
    config: ManufacturingConfig,
    pitch: Optional[float] = None,
) -> PrintabilityCheckResult:
    """
    Check that all channels meet minimum diameter requirements.
    
    Channels smaller than the minimum printable diameter will not
    print correctly and may be blocked or collapsed.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        The embedded mesh (domain with void)
    config : ManufacturingConfig
        Manufacturing constraints
    pitch : float, optional
        Voxel pitch for analysis (default: min_channel_diameter / 4)
        
    Returns
    -------
    PrintabilityCheckResult
        Result with pass/fail status and details
    """
    min_diameter = config.min_channel_diameter
    
    if pitch is None:
        pitch = min_diameter / 4.0
    
    try:
        vox = mesh.voxelized(pitch)
        vox_filled = vox.fill()
        fluid_mask = vox_filled.matrix.astype(bool)
    except Exception as e:
        return PrintabilityCheckResult(
            passed=False,
            check_name="min_channel_diameter",
            message=f"Voxelization failed: {e}",
            details={"error": str(e)},
            warnings=[str(e)],
        )
    
    if not fluid_mask.any():
        return PrintabilityCheckResult(
            passed=False,
            check_name="min_channel_diameter",
            message="No fluid volume found",
            details={"error": "Empty fluid mask"},
            warnings=["No fluid volume detected in mesh"],
        )
    
    # Compute distance transform to find channel radii
    distance = ndimage.distance_transform_edt(fluid_mask)
    
    # Convert to physical units
    distance_mm = distance * pitch
    
    # Find minimum channel radius (max distance from wall at any point)
    # For a channel, the radius is the distance from center to wall
    # We look at the skeleton (local maxima of distance transform)
    
    # Simple approach: find the minimum of the local maxima
    # A more sophisticated approach would extract the skeleton
    
    # Find local maxima (channel centers)
    from scipy.ndimage import maximum_filter
    local_max = maximum_filter(distance_mm, size=3)
    is_local_max = (distance_mm == local_max) & (distance_mm > 0)
    
    if not is_local_max.any():
        return PrintabilityCheckResult(
            passed=True,
            check_name="min_channel_diameter",
            message="No channel centers found",
            details={"warning": "Could not identify channel centers"},
            warnings=["Could not identify channel centers for diameter analysis"],
        )
    
    # Get radii at channel centers
    channel_radii = distance_mm[is_local_max]
    channel_diameters = 2 * channel_radii
    
    min_found_diameter = float(np.min(channel_diameters))
    mean_diameter = float(np.mean(channel_diameters))
    max_diameter = float(np.max(channel_diameters))
    
    # Count channels below minimum
    below_min = channel_diameters < min_diameter
    num_below_min = int(np.sum(below_min))
    fraction_below_min = float(np.mean(below_min))
    
    passed = min_found_diameter >= min_diameter
    
    details = {
        "min_required_diameter": min_diameter,
        "min_found_diameter": min_found_diameter,
        "mean_diameter": mean_diameter,
        "max_diameter": max_diameter,
        "num_channel_points": len(channel_diameters),
        "num_below_min": num_below_min,
        "fraction_below_min": fraction_below_min,
        "pitch": pitch,
        "units": config.units,
    }
    
    warnings = []
    if not passed:
        warnings.append(f"Min channel diameter {min_found_diameter:.2f}mm < required {min_diameter:.2f}mm")
    if fraction_below_min > 0.1:
        warnings.append(f"{fraction_below_min:.1%} of channel points below minimum diameter")
    
    return PrintabilityCheckResult(
        passed=passed,
        check_name="min_channel_diameter",
        message=f"Min diameter: {min_found_diameter:.2f}mm (required: {min_diameter:.2f}mm)",
        details=details,
        warnings=warnings,
    )


def check_wall_thickness(
    mesh: trimesh.Trimesh,
    config: ManufacturingConfig,
    pitch: Optional[float] = None,
) -> PrintabilityCheckResult:
    """
    Check that all walls meet minimum thickness requirements.
    
    Walls thinner than the minimum will not print correctly
    and may break or deform.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        The embedded mesh (domain with void)
    config : ManufacturingConfig
        Manufacturing constraints
    pitch : float, optional
        Voxel pitch for analysis (default: min_wall_thickness / 4)
        
    Returns
    -------
    PrintabilityCheckResult
        Result with pass/fail status and details
    """
    min_thickness = config.min_wall_thickness
    
    if pitch is None:
        pitch = min_thickness / 4.0
    
    try:
        vox = mesh.voxelized(pitch)
        vox_filled = vox.fill()
        fluid_mask = vox_filled.matrix.astype(bool)
    except Exception as e:
        return PrintabilityCheckResult(
            passed=False,
            check_name="wall_thickness",
            message=f"Voxelization failed: {e}",
            details={"error": str(e)},
            warnings=[str(e)],
        )
    
    # Wall is the solid part (inverse of fluid)
    solid_mask = ~fluid_mask
    
    if not solid_mask.any():
        return PrintabilityCheckResult(
            passed=False,
            check_name="wall_thickness",
            message="No solid volume found",
            details={"error": "Empty solid mask"},
            warnings=["No solid volume detected - structure may be invalid"],
        )
    
    # Compute distance transform for solid
    distance = ndimage.distance_transform_edt(solid_mask)
    distance_mm = distance * pitch
    
    # Find local maxima (wall centers)
    from scipy.ndimage import maximum_filter
    local_max = maximum_filter(distance_mm, size=3)
    is_local_max = (distance_mm == local_max) & (distance_mm > 0)
    
    if not is_local_max.any():
        return PrintabilityCheckResult(
            passed=True,
            check_name="wall_thickness",
            message="No wall centers found",
            details={"warning": "Could not identify wall centers"},
            warnings=["Could not identify wall centers for thickness analysis"],
        )
    
    # Get half-thickness at wall centers (distance to nearest surface)
    wall_half_thickness = distance_mm[is_local_max]
    wall_thickness = 2 * wall_half_thickness
    
    min_found_thickness = float(np.min(wall_thickness))
    mean_thickness = float(np.mean(wall_thickness))
    max_thickness = float(np.max(wall_thickness))
    
    # Count walls below minimum
    below_min = wall_thickness < min_thickness
    num_below_min = int(np.sum(below_min))
    fraction_below_min = float(np.mean(below_min))
    
    passed = min_found_thickness >= min_thickness
    
    details = {
        "min_required_thickness": min_thickness,
        "min_found_thickness": min_found_thickness,
        "mean_thickness": mean_thickness,
        "max_thickness": max_thickness,
        "num_wall_points": len(wall_thickness),
        "num_below_min": num_below_min,
        "fraction_below_min": fraction_below_min,
        "pitch": pitch,
        "units": config.units,
    }
    
    warnings = []
    if not passed:
        warnings.append(f"Min wall thickness {min_found_thickness:.2f}mm < required {min_thickness:.2f}mm")
    if fraction_below_min > 0.1:
        warnings.append(f"{fraction_below_min:.1%} of wall points below minimum thickness")
    
    return PrintabilityCheckResult(
        passed=passed,
        check_name="wall_thickness",
        message=f"Min thickness: {min_found_thickness:.2f}mm (required: {min_thickness:.2f}mm)",
        details=details,
        warnings=warnings,
    )


def check_unsupported_features(
    mesh: trimesh.Trimesh,
    config: ManufacturingConfig,
) -> PrintabilityCheckResult:
    """
    Check for unsupported overhangs that may cause print failures.
    
    Overhangs beyond the maximum angle require support structures
    or may fail to print correctly.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        The embedded mesh (domain with void)
    config : ManufacturingConfig
        Manufacturing constraints
        
    Returns
    -------
    PrintabilityCheckResult
        Result with pass/fail status and details
    """
    max_angle = config.max_overhang_angle
    
    # Get face normals
    normals = mesh.face_normals
    
    # Calculate angle from vertical (Z-up assumed)
    # Overhang angle is angle between normal and -Z axis
    z_down = np.array([0, 0, -1])
    
    # Dot product gives cos(angle)
    cos_angles = np.dot(normals, z_down)
    angles_rad = np.arccos(np.clip(cos_angles, -1, 1))
    angles_deg = np.degrees(angles_rad)
    
    # Overhang faces are those pointing downward (angle < 90 from -Z)
    # and beyond the maximum overhang angle
    is_overhang = angles_deg < (90 - max_angle)
    
    num_overhang_faces = int(np.sum(is_overhang))
    total_faces = len(normals)
    overhang_fraction = num_overhang_faces / total_faces if total_faces > 0 else 0.0
    
    # Calculate overhang area
    face_areas = mesh.area_faces
    overhang_area = float(np.sum(face_areas[is_overhang]))
    total_area = float(np.sum(face_areas))
    overhang_area_fraction = overhang_area / total_area if total_area > 0 else 0.0
    
    # Pass if overhang area is small (< 10% of total)
    passed = overhang_area_fraction < 0.1
    
    details = {
        "max_overhang_angle": max_angle,
        "num_overhang_faces": num_overhang_faces,
        "total_faces": total_faces,
        "overhang_fraction": float(overhang_fraction),
        "overhang_area": overhang_area,
        "total_area": total_area,
        "overhang_area_fraction": float(overhang_area_fraction),
        "printer_type": config.printer_type,
    }
    
    warnings = []
    if overhang_area_fraction >= 0.1:
        warnings.append(f"{overhang_area_fraction:.1%} of surface area is unsupported overhang")
    if num_overhang_faces > 0:
        warnings.append(f"{num_overhang_faces} faces exceed max overhang angle of {max_angle}deg")
    
    return PrintabilityCheckResult(
        passed=passed,
        check_name="unsupported_features",
        message=f"Overhang area: {overhang_area_fraction:.1%} (max angle: {max_angle}deg)",
        details=details,
        warnings=warnings,
    )


@dataclass
class PrintabilityCheckReport:
    """Aggregated report of all printability checks."""
    passed: bool
    status: str  # "ok", "warnings", "fail"
    checks: List[PrintabilityCheckResult]
    summary: Dict[str, Any]
    config: ManufacturingConfig


def run_all_printability_checks(
    mesh: trimesh.Trimesh,
    config: Optional[ManufacturingConfig] = None,
) -> PrintabilityCheckReport:
    """
    Run all post-embedding printability checks.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        The embedded mesh (domain with void)
    config : ManufacturingConfig, optional
        Manufacturing constraints (uses defaults if not provided)
        
    Returns
    -------
    PrintabilityCheckReport
        Aggregated report with all check results
    """
    if config is None:
        config = ManufacturingConfig()
    
    checks = [
        check_min_channel_diameter(mesh, config),
        check_wall_thickness(mesh, config),
        check_unsupported_features(mesh, config),
    ]
    
    all_passed = all(c.passed for c in checks)
    has_warnings = any(len(c.warnings) > 0 for c in checks)
    
    if all_passed and not has_warnings:
        status = "ok"
    elif all_passed:
        status = "warnings"
    else:
        status = "fail"
    
    summary = {
        "total_checks": len(checks),
        "passed_checks": sum(1 for c in checks if c.passed),
        "failed_checks": sum(1 for c in checks if not c.passed),
        "total_warnings": sum(len(c.warnings) for c in checks),
    }
    
    return PrintabilityCheckReport(
        passed=all_passed,
        status=status,
        checks=checks,
        summary=summary,
        config=config,
    )
