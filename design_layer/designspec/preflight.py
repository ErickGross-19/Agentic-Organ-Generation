"""
Preflight validation and default policy filling for DesignSpec.

This module provides:
1. Default policy filling - ensures no null policies at runtime
2. Preflight validation - catches issues before pipeline execution
3. Unit audit reporting - debug mode for tracking normalization

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


# Environment variable to enable debug mode
DEBUG_ENV_VAR = "AOG_NORMALIZATION_DEBUG"


@dataclass
class PreflightError:
    """A single preflight validation error."""
    code: str
    message: str
    path: str
    severity: str = "error"  # "error" or "warning"
    suggestion: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "severity": self.severity,
            "suggestion": self.suggestion,
        }


@dataclass
class PreflightResult:
    """Result of preflight validation."""
    success: bool
    errors: List[PreflightError] = field(default_factory=list)
    warnings: List[PreflightError] = field(default_factory=list)
    policies_filled: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [e.to_dict() for e in self.warnings],
            "policies_filled": self.policies_filled,
        }


@dataclass
class UnitAuditEntry:
    """A single entry in the unit audit report."""
    path: str
    field_name: str
    original_value: Any
    normalized_value: Any
    scale_type: str  # "length", "area", "volume"
    scale_factor: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "field_name": self.field_name,
            "original_value": self.original_value,
            "normalized_value": self.normalized_value,
            "scale_type": self.scale_type,
            "scale_factor": self.scale_factor,
        }


@dataclass
class UnitAuditReport:
    """Report of all unit conversions performed during normalization."""
    input_units: str
    scale_factor: float
    entries: List[UnitAuditEntry] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_units": self.input_units,
            "scale_factor": self.scale_factor,
            "total_conversions": len(self.entries),
            "entries": [e.to_dict() for e in self.entries],
        }
    
    def save(self, path: Path) -> None:
        """Save the audit report to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# Default policy configurations (in METERS)
# These are used when a policy is null or missing required keys
DEFAULT_POLICIES: Dict[str, Dict[str, Any]] = {
    "resolution": {
        "min_channel_diameter": 2e-4,  # 0.2mm
        "min_voxels_across_feature": 4,
        "max_voxels": 5_000_000,
        "min_pitch": 1e-6,  # 1um
        "max_pitch": 1e-3,  # 1mm
        "auto_relax_pitch": True,
        "pitch_step_factor": 1.5,
    },
    "growth": {
        "enabled": True,
        "backend": "space_colonization",
        "target_terminals": None,
        "terminal_tolerance": 0.1,
        "max_iterations": 500,
        "min_segment_length": 2e-4,  # 0.2mm
        "max_segment_length": 2e-3,  # 2mm
        "min_radius": 1e-4,  # 0.1mm
        "step_size": 3e-4,  # 0.3mm
    },
    "collision": {
        "enabled": True,
        "check_collisions": True,
        "collision_clearance": 2e-4,  # 0.2mm
    },
    "unified_collision": {
        "enabled": True,
        "min_clearance": 2e-4,  # 0.2mm
        "min_radius": 1e-4,  # 0.1mm
        "strategy_order": ["reroute", "shrink", "terminate"],
    },
    "radius": {
        "mode": "murray",
        "murray_exponent": 3.0,
        "taper_factor": 0.8,
        "min_radius": 1e-4,  # 0.1mm
        "max_radius": 5e-3,  # 5mm
    },
    "ports": {
        "enabled": True,
        "face": "top",
        "pattern": "circle",
        "ridge_width": 1e-4,  # 0.1mm
        "ridge_clearance": 1e-4,  # 0.1mm
        "port_margin": 5e-4,  # 0.5mm
    },
    "channels": {
        "enabled": False,
        "profile": "cylinder",
        "hook_depth": 2e-3,  # 2mm
    },
    "network_cleanup": {
        "enable_snap": True,
        "snap_tol": 1e-4,  # 0.1mm
        "enable_prune": True,
        "min_segment_length": 1e-4,  # 0.1mm
        "enable_merge": True,
        "merge_tol": 1e-4,  # 0.1mm
    },
    "mesh_synthesis": {
        "add_node_spheres": True,
        "cap_ends": True,
        "voxel_repair_synthesis": False,
        "voxel_repair_pitch": 1e-4,  # 0.1mm
        "segments_per_circle": 16,
    },
    "mesh_merge": {
        "mode": "auto",
        "voxel_pitch": 5e-5,  # 50um
        "auto_adjust_pitch": True,
        "max_pitch_steps": 4,
        "pitch_step_factor": 1.5,
        "fallback_boolean": True,
        "keep_largest_component": True,
        "min_component_faces": 100,
        "min_component_volume": 1e-12,  # 1 cubic mm
        "fill_voxels": True,
        "max_voxels": 100_000_000,
    },
    "composition": {
        "repair_enabled": True,
        "union_before_embed": True,
        "repair_fill_holes": True,
        "repair_voxel_pitch": 5e-5,  # 50um
        "keep_largest_component": True,
        "min_component_volume": 1e-12,  # 1 cubic mm
    },
    "embedding": {
        "voxel_pitch": None,  # Use resolution policy
        "shell_thickness": 5e-4,  # 0.5mm
        "use_resolution_policy": True,
        "auto_adjust_pitch": True,
        "max_pitch_steps": 4,
        "pitch_step_factor": 1.5,
        "max_voxels": 3_000_000,
        "preserve_ports_enabled": True,
        "preserve_mode": "recarve",
        "carve_radius_factor": 1.2,
        "carve_depth": 2e-3,  # 2mm
    },
    "validity": {
        "check_watertight": True,
        "check_components": True,
        "check_min_diameter": True,
        "check_open_ports": True,
        "check_bounds": True,
        "min_diameter_threshold": 5e-4,  # 0.5mm
        "max_components": 1,
    },
    "open_port": {
        "enabled": True,
        "probe_radius_factor": 1.2,
        "probe_length": 2e-3,  # 2mm
        "min_connected_volume_voxels": 10,
        "mode": "voxel_connectivity",
        "local_region_size": 5e-3,  # 5mm
        "max_voxels_roi": 1_000_000,
        "auto_relax_pitch": True,
    },
    "repair": {
        "voxel_repair_enabled": True,
        "voxel_pitch": 1e-4,  # 0.1mm
        "auto_adjust_pitch": True,
        "max_pitch_steps": 4,
        "pitch_step_factor": 1.5,
        "fill_voxels": True,
        "remove_small_components_enabled": True,
        "min_component_faces": 500,
        "min_component_volume": 1e-12,  # 1 cubic mm
        "fill_holes_enabled": True,
    },
    "pathfinding": {
        "use_resolution_policy": True,
        "coarse_pitch_factor": 4.0,
        "corridor_radius_factor": 2.0,
        "pitch_coarse": 1e-4,  # 0.1mm
        "pitch_fine": 5e-6,  # 5um
        "clearance": 2e-4,  # 0.2mm
        "local_radius": 2e-4,  # 0.2mm
    },
    "domain_meshing": {
        "cache_meshes": True,
        "emit_warnings": True,
        "target_face_count": 50000,
    },
    "output": {
        "output_dir": "./out",
        "output_units": "mm",
        "naming_convention": "default",
        "save_intermediates": False,
        "save_reports": True,
    },
}

# Nested policy defaults (for policies that contain sub-policies)
NESTED_POLICY_DEFAULTS: Dict[str, Dict[str, str]] = {
    "composition": {
        "synthesis_policy": "mesh_synthesis",
        "merge_policy": "mesh_merge",
        "repair_policy": "repair",
    },
    "domain_meshing": {
        "mesh_policy": "mesh_synthesis",
        "implicit_policy": "mesh_merge",
    },
}

# Required policies for each pipeline stage
STAGE_REQUIRED_POLICIES: Dict[str, List[str]] = {
    "compile_policies": [],
    "compile_domains": [],
    "component_ports": ["ports"],
    "component_build": ["growth"],
    "component_mesh": ["mesh_synthesis"],
    "union_voids": ["composition"],
    "mesh_domain": ["domain_meshing"],
    "embed": ["embedding"],
    "validity": ["validity"],
    "export": ["output"],
}


def is_debug_mode() -> bool:
    """Check if debug mode is enabled via environment variable."""
    return os.environ.get(DEBUG_ENV_VAR, "").lower() in ("1", "true", "yes")


def fill_policy_defaults(
    policies: Dict[str, Any],
    fill_nested: bool = True,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Fill missing policies and missing keys with defaults.
    
    Parameters
    ----------
    policies : dict
        The policies dict from the spec (already normalized to meters)
    fill_nested : bool
        If True, also fill nested policy objects
        
    Returns
    -------
    tuple
        (filled_policies, list of policy names that were filled)
    """
    import copy
    filled = copy.deepcopy(policies)
    filled_names = []
    
    # Fill top-level policies
    for policy_name, default_policy in DEFAULT_POLICIES.items():
        if policy_name not in filled or filled[policy_name] is None:
            # Policy is completely missing or null - use full default
            filled[policy_name] = copy.deepcopy(default_policy)
            filled_names.append(f"{policy_name} (full)")
        elif isinstance(filled[policy_name], dict):
            # Policy exists but may be missing keys - fill missing keys
            policy_dict = filled[policy_name]
            keys_filled = []
            for key, default_value in default_policy.items():
                if key not in policy_dict or policy_dict[key] is None:
                    # Check if this is a pitch field with use_resolution_policy=true
                    if key in ("voxel_pitch", "validation_pitch") and policy_dict.get("use_resolution_policy", False):
                        # OK to leave pitch as None when use_resolution_policy is true
                        continue
                    policy_dict[key] = default_value
                    keys_filled.append(key)
            if keys_filled:
                filled_names.append(f"{policy_name}.{{{', '.join(keys_filled)}}}")
    
    # Fill nested policies
    if fill_nested:
        for parent_policy, nested_mappings in NESTED_POLICY_DEFAULTS.items():
            if parent_policy in filled and isinstance(filled[parent_policy], dict):
                parent_dict = filled[parent_policy]
                for nested_key, default_policy_name in nested_mappings.items():
                    if nested_key not in parent_dict or parent_dict[nested_key] is None:
                        # Nested policy is missing - use full default
                        if default_policy_name in DEFAULT_POLICIES:
                            parent_dict[nested_key] = copy.deepcopy(DEFAULT_POLICIES[default_policy_name])
                            filled_names.append(f"{parent_policy}.{nested_key} (full)")
                    elif isinstance(parent_dict[nested_key], dict):
                        # Nested policy exists but may be missing keys
                        nested_dict = parent_dict[nested_key]
                        if default_policy_name in DEFAULT_POLICIES:
                            default_nested = DEFAULT_POLICIES[default_policy_name]
                            keys_filled = []
                            for key, default_value in default_nested.items():
                                if key not in nested_dict or nested_dict[key] is None:
                                    # Check if this is a pitch field with use_resolution_policy=true
                                    if key in ("voxel_pitch", "validation_pitch") and nested_dict.get("use_resolution_policy", False):
                                        continue
                                    nested_dict[key] = default_value
                                    keys_filled.append(key)
                            if keys_filled:
                                filled_names.append(f"{parent_policy}.{nested_key}.{{{', '.join(keys_filled)}}}")
    
    return filled, filled_names


def _get_domain_scale(spec: Dict[str, Any]) -> float:
    """
    Get the approximate scale of the domain in meters.
    
    Used for sanity checks on voxel pitches.
    """
    domains = spec.get("domains", {})
    if not domains:
        return 0.01  # Default 10mm
    
    # Get the first domain
    domain = next(iter(domains.values()))
    if not isinstance(domain, dict):
        return 0.01
    
    domain_type = domain.get("type", "").lower()
    
    if domain_type == "cylinder":
        radius = domain.get("radius", 0.01)
        height = domain.get("height", 0.01)
        return max(radius * 2, height)
    elif domain_type == "box":
        x_size = abs(domain.get("x_max", 0.01) - domain.get("x_min", -0.01))
        y_size = abs(domain.get("y_max", 0.01) - domain.get("y_min", -0.01))
        z_size = abs(domain.get("z_max", 0.01) - domain.get("z_min", -0.01))
        return max(x_size, y_size, z_size)
    elif domain_type == "ellipsoid":
        radii = domain.get("radii", [0.01, 0.01, 0.01])
        if isinstance(radii, list) and len(radii) >= 3:
            return max(radii) * 2
        return 0.01
    
    return 0.01


def run_preflight_checks(
    spec: Dict[str, Any],
    stages_to_run: Optional[List[str]] = None,
) -> PreflightResult:
    """
    Run preflight validation checks on a normalized spec.
    
    Parameters
    ----------
    spec : dict
        The normalized spec dict (all values in meters)
    stages_to_run : list of str, optional
        List of stages that will be run. If None, checks all stages.
        
    Returns
    -------
    PreflightResult
        Result containing errors and warnings
    """
    errors: List[PreflightError] = []
    warnings: List[PreflightError] = []
    
    policies = spec.get("policies", {})
    domains = spec.get("domains", {})
    components = spec.get("components", [])
    
    # Get domain scale for sanity checks
    domain_scale = _get_domain_scale(spec)
    
    # Check 1: Required policies for stages
    if stages_to_run is None:
        stages_to_run = list(STAGE_REQUIRED_POLICIES.keys())
    
    for stage in stages_to_run:
        required = STAGE_REQUIRED_POLICIES.get(stage, [])
        for policy_name in required:
            if policy_name not in policies or policies[policy_name] is None:
                errors.append(PreflightError(
                    code="MISSING_REQUIRED_POLICY",
                    message=f"Policy '{policy_name}' is null but required for stage '{stage}'",
                    path=f"policies.{policy_name}",
                    suggestion=f"Add a '{policy_name}' policy or use default policy filling",
                ))
    
    # Check 2: Voxel pitch sanity checks
    pitch_fields = [
        ("policies.embedding.voxel_pitch", policies.get("embedding", {}).get("voxel_pitch")),
        ("policies.mesh_merge.voxel_pitch", policies.get("mesh_merge", {}).get("voxel_pitch")),
        ("policies.composition.merge_policy.voxel_pitch", 
         policies.get("composition", {}).get("merge_policy", {}).get("voxel_pitch") if isinstance(policies.get("composition", {}).get("merge_policy"), dict) else None),
        ("policies.composition.repair_policy.voxel_pitch",
         policies.get("composition", {}).get("repair_policy", {}).get("voxel_pitch") if isinstance(policies.get("composition", {}).get("repair_policy"), dict) else None),
        ("policies.repair.voxel_pitch", policies.get("repair", {}).get("voxel_pitch")),
        ("policies.domain_meshing.voxel_pitch", policies.get("domain_meshing", {}).get("voxel_pitch")),
        ("policies.domain_meshing.mesh_policy.repair_voxel_pitch",
         policies.get("domain_meshing", {}).get("mesh_policy", {}).get("repair_voxel_pitch") if isinstance(policies.get("domain_meshing", {}).get("mesh_policy"), dict) else None),
        ("policies.domain_meshing.implicit_policy.voxel_pitch",
         policies.get("domain_meshing", {}).get("implicit_policy", {}).get("voxel_pitch") if isinstance(policies.get("domain_meshing", {}).get("implicit_policy"), dict) else None),
    ]
    
    for path, pitch in pitch_fields:
        if pitch is None:
            # Check if use_resolution_policy is true
            policy_name = path.split(".")[1]
            policy = policies.get(policy_name, {})
            if isinstance(policy, dict) and policy.get("use_resolution_policy", False):
                continue  # OK to have null pitch with use_resolution_policy
            # Otherwise, it's a warning (will use default)
            continue
        
        if not isinstance(pitch, (int, float)):
            continue
            
        # Check if pitch seems too large (likely not normalized)
        if pitch > 0.01:  # > 10mm
            errors.append(PreflightError(
                code="PITCH_NOT_NORMALIZED",
                message=f"{path} appears to be in mm but was not normalized (value={pitch}m is too large)",
                path=path,
                suggestion="Ensure meta.input_units is set correctly and normalization is applied",
            ))
        
        # Check if pitch is too large relative to domain
        if pitch > domain_scale / 10:
            warnings.append(PreflightError(
                code="PITCH_TOO_LARGE",
                message=f"domain scale is ~{domain_scale:.4f}m but {path} is {pitch:.6f}m (ratio too large); union likely to degenerate",
                path=path,
                severity="warning",
                suggestion=f"Consider reducing voxel_pitch to at most {domain_scale / 20:.6f}m",
            ))
    
    # Check 3: Domain validation
    if not domains:
        errors.append(PreflightError(
            code="NO_DOMAINS",
            message="No domains defined in spec",
            path="domains",
            suggestion="Add at least one domain definition",
        ))
    
    for domain_name, domain in domains.items():
        if not isinstance(domain, dict):
            continue
        
        domain_type = domain.get("type", "").lower()
        
        if domain_type == "cylinder":
            radius = domain.get("radius")
            height = domain.get("height")
            if radius is not None and radius > 1.0:  # > 1m
                warnings.append(PreflightError(
                    code="DOMAIN_NOT_NORMALIZED",
                    message=f"Domain '{domain_name}' radius={radius}m seems too large (likely not normalized from mm)",
                    path=f"domains.{domain_name}.radius",
                    severity="warning",
                ))
            if height is not None and height > 1.0:  # > 1m
                warnings.append(PreflightError(
                    code="DOMAIN_NOT_NORMALIZED",
                    message=f"Domain '{domain_name}' height={height}m seems too large (likely not normalized from mm)",
                    path=f"domains.{domain_name}.height",
                    severity="warning",
                ))
    
    # Check 4: Component validation
    for i, component in enumerate(components):
        if not isinstance(component, dict):
            continue
        
        comp_id = component.get("id", f"component_{i}")
        domain_ref = component.get("domain_ref")
        
        if domain_ref and domain_ref not in domains:
            errors.append(PreflightError(
                code="INVALID_DOMAIN_REF",
                message=f"Component '{comp_id}' references domain '{domain_ref}' which does not exist",
                path=f"components[{i}].domain_ref",
                suggestion=f"Available domains: {list(domains.keys())}",
            ))
        
        # Check port radii
        ports = component.get("ports", {})
        for port_type in ["inlets", "outlets"]:
            port_list = ports.get(port_type, [])
            for j, port in enumerate(port_list):
                if not isinstance(port, dict):
                    continue
                radius = port.get("radius")
                if radius is not None and radius > 0.1:  # > 100mm
                    warnings.append(PreflightError(
                        code="PORT_RADIUS_NOT_NORMALIZED",
                        message=f"Port radius={radius}m in component '{comp_id}' seems too large",
                        path=f"components[{i}].ports.{port_type}[{j}].radius",
                        severity="warning",
                    ))
    
    # Check 5: Volume field sanity
    volume_fields = [
        ("policies.repair.min_component_volume", policies.get("repair", {}).get("min_component_volume")),
        ("policies.composition.min_component_volume", policies.get("composition", {}).get("min_component_volume")),
        ("policies.mesh_merge.min_component_volume", policies.get("mesh_merge", {}).get("min_component_volume")),
        ("policies.composition.merge_policy.min_component_volume",
         policies.get("composition", {}).get("merge_policy", {}).get("min_component_volume") if isinstance(policies.get("composition", {}).get("merge_policy"), dict) else None),
        ("policies.composition.repair_policy.min_component_volume",
         policies.get("composition", {}).get("repair_policy", {}).get("min_component_volume") if isinstance(policies.get("composition", {}).get("repair_policy"), dict) else None),
    ]
    
    for path, volume in volume_fields:
        if volume is None:
            continue
        if not isinstance(volume, (int, float)):
            continue
        
        # Volume should be in cubic meters - 1mm³ = 1e-9 m³
        # If volume > 1e-3 (1000 mm³ in m³ terms), it's likely not normalized
        if volume > 1e-3:
            warnings.append(PreflightError(
                code="VOLUME_NOT_NORMALIZED",
                message=f"{path}={volume}m³ seems too large (likely not normalized from mm³)",
                path=path,
                severity="warning",
                suggestion="Volume fields should scale by scale³ (e.g., 1mm³ = 1e-9m³)",
            ))
    
    success = len(errors) == 0
    
    return PreflightResult(
        success=success,
        errors=errors,
        warnings=warnings,
    )


def create_unit_audit_report(
    original_spec: Dict[str, Any],
    normalized_spec: Dict[str, Any],
    input_units: str,
    scale: float,
) -> UnitAuditReport:
    """
    Create an audit report comparing original and normalized values.
    
    Parameters
    ----------
    original_spec : dict
        The original spec before normalization
    normalized_spec : dict
        The spec after normalization
    input_units : str
        The input units (e.g., "mm")
    scale : float
        The scale factor used for conversion
        
    Returns
    -------
    UnitAuditReport
        Report containing all conversion entries
    """
    report = UnitAuditReport(
        input_units=input_units,
        scale_factor=scale,
    )
    
    if scale == 1.0:
        return report  # No conversions needed
    
    def compare_dicts(orig: Dict, norm: Dict, path: str = "") -> None:
        """Recursively compare dicts and record differences."""
        for key in orig:
            if key not in norm:
                continue
            
            orig_val = orig[key]
            norm_val = norm[key]
            current_path = f"{path}.{key}" if path else key
            
            if isinstance(orig_val, dict) and isinstance(norm_val, dict):
                compare_dicts(orig_val, norm_val, current_path)
            elif isinstance(orig_val, list) and isinstance(norm_val, list):
                for i, (o, n) in enumerate(zip(orig_val, norm_val)):
                    if isinstance(o, dict) and isinstance(n, dict):
                        compare_dicts(o, n, f"{current_path}[{i}]")
                    elif isinstance(o, (int, float)) and isinstance(n, (int, float)):
                        if abs(o - n) > 1e-15:
                            # Determine scale type
                            ratio = n / o if o != 0 else 0
                            if abs(ratio - scale) < 1e-10:
                                scale_type = "length"
                            elif abs(ratio - scale**2) < 1e-10:
                                scale_type = "area"
                            elif abs(ratio - scale**3) < 1e-10:
                                scale_type = "volume"
                            else:
                                scale_type = "unknown"
                            
                            report.entries.append(UnitAuditEntry(
                                path=f"{current_path}[{i}]",
                                field_name=key,
                                original_value=o,
                                normalized_value=n,
                                scale_type=scale_type,
                                scale_factor=ratio,
                            ))
            elif isinstance(orig_val, (int, float)) and isinstance(norm_val, (int, float)):
                if abs(orig_val - norm_val) > 1e-15:
                    # Determine scale type
                    ratio = norm_val / orig_val if orig_val != 0 else 0
                    if abs(ratio - scale) < 1e-10:
                        scale_type = "length"
                    elif abs(ratio - scale**2) < 1e-10:
                        scale_type = "area"
                    elif abs(ratio - scale**3) < 1e-10:
                        scale_type = "volume"
                    else:
                        scale_type = "unknown"
                    
                    report.entries.append(UnitAuditEntry(
                        path=current_path,
                        field_name=key,
                        original_value=orig_val,
                        normalized_value=norm_val,
                        scale_type=scale_type,
                        scale_factor=ratio,
                    ))
    
    compare_dicts(original_spec, normalized_spec)
    
    return report


class PreflightValidationError(Exception):
    """Raised when preflight validation fails."""
    
    def __init__(self, result: PreflightResult):
        self.result = result
        error_messages = [f"  - {e.message}" for e in result.errors]
        message = f"Preflight validation failed with {len(result.errors)} error(s):\n" + "\n".join(error_messages)
        super().__init__(message)


__all__ = [
    "PreflightError",
    "PreflightResult",
    "PreflightValidationError",
    "UnitAuditEntry",
    "UnitAuditReport",
    "DEFAULT_POLICIES",
    "NESTED_POLICY_DEFAULTS",
    "STAGE_REQUIRED_POLICIES",
    "DEBUG_ENV_VAR",
    "is_debug_mode",
    "fill_policy_defaults",
    "run_preflight_checks",
    "create_unit_audit_report",
]
