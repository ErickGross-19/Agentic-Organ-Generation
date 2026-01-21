"""
DesignSpec loader, validator, and normalizer.

This module provides the DesignSpec class that:
- Loads from dict or JSON file
- Validates top-level structure
- Applies backward-compatible aliases
- Normalizes all length values to meters
- Computes stable content hashes for reproducibility

UNIT CONVENTIONS
----------------
All geometric values are normalized to METERS internally.
Input units are specified via meta.input_units and converted on load.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union, Set, Tuple
from pathlib import Path
import json
import hashlib
import logging
import copy
import os

from .schema import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    validate_schema_block,
    SchemaInfo,
    SchemaValidationError,
)
from .compat import apply_all_aliases
from .preflight import (
    fill_policy_defaults,
    run_preflight_checks,
    create_unit_audit_report,
    is_debug_mode,
    PreflightValidationError,
    UnitAuditReport,
    PreflightResult,
    DEBUG_ENV_VAR,
)

logger = logging.getLogger(__name__)


class DesignSpecError(Exception):
    """Base exception for DesignSpec errors."""
    pass


class DesignSpecValidationError(DesignSpecError):
    """Raised when spec validation fails."""
    pass


REQUIRED_TOP_LEVEL_KEYS = {"schema", "meta", "policies", "domains", "components"}

POLICY_LENGTH_FIELDS: Dict[str, Set[str]] = {
    "resolution": {
        "min_channel_diameter",
        "min_pitch",
        "max_pitch",
    },
    "pathfinding": {
        "clearance",
        "local_radius",
        "pitch_coarse",
        "pitch_fine",
        "corridor_radius_buffer",
    },
    "ports": {
        "ridge_width",
        "ridge_clearance",
        "port_margin",
    },
    "channels": {
        "length",
        "start_offset",
        "stop_before_boundary",
        "radius_end",
        "hook_depth",
        "ring_sigma",
    },
    "embedding": {
        "voxel_pitch",
        "shell_thickness",
        "carve_depth",
    },
    "validity": {
        "min_diameter_threshold",
        "surface_opening_tolerance",
    },
    "open_port": {
        "probe_length",
        "local_region_size",
        "validation_pitch",
    },
    "repair": {
        "voxel_pitch",
    },
    "growth": {
        "min_segment_length",
        "max_segment_length",
        "step_size",
        "min_radius",
    },
    "collision": {
        "collision_clearance",
    },
    "unified_collision": {
        "min_clearance",
        "min_radius",
    },
    "composition": {
        "repair_voxel_pitch",
    },
    "mesh_synthesis": {
        "voxel_repair_pitch",
        "radius_clamp_min",
        "radius_clamp_max",
    },
    "mesh_merge": {
        "voxel_pitch",
        "min_channel_diameter",
    },
    "network_cleanup": {
        "snap_tol",
        "min_segment_length",
        "merge_tol",
    },
    "radius": {
        "min_radius",
        "max_radius",
    },
    "domain_meshing": {
    },
    "output": {
    },
    "ridge": {
        "height",
        "thickness",
        "inset",
        "overlap",
    },
}

# Volume fields need scale³ conversion (cubic units)
POLICY_VOLUME_FIELDS: Dict[str, Set[str]] = {
    "repair": {
        "min_component_volume",
    },
    "composition": {
        "min_component_volume",
    },
    "mesh_merge": {
        "min_component_volume",
    },
}

# Nested policy mappings for recursive normalization
NESTED_POLICY_FIELDS: Dict[str, Dict[str, str]] = {
    "composition": {
        "synthesis_policy": "mesh_synthesis",
        "merge_policy": "mesh_merge",
        "repair_policy": "repair",
    },
}

DOMAIN_LENGTH_FIELDS = {
    "center", "radii", "radius", "height", "min_corner", "max_corner",
    "size", "start", "end", "top_radius", "bottom_radius", "translation",
    "x_min", "x_max", "y_min", "y_max", "z_min", "z_max",
    "length", "radius_top", "radius_bottom",
    "semi_axis_a", "semi_axis_b", "semi_axis_c",
}

PORT_LENGTH_FIELDS = {"position", "radius"}

COMPONENT_LENGTH_FIELDS = {"position", "radius", "direction"}


def _get_unit_scale(input_units: str) -> float:
    """
    Get scale factor to convert from input_units to meters.
    
    Parameters
    ----------
    input_units : str
        Unit string: "m", "mm", "cm", or "um"
        
    Returns
    -------
    float
        Scale factor to multiply values by to convert to meters
        
    Raises
    ------
    ValueError
        If input_units is not a recognized unit
    """
    if input_units == "m":
        return 1.0
    elif input_units == "mm":
        return 1e-3
    elif input_units == "cm":
        return 1e-2
    elif input_units == "um":
        return 1e-6
    else:
        raise ValueError(f"Unknown unit: '{input_units}'. Supported units: m, mm, cm, um")


def _convert_value(value: Any, scale: float) -> Any:
    """Convert a value by scale factor, handling nested structures."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value * scale
    elif isinstance(value, list):
        return [_convert_value(v, scale) for v in value]
    elif isinstance(value, tuple):
        return tuple(_convert_value(v, scale) for v in value)
    elif isinstance(value, dict):
        if "x" in value and "y" in value and "z" in value:
            return {
                k: v * scale if k in ("x", "y", "z") and isinstance(v, (int, float)) else v
                for k, v in value.items()
            }
        return value
    return value


def _normalize_tissue_sampling(
    tissue_sampling: Dict[str, Any],
    scale: float,
) -> Dict[str, Any]:
    """
    Normalize length fields in tissue_sampling dict.
    
    Length fields:
    - depth_min, depth_max
    - min_distance_to_ports
    - r_min, r_max
    - ring_r0, ring_sigma
    - shell_thickness
    - gaussian_mean, gaussian_sigma (lists of lengths)
    - mixture_components (recursive)
    """
    result = dict(tissue_sampling)
    
    length_fields = {
        "depth_min", "depth_max",
        "min_distance_to_ports",
        "r_min", "r_max",
        "ring_r0", "ring_sigma",
        "shell_thickness",
    }
    
    for field_name in length_fields:
        if field_name in result and result[field_name] is not None:
            result[field_name] = _convert_value(result[field_name], scale)
    
    # gaussian_mean and gaussian_sigma are lists of lengths
    if "gaussian_mean" in result and result["gaussian_mean"] is not None:
        result["gaussian_mean"] = _convert_value(result["gaussian_mean"], scale)
    if "gaussian_sigma" in result and result["gaussian_sigma"] is not None:
        result["gaussian_sigma"] = _convert_value(result["gaussian_sigma"], scale)
    
    # mixture_components may contain nested policy objects with length fields
    if "mixture_components" in result and isinstance(result["mixture_components"], list):
        normalized_components = []
        for comp in result["mixture_components"]:
            if isinstance(comp, dict):
                normalized_comp = dict(comp)
                # Each component may have its own policy subobject
                if "policy" in normalized_comp and isinstance(normalized_comp["policy"], dict):
                    normalized_comp["policy"] = _normalize_tissue_sampling(
                        normalized_comp["policy"], scale
                    )
                normalized_components.append(normalized_comp)
            else:
                normalized_components.append(comp)
        result["mixture_components"] = normalized_components
    
    return result


def _normalize_space_colonization_policy(
    sc_policy: Dict[str, Any],
    scale: float,
) -> Dict[str, Any]:
    """
    Normalize length fields in space_colonization_policy dict.
    
    Length fields:
    - branch_enable_after_distance
    - min_branch_segment_length
    """
    result = dict(sc_policy)
    
    length_fields = {
        "branch_enable_after_distance",
        "min_branch_segment_length",
    }
    
    for field_name in length_fields:
        if field_name in result and result[field_name] is not None:
            result[field_name] = _convert_value(result[field_name], scale)
    
    return result


def _normalize_backend_params(
    backend_params: Dict[str, Any],
    scale: float,
) -> Dict[str, Any]:
    """
    Normalize length fields in backend_params dict.
    
    Handles:
    - Top-level length fields (influence_radius, kill_radius, etc.)
    - collision_merge_distance, multi_inlet_blend_sigma
    - Nested tissue_sampling dict
    - Nested space_colonization_policy dict
    """
    result = dict(backend_params)
    
    # Known length fields in backend_params
    backend_length_fields = {
        "step_size", "min_segment_length", "max_segment_length",
        "influence_radius", "kill_radius", "perception_radius",
        "clearance", "min_radius", "max_radius",
        "wall_margin_m", "terminal_radius",  # K-ary tree specific
        "collision_clearance", "min_terminal_separation",  # CCO specific
        "collision_merge_distance",  # Multi-inlet merge distance
        "multi_inlet_blend_sigma",  # Multi-inlet blend sigma
    }
    
    for field_name in backend_length_fields:
        if field_name in result and result[field_name] is not None:
            result[field_name] = _convert_value(result[field_name], scale)
    
    # Normalize nested tissue_sampling dict
    if "tissue_sampling" in result and isinstance(result["tissue_sampling"], dict):
        result["tissue_sampling"] = _normalize_tissue_sampling(result["tissue_sampling"], scale)
    
    # Normalize nested space_colonization_policy dict
    if "space_colonization_policy" in result and isinstance(result["space_colonization_policy"], dict):
        result["space_colonization_policy"] = _normalize_space_colonization_policy(
            result["space_colonization_policy"], scale
        )
    
    return result


def _normalize_policy_to_meters(
    policy_name: str,
    policy_dict: Dict[str, Any],
    scale: float,
) -> Dict[str, Any]:
    """
    Normalize length and volume fields in a policy dict to meters.
    
    Handles:
    - Length fields (multiply by scale)
    - Volume fields (multiply by scale³)
    - Nested policy dicts (recursive normalization)
    - Domain meshing sub-policies
    - Backend params in growth policy
    """
    result = dict(policy_dict)
    length_fields = POLICY_LENGTH_FIELDS.get(policy_name, set())
    volume_fields = POLICY_VOLUME_FIELDS.get(policy_name, set())
    nested_policies = NESTED_POLICY_FIELDS.get(policy_name, {})
    
    # Normalize length fields (scale)
    for field_name in length_fields:
        if field_name in result and result[field_name] is not None:
            result[field_name] = _convert_value(result[field_name], scale)
    
    # Normalize volume fields (scale³)
    volume_scale = scale ** 3
    for field_name in volume_fields:
        if field_name in result and result[field_name] is not None:
            result[field_name] = _convert_value(result[field_name], volume_scale)
    
    # Recursively normalize nested policy dicts
    for nested_field, nested_policy_name in nested_policies.items():
        if nested_field in result and isinstance(result[nested_field], dict):
            result[nested_field] = _normalize_policy_to_meters(
                nested_policy_name, result[nested_field], scale
            )
    
    if policy_name == "resolution" and "input_units" in result:
        result["input_units"] = "m"
    
    # Handle backend_params in growth policy
    if policy_name == "growth" and "backend_params" in result:
        backend_params = result["backend_params"]
        if isinstance(backend_params, dict):
            result["backend_params"] = _normalize_backend_params(backend_params, scale)
    
    # Handle domain_meshing sub-policies
    if policy_name == "domain_meshing":
        # mesh_policy has repair_voxel_pitch
        if "mesh_policy" in result and isinstance(result["mesh_policy"], dict):
            mesh_policy = result["mesh_policy"]
            if "repair_voxel_pitch" in mesh_policy and mesh_policy["repair_voxel_pitch"] is not None:
                mesh_policy["repair_voxel_pitch"] = _convert_value(mesh_policy["repair_voxel_pitch"], scale)
            result["mesh_policy"] = mesh_policy
        
        # implicit_policy has voxel_pitch
        if "implicit_policy" in result and isinstance(result["implicit_policy"], dict):
            implicit_policy = result["implicit_policy"]
            if "voxel_pitch" in implicit_policy and implicit_policy["voxel_pitch"] is not None:
                implicit_policy["voxel_pitch"] = _convert_value(implicit_policy["voxel_pitch"], scale)
            result["implicit_policy"] = implicit_policy
        
        # top-level voxel_pitch in domain_meshing
        if "voxel_pitch" in result and result["voxel_pitch"] is not None:
            result["voxel_pitch"] = _convert_value(result["voxel_pitch"], scale)
    
    return result


def _normalize_domain_to_meters(
    domain_dict: Dict[str, Any],
    scale: float,
) -> Dict[str, Any]:
    """Normalize length fields in a domain dict to meters."""
    result = dict(domain_dict)
    
    for field_name in DOMAIN_LENGTH_FIELDS:
        if field_name in result and result[field_name] is not None:
            result[field_name] = _convert_value(result[field_name], scale)
    
    if "base_domain" in result:
        result["base_domain"] = _normalize_domain_to_meters(result["base_domain"], scale)
    
    if "children" in result and isinstance(result["children"], list):
        result["children"] = [
            _normalize_domain_to_meters(child, scale)
            for child in result["children"]
        ]
    
    return result


def _normalize_port_to_meters(
    port_dict: Dict[str, Any],
    scale: float,
) -> Dict[str, Any]:
    """Normalize length fields in a port dict to meters."""
    result = dict(port_dict)
    
    for field_name in PORT_LENGTH_FIELDS:
        if field_name in result and result[field_name] is not None:
            result[field_name] = _convert_value(result[field_name], scale)
    
    return result


def _normalize_component_to_meters(
    component_dict: Dict[str, Any],
    scale: float,
) -> Dict[str, Any]:
    """Normalize length fields in a component dict to meters."""
    result = copy.deepcopy(component_dict)
    
    if "ports" in result and isinstance(result["ports"], dict):
        for port_type in ["inlets", "outlets"]:
            if port_type in result["ports"] and isinstance(result["ports"][port_type], list):
                result["ports"][port_type] = [
                    _normalize_port_to_meters(port, scale)
                    for port in result["ports"][port_type]
                ]
    
    # Normalize policy_overrides if present
    if "policy_overrides" in result and isinstance(result["policy_overrides"], dict):
        for policy_name, policy_dict in result["policy_overrides"].items():
            if isinstance(policy_dict, dict):
                result["policy_overrides"][policy_name] = _normalize_policy_to_meters(
                    policy_name, policy_dict, scale
                )
    
    # Normalize known length fields in build.backend_params
    if "build" in result and isinstance(result["build"], dict):
        backend_params = result["build"].get("backend_params", {})
        if isinstance(backend_params, dict):
            result["build"]["backend_params"] = _normalize_backend_params(backend_params, scale)
    
    return result


def _normalize_spec_to_meters(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize all length values in a spec from input_units to meters.
    
    Parameters
    ----------
    spec : dict
        The spec dict with input_units specified in meta
        
    Returns
    -------
    dict
        Spec with all length values converted to meters
    """
    result = copy.deepcopy(spec)
    
    input_units = result.get("meta", {}).get("input_units", "m")
    scale = _get_unit_scale(input_units)
    
    if scale == 1.0:
        return result
    
    if "policies" in result and isinstance(result["policies"], dict):
        for policy_name, policy_dict in result["policies"].items():
            if isinstance(policy_dict, dict):
                result["policies"][policy_name] = _normalize_policy_to_meters(
                    policy_name, policy_dict, scale
                )
    
    if "domains" in result and isinstance(result["domains"], dict):
        for domain_name, domain_dict in result["domains"].items():
            if isinstance(domain_dict, dict):
                result["domains"][domain_name] = _normalize_domain_to_meters(
                    domain_dict, scale
                )
    
    if "components" in result and isinstance(result["components"], list):
        result["components"] = [
            _normalize_component_to_meters(comp, scale)
            for comp in result["components"]
        ]
    
    result["meta"]["input_units"] = "m"
    result["meta"]["original_input_units"] = input_units
    
    return result


def _compute_spec_hash(spec: Dict[str, Any]) -> str:
    """
    Compute a stable hash of the spec for reproducibility.
    
    Uses canonical JSON serialization (sorted keys) to ensure
    the same spec always produces the same hash regardless of
    key ordering.
    
    Parameters
    ----------
    spec : dict
        The normalized spec dict
        
    Returns
    -------
    str
        Hex digest of the spec hash (first 16 characters)
    """
    canonical_json = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode()).hexdigest()[:16]


@dataclass
class DesignSpec:
    """
    Loaded and validated design specification.
    
    This class represents a fully loaded, validated, and normalized
    design specification ready for execution by DesignSpecRunner.
    
    Attributes
    ----------
    raw : dict
        The original spec dict as loaded (before normalization)
    normalized : dict
        The spec dict with all values normalized to meters
    schema_info : SchemaInfo
        Parsed schema information
    spec_hash : str
        Stable content hash for reproducibility
    warnings : list of str
        Warnings generated during loading (aliases applied, etc.)
    preflight_result : PreflightResult, optional
        Result of preflight validation checks
    unit_audit_report : UnitAuditReport, optional
        Report of unit conversions (only in debug mode)
    policies_filled : list of str
        List of policies that were filled with defaults
    """
    raw: Dict[str, Any]
    normalized: Dict[str, Any]
    schema_info: SchemaInfo
    spec_hash: str
    warnings: List[str] = field(default_factory=list)
    preflight_result: Optional[PreflightResult] = None
    unit_audit_report: Optional[UnitAuditReport] = None
    policies_filled: List[str] = field(default_factory=list)
    
    @property
    def meta(self) -> Dict[str, Any]:
        """Get the meta section."""
        return self.normalized.get("meta", {})
    
    @property
    def seed(self) -> Optional[int]:
        """Get the seed from meta."""
        return self.meta.get("seed")
    
    @property
    def input_units(self) -> str:
        """Get the original input units."""
        return self.meta.get("original_input_units", "m")
    
    @property
    def policies(self) -> Dict[str, Any]:
        """Get the policies section (normalized to meters)."""
        return self.normalized.get("policies", {})
    
    @property
    def domains(self) -> Dict[str, Any]:
        """Get the domains section (normalized to meters)."""
        return self.normalized.get("domains", {})
    
    @property
    def components(self) -> List[Dict[str, Any]]:
        """Get the components list (normalized to meters)."""
        return self.normalized.get("components", [])
    
    @property
    def composition(self) -> Dict[str, Any]:
        """Get the composition section."""
        return self.normalized.get("composition", {})
    
    @property
    def embedding(self) -> Dict[str, Any]:
        """Get the embedding section."""
        return self.normalized.get("embedding", {})
    
    @property
    def validity(self) -> Dict[str, Any]:
        """Get the validity section."""
        return self.normalized.get("validity", {})
    
    @property
    def outputs(self) -> Dict[str, Any]:
        """Get the outputs section."""
        return self.normalized.get("outputs", {})
    
    def get_component(self, component_id: str) -> Optional[Dict[str, Any]]:
        """Get a component by ID."""
        for comp in self.components:
            if comp.get("id") == component_id:
                return comp
        return None
    
    def get_domain(self, domain_ref: str) -> Optional[Dict[str, Any]]:
        """Get a domain by reference name."""
        return self.domains.get(domain_ref)
    
    def to_dict(self) -> Dict[str, Any]:
        """Return the normalized spec as a dict."""
        return self.normalized
    
    def to_json(self, indent: int = 2) -> str:
        """Return the normalized spec as JSON string."""
        return json.dumps(self.normalized, indent=indent)
    
    @classmethod
    def from_dict(
        cls,
        spec_dict: Dict[str, Any],
        strict_schema: bool = False,
        allow_unknown_fields: bool = True,
        fill_defaults: bool = True,
        run_preflight: bool = True,
        preflight_strict: bool = False,
    ) -> "DesignSpec":
        """
        Load and validate a spec from a dictionary.
        
        Parameters
        ----------
        spec_dict : dict
            The spec dictionary to load
        strict_schema : bool
            If True, unknown fields raise errors. Default False.
        allow_unknown_fields : bool
            If True (default), unknown fields generate warnings.
            If False and strict_schema=True, unknown fields raise errors.
        fill_defaults : bool
            If True (default), fill missing policies with defaults.
        run_preflight : bool
            If True (default), run preflight validation checks.
        preflight_strict : bool
            If True, preflight errors raise exceptions. Default False (warnings only).
            
        Returns
        -------
        DesignSpec
            Loaded and validated spec
            
        Raises
        ------
        DesignSpecValidationError
            If validation fails
        PreflightValidationError
            If preflight_strict=True and preflight validation fails
        """
        warnings = []
        policies_filled: List[str] = []
        preflight_result: Optional[PreflightResult] = None
        unit_audit_report: Optional[UnitAuditReport] = None
        
        spec, alias_warnings = apply_all_aliases(spec_dict)
        warnings.extend(alias_warnings)
        
        schema_errors = validate_schema_block(spec.get("schema", {}))
        if schema_errors:
            raise DesignSpecValidationError(
                f"Schema validation failed: {'; '.join(schema_errors)}"
            )
        
        missing_keys = REQUIRED_TOP_LEVEL_KEYS - set(spec.keys())
        if missing_keys:
            raise DesignSpecValidationError(
                f"Missing required top-level keys: {missing_keys}"
            )
        
        meta = spec.get("meta", {})
        if "seed" not in meta:
            warnings.append("meta.seed not specified, results may not be reproducible")
        if "input_units" not in meta:
            warnings.append("meta.input_units not specified, assuming meters")
            spec["meta"]["input_units"] = "m"
        
        if strict_schema and not allow_unknown_fields:
            known_keys = REQUIRED_TOP_LEVEL_KEYS | {
                "composition", "embedding", "validity", "outputs"
            }
            unknown_keys = set(spec.keys()) - known_keys
            if unknown_keys:
                raise DesignSpecValidationError(
                    f"Unknown top-level keys (strict_schema=True): {unknown_keys}"
                )
        
        original_spec = copy.deepcopy(spec) if is_debug_mode() else None
        
        normalized = _normalize_spec_to_meters(spec)
        
        if is_debug_mode() and original_spec is not None:
            input_units = spec_dict.get("meta", {}).get("input_units", "m")
            scale = _get_unit_scale(input_units)
            unit_audit_report = create_unit_audit_report(
                original_spec, normalized, input_units, scale
            )
            logger.info(
                f"Unit audit: {len(unit_audit_report.entries)} fields converted "
                f"from {input_units} to meters (scale={scale})"
            )
        
        if fill_defaults:
            filled_policies, filled_names = fill_policy_defaults(
                normalized.get("policies", {}),
                fill_nested=True,
            )
            normalized["policies"] = filled_policies
            policies_filled = filled_names
            if filled_names:
                logger.info(f"Filled default policies: {', '.join(filled_names)}")
        
        if run_preflight:
            preflight_result = run_preflight_checks(normalized)
            
            for error in preflight_result.errors:
                logger.error(f"Preflight error [{error.code}]: {error.message}")
                if error.suggestion:
                    logger.error(f"  Suggestion: {error.suggestion}")
            
            for warning in preflight_result.warnings:
                logger.warning(f"Preflight warning [{warning.code}]: {warning.message}")
                warnings.append(f"[{warning.code}] {warning.message}")
            
            if preflight_strict and not preflight_result.success:
                raise PreflightValidationError(preflight_result)
        
        schema_info = SchemaInfo.from_dict(spec.get("schema", {}))
        
        spec_hash = _compute_spec_hash(normalized)
        
        return cls(
            raw=spec_dict,
            normalized=normalized,
            schema_info=schema_info,
            spec_hash=spec_hash,
            warnings=warnings,
            preflight_result=preflight_result,
            unit_audit_report=unit_audit_report,
            policies_filled=policies_filled,
        )
    
    @classmethod
    def from_json(
        cls,
        path: Union[str, Path],
        strict_schema: bool = False,
        allow_unknown_fields: bool = True,
        fill_defaults: bool = True,
        run_preflight: bool = True,
        preflight_strict: bool = False,
    ) -> "DesignSpec":
        """
        Load and validate a spec from a JSON file.
        
        Parameters
        ----------
        path : str or Path
            Path to the JSON file
        strict_schema : bool
            If True, unknown fields raise errors
        allow_unknown_fields : bool
            If True (default), unknown fields generate warnings
        fill_defaults : bool
            If True (default), fill missing policies with defaults.
        run_preflight : bool
            If True (default), run preflight validation checks.
        preflight_strict : bool
            If True, preflight errors raise exceptions. Default False (warnings only).
            
        Returns
        -------
        DesignSpec
            Loaded and validated spec
        """
        path = Path(path)
        if not path.exists():
            raise DesignSpecError(f"Spec file not found: {path}")
        
        with open(path, "r") as f:
            spec_dict = json.load(f)
        
        return cls.from_dict(
            spec_dict,
            strict_schema=strict_schema,
            allow_unknown_fields=allow_unknown_fields,
            fill_defaults=fill_defaults,
            run_preflight=run_preflight,
            preflight_strict=preflight_strict,
        )


__all__ = [
    "DesignSpec",
    "DesignSpecError",
    "DesignSpecValidationError",
    "POLICY_LENGTH_FIELDS",
    "POLICY_VOLUME_FIELDS",
    "NESTED_POLICY_FIELDS",
    "DOMAIN_LENGTH_FIELDS",
    "_get_unit_scale",
]
