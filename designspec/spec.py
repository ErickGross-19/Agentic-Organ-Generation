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
from typing import Dict, Any, List, Optional, Union, Set
from pathlib import Path
import json
import hashlib
import logging
import copy

from .schema import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    validate_schema_block,
    SchemaInfo,
    SchemaValidationError,
)
from .compat import apply_all_aliases

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
    },
    "open_port": {
        "probe_length",
        "local_region_size",
        "validation_pitch",
    },
    "repair": {
        "voxel_pitch",
        "min_component_volume",
    },
    "growth": {
        "min_segment_length",
        "max_segment_length",
        "step_size",
    },
    "collision": {
        "collision_clearance",
    },
    "composition": {
        "min_component_volume",
    },
    "domain_meshing": {
    },
    "output": {
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
        Unit string: "m", "mm", or "um"
        
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
    elif input_units == "um":
        return 1e-6
    else:
        raise ValueError(f"Unknown unit: '{input_units}'. Supported units: m, mm, um")


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


def _normalize_policy_to_meters(
    policy_name: str,
    policy_dict: Dict[str, Any],
    scale: float,
) -> Dict[str, Any]:
    """Normalize length fields in a policy dict to meters."""
    result = dict(policy_dict)
    length_fields = POLICY_LENGTH_FIELDS.get(policy_name, set())
    
    for field_name in length_fields:
        if field_name in result and result[field_name] is not None:
            result[field_name] = _convert_value(result[field_name], scale)
    
    if policy_name == "resolution" and "input_units" in result:
        result["input_units"] = "m"
    
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
            # Known length fields in backend_params
            backend_length_fields = {
                "step_size", "min_segment_length", "max_segment_length",
                "influence_radius", "kill_radius", "perception_radius",
                "clearance", "min_radius", "max_radius",
            }
            for field_name in backend_length_fields:
                if field_name in backend_params and backend_params[field_name] is not None:
                    backend_params[field_name] = _convert_value(backend_params[field_name], scale)
            result["build"]["backend_params"] = backend_params
    
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
    """
    raw: Dict[str, Any]
    normalized: Dict[str, Any]
    schema_info: SchemaInfo
    spec_hash: str
    warnings: List[str] = field(default_factory=list)
    
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
            
        Returns
        -------
        DesignSpec
            Loaded and validated spec
            
        Raises
        ------
        DesignSpecValidationError
            If validation fails
        """
        warnings = []
        
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
        
        normalized = _normalize_spec_to_meters(spec)
        
        schema_info = SchemaInfo.from_dict(spec.get("schema", {}))
        
        spec_hash = _compute_spec_hash(normalized)
        
        return cls(
            raw=spec_dict,
            normalized=normalized,
            schema_info=schema_info,
            spec_hash=spec_hash,
            warnings=warnings,
        )
    
    @classmethod
    def from_json(
        cls,
        path: Union[str, Path],
        strict_schema: bool = False,
        allow_unknown_fields: bool = True,
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
        )


__all__ = [
    "DesignSpec",
    "DesignSpecError",
    "DesignSpecValidationError",
    "POLICY_LENGTH_FIELDS",
    "DOMAIN_LENGTH_FIELDS",
]
