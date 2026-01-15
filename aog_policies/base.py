"""
Base utilities for AOG policies.

This module provides shared helpers and the OperationReport dataclass
used across all policy-driven operations.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Union, Tuple
import json


def validate_policy(policy: Any, required_fields: Optional[List[str]] = None) -> List[str]:
    """
    Validate a policy object.
    
    Parameters
    ----------
    policy : Any
        Policy dataclass instance to validate
    required_fields : List[str], optional
        List of field names that must be non-None
        
    Returns
    -------
    List[str]
        List of validation error messages (empty if valid)
    """
    errors = []
    
    if required_fields:
        for field_name in required_fields:
            if not hasattr(policy, field_name):
                errors.append(f"Missing required field: {field_name}")
            elif getattr(policy, field_name) is None:
                errors.append(f"Required field is None: {field_name}")
    
    return errors


def coerce_float(value: Any, default: float = 0.0) -> float:
    """
    Coerce a value to float, with fallback to default.
    
    Parameters
    ----------
    value : Any
        Value to coerce
    default : float
        Default value if coercion fails
        
    Returns
    -------
    float
        Coerced float value
    """
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def coerce_vec3(
    value: Any,
    default: Tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> Tuple[float, float, float]:
    """
    Coerce a value to a 3D vector tuple.
    
    Accepts:
    - tuple/list of 3 numbers
    - Point3D-like object with x, y, z attributes
    - dict with x, y, z keys
    
    Parameters
    ----------
    value : Any
        Value to coerce
    default : tuple
        Default value if coercion fails
        
    Returns
    -------
    Tuple[float, float, float]
        Coerced 3D vector
    """
    if value is None:
        return default
    
    # Handle tuple/list
    if isinstance(value, (tuple, list)) and len(value) >= 3:
        try:
            return (float(value[0]), float(value[1]), float(value[2]))
        except (TypeError, ValueError):
            return default
    
    # Handle Point3D-like object
    if hasattr(value, 'x') and hasattr(value, 'y') and hasattr(value, 'z'):
        try:
            return (float(value.x), float(value.y), float(value.z))
        except (TypeError, ValueError):
            return default
    
    # Handle dict
    if isinstance(value, dict) and 'x' in value and 'y' in value and 'z' in value:
        try:
            return (float(value['x']), float(value['y']), float(value['z']))
        except (TypeError, ValueError):
            return default
    
    return default


def alias_fields(d: Dict[str, Any], aliases: Dict[str, str]) -> Dict[str, Any]:
    """
    Apply field aliases to a dictionary.
    
    This allows legacy field names to be mapped to canonical names.
    
    Parameters
    ----------
    d : dict
        Input dictionary
    aliases : dict
        Mapping of legacy_name -> canonical_name
        
    Returns
    -------
    dict
        Dictionary with aliases applied
    """
    result = d.copy()
    for legacy_name, canonical_name in aliases.items():
        if legacy_name in result and canonical_name not in result:
            result[canonical_name] = result.pop(legacy_name)
    return result


@dataclass
class OperationReport:
    """
    Standard report structure for all operations.
    
    Every operation returns a report with requested vs effective policy,
    warnings, and operation-specific metadata.
    
    The "requested vs effective" pattern allows tracking of runtime
    adjustments (e.g., pitch stepping, constraint enforcement).
    """
    operation: str = "unknown"  # Default to "unknown" for convenience
    success: bool = True
    requested_policy: Dict[str, Any] = field(default_factory=dict)
    effective_policy: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
    
    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)
    
    def add_error(self, message: str) -> None:
        """Add an error message and mark as failed."""
        self.errors.append(message)
        self.success = False
    
    def merge(self, other: "OperationReport") -> None:
        """Merge another report into this one."""
        self.warnings.extend(other.warnings)
        self.errors.extend(other.errors)
        if not other.success:
            self.success = False
        self.metadata.update(other.metadata)


__all__ = [
    "OperationReport",
    "validate_policy",
    "coerce_float",
    "coerce_vec3",
    "alias_fields",
]
