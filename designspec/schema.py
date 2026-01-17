"""
Schema versioning and compatibility for DesignSpec.

This module defines the schema name, version, and compatibility rules
for the aog_designspec format.

VERSIONING RULES
----------------
- Major version: Breaking changes (incompatible)
- Minor version: New features (must match exactly for now)
- Patch version: Bug fixes (compatible if listed in compatible_with)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set
import re


SCHEMA_NAME = "aog_designspec"
SCHEMA_VERSION = "1.0.0"

SUPPORTED_VERSIONS: Dict[str, Dict[str, Any]] = {
    "1.0.0": {
        "compatible_with": ["1.0.x"],
        "deprecated": False,
    },
}


class SchemaValidationError(Exception):
    """Raised when schema validation fails."""
    pass


def parse_version(version_str: str) -> tuple:
    """
    Parse a version string into (major, minor, patch) tuple.
    
    Parameters
    ----------
    version_str : str
        Version string like "1.0.0" or "1.0.x"
        
    Returns
    -------
    tuple
        (major, minor, patch) where patch may be None for wildcards
    """
    match = re.match(r"^(\d+)\.(\d+)\.([\dx]+)$", version_str)
    if not match:
        raise ValueError(f"Invalid version format: {version_str}")
    
    major = int(match.group(1))
    minor = int(match.group(2))
    patch_str = match.group(3)
    patch = None if patch_str == "x" else int(patch_str)
    
    return (major, minor, patch)


def is_version_compatible(spec_version: str, target_version: str = SCHEMA_VERSION) -> bool:
    """
    Check if a spec version is compatible with the target version.
    
    Compatibility rules:
    - Major version must match exactly
    - Minor version must match exactly
    - Patch version is compatible if target's compatible_with includes it
    
    Parameters
    ----------
    spec_version : str
        Version from the spec being validated
    target_version : str
        Target version to check compatibility against (default: current)
        
    Returns
    -------
    bool
        True if versions are compatible
    """
    try:
        spec_parsed = parse_version(spec_version)
        target_parsed = parse_version(target_version)
    except ValueError:
        return False
    
    spec_major, spec_minor, spec_patch = spec_parsed
    target_major, target_minor, target_patch = target_parsed
    
    if spec_major != target_major:
        return False
    
    if spec_minor != target_minor:
        return False
    
    if spec_version == target_version:
        return True
    
    target_info = SUPPORTED_VERSIONS.get(target_version, {})
    compatible_with = target_info.get("compatible_with", [])
    
    for pattern in compatible_with:
        pattern_parsed = parse_version(pattern)
        pattern_major, pattern_minor, pattern_patch = pattern_parsed
        
        if spec_major == pattern_major and spec_minor == pattern_minor:
            if pattern_patch is None:
                return True
            if spec_patch == pattern_patch:
                return True
    
    return False


def validate_schema_block(schema_block: Dict[str, Any]) -> List[str]:
    """
    Validate the schema block of a spec.
    
    Parameters
    ----------
    schema_block : dict
        The "schema" section of a spec
        
    Returns
    -------
    list of str
        List of validation error messages (empty if valid)
    """
    errors = []
    
    if not isinstance(schema_block, dict):
        errors.append("schema must be a dict")
        return errors
    
    name = schema_block.get("name")
    if name != SCHEMA_NAME:
        errors.append(f"schema.name must be '{SCHEMA_NAME}', got '{name}'")
    
    version = schema_block.get("version")
    if not version:
        errors.append("schema.version is required")
    elif not is_version_compatible(version):
        errors.append(
            f"schema.version '{version}' is not compatible with "
            f"supported version '{SCHEMA_VERSION}'"
        )
    
    return errors


@dataclass
class SchemaInfo:
    """
    Parsed schema information from a spec.
    """
    name: str
    version: str
    compatible_with: List[str] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SchemaInfo":
        """Create SchemaInfo from schema block dict."""
        return cls(
            name=d.get("name", ""),
            version=d.get("version", ""),
            compatible_with=d.get("compatible_with", []),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
        return {
            "name": self.name,
            "version": self.version,
            "compatible_with": self.compatible_with,
        }


__all__ = [
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "SUPPORTED_VERSIONS",
    "SchemaValidationError",
    "parse_version",
    "is_version_compatible",
    "validate_schema_block",
    "SchemaInfo",
]
