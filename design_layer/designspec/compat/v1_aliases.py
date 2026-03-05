"""
V1 alias mappings for DesignSpec backward compatibility.

This module provides alias mappings for legacy field names used in v1 specs.
When a spec uses a legacy name, it is transparently mapped to the canonical name
and a warning is recorded.

ALIAS CATEGORIES
----------------
1. Top-level keys: Legacy names for top-level spec sections
2. Per-policy keys: Legacy names within policy dicts
3. Per-component keys: Legacy names within component dicts
4. Domain keys: Legacy names within domain dicts (minimal, domain_from_dict handles most)
"""

from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


V1_TOP_LEVEL_ALIASES: Dict[str, str] = {
    "input_units": "meta.input_units",
}


V1_POLICY_ALIASES: Dict[str, Dict[str, str]] = {
    "resolution": {
        "voxels_across_min_diameter": "min_voxels_across_feature",
    },
    "ports": {
        "placement_pattern": "pattern",
    },
    "channels": {
        "channel_type": "profile",
        "min_length": "length",
        "hook_strategy": "constraint_strategy",
        "inlet_radius": "radius_end",
        "outlet_radius": "radius_end",
        "target_depth": "hook_depth",
        "hook_angle": "hook_angle_deg",
    },
    "embedding": {
        "preserve_ports_mode": "preserve_mode",
        "recarve_enabled": "preserve_ports_enabled",
    },
    "validity": {
    },
    "open_port": {
        "roi_size_factor": "local_region_size",
    },
    "growth": {
    },
    "pathfinding": {
    },
    "composition": {
    },
    "repair": {
    },
    "output": {
    },
    "domain_meshing": {
    },
}


V1_COMPONENT_ALIASES: Dict[str, str] = {
    "domain_id": "domain_ref",
    "generator": "build.type",
    "generator_kind": "build.type",
}


V1_DOMAIN_ALIASES: Dict[str, str] = {
}


def apply_aliases(
    d: Dict[str, Any],
    aliases: Dict[str, str],
    context: str = "",
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Apply alias mappings to a dictionary.
    
    Parameters
    ----------
    d : dict
        Input dictionary to transform
    aliases : dict
        Mapping of legacy_name -> canonical_name
    context : str
        Context string for warning messages (e.g., "resolution policy")
        
    Returns
    -------
    result : dict
        Dictionary with aliases applied
    warnings : list of str
        List of warning messages for applied aliases
    """
    result = dict(d)
    warnings = []
    
    for legacy_name, canonical_name in aliases.items():
        if legacy_name in result:
            if "." in canonical_name:
                parts = canonical_name.split(".")
                target_key = parts[-1]
                if target_key not in result:
                    result[target_key] = result.pop(legacy_name)
                    warnings.append(
                        f"Alias applied in {context}: '{legacy_name}' -> '{target_key}'"
                    )
                else:
                    result.pop(legacy_name)
            else:
                if canonical_name not in result:
                    result[canonical_name] = result.pop(legacy_name)
                    warnings.append(
                        f"Alias applied in {context}: '{legacy_name}' -> '{canonical_name}'"
                    )
                else:
                    result.pop(legacy_name)
    
    return result, warnings


def apply_policy_aliases(
    policies: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Apply aliases to all policy dicts.
    
    Parameters
    ----------
    policies : dict
        The "policies" section of a spec
        
    Returns
    -------
    result : dict
        Policies with aliases applied
    warnings : list of str
        List of warning messages
    """
    result = dict(policies)
    all_warnings = []
    
    for policy_name, policy_aliases in V1_POLICY_ALIASES.items():
        if policy_name in result and isinstance(result[policy_name], dict):
            result[policy_name], warnings = apply_aliases(
                result[policy_name],
                policy_aliases,
                context=f"{policy_name} policy",
            )
            all_warnings.extend(warnings)
    
    return result, all_warnings


def apply_component_aliases(
    component: Dict[str, Any],
    component_id: str = "",
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Apply aliases to a component dict.
    
    Parameters
    ----------
    component : dict
        A single component from the "components" list
    component_id : str
        Component ID for warning messages
        
    Returns
    -------
    result : dict
        Component with aliases applied
    warnings : list of str
        List of warning messages
    """
    result = dict(component)
    all_warnings = []
    
    for legacy_name, canonical_name in V1_COMPONENT_ALIASES.items():
        if legacy_name in result:
            if "." in canonical_name:
                parts = canonical_name.split(".")
                parent_key = parts[0]
                child_key = parts[1]
                
                if parent_key not in result:
                    result[parent_key] = {}
                
                if child_key not in result[parent_key]:
                    result[parent_key][child_key] = result.pop(legacy_name)
                    all_warnings.append(
                        f"Alias applied in component '{component_id}': "
                        f"'{legacy_name}' -> '{canonical_name}'"
                    )
                else:
                    result.pop(legacy_name)
            else:
                if canonical_name not in result:
                    result[canonical_name] = result.pop(legacy_name)
                    all_warnings.append(
                        f"Alias applied in component '{component_id}': "
                        f"'{legacy_name}' -> '{canonical_name}'"
                    )
                else:
                    result.pop(legacy_name)
    
    return result, all_warnings


def apply_all_aliases(spec: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Apply all alias transformations to a spec.
    
    Parameters
    ----------
    spec : dict
        The full spec dict
        
    Returns
    -------
    result : dict
        Spec with all aliases applied
    warnings : list of str
        List of all warning messages
    """
    result = dict(spec)
    all_warnings = []
    
    result, warnings = apply_aliases(result, V1_TOP_LEVEL_ALIASES, context="top-level")
    all_warnings.extend(warnings)
    
    if "policies" in result and isinstance(result["policies"], dict):
        result["policies"], warnings = apply_policy_aliases(result["policies"])
        all_warnings.extend(warnings)
    
    if "components" in result and isinstance(result["components"], list):
        new_components = []
        for i, component in enumerate(result["components"]):
            if isinstance(component, dict):
                comp_id = component.get("id", f"component_{i}")
                component, warnings = apply_component_aliases(component, comp_id)
                all_warnings.extend(warnings)
            new_components.append(component)
        result["components"] = new_components
    
    return result, all_warnings


__all__ = [
    "V1_TOP_LEVEL_ALIASES",
    "V1_POLICY_ALIASES",
    "V1_COMPONENT_ALIASES",
    "V1_DOMAIN_ALIASES",
    "apply_aliases",
    "apply_policy_aliases",
    "apply_component_aliases",
    "apply_all_aliases",
]
