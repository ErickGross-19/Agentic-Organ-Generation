"""
Schema Patch Validator

Validates schema patches proposed by the LLM to ensure they are safe and valid.

Validation rules:
1. Only allow known modules
2. Only allow custom fields under custom.* namespace with required metadata
3. Ensure no reserved field collisions
4. Ensure required fields remain coherent
"""

from typing import Tuple, List, Set, Dict, Any, Optional

from .schema_modules import (
    ALL_MODULES,
    CORE_FIELD_GROUPS,
    FieldDefinition,
    FieldType,
    get_all_core_fields,
)
from .schema_manager import SchemaPatch, ModulePatch, FieldPatch, ActiveSchema


# =============================================================================
# Reserved Fields
# =============================================================================

RESERVED_FIELD_PREFIXES = [
    "system.",
    "internal.",
    "_",
]

RESERVED_FIELD_NAMES = {
    "id",
    "uuid",
    "created_at",
    "updated_at",
    "version",
    "schema_version",
}


def get_all_known_field_names() -> Set[str]:
    """Get all known field names from core and modules."""
    names = set()
    
    # Core fields
    for f in get_all_core_fields():
        names.add(f.name)
    
    # Module fields
    for module in ALL_MODULES.values():
        for f in module.fields:
            names.add(f.name)
    
    return names


# =============================================================================
# Validation Functions
# =============================================================================

def validate_module_patch(
    patch: ModulePatch,
    schema: ActiveSchema
) -> Tuple[bool, str]:
    """
    Validate a module patch.
    
    Parameters
    ----------
    patch : ModulePatch
        The module patch to validate
    schema : ActiveSchema
        Current schema state
        
    Returns
    -------
    Tuple[bool, str]
        (is_valid, error_message)
    """
    # Check action is valid
    if patch.action not in ("activate", "deactivate"):
        return False, f"Invalid module action: {patch.action}"
    
    # Check module exists
    if patch.module_name not in ALL_MODULES:
        return False, f"Unknown module: {patch.module_name}. Valid modules: {', '.join(ALL_MODULES.keys())}"
    
    # Check dependencies for activation
    if patch.action == "activate":
        module = ALL_MODULES[patch.module_name]
        for dep in module.dependencies:
            if dep not in schema.active_modules and dep != patch.module_name:
                return False, f"Module {patch.module_name} requires {dep} to be active first"
    
    # Check dependents for deactivation
    if patch.action == "deactivate":
        for name, module in ALL_MODULES.items():
            if name in schema.active_modules and patch.module_name in module.dependencies:
                return False, f"Cannot deactivate {patch.module_name}: {name} depends on it"
    
    return True, ""


def validate_field_patch(
    patch: FieldPatch,
    schema: ActiveSchema
) -> Tuple[bool, str]:
    """
    Validate a field patch.
    
    Parameters
    ----------
    patch : FieldPatch
        The field patch to validate
    schema : ActiveSchema
        Current schema state
        
    Returns
    -------
    Tuple[bool, str]
        (is_valid, error_message)
    """
    # Check action is valid
    valid_actions = ("set_value", "set_required", "set_default", "remove")
    if patch.action not in valid_actions:
        return False, f"Invalid field action: {patch.action}. Valid: {', '.join(valid_actions)}"
    
    # Check field name is not reserved
    for prefix in RESERVED_FIELD_PREFIXES:
        if patch.field_name.startswith(prefix):
            return False, f"Field name '{patch.field_name}' uses reserved prefix '{prefix}'"
    
    if patch.field_name in RESERVED_FIELD_NAMES:
        return False, f"Field name '{patch.field_name}' is reserved"
    
    # Get known field names
    known_fields = get_all_known_field_names()
    
    # Check if field exists (for non-custom fields)
    if not patch.field_name.startswith("custom."):
        if patch.field_name not in known_fields:
            # Check if it's in an active module
            field_found = False
            for f in schema.get_all_fields():
                if f.name == patch.field_name:
                    field_found = True
                    break
            
            if not field_found:
                return False, f"Unknown field: {patch.field_name}. Use 'custom.{patch.field_name}' for custom fields"
    
    # Validate value type for set_value
    if patch.action == "set_value" and patch.value is not None:
        # Find field definition
        field_def = None
        for f in schema.get_all_fields():
            if f.name == patch.field_name:
                field_def = f
                break
        
        if field_def:
            is_valid, error = validate_field_value(field_def, patch.value)
            if not is_valid:
                return False, error
    
    # Validate set_required value is boolean
    if patch.action == "set_required":
        if not isinstance(patch.value, bool):
            return False, f"set_required action requires boolean value, got {type(patch.value).__name__}"
    
    return True, ""


def validate_field_value(
    field: FieldDefinition,
    value: Any
) -> Tuple[bool, str]:
    """
    Validate a value against a field definition.
    
    Parameters
    ----------
    field : FieldDefinition
        The field definition
    value : Any
        The value to validate
        
    Returns
    -------
    Tuple[bool, str]
        (is_valid, error_message)
    """
    # Type checking
    if field.field_type == FieldType.STRING:
        if not isinstance(value, str):
            return False, f"Field {field.name} expects string, got {type(value).__name__}"
    
    elif field.field_type == FieldType.INTEGER:
        if not isinstance(value, int) or isinstance(value, bool):
            return False, f"Field {field.name} expects integer, got {type(value).__name__}"
        
        if field.min_value is not None and value < field.min_value:
            return False, f"Field {field.name} value {value} is below minimum {field.min_value}"
        if field.max_value is not None and value > field.max_value:
            return False, f"Field {field.name} value {value} is above maximum {field.max_value}"
    
    elif field.field_type == FieldType.FLOAT:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False, f"Field {field.name} expects number, got {type(value).__name__}"
        
        if field.min_value is not None and value < field.min_value:
            return False, f"Field {field.name} value {value} is below minimum {field.min_value}"
        if field.max_value is not None and value > field.max_value:
            return False, f"Field {field.name} value {value} is above maximum {field.max_value}"
    
    elif field.field_type == FieldType.BOOLEAN:
        if not isinstance(value, bool):
            return False, f"Field {field.name} expects boolean, got {type(value).__name__}"
    
    elif field.field_type == FieldType.ENUM:
        if field.enum_values and value not in field.enum_values:
            return False, f"Field {field.name} value '{value}' not in allowed values: {field.enum_values}"
    
    elif field.field_type == FieldType.ARRAY:
        if not isinstance(value, (list, tuple)):
            return False, f"Field {field.name} expects array, got {type(value).__name__}"
    
    elif field.field_type == FieldType.OBJECT:
        if not isinstance(value, dict):
            return False, f"Field {field.name} expects object, got {type(value).__name__}"
    
    elif field.field_type in (FieldType.POINT3D, FieldType.SIZE3D):
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            return False, f"Field {field.name} expects 3D coordinate/size (x, y, z), got {value}"
        for i, v in enumerate(value):
            if not isinstance(v, (int, float)):
                return False, f"Field {field.name} coordinate {i} must be numeric, got {type(v).__name__}"
    
    return True, ""


def validate_custom_field(
    name: str,
    field_def: FieldDefinition
) -> Tuple[bool, str]:
    """
    Validate a custom field definition.
    
    Parameters
    ----------
    name : str
        The custom field name
    field_def : FieldDefinition
        The field definition
        
    Returns
    -------
    Tuple[bool, str]
        (is_valid, error_message)
    """
    # Must be under custom.* namespace
    if not name.startswith("custom."):
        return False, f"Custom field '{name}' must be under 'custom.' namespace"
    
    # Check for reserved prefixes in the custom name
    custom_name = name[7:]  # Remove "custom." prefix
    for prefix in RESERVED_FIELD_PREFIXES:
        if custom_name.startswith(prefix):
            return False, f"Custom field name '{custom_name}' uses reserved prefix '{prefix}'"
    
    # Must have description
    if not field_def.description:
        return False, f"Custom field '{name}' must have a description"
    
    # Must have valid type
    if not isinstance(field_def.field_type, FieldType):
        return False, f"Custom field '{name}' has invalid type"
    
    # Check for collision with known fields
    known_fields = get_all_known_field_names()
    if name in known_fields:
        return False, f"Custom field '{name}' collides with existing field"
    
    return True, ""


def validate_schema_coherence(
    schema: ActiveSchema,
    patch: SchemaPatch
) -> Tuple[bool, str]:
    """
    Validate that the schema remains coherent after applying the patch.
    
    Parameters
    ----------
    schema : ActiveSchema
        Current schema state
    patch : SchemaPatch
        The patch to apply
        
    Returns
    -------
    Tuple[bool, str]
        (is_valid, error_message)
    """
    # Simulate applying the patch
    simulated_modules = set(schema.active_modules)
    simulated_values = dict(schema._field_values)
    
    # Apply module patches
    for mp in patch.module_patches:
        if mp.action == "activate":
            simulated_modules.add(mp.module_name)
        elif mp.action == "deactivate":
            simulated_modules.discard(mp.module_name)
    
    # Apply field patches
    for fp in patch.field_patches:
        if fp.action == "set_value":
            simulated_values[fp.field_name] = fp.value
        elif fp.action == "remove":
            simulated_values.pop(fp.field_name, None)
    
    # Check I/O coherence
    num_inlets = simulated_values.get("num_inlets")
    inlet_positions = simulated_values.get("inlet_positions_m")
    
    if num_inlets and inlet_positions:
        if len(inlet_positions) != num_inlets:
            return False, f"num_inlets ({num_inlets}) doesn't match inlet_positions count ({len(inlet_positions)})"
    
    num_outlets = simulated_values.get("num_outlets")
    outlet_positions = simulated_values.get("outlet_positions_m")
    
    if num_outlets and outlet_positions:
        if len(outlet_positions) != num_outlets:
            return False, f"num_outlets ({num_outlets}) doesn't match outlet_positions count ({len(outlet_positions)})"
    
    # Check domain coherence
    domain_type = simulated_values.get("domain_type")
    domain_size = simulated_values.get("domain_size_m")
    
    if domain_type and domain_size:
        if domain_type == "ellipsoid":
            # Ellipsoid uses semi-axes, all must be positive
            if any(s <= 0 for s in domain_size):
                return False, "Ellipsoid domain requires positive semi-axes"
        elif domain_type == "box":
            # Box uses dimensions, all must be positive
            if any(s <= 0 for s in domain_size):
                return False, "Box domain requires positive dimensions"
    
    # Check constraint coherence
    min_radius = simulated_values.get("min_radius_m")
    inlet_radius = simulated_values.get("inlet_radius_m")
    
    if min_radius and inlet_radius:
        if min_radius > inlet_radius:
            return False, f"min_radius ({min_radius}) cannot be greater than inlet_radius ({inlet_radius})"
    
    return True, ""


def validate_schema_patch(
    patch: SchemaPatch,
    schema: ActiveSchema
) -> Tuple[bool, str]:
    """
    Validate a complete schema patch.
    
    Parameters
    ----------
    patch : SchemaPatch
        The patch to validate
    schema : ActiveSchema
        Current schema state
        
    Returns
    -------
    Tuple[bool, str]
        (is_valid, error_message)
    """
    errors = []
    
    # Validate module patches
    for mp in patch.module_patches:
        is_valid, error = validate_module_patch(mp, schema)
        if not is_valid:
            errors.append(f"Module patch error: {error}")
    
    # Validate field patches
    for fp in patch.field_patches:
        is_valid, error = validate_field_patch(fp, schema)
        if not is_valid:
            errors.append(f"Field patch error: {error}")
    
    # Validate custom fields
    for name, field_def in patch.custom_fields.items():
        is_valid, error = validate_custom_field(name, field_def)
        if not is_valid:
            errors.append(f"Custom field error: {error}")
    
    # Validate schema coherence
    is_valid, error = validate_schema_coherence(schema, patch)
    if not is_valid:
        errors.append(f"Coherence error: {error}")
    
    if errors:
        return False, "; ".join(errors)
    
    return True, ""


def suggest_patch_fixes(
    patch: SchemaPatch,
    schema: ActiveSchema
) -> List[str]:
    """
    Suggest fixes for an invalid patch.
    
    Parameters
    ----------
    patch : SchemaPatch
        The invalid patch
    schema : ActiveSchema
        Current schema state
        
    Returns
    -------
    List[str]
        List of suggested fixes
    """
    suggestions = []
    
    # Check for unknown modules
    for mp in patch.module_patches:
        if mp.module_name not in ALL_MODULES:
            similar = [m for m in ALL_MODULES.keys() if mp.module_name.lower() in m.lower()]
            if similar:
                suggestions.append(f"Did you mean module '{similar[0]}' instead of '{mp.module_name}'?")
            else:
                suggestions.append(f"Available modules: {', '.join(ALL_MODULES.keys())}")
    
    # Check for unknown fields
    known_fields = get_all_known_field_names()
    for fp in patch.field_patches:
        if not fp.field_name.startswith("custom.") and fp.field_name not in known_fields:
            similar = [f for f in known_fields if fp.field_name.lower() in f.lower()]
            if similar:
                suggestions.append(f"Did you mean field '{similar[0]}' instead of '{fp.field_name}'?")
            else:
                suggestions.append(f"Use 'custom.{fp.field_name}' for custom fields")
    
    # Check for missing custom field metadata
    for name, field_def in patch.custom_fields.items():
        if not field_def.description:
            suggestions.append(f"Add description for custom field '{name}'")
    
    return suggestions
