"""
Schema Manager

Manages the active schema for each object, handling module activation/deactivation,
field validation, and question generation based on missing required fields.

The schema manager:
1. Maintains the ActiveSchema for the current object
2. Applies schema patches from LLM (after validation)
3. Produces a list of missing required fields
4. Produces question candidates ranked by rework cost
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from enum import Enum
import copy

from .schema_modules import (
    SchemaModule,
    FieldDefinition,
    FieldType,
    ALL_MODULES,
    CORE_FIELD_GROUPS,
    get_all_core_fields,
    get_module,
    get_modules_for_triggers,
)


# =============================================================================
# Schema Patch Types
# =============================================================================

@dataclass
class FieldPatch:
    """A patch to a single field."""
    field_name: str
    action: str  # "set_value", "set_required", "set_default", "remove"
    value: Any = None


@dataclass
class ModulePatch:
    """A patch to activate/deactivate a module."""
    module_name: str
    action: str  # "activate", "deactivate"


@dataclass
class SchemaPatch:
    """
    A complete schema patch proposed by the LLM.
    
    Contains module changes and field changes that should be applied
    to the active schema.
    """
    module_patches: List[ModulePatch] = field(default_factory=list)
    field_patches: List[FieldPatch] = field(default_factory=list)
    custom_fields: Dict[str, FieldDefinition] = field(default_factory=dict)
    reasoning: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "module_patches": [
                {"module_name": p.module_name, "action": p.action}
                for p in self.module_patches
            ],
            "field_patches": [
                {"field_name": p.field_name, "action": p.action, "value": p.value}
                for p in self.field_patches
            ],
            "custom_fields": {
                name: f.to_dict() for name, f in self.custom_fields.items()
            },
            "reasoning": self.reasoning,
        }


# =============================================================================
# Question Generation
# =============================================================================

@dataclass
class Question:
    """A question to ask the user."""
    field_name: str
    question_text: str
    field_type: FieldType
    options: Optional[List[str]] = None
    default: Any = None
    rework_cost: str = "medium"
    reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_name": self.field_name,
            "question_text": self.question_text,
            "field_type": self.field_type.value,
            "options": self.options,
            "default": self.default,
            "rework_cost": self.rework_cost,
            "reason": self.reason,
        }


# =============================================================================
# Active Schema
# =============================================================================

class ActiveSchema:
    """
    Maintains the current schema state for an object.
    
    The active schema consists of:
    - Core fields (always present)
    - Active modules (opt-in)
    - Field values (user-provided or defaults)
    - Custom fields (LLM-proposed, under custom.* namespace)
    """
    
    def __init__(self):
        self._active_modules: Set[str] = set()
        self._field_values: Dict[str, Any] = {}
        self._custom_fields: Dict[str, FieldDefinition] = {}
        self._field_required_overrides: Dict[str, bool] = {}
    
    @property
    def active_modules(self) -> List[str]:
        """Get list of active module names."""
        return sorted(self._active_modules)
    
    def activate_module(self, module_name: str) -> bool:
        """
        Activate a schema module.
        
        Returns True if module was activated, False if already active or invalid.
        """
        if module_name not in ALL_MODULES:
            return False
        
        if module_name in self._active_modules:
            return False
        
        # Check dependencies
        module = ALL_MODULES[module_name]
        for dep in module.dependencies:
            if dep not in self._active_modules:
                self.activate_module(dep)
        
        self._active_modules.add(module_name)
        return True
    
    def deactivate_module(self, module_name: str) -> bool:
        """
        Deactivate a schema module.
        
        Returns True if module was deactivated, False if not active.
        """
        if module_name not in self._active_modules:
            return False
        
        # Check if other modules depend on this one
        for name, module in ALL_MODULES.items():
            if name in self._active_modules and module_name in module.dependencies:
                return False
        
        self._active_modules.discard(module_name)
        return True
    
    def set_field_value(self, field_name: str, value: Any) -> None:
        """Set a field value."""
        self._field_values[field_name] = value
    
    def get_field_value(self, field_name: str) -> Any:
        """Get a field value, or None if not set."""
        return self._field_values.get(field_name)
    
    def has_field_value(self, field_name: str) -> bool:
        """Check if a field has a value set."""
        return field_name in self._field_values
    
    def set_field_required(self, field_name: str, required: bool) -> None:
        """Override the required status of a field."""
        self._field_required_overrides[field_name] = required
    
    def add_custom_field(self, name: str, field_def: FieldDefinition) -> None:
        """Add a custom field (must be under custom.* namespace)."""
        if not name.startswith("custom."):
            name = f"custom.{name}"
        self._custom_fields[name] = field_def
    
    def get_all_fields(self) -> List[FieldDefinition]:
        """Get all fields (core + active modules + custom)."""
        fields = []
        
        # Core fields
        fields.extend(get_all_core_fields())
        
        # Active module fields
        for module_name in self._active_modules:
            module = ALL_MODULES.get(module_name)
            if module:
                fields.extend(module.fields)
        
        # Custom fields
        fields.extend(self._custom_fields.values())
        
        return fields
    
    def get_required_fields(self) -> List[FieldDefinition]:
        """Get all required fields."""
        required = []
        
        for f in self.get_all_fields():
            # Check for override
            if f.name in self._field_required_overrides:
                if self._field_required_overrides[f.name]:
                    required.append(f)
            elif f.required:
                required.append(f)
        
        return required
    
    def missing_required_fields(self) -> List[FieldDefinition]:
        """Get required fields that don't have values."""
        missing = []
        
        for f in self.get_required_fields():
            if not self.has_field_value(f.name):
                # Check if there's a default
                if f.default is None:
                    missing.append(f)
        
        return missing
    
    def to_dict(self) -> Dict[str, Any]:
        """Export schema state as dictionary."""
        return {
            "active_modules": self.active_modules,
            "field_values": copy.deepcopy(self._field_values),
            "custom_fields": {
                name: f.to_dict() for name, f in self._custom_fields.items()
            },
            "required_overrides": copy.deepcopy(self._field_required_overrides),
        }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load schema state from dictionary."""
        self._active_modules = set(data.get("active_modules", []))
        self._field_values = data.get("field_values", {})
        self._field_required_overrides = data.get("required_overrides", {})
        
        # Reconstruct custom fields
        custom_data = data.get("custom_fields", {})
        for name, field_data in custom_data.items():
            self._custom_fields[name] = FieldDefinition(
                name=name,
                field_type=FieldType(field_data["type"]),
                description=field_data.get("description", ""),
                required=field_data.get("required", False),
                default=field_data.get("default"),
                unit=field_data.get("unit"),
            )


# =============================================================================
# Schema Manager
# =============================================================================

class SchemaManager:
    """
    Manages schema for an object throughout its lifecycle.
    
    Responsibilities:
    - Maintain ActiveSchema for the current object
    - Apply schema patches from LLM (after validation)
    - Produce list of missing required fields
    - Produce question candidates ranked by rework cost
    - Update schema based on generation results
    """
    
    def __init__(self):
        self.schema = ActiveSchema()
        self._patch_history: List[SchemaPatch] = []
    
    def reset(self) -> None:
        """Reset schema to initial state."""
        self.schema = ActiveSchema()
        self._patch_history = []
    
    def activate_module(self, module_name: str) -> bool:
        """
        Activate a schema module.
        
        Wrapper around ActiveSchema.activate_module() for direct access.
        
        Parameters
        ----------
        module_name : str
            Name of the module to activate (e.g., "TopologyModule")
            
        Returns
        -------
        bool
            True if module was activated, False if already active or invalid
        """
        return self.schema.activate_module(module_name)
    
    def deactivate_module(self, module_name: str) -> bool:
        """
        Deactivate a schema module.
        
        Wrapper around ActiveSchema.deactivate_module() for direct access.
        
        Parameters
        ----------
        module_name : str
            Name of the module to deactivate
            
        Returns
        -------
        bool
            True if module was deactivated, False if not active or has dependents
        """
        return self.schema.deactivate_module(module_name)
    
    def is_module_active(self, module_name: str) -> bool:
        """
        Check if a module is currently active.
        
        Parameters
        ----------
        module_name : str
            Name of the module to check
            
        Returns
        -------
        bool
            True if module is active
        """
        return module_name in self.schema._active_modules
    
    def get_active_modules(self) -> List[str]:
        """
        Get list of currently active module names.
        
        Returns
        -------
        List[str]
            Sorted list of active module names
        """
        return self.schema.active_modules
    
    def missing_required_fields(self) -> List[Any]:
        """
        Get required fields that don't have values.
        
        Wrapper around ActiveSchema.missing_required_fields() for direct access.
        
        Returns
        -------
        List[FieldDefinition]
            List of missing required fields
        """
        return self.schema.missing_required_fields()
    
    def set_field_value(self, field_name: str, value: Any) -> None:
        """
        Set a field value in the schema.
        
        Wrapper around ActiveSchema.set_field_value() for direct access.
        
        Parameters
        ----------
        field_name : str
            Name of the field to set
        value : Any
            Value to set
        """
        self.schema.set_field_value(field_name, value)
    
    def get_field_value(self, field_name: str) -> Any:
        """
        Get a field value from the schema.
        
        Wrapper around ActiveSchema.get_field_value() for direct access.
        
        Parameters
        ----------
        field_name : str
            Name of the field to get
            
        Returns
        -------
        Any
            The field value, or None if not set
        """
        return self.schema.get_field_value(field_name)
    
    def has_field_value(self, field_name: str) -> bool:
        """
        Check if a field has a value set.
        
        Wrapper around ActiveSchema.has_field_value() for direct access.
        
        Parameters
        ----------
        field_name : str
            Name of the field to check
            
        Returns
        -------
        bool
            True if the field has a value set
        """
        return self.schema.has_field_value(field_name)
    
    def update_from_user_turn(self, text: str) -> List[str]:
        """
        Update schema based on user input.
        
        Scans text for trigger words and activates relevant modules.
        
        Parameters
        ----------
        text : str
            User input text
            
        Returns
        -------
        List[str]
            Names of modules that were activated
        """
        activated = []
        
        # Find modules to activate based on triggers
        module_names = get_modules_for_triggers(text)
        
        for name in module_names:
            if self.schema.activate_module(name):
                activated.append(name)
        
        return activated
    
    def apply_patch(self, patch: SchemaPatch, validate: bool = True) -> Tuple[bool, str]:
        """
        Apply a schema patch.
        
        Parameters
        ----------
        patch : SchemaPatch
            The patch to apply
        validate : bool
            Whether to validate the patch first
            
        Returns
        -------
        Tuple[bool, str]
            (success, message)
        """
        if validate:
            from .schema_patch_validator import validate_schema_patch
            is_valid, error = validate_schema_patch(patch, self.schema)
            if not is_valid:
                return False, error
        
        # Apply module patches
        for mp in patch.module_patches:
            if mp.action == "activate":
                self.schema.activate_module(mp.module_name)
            elif mp.action == "deactivate":
                self.schema.deactivate_module(mp.module_name)
        
        # Apply field patches
        for fp in patch.field_patches:
            if fp.action == "set_value":
                self.schema.set_field_value(fp.field_name, fp.value)
            elif fp.action == "set_required":
                self.schema.set_field_required(fp.field_name, fp.value)
            elif fp.action == "set_default":
                # Find the field and update its default
                pass  # Defaults are handled at field definition level
            elif fp.action == "remove":
                if fp.field_name in self.schema._field_values:
                    del self.schema._field_values[fp.field_name]
        
        # Add custom fields
        for name, field_def in patch.custom_fields.items():
            self.schema.add_custom_field(name, field_def)
        
        self._patch_history.append(patch)
        return True, "Patch applied successfully"
    
    def plan_questions(
        self,
        missing: Optional[List[FieldDefinition]] = None,
        ambiguities: Optional[List[Dict[str, Any]]] = None,
        conflicts: Optional[List[Dict[str, Any]]] = None,
        max_questions: int = 5
    ) -> List[Question]:
        """
        Plan questions to ask the user.
        
        Questions are ranked by rework cost (high cost = ask first).
        
        Parameters
        ----------
        missing : List[FieldDefinition], optional
            Missing required fields (defaults to self.schema.missing_required_fields())
        ambiguities : List[Dict], optional
            Ambiguities from understanding phase
        conflicts : List[Dict], optional
            Conflicts detected in current values
        max_questions : int
            Maximum number of questions to return
            
        Returns
        -------
        List[Question]
            Ranked list of questions
        """
        questions = []
        
        # Get missing fields if not provided
        if missing is None:
            missing = self.schema.missing_required_fields()
        
        # Create questions for missing fields
        for f in missing:
            q = Question(
                field_name=f.name,
                question_text=self._generate_question_text(f),
                field_type=f.field_type,
                options=f.enum_values,
                default=f.default,
                rework_cost=f.rework_cost,
                reason=f"Required field '{f.name}' is missing",
            )
            questions.append(q)
        
        # Add questions for ambiguities
        if ambiguities:
            for amb in ambiguities:
                q = Question(
                    field_name=amb.get("field", "unknown"),
                    question_text=amb.get("description", "Please clarify"),
                    field_type=FieldType.STRING,
                    options=amb.get("options"),
                    rework_cost=amb.get("impact", "medium"),
                    reason="Ambiguity detected",
                )
                questions.append(q)
        
        # Add questions for conflicts
        if conflicts:
            for conf in conflicts:
                q = Question(
                    field_name=conf.get("field", "unknown"),
                    question_text=f"Conflict detected: {conf.get('description', 'Please resolve')}",
                    field_type=FieldType.STRING,
                    rework_cost="high",
                    reason="Conflict detected",
                )
                questions.append(q)
        
        # Sort by rework cost (high first)
        cost_order = {"high": 0, "medium": 1, "low": 2}
        questions.sort(key=lambda q: cost_order.get(q.rework_cost, 1))
        
        return questions[:max_questions]
    
    def _generate_question_text(self, field: FieldDefinition) -> str:
        """Generate a natural question for a field."""
        name_readable = field.name.replace("_", " ").replace(".", " ")
        
        if field.field_type == FieldType.ENUM and field.enum_values:
            options = ", ".join(field.enum_values)
            return f"What {name_readable} would you like? Options: {options}"
        elif field.field_type == FieldType.BOOLEAN:
            return f"Should {name_readable} be enabled?"
        elif field.field_type in (FieldType.INTEGER, FieldType.FLOAT):
            unit_str = f" (in {field.unit})" if field.unit else ""
            return f"What value for {name_readable}{unit_str}?"
        elif field.field_type in (FieldType.POINT3D, FieldType.SIZE3D):
            unit_str = f" (in {field.unit})" if field.unit else ""
            return f"What are the coordinates/dimensions for {name_readable}{unit_str}?"
        else:
            return f"Please specify {name_readable}: {field.description}"
    
    def update_from_results(
        self,
        results: Dict[str, Any]
    ) -> List[str]:
        """
        Update schema based on generation/validation results.
        
        This enables the schema to evolve based on real outcomes.
        
        Parameters
        ----------
        results : Dict[str, Any]
            Results from generation or validation containing:
            - collision_count: number of vessel collisions
            - mesh_faces: number of mesh faces
            - embedding_success: whether embedding succeeded
            - validation_failures: list of validation failures
            
        Returns
        -------
        List[str]
            Names of modules that were activated
        """
        activated = []
        
        # Collisions -> activate ComplexityBudget + stricter clearance
        collision_count = results.get("collision_count", 0)
        if collision_count > 0:
            if self.schema.activate_module("ComplexityBudgetModule"):
                activated.append("ComplexityBudgetModule")
            # Make min_clearance_m required with higher value
            self.schema.set_field_required("min_clearance_m", True)
        
        # Mesh too heavy -> activate MeshQuality
        mesh_faces = results.get("mesh_faces", 0)
        if mesh_faces > 500000:
            if self.schema.activate_module("MeshQualityModule"):
                activated.append("MeshQualityModule")
        
        # Embedding failure -> require voxel_pitch + memory_policy
        embedding_success = results.get("embedding_success", True)
        if not embedding_success:
            if self.schema.activate_module("EmbeddingModule"):
                activated.append("EmbeddingModule")
            self.schema.set_field_required("voxel_pitch_m", True)
            
            if self.schema.activate_module("ComplexityBudgetModule"):
                activated.append("ComplexityBudgetModule")
            self.schema.set_field_required("memory_policy", True)
        
        # Flow optimization requested -> activate FlowPhysics
        validation_failures = results.get("validation_failures", [])
        for failure in validation_failures:
            if "flow" in str(failure).lower() or "pressure" in str(failure).lower():
                if self.schema.activate_module("FlowPhysicsModule"):
                    activated.append("FlowPhysicsModule")
                break
        
        return activated
    
    def get_spec_summary(self) -> str:
        """
        Get a human-readable summary of the current spec.
        
        Returns
        -------
        str
            Formatted spec summary
        """
        lines = []
        
        # Active modules
        if self.schema.active_modules:
            lines.append(f"Active modules: {', '.join(self.schema.active_modules)}")
        
        # Field values by group
        for group_name, group_fields in CORE_FIELD_GROUPS.items():
            group_values = []
            for f in group_fields:
                if self.schema.has_field_value(f.name):
                    value = self.schema.get_field_value(f.name)
                    group_values.append(f"{f.name}={value}")
            
            if group_values:
                lines.append(f"{group_name}: {', '.join(group_values)}")
        
        # Module field values
        for module_name in self.schema.active_modules:
            module = ALL_MODULES.get(module_name)
            if module:
                module_values = []
                for f in module.fields:
                    if self.schema.has_field_value(f.name):
                        value = self.schema.get_field_value(f.name)
                        module_values.append(f"{f.name}={value}")
                
                if module_values:
                    lines.append(f"{module_name}: {', '.join(module_values)}")
        
        # Missing required fields
        missing = self.schema.missing_required_fields()
        if missing:
            missing_names = [f.name for f in missing]
            lines.append(f"Missing: {', '.join(missing_names)}")
        
        return "\n".join(lines) if lines else "No fields set yet"
    
    def export_design_spec_dict(self) -> Dict[str, Any]:
        """
        Export current schema values as a DesignSpec-compatible dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary that can be used to create a DesignSpec
        """
        spec = {}
        
        # Domain
        domain_type = self.schema.get_field_value("domain_type")
        domain_size = self.schema.get_field_value("domain_size_m")
        domain_center = self.schema.get_field_value("domain_center_m")
        
        if domain_type and domain_size:
            spec["domain"] = {
                "type": domain_type,
                "size_m": domain_size,
                "center_m": domain_center or [0.0, 0.0, 0.0],
            }
        
        # Trees (inlets/outlets)
        num_inlets = self.schema.get_field_value("num_inlets") or 1
        inlet_radius = self.schema.get_field_value("inlet_radius_m")
        inlet_positions = self.schema.get_field_value("inlet_positions_m")
        
        trees = []
        for i in range(num_inlets):
            tree = {
                "inlet_radius_m": inlet_radius,
            }
            if inlet_positions and i < len(inlet_positions):
                tree["inlet_position_m"] = inlet_positions[i]
            
            # Add topology params if TopologyModule is active
            if "TopologyModule" in self.schema.active_modules:
                target_terminals = self.schema.get_field_value("target_terminals")
                if target_terminals:
                    tree["target_terminals"] = target_terminals
            
            trees.append(tree)
        
        if trees:
            spec["trees"] = trees
        
        # Constraints
        min_radius = self.schema.get_field_value("min_radius_m")
        min_clearance = self.schema.get_field_value("min_clearance_m")
        
        if min_radius or min_clearance:
            spec["constraints"] = {}
            if min_radius:
                spec["constraints"]["min_radius_m"] = min_radius
            if min_clearance:
                spec["constraints"]["min_clearance_m"] = min_clearance
        
        # Embedding params
        if "EmbeddingModule" in self.schema.active_modules:
            voxel_pitch = self.schema.get_field_value("voxel_pitch_m")
            if voxel_pitch:
                spec["embedding"] = {"voxel_pitch_m": voxel_pitch}
        
        # Output contract
        output_format = self.schema.get_field_value("output_format")
        output_units = self.schema.get_field_value("output_units")
        
        if output_format or output_units:
            spec["output"] = {}
            if output_format:
                spec["output"]["format"] = output_format
            if output_units:
                spec["output"]["units"] = output_units
        
        return spec
