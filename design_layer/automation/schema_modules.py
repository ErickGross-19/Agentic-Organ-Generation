"""
Schema Modules Definition

Defines the whitelisted schema modules that can be activated/deactivated
by the LLM based on the object type and conversation context.

Core Schema (always required):
- object identity + version
- frame of reference
- domain
- I/O definitions (inlets/outlets)
- hard constraints: min radius + min clearance
- output contract: expected artifacts, units
- minimal acceptance criteria

Schema Modules (opt-in bundles):
- TopologyModule: terminals, depth, branching
- GeometryStyleModule: tortuosity, angles, length ranges
- PerfusionZonesModule: regions, density multipliers, keep-outs
- ComplexityBudgetModule: max nodes/segments/runtime/memory policy
- MeshQualityModule: watertight requirement, decimation caps
- EmbeddingModule: voxel pitch, shell thickness, void/shell choices
- FlowPhysicsModule: only when user asks or when iterating for flow
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from enum import Enum


# =============================================================================
# Field Definitions
# =============================================================================

class FieldType(Enum):
    """Types of schema fields."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    ENUM = "enum"
    POINT3D = "point3d"
    SIZE3D = "size3d"


@dataclass
class FieldDefinition:
    """Definition of a schema field."""
    name: str
    field_type: FieldType
    description: str
    required: bool = False
    default: Any = None
    unit: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    enum_values: Optional[List[str]] = None
    rework_cost: str = "medium"  # "high", "medium", "low" - for question ranking
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.field_type.value,
            "description": self.description,
            "required": self.required,
            "default": self.default,
            "unit": self.unit,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "enum_values": self.enum_values,
            "rework_cost": self.rework_cost,
        }


# =============================================================================
# Module Definitions
# =============================================================================

@dataclass
class SchemaModule:
    """Definition of a schema module."""
    name: str
    description: str
    fields: List[FieldDefinition]
    activation_triggers: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    def get_required_fields(self) -> List[FieldDefinition]:
        return [f for f in self.fields if f.required]
    
    def get_optional_fields(self) -> List[FieldDefinition]:
        return [f for f in self.fields if not f.required]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "fields": [f.to_dict() for f in self.fields],
            "activation_triggers": self.activation_triggers,
            "dependencies": self.dependencies,
        }


# =============================================================================
# Core Schema (Always Required)
# =============================================================================

CORE_IDENTITY_FIELDS = [
    FieldDefinition(
        name="object_name",
        field_type=FieldType.STRING,
        description="Unique name for this object within the project",
        required=True,
        rework_cost="high",
    ),
    FieldDefinition(
        name="object_type",
        field_type=FieldType.ENUM,
        description="Type of organ/structure being generated",
        required=True,
        enum_values=["liver", "kidney", "lung", "heart", "generic"],
        rework_cost="high",
    ),
    FieldDefinition(
        name="version",
        field_type=FieldType.STRING,
        description="Version identifier for this design iteration",
        required=False,
        default="1.0",
        rework_cost="low",
    ),
]

CORE_FRAME_OF_REFERENCE_FIELDS = [
    FieldDefinition(
        name="coordinate_system",
        field_type=FieldType.ENUM,
        description="Coordinate system convention",
        required=False,
        default="xyz_standard",
        enum_values=["xyz_standard", "anatomical", "custom"],
        rework_cost="high",
    ),
    FieldDefinition(
        name="x_axis_meaning",
        field_type=FieldType.STRING,
        description="What the X axis represents (e.g., 'width', 'left-right')",
        required=False,
        default="width",
        rework_cost="high",
    ),
    FieldDefinition(
        name="y_axis_meaning",
        field_type=FieldType.STRING,
        description="What the Y axis represents (e.g., 'depth', 'anterior-posterior')",
        required=False,
        default="depth",
        rework_cost="high",
    ),
    FieldDefinition(
        name="z_axis_meaning",
        field_type=FieldType.STRING,
        description="What the Z axis represents (e.g., 'height', 'superior-inferior')",
        required=False,
        default="height",
        rework_cost="high",
    ),
]

CORE_DOMAIN_FIELDS = [
    FieldDefinition(
        name="domain_type",
        field_type=FieldType.ENUM,
        description="Shape of the domain volume",
        required=True,
        enum_values=["box", "ellipsoid", "cylinder", "custom_mesh"],
        rework_cost="high",
    ),
    FieldDefinition(
        name="domain_size_m",
        field_type=FieldType.SIZE3D,
        description="Size of domain bounding box in meters (x, y, z)",
        required=True,
        unit="m",
        rework_cost="high",
    ),
    FieldDefinition(
        name="domain_center_m",
        field_type=FieldType.POINT3D,
        description="Center position of domain in meters",
        required=False,
        default=[0.0, 0.0, 0.0],
        unit="m",
        rework_cost="medium",
    ),
]

CORE_IO_FIELDS = [
    FieldDefinition(
        name="num_inlets",
        field_type=FieldType.INTEGER,
        description="Number of inlet ports",
        required=True,
        min_value=1,
        rework_cost="high",
    ),
    FieldDefinition(
        name="num_outlets",
        field_type=FieldType.INTEGER,
        description="Number of outlet ports",
        required=True,
        min_value=0,
        rework_cost="high",
    ),
    FieldDefinition(
        name="inlet_positions_m",
        field_type=FieldType.ARRAY,
        description="Positions of inlet ports in meters",
        required=False,
        unit="m",
        rework_cost="high",
    ),
    FieldDefinition(
        name="outlet_positions_m",
        field_type=FieldType.ARRAY,
        description="Positions of outlet ports in meters",
        required=False,
        unit="m",
        rework_cost="high",
    ),
    FieldDefinition(
        name="inlet_radius_m",
        field_type=FieldType.FLOAT,
        description="Radius of inlet ports in meters",
        required=True,
        unit="m",
        min_value=0.0001,
        rework_cost="high",
    ),
    FieldDefinition(
        name="outlet_radius_m",
        field_type=FieldType.FLOAT,
        description="Radius of outlet ports in meters",
        required=False,
        unit="m",
        min_value=0.0001,
        rework_cost="medium",
    ),
]

CORE_CONSTRAINTS_FIELDS = [
    FieldDefinition(
        name="min_radius_m",
        field_type=FieldType.FLOAT,
        description="Minimum vessel radius in meters",
        required=True,
        unit="m",
        min_value=0.00005,
        default=0.0001,
        rework_cost="high",
    ),
    FieldDefinition(
        name="min_clearance_m",
        field_type=FieldType.FLOAT,
        description="Minimum clearance between vessels in meters",
        required=True,
        unit="m",
        min_value=0.0001,
        default=0.0002,
        rework_cost="high",
    ),
]

CORE_OUTPUT_CONTRACT_FIELDS = [
    FieldDefinition(
        name="output_format",
        field_type=FieldType.ENUM,
        description="Primary output format",
        required=True,
        default="stl",
        enum_values=["stl", "obj", "ply", "vtk"],
        rework_cost="low",
    ),
    FieldDefinition(
        name="output_units",
        field_type=FieldType.ENUM,
        description="Units for output files",
        required=True,
        default="mm",
        enum_values=["m", "mm", "um"],
        rework_cost="low",
    ),
    FieldDefinition(
        name="generate_domain_with_void",
        field_type=FieldType.BOOLEAN,
        description="Generate domain mesh with vascular void",
        required=False,
        default=True,
        rework_cost="low",
    ),
    FieldDefinition(
        name="generate_surface_mesh",
        field_type=FieldType.BOOLEAN,
        description="Generate surface mesh of vascular network",
        required=False,
        default=True,
        rework_cost="low",
    ),
]

CORE_ACCEPTANCE_FIELDS = [
    FieldDefinition(
        name="min_coverage_fraction",
        field_type=FieldType.FLOAT,
        description="Minimum tissue coverage fraction (0-1)",
        required=False,
        default=0.8,
        min_value=0.0,
        max_value=1.0,
        rework_cost="medium",
    ),
    FieldDefinition(
        name="max_collision_count",
        field_type=FieldType.INTEGER,
        description="Maximum allowed vessel collisions",
        required=False,
        default=0,
        min_value=0,
        rework_cost="medium",
    ),
]


# =============================================================================
# Optional Schema Modules
# =============================================================================

TOPOLOGY_MODULE = SchemaModule(
    name="TopologyModule",
    description="Controls network topology: terminals, depth, branching patterns",
    fields=[
        FieldDefinition(
            name="target_terminals",
            field_type=FieldType.INTEGER,
            description="Target number of terminal branches",
            required=True,
            min_value=1,
            rework_cost="medium",
        ),
        FieldDefinition(
            name="max_depth",
            field_type=FieldType.INTEGER,
            description="Maximum branching depth from inlet",
            required=False,
            min_value=1,
            rework_cost="medium",
        ),
        FieldDefinition(
            name="branching_factor",
            field_type=FieldType.FLOAT,
            description="Average number of children per branch",
            required=False,
            default=2.0,
            min_value=1.5,
            max_value=4.0,
            rework_cost="low",
        ),
        FieldDefinition(
            name="topology_style",
            field_type=FieldType.ENUM,
            description="Overall topology pattern",
            required=False,
            default="tree",
            enum_values=["tree", "mesh", "hybrid"],
            rework_cost="high",
        ),
    ],
    activation_triggers=["terminal", "branch", "depth", "tree", "network"],
)

GEOMETRY_STYLE_MODULE = SchemaModule(
    name="GeometryStyleModule",
    description="Controls vessel geometry: tortuosity, angles, lengths",
    fields=[
        FieldDefinition(
            name="tortuosity",
            field_type=FieldType.FLOAT,
            description="Vessel tortuosity factor (1.0 = straight)",
            required=False,
            default=1.0,
            min_value=1.0,
            max_value=2.0,
            rework_cost="low",
        ),
        FieldDefinition(
            name="min_bifurcation_angle_deg",
            field_type=FieldType.FLOAT,
            description="Minimum angle between child branches in degrees",
            required=False,
            default=30.0,
            unit="degrees",
            min_value=15.0,
            max_value=90.0,
            rework_cost="low",
        ),
        FieldDefinition(
            name="max_bifurcation_angle_deg",
            field_type=FieldType.FLOAT,
            description="Maximum angle between child branches in degrees",
            required=False,
            default=90.0,
            unit="degrees",
            min_value=30.0,
            max_value=180.0,
            rework_cost="low",
        ),
        FieldDefinition(
            name="length_to_radius_ratio",
            field_type=FieldType.FLOAT,
            description="Typical segment length as multiple of radius",
            required=False,
            default=10.0,
            min_value=2.0,
            max_value=50.0,
            rework_cost="low",
        ),
        FieldDefinition(
            name="murray_exponent",
            field_type=FieldType.FLOAT,
            description="Murray's law exponent (typically 3.0)",
            required=False,
            default=3.0,
            min_value=2.0,
            max_value=4.0,
            rework_cost="low",
        ),
    ],
    activation_triggers=["tortuous", "angle", "smooth", "curved", "straight", "geometry"],
)

PERFUSION_ZONES_MODULE = SchemaModule(
    name="PerfusionZonesModule",
    description="Defines perfusion zones with different density requirements",
    fields=[
        FieldDefinition(
            name="zones",
            field_type=FieldType.ARRAY,
            description="List of perfusion zone definitions",
            required=True,
            rework_cost="medium",
        ),
        FieldDefinition(
            name="default_density",
            field_type=FieldType.ENUM,
            description="Default density for unzoned regions",
            required=False,
            default="medium",
            enum_values=["sparse", "medium", "dense"],
            rework_cost="low",
        ),
        FieldDefinition(
            name="keepout_regions",
            field_type=FieldType.ARRAY,
            description="Regions where vessels should not grow",
            required=False,
            rework_cost="medium",
        ),
    ],
    activation_triggers=["zone", "region", "density", "perfusion", "area", "keepout"],
)

COMPLEXITY_BUDGET_MODULE = SchemaModule(
    name="ComplexityBudgetModule",
    description="Limits computational complexity and resource usage",
    fields=[
        FieldDefinition(
            name="max_nodes",
            field_type=FieldType.INTEGER,
            description="Maximum number of nodes in network",
            required=False,
            min_value=10,
            rework_cost="low",
        ),
        FieldDefinition(
            name="max_segments",
            field_type=FieldType.INTEGER,
            description="Maximum number of segments in network",
            required=False,
            min_value=10,
            rework_cost="low",
        ),
        FieldDefinition(
            name="max_generation_time_s",
            field_type=FieldType.FLOAT,
            description="Maximum generation time in seconds",
            required=False,
            default=300.0,
            unit="s",
            min_value=10.0,
            rework_cost="low",
        ),
        FieldDefinition(
            name="memory_policy",
            field_type=FieldType.ENUM,
            description="Memory usage policy",
            required=False,
            default="balanced",
            enum_values=["minimal", "balanced", "performance"],
            rework_cost="low",
        ),
    ],
    activation_triggers=["budget", "limit", "max", "complexity", "performance", "memory", "time"],
)

MESH_QUALITY_MODULE = SchemaModule(
    name="MeshQualityModule",
    description="Controls mesh quality and export settings",
    fields=[
        FieldDefinition(
            name="require_watertight",
            field_type=FieldType.BOOLEAN,
            description="Require watertight mesh output",
            required=False,
            default=True,
            rework_cost="low",
        ),
        FieldDefinition(
            name="max_faces",
            field_type=FieldType.INTEGER,
            description="Maximum number of mesh faces",
            required=False,
            min_value=1000,
            rework_cost="low",
        ),
        FieldDefinition(
            name="decimation_ratio",
            field_type=FieldType.FLOAT,
            description="Mesh decimation ratio (1.0 = no decimation)",
            required=False,
            default=1.0,
            min_value=0.1,
            max_value=1.0,
            rework_cost="low",
        ),
        FieldDefinition(
            name="mesh_repair",
            field_type=FieldType.BOOLEAN,
            description="Attempt automatic mesh repair",
            required=False,
            default=True,
            rework_cost="low",
        ),
    ],
    activation_triggers=["mesh", "quality", "watertight", "faces", "decimation", "repair"],
)

EMBEDDING_MODULE = SchemaModule(
    name="EmbeddingModule",
    description="Controls domain embedding and voxelization",
    fields=[
        FieldDefinition(
            name="voxel_pitch_m",
            field_type=FieldType.FLOAT,
            description="Voxel pitch for embedding in meters",
            required=True,
            default=0.0003,
            unit="m",
            min_value=0.00001,
            max_value=0.01,
            rework_cost="medium",
        ),
        FieldDefinition(
            name="shell_thickness_m",
            field_type=FieldType.FLOAT,
            description="Shell thickness around domain in meters",
            required=False,
            default=0.001,
            unit="m",
            min_value=0.0001,
            rework_cost="low",
        ),
        FieldDefinition(
            name="void_style",
            field_type=FieldType.ENUM,
            description="How to create vascular void",
            required=False,
            default="boolean_subtract",
            enum_values=["boolean_subtract", "voxel_carve"],
            rework_cost="low",
        ),
        FieldDefinition(
            name="port_extension_m",
            field_type=FieldType.FLOAT,
            description="Extension of ports beyond domain surface",
            required=False,
            default=0.002,
            unit="m",
            min_value=0.0,
            rework_cost="low",
        ),
    ],
    activation_triggers=["embed", "voxel", "shell", "void", "pitch", "resolution"],
)

FLOW_PHYSICS_MODULE = SchemaModule(
    name="FlowPhysicsModule",
    description="Flow physics constraints and optimization",
    fields=[
        FieldDefinition(
            name="target_reynolds",
            field_type=FieldType.FLOAT,
            description="Target Reynolds number for flow",
            required=False,
            min_value=0.1,
            max_value=2300.0,
            rework_cost="medium",
        ),
        FieldDefinition(
            name="inlet_flow_rate_m3s",
            field_type=FieldType.FLOAT,
            description="Inlet volumetric flow rate in m³/s",
            required=False,
            unit="m³/s",
            min_value=0.0,
            rework_cost="medium",
        ),
        FieldDefinition(
            name="pressure_drop_pa",
            field_type=FieldType.FLOAT,
            description="Target pressure drop in Pascals",
            required=False,
            unit="Pa",
            min_value=0.0,
            rework_cost="medium",
        ),
        FieldDefinition(
            name="fluid_viscosity_pas",
            field_type=FieldType.FLOAT,
            description="Fluid dynamic viscosity in Pa·s",
            required=False,
            default=0.001,
            unit="Pa·s",
            min_value=0.0001,
            rework_cost="low",
        ),
        FieldDefinition(
            name="optimize_for_flow",
            field_type=FieldType.BOOLEAN,
            description="Optimize network for flow characteristics",
            required=False,
            default=False,
            rework_cost="medium",
        ),
    ],
    activation_triggers=["flow", "pressure", "reynolds", "viscosity", "optimize", "physics"],
    dependencies=["TopologyModule"],
)


# =============================================================================
# Module Registry
# =============================================================================

ALL_MODULES: Dict[str, SchemaModule] = {
    "TopologyModule": TOPOLOGY_MODULE,
    "GeometryStyleModule": GEOMETRY_STYLE_MODULE,
    "PerfusionZonesModule": PERFUSION_ZONES_MODULE,
    "ComplexityBudgetModule": COMPLEXITY_BUDGET_MODULE,
    "MeshQualityModule": MESH_QUALITY_MODULE,
    "EmbeddingModule": EMBEDDING_MODULE,
    "FlowPhysicsModule": FLOW_PHYSICS_MODULE,
}

CORE_FIELD_GROUPS = {
    "identity": CORE_IDENTITY_FIELDS,
    "frame_of_reference": CORE_FRAME_OF_REFERENCE_FIELDS,
    "domain": CORE_DOMAIN_FIELDS,
    "inlets_outlets": CORE_IO_FIELDS,
    "constraints": CORE_CONSTRAINTS_FIELDS,
    "output_contract": CORE_OUTPUT_CONTRACT_FIELDS,
    "acceptance_criteria": CORE_ACCEPTANCE_FIELDS,
}


def get_all_core_fields() -> List[FieldDefinition]:
    """Get all core schema fields."""
    fields = []
    for group_fields in CORE_FIELD_GROUPS.values():
        fields.extend(group_fields)
    return fields


def get_module(name: str) -> Optional[SchemaModule]:
    """Get a module by name."""
    return ALL_MODULES.get(name)


def get_modules_for_triggers(text: str) -> List[str]:
    """
    Get module names that should be activated based on trigger words in text.
    
    Parameters
    ----------
    text : str
        User text to scan for trigger words
        
    Returns
    -------
    List[str]
        Names of modules that should be activated
    """
    text_lower = text.lower()
    activated = []
    
    for name, module in ALL_MODULES.items():
        for trigger in module.activation_triggers:
            if trigger in text_lower:
                activated.append(name)
                break
    
    return activated
