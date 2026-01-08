"""
Single Agent Organ Generator V2 Workflow

This module implements the "Single Agent Organ Generator V2" workflow - a stateful,
interactive workflow for organ structure generation using LLM agents.

V2 introduces the "Interpret -> Plan -> Ask" agent pattern:
- Agent dialogue system for understanding user intent
- Dynamic schema with activatable modules
- LLM healthcheck and circuit breaker for reliability
- Iteration feedback integration

The workflow follows these steps:
0. PROJECT_INIT: Ask user for project name and global defaults
1. OBJECT_PLANNING: Ask how many objects and create object folders
2. FRAME_OF_REFERENCE: Establish coordinate conventions per object
3. REQUIREMENTS_CAPTURE: Schema-gated requirements gathering
4. SPEC_COMPILATION: Compile requirements to DesignSpec
5. GENERATION: Execute generation within object folder
6. ANALYSIS_VALIDATION: Analyze and validate generated structure
7. ITERATION: Accept user critique and iterate
8. FINALIZATION: Embed and produce final outputs
9. COMPLETE: Close project

Key Features:
- Per-object folder structure with versioned artifacts
- Schema-gated requirements capture with dynamic modules
- Agent dialogue: Understand -> Plan -> Ask flow
- Fixed frame of reference before spatial discussions
- Ability to pull context from previous attempts
- Deterministic, reproducible generation
- LLM healthcheck prevents infinite loops

Usage:
    from automation.workflow import SingleAgentOrganGeneratorV2
    from automation.agent_runner import create_agent
    
    agent = create_agent(provider="openai", model="gpt-4")
    workflow = SingleAgentOrganGeneratorV2(agent)
    workflow.run()
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
from pathlib import Path
import json
import time
import os

from .agent_runner import AgentRunner, TaskResult, TaskStatus
from .execution_modes import (
    ExecutionMode,
    DEFAULT_EXECUTION_MODE,
    parse_execution_mode,
    should_write_script,
    should_pause_for_review,
    should_execute,
)
from .script_writer import write_script, get_run_command
from .review_gate import run_review_gate, ReviewAction
from .subprocess_runner import run_script, print_run_summary
from .artifact_verifier import verify_generation_stage, save_manifest, print_verification_summary
from .llm_healthcheck import assert_llm_ready, MissingCredentialsError, ProviderMisconfiguredError, FatalLLMError, QuotaExhaustedError

# Import DesignSpec classes for deterministic spec compilation
from generation.specs.design_spec import (
    DesignSpec,
    BoxSpec,
    EllipsoidSpec,
    InletSpec,
    OutletSpec,
    ColonizationSpec,
    TreeSpec,
)


class WorkflowStuckError(Exception):
    """
    Raised when the workflow is stuck in an infinite loop.
    
    This error provides diagnostic information about why the workflow
    is stuck, such as repeated state/question combinations.
    """
    
    def __init__(
        self,
        state: str,
        question_id: Optional[str],
        repetition_count: int,
        diagnosis: str,
    ):
        self.state = state
        self.question_id = question_id
        self.repetition_count = repetition_count
        self.diagnosis = diagnosis
        message = (
            f"Workflow stuck: state '{state}' "
            f"(question: {question_id or 'N/A'}) "
            f"repeated {repetition_count} times.\n"
            f"Diagnosis: {diagnosis}"
        )
        super().__init__(message)
from .agent_dialogue import (
    understand_object,
    propose_plans,
    format_user_summary,
    format_living_spec,
    generate_requirements_summary,
    UnderstandingReport,
    PlanOption,
)
from .schema_manager import SchemaManager, ActiveSchema
from .reactive_prompt import (
    ReactivePromptSession,
    TurnIntent,
    interpret_user_turn,
    GlobalDefaults,
    get_question_help,
    create_question_help,
    WORKFLOW_QUESTION_HELP,
)


class WorkflowState(Enum):
    """States in the Single Agent Organ Generator V2 workflow."""
    PROJECT_INIT = "project_init"
    OBJECT_PLANNING = "object_planning"
    FRAME_OF_REFERENCE = "frame_of_reference"
    REQUIREMENTS_CAPTURE = "requirements_capture"
    SPEC_COMPILATION = "spec_compilation"
    GENERATION = "generation"
    ANALYSIS_VALIDATION = "analysis_validation"
    ITERATION = "iteration"
    FINALIZATION = "finalization"
    COMPLETE = "complete"


# =============================================================================
# Requirements Schema Data Classes (9 Sections)
# =============================================================================

@dataclass
class IdentitySection:
    """Section 1: Identity information for an object."""
    object_name: str = ""
    object_slug: str = ""
    version: int = 1
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "IdentitySection":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class FrameOfReferenceSection:
    """Section 2: Frame of reference and units.
    
    The `confirmed` field tracks whether the user has explicitly confirmed
    the coordinate convention. This is important because spatial terms like
    "left/right/top/bottom" are ambiguous until the frame is confirmed.
    """
    origin: str = "domain.center"
    axes: Dict[str, str] = field(default_factory=lambda: {
        "x": "width",
        "y": "depth", 
        "z": "height"
    })
    viewpoint: str = "front"
    units_internal: str = "m"
    units_export: str = "mm"
    confirmed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "origin": self.origin,
            "axes": self.axes,
            "viewpoint": self.viewpoint,
            "units_internal": self.units_internal,
            "units_export": self.units_export,
            "confirmed": self.confirmed,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FrameOfReferenceSection":
        return cls(
            origin=d.get("origin", "domain.center"),
            axes=d.get("axes", {"x": "width", "y": "depth", "z": "height"}),
            viewpoint=d.get("viewpoint", "front"),
            units_internal=d.get("units_internal", "m"),
            units_export=d.get("units_export", "mm"),
            confirmed=d.get("confirmed", False),
        )


@dataclass
class DomainSection:
    """Section 3: Domain specification."""
    type: str = "box"
    size_m: Tuple[float, float, float] = (0.02, 0.06, 0.03)
    center_m: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    margin_m: float = 0.001
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "size_m": list(self.size_m),
            "center_m": list(self.center_m),
            "margin_m": self.margin_m,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DomainSection":
        return cls(
            type=d.get("type", "box"),
            size_m=tuple(d.get("size_m", [0.02, 0.06, 0.03])),
            center_m=tuple(d.get("center_m", [0.0, 0.0, 0.0])),
            margin_m=d.get("margin_m", 0.001),
        )


@dataclass
class PortSpec:
    """Specification for an inlet or outlet port."""
    name: str = ""
    position_m: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius_m: float = 0.001
    direction_unit: Optional[Tuple[float, float, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "position_m": list(self.position_m),
            "radius_m": self.radius_m,
            "direction_unit": list(self.direction_unit) if self.direction_unit else None,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PortSpec":
        return cls(
            name=d.get("name", ""),
            position_m=tuple(d.get("position_m", [0.0, 0.0, 0.0])),
            radius_m=d.get("radius_m", 0.001),
            direction_unit=tuple(d["direction_unit"]) if d.get("direction_unit") else None,
        )


@dataclass
class InletsOutletsSection:
    """Section 4: Inlets and outlets specification."""
    inlets: List[PortSpec] = field(default_factory=list)
    outlets: List[PortSpec] = field(default_factory=list)
    placement_rule: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "inlets": [i.to_dict() for i in self.inlets],
            "outlets": [o.to_dict() for o in self.outlets],
            "placement_rule": self.placement_rule,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "InletsOutletsSection":
        return cls(
            inlets=[PortSpec.from_dict(i) for i in d.get("inlets", [])],
            outlets=[PortSpec.from_dict(o) for o in d.get("outlets", [])],
            placement_rule=d.get("placement_rule", ""),
        )


@dataclass
class ZoneDensity:
    """Subregion with density multiplier for perfusion zones."""
    name: str = ""
    bounds_m: Optional[Dict[str, Tuple[float, float]]] = None
    density_multiplier: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "bounds_m": {k: list(v) for k, v in self.bounds_m.items()} if self.bounds_m else None,
            "density_multiplier": self.density_multiplier,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ZoneDensity":
        bounds = None
        if d.get("bounds_m"):
            bounds = {k: tuple(v) for k, v in d["bounds_m"].items()}
        return cls(
            name=d.get("name", ""),
            bounds_m=bounds,
            density_multiplier=d.get("density_multiplier", 1.0),
        )


@dataclass
class TopologySection:
    """Section 5: Topology intent.
    
    topology_kind determines which questions are relevant:
    - "path": Simple inlet→outlet channel, no branching questions
    - "tree": Full tree structure, all branching/terminal questions relevant
    - "loop": Looping structure with recirculation
    - "multi_tree": Multiple independent trees
    """
    style: str = "tree"  # branching style: balanced, aggressive_early, space_filling
    topology_kind: str = "tree"  # "path", "tree", "loop", "multi_tree"
    target_terminals: Optional[int] = None
    max_depth: Optional[int] = None
    branching_factor_range: Tuple[int, int] = (2, 2)
    zone_density: List[ZoneDensity] = field(default_factory=list)
    # New fields for topology-specific questions
    terminal_mode: str = "a"  # "a" = inlet→terminals, "b" = inlet→outlet with branches
    loop_count: Optional[int] = None  # for loop topology
    loop_style: Optional[str] = None  # single_loop, mesh, ladder
    tree_count: Optional[int] = None  # for multi_tree topology
    shared_inlet: bool = False  # for multi_tree: do trees share an inlet?
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "style": self.style,
            "topology_kind": self.topology_kind,
            "target_terminals": self.target_terminals,
            "max_depth": self.max_depth,
            "branching_factor_range": list(self.branching_factor_range),
            "zone_density": [z.to_dict() for z in self.zone_density],
            "terminal_mode": self.terminal_mode,
            "loop_count": self.loop_count,
            "loop_style": self.loop_style,
            "tree_count": self.tree_count,
            "shared_inlet": self.shared_inlet,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TopologySection":
        return cls(
            style=d.get("style", "tree"),
            topology_kind=d.get("topology_kind", "tree"),
            target_terminals=d.get("target_terminals"),
            max_depth=d.get("max_depth"),
            branching_factor_range=tuple(d.get("branching_factor_range", [2, 2])),
            zone_density=[ZoneDensity.from_dict(z) for z in d.get("zone_density", [])],
            terminal_mode=d.get("terminal_mode", "a"),
            loop_count=d.get("loop_count"),
            loop_style=d.get("loop_style"),
            tree_count=d.get("tree_count"),
            shared_inlet=d.get("shared_inlet", False),
        )


@dataclass
class GeometrySection:
    """Section 6: Geometry intent."""
    segment_length_m: Dict[str, float] = field(default_factory=lambda: {"min": 0.0005, "max": 0.005})
    tortuosity: float = 0.0
    branch_angle_deg: Dict[str, float] = field(default_factory=lambda: {"min": 30.0, "max": 90.0})
    radius_profile: str = "murray"
    radius_bounds_m: Dict[str, Optional[float]] = field(default_factory=lambda: {"min": 0.0001, "max": None})
    # New fields for topology-specific questions
    tapering: str = "murray"  # murray, linear, fixed
    route_type: Optional[str] = None  # for path: straight, single_bend, s_curve, via_points
    bend_radius_m: Optional[float] = None  # minimum bend radius for curved paths
    wall_thickness_m: Optional[float] = None  # wall thickness for hollow channels
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_length_m": self.segment_length_m,
            "tortuosity": self.tortuosity,
            "branch_angle_deg": self.branch_angle_deg,
            "radius_profile": self.radius_profile,
            "radius_bounds_m": self.radius_bounds_m,
            "tapering": self.tapering,
            "route_type": self.route_type,
            "bend_radius_m": self.bend_radius_m,
            "wall_thickness_m": self.wall_thickness_m,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GeometrySection":
        return cls(
            segment_length_m=d.get("segment_length_m", {"min": 0.0005, "max": 0.005}),
            tortuosity=d.get("tortuosity", 0.0),
            branch_angle_deg=d.get("branch_angle_deg", {"min": 30.0, "max": 90.0}),
            radius_profile=d.get("radius_profile", "murray"),
            radius_bounds_m=d.get("radius_bounds_m", {"min": 0.0001, "max": None}),
            tapering=d.get("tapering", "murray"),
            route_type=d.get("route_type"),
            bend_radius_m=d.get("bend_radius_m"),
            wall_thickness_m=d.get("wall_thickness_m"),
        )


@dataclass
class ConstraintsSection:
    """Section 7: Constraints."""
    min_radius_m: float = 0.0001
    min_clearance_m: float = 0.0002
    avoid_self_intersection: bool = True
    boundary_buffer_m: float = 0.001
    max_segments: Optional[int] = None
    max_runtime_s: Optional[float] = None
    # New fields for topology-specific questions
    boundary_margin_m: Optional[float] = None  # how far structures stay from walls
    ban_self_intersection: bool = True  # hard ban on self-intersections
    enforce_clearance: bool = True  # hard min clearance enforcement
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ConstraintsSection":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class EmbeddingExportSection:
    """Section 8: Embedding and export settings."""
    domain_override: Optional[Dict[str, Any]] = None
    voxel_pitch_m: Optional[float] = None
    shell_thickness_m: Optional[float] = None
    output_void: bool = True
    stl_units: str = "mm"
    # New fields for topology-specific questions
    embed_requested: bool = False  # whether embedding is requested
    min_feature_size_m: Optional[float] = None  # minimum printable feature size / nozzle diameter
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain_override": self.domain_override,
            "voxel_pitch_m": self.voxel_pitch_m,
            "shell_thickness_m": self.shell_thickness_m,
            "output_void": self.output_void,
            "stl_units": self.stl_units,
            "embed_requested": self.embed_requested,
            "min_feature_size_m": self.min_feature_size_m,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EmbeddingExportSection":
        return cls(
            domain_override=d.get("domain_override"),
            voxel_pitch_m=d.get("voxel_pitch_m"),
            shell_thickness_m=d.get("shell_thickness_m"),
            output_void=d.get("output_void", True),
            stl_units=d.get("stl_units", "mm"),
            embed_requested=d.get("embed_requested", False),
            min_feature_size_m=d.get("min_feature_size_m"),
        )


@dataclass
class AcceptanceCriteriaSection:
    """Section 9: Acceptance criteria."""
    min_radius_m: Optional[float] = None
    min_clearance_m: Optional[float] = None
    terminals_range: Optional[Tuple[int, int]] = None
    mesh_watertight_required: bool = True
    flow_metrics_enabled: bool = False
    custom_criteria: Dict[str, Any] = field(default_factory=dict)
    # New fields for topology-specific questions
    watertight_required: bool = True  # require watertight mesh
    terminal_range: Optional[str] = None  # acceptable terminal count range as string (e.g., "50-100")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_radius_m": self.min_radius_m,
            "min_clearance_m": self.min_clearance_m,
            "terminals_range": list(self.terminals_range) if self.terminals_range else None,
            "mesh_watertight_required": self.mesh_watertight_required,
            "flow_metrics_enabled": self.flow_metrics_enabled,
            "custom_criteria": self.custom_criteria,
            "watertight_required": self.watertight_required,
            "terminal_range": self.terminal_range,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AcceptanceCriteriaSection":
        return cls(
            min_radius_m=d.get("min_radius_m"),
            min_clearance_m=d.get("min_clearance_m"),
            terminals_range=tuple(d["terminals_range"]) if d.get("terminals_range") else None,
            mesh_watertight_required=d.get("mesh_watertight_required", True),
            flow_metrics_enabled=d.get("flow_metrics_enabled", False),
            custom_criteria=d.get("custom_criteria", {}),
            watertight_required=d.get("watertight_required", True),
            terminal_range=d.get("terminal_range"),
        )


@dataclass
class ObjectRequirements:
    """Complete requirements schema for a single object (all 9 sections)."""
    identity: IdentitySection = field(default_factory=IdentitySection)
    frame_of_reference: FrameOfReferenceSection = field(default_factory=FrameOfReferenceSection)
    domain: DomainSection = field(default_factory=DomainSection)
    inlets_outlets: InletsOutletsSection = field(default_factory=InletsOutletsSection)
    topology: TopologySection = field(default_factory=TopologySection)
    geometry: GeometrySection = field(default_factory=GeometrySection)
    constraints: ConstraintsSection = field(default_factory=ConstraintsSection)
    embedding_export: EmbeddingExportSection = field(default_factory=EmbeddingExportSection)
    acceptance_criteria: AcceptanceCriteriaSection = field(default_factory=AcceptanceCriteriaSection)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "identity": self.identity.to_dict(),
            "frame_of_reference": self.frame_of_reference.to_dict(),
            "domain": self.domain.to_dict(),
            "inlets_outlets": self.inlets_outlets.to_dict(),
            "topology": self.topology.to_dict(),
            "geometry": self.geometry.to_dict(),
            "constraints": self.constraints.to_dict(),
            "embedding_export": self.embedding_export.to_dict(),
            "acceptance_criteria": self.acceptance_criteria.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ObjectRequirements":
        return cls(
            identity=IdentitySection.from_dict(d.get("identity", {})),
            frame_of_reference=FrameOfReferenceSection.from_dict(d.get("frame_of_reference", {})),
            domain=DomainSection.from_dict(d.get("domain", {})),
            inlets_outlets=InletsOutletsSection.from_dict(d.get("inlets_outlets", {})),
            topology=TopologySection.from_dict(d.get("topology", {})),
            geometry=GeometrySection.from_dict(d.get("geometry", {})),
            constraints=ConstraintsSection.from_dict(d.get("constraints", {})),
            embedding_export=EmbeddingExportSection.from_dict(d.get("embedding_export", {})),
            acceptance_criteria=AcceptanceCriteriaSection.from_dict(d.get("acceptance_criteria", {})),
        )
    
    def to_json(self, path: str) -> None:
        """Save requirements to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, path: str) -> "ObjectRequirements":
        """Load requirements from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


# =============================================================================
# Context Classes
# =============================================================================

@dataclass
class ObjectContext:
    """Tracks state for a single object within a project."""
    name: str = ""
    slug: str = ""
    status: str = "draft"
    created_at: float = field(default_factory=time.time)
    
    object_dir: str = ""
    intent_dir: str = ""
    spec_dir: str = ""
    code_dir: str = ""
    outputs_dir: str = ""
    mesh_dir: str = ""
    analysis_dir: str = ""
    validation_dir: str = ""
    iterations_dir: str = ""
    final_dir: str = ""
    
    version: int = 1
    requirements: Optional[ObjectRequirements] = None
    raw_intent: str = ""
    
    spec_path: Optional[str] = None
    code_path: Optional[str] = None
    network_path: Optional[str] = None
    mesh_path: Optional[str] = None
    analysis_path: Optional[str] = None
    validation_path: Optional[str] = None
    
    final_void_stl: Optional[str] = None
    final_domain_stl: Optional[str] = None
    final_manifest: Optional[str] = None
    
    feedback_history: List[str] = field(default_factory=list)
    frame_locked: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "slug": self.slug,
            "status": self.status,
            "created_at": self.created_at,
            "object_dir": self.object_dir,
            "version": self.version,
            "requirements": self.requirements.to_dict() if self.requirements else None,
            "raw_intent": self.raw_intent,
            "spec_path": self.spec_path,
            "code_path": self.code_path,
            "network_path": self.network_path,
            "mesh_path": self.mesh_path,
            "analysis_path": self.analysis_path,
            "validation_path": self.validation_path,
            "final_void_stl": self.final_void_stl,
            "final_domain_stl": self.final_domain_stl,
            "final_manifest": self.final_manifest,
            "feedback_history": self.feedback_history,
            "frame_locked": self.frame_locked,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ObjectContext":
        obj = cls(
            name=d.get("name", ""),
            slug=d.get("slug", ""),
            status=d.get("status", "draft"),
            created_at=d.get("created_at", time.time()),
            object_dir=d.get("object_dir", ""),
            version=d.get("version", 1),
            raw_intent=d.get("raw_intent", ""),
            spec_path=d.get("spec_path"),
            code_path=d.get("code_path"),
            network_path=d.get("network_path"),
            mesh_path=d.get("mesh_path"),
            analysis_path=d.get("analysis_path"),
            validation_path=d.get("validation_path"),
            final_void_stl=d.get("final_void_stl"),
            final_domain_stl=d.get("final_domain_stl"),
            final_manifest=d.get("final_manifest"),
            feedback_history=d.get("feedback_history", []),
            frame_locked=d.get("frame_locked", False),
        )
        if d.get("requirements"):
            obj.requirements = ObjectRequirements.from_dict(d["requirements"])
        return obj
    
    def setup_folders(self) -> None:
        """Create the folder structure for this object."""
        self.intent_dir = os.path.join(self.object_dir, "00_intent")
        self.spec_dir = os.path.join(self.object_dir, "01_spec")
        self.code_dir = os.path.join(self.object_dir, "02_code")
        self.outputs_dir = os.path.join(self.object_dir, "03_outputs")
        self.mesh_dir = os.path.join(self.object_dir, "04_mesh")
        self.analysis_dir = os.path.join(self.object_dir, "05_analysis")
        self.validation_dir = os.path.join(self.object_dir, "06_validation")
        self.iterations_dir = os.path.join(self.object_dir, "07_iterations")
        self.final_dir = os.path.join(self.object_dir, "08_final")
        
        for folder in [
            self.intent_dir, self.spec_dir, self.code_dir, self.outputs_dir,
            self.mesh_dir, self.analysis_dir, self.validation_dir,
            self.iterations_dir, self.final_dir,
            os.path.join(self.final_dir, "embed"),
        ]:
            os.makedirs(folder, exist_ok=True)
    
    def get_versioned_path(self, folder: str, prefix: str, ext: str) -> str:
        """Get versioned file path."""
        return os.path.join(folder, f"{prefix}_v{self.version:03d}.{ext}")
    
    def increment_version(self) -> None:
        """Increment version number."""
        self.version += 1


@dataclass
class ProjectContext:
    """Tracks project state across the workflow."""
    project_name: str = ""
    project_slug: str = ""
    output_dir: str = ""
    
    units_internal: str = "m"
    units_export: str = "mm"
    default_embed_domain: Tuple[float, float, float] = (0.02, 0.06, 0.03)
    flow_solver_enabled: bool = False
    
    objects: List[ObjectContext] = field(default_factory=list)
    current_object_index: int = 0
    variant_mode: bool = False
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_name": self.project_name,
            "project_slug": self.project_slug,
            "output_dir": self.output_dir,
            "units_internal": self.units_internal,
            "units_export": self.units_export,
            "default_embed_domain": list(self.default_embed_domain),
            "flow_solver_enabled": self.flow_solver_enabled,
            "objects": [obj.to_dict() for obj in self.objects],
            "current_object_index": self.current_object_index,
            "variant_mode": self.variant_mode,
            "created_at": self.created_at,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProjectContext":
        ctx = cls(
            project_name=d.get("project_name", ""),
            project_slug=d.get("project_slug", ""),
            output_dir=d.get("output_dir", ""),
            units_internal=d.get("units_internal", "m"),
            units_export=d.get("units_export", "mm"),
            default_embed_domain=tuple(d.get("default_embed_domain", [0.02, 0.06, 0.03])),
            flow_solver_enabled=d.get("flow_solver_enabled", False),
            current_object_index=d.get("current_object_index", 0),
            variant_mode=d.get("variant_mode", False),
            created_at=d.get("created_at", time.time()),
        )
        ctx.objects = [ObjectContext.from_dict(obj) for obj in d.get("objects", [])]
        return ctx
    
    def get_current_object(self) -> Optional[ObjectContext]:
        """Get the current object being worked on."""
        if 0 <= self.current_object_index < len(self.objects):
            return self.objects[self.current_object_index]
        return None
    
    def add_object(self, name: str, slug: str) -> ObjectContext:
        """Add a new object to the project."""
        obj = ObjectContext(
            name=name,
            slug=slug,
            object_dir=os.path.join(self.output_dir, "objects", slug),
        )
        obj.setup_folders()
        
        obj_json = {
            "name": name,
            "slug": slug,
            "status": "draft",
            "created_at": obj.created_at,
        }
        with open(os.path.join(obj.object_dir, "object.json"), 'w') as f:
            json.dump(obj_json, f, indent=2)
        
        self.objects.append(obj)
        return obj
    
    def save_project_json(self) -> None:
        """Save project.json file."""
        project_data = {
            "project_name": self.project_name,
            "project_slug": self.project_slug,
            "units_internal": self.units_internal,
            "units_export": self.units_export,
            "default_embed_domain": list(self.default_embed_domain),
            "flow_solver_enabled": self.flow_solver_enabled,
            "variant_mode": self.variant_mode,
            "created_at": self.created_at,
            "objects": [{"name": obj.name, "slug": obj.slug} for obj in self.objects],
        }
        with open(os.path.join(self.output_dir, "project.json"), 'w') as f:
            json.dump(project_data, f, indent=2)


# =============================================================================
# Organ-Specific Question Templates for Dynamic Variance
# =============================================================================

ORGAN_QUESTION_VARIANTS = {
    "liver": {
        "description": "Liver vascular network with hepatic structures",
        "C": {
            "name": "Hepatic Inlets/Outlets",
            "questions": [
                ("num_inlets", "How many hepatic artery inlets?", "1"),
                ("num_outlets", "How many portal vein outlets (or hepatic vein drainage)?", "1"),
                ("inlet_outlet_location", "Hepatic artery entry point (hilum/distributed)?", "hilum"),
                ("inlet_radius", "Hepatic artery inlet radius (mm)?", "2.0"),
                ("outlet_radius", "Portal/hepatic vein outlet radius (mm)?", "3.0"),
                ("direction_vectors", "Vessel direction: follow lobular structure? (yes/infer)", "infer"),
                ("dual_supply", "Include both arterial and portal venous trees?", "yes"),
            ],
        },
        "D": {
            "name": "Hepatic Topology",
            "questions": [
                ("strict_tree", "Strict tree (no portal-arterial anastomoses)?", "yes"),
                ("target_terminals", "Target sinusoidal terminal count?", "100"),
                ("max_depth", "Maximum branching depth (lobular levels)?", "6"),
                ("branching_style", "Branching: lobular-aligned or space-filling?", "lobular-aligned"),
                ("perfusion_zones", "Perfusion zones (periportal/centrilobular emphasis)?", "uniform"),
            ],
        },
        "E": {
            "name": "Hepatic Geometry",
            "questions": [
                ("tortuosity", "Vessel tortuosity (hepatic vessels are typically low)?", "low"),
                ("branch_angle_range", "Branch angle range (hepatic: 45-75 typical)?", "45-75"),
                ("segment_length_range", "Segment length (lobule scale: 0.5-2mm)?", "0.5-2"),
                ("tapering", "Radius tapering (Murray's law for hepatic)?", "murray"),
            ],
        },
    },
    "kidney": {
        "description": "Renal vascular network with cortex/medulla structure",
        "C": {
            "name": "Renal Inlets/Outlets",
            "questions": [
                ("num_inlets", "How many renal artery inlets?", "1"),
                ("num_outlets", "How many renal vein outlets?", "1"),
                ("inlet_outlet_location", "Entry/exit at renal hilum? (hilum/distributed)", "hilum"),
                ("inlet_radius", "Renal artery inlet radius (mm)?", "2.5"),
                ("outlet_radius", "Renal vein outlet radius (mm)?", "3.0"),
                ("direction_vectors", "Vessel direction: radial from hilum? (radial/infer)", "radial"),
                ("arcuate_vessels", "Include arcuate vessel layer at corticomedullary junction?", "yes"),
            ],
        },
        "D": {
            "name": "Renal Topology",
            "questions": [
                ("strict_tree", "Strict arterial tree (no arteriovenous shunts)?", "yes"),
                ("target_terminals", "Target glomerular/terminal count?", "200"),
                ("max_depth", "Maximum depth (interlobar->arcuate->interlobular->afferent)?", "5"),
                ("branching_style", "Branching: cortical-dense or uniform?", "cortical-dense"),
                ("perfusion_zones", "Perfusion emphasis (cortex/medulla ratio)?", "cortex-heavy"),
            ],
        },
        "E": {
            "name": "Renal Geometry",
            "questions": [
                ("tortuosity", "Vessel tortuosity (renal: low-medium)?", "low"),
                ("branch_angle_range", "Branch angle range (renal: 30-60 typical)?", "30-60"),
                ("segment_length_range", "Segment length (nephron scale: 0.3-1.5mm)?", "0.3-1.5"),
                ("tapering", "Radius tapering profile?", "murray"),
            ],
        },
    },
    "lung": {
        "description": "Pulmonary vascular network with bronchial alignment",
        "C": {
            "name": "Pulmonary Inlets/Outlets",
            "questions": [
                ("num_inlets", "How many pulmonary artery inlets?", "1"),
                ("num_outlets", "How many pulmonary vein outlets?", "2"),
                ("inlet_outlet_location", "Entry at hilum, exit distributed? (hilum/lobar)", "hilum"),
                ("inlet_radius", "Pulmonary artery inlet radius (mm)?", "3.0"),
                ("outlet_radius", "Pulmonary vein outlet radius (mm)?", "2.5"),
                ("direction_vectors", "Follow bronchial tree alignment? (bronchial/infer)", "bronchial"),
                ("bronchial_circulation", "Include bronchial arterial supply?", "no"),
            ],
        },
        "D": {
            "name": "Pulmonary Topology",
            "questions": [
                ("strict_tree", "Strict tree (no pulmonary shunts)?", "yes"),
                ("target_terminals", "Target alveolar capillary terminal count?", "500"),
                ("max_depth", "Maximum depth (lobar->segmental->subsegmental->acinar)?", "8"),
                ("branching_style", "Branching: dichotomous (lung-typical) or asymmetric?", "dichotomous"),
                ("perfusion_zones", "Perfusion zones (apical/basal gradient)?", "basal-heavy"),
            ],
        },
        "E": {
            "name": "Pulmonary Geometry",
            "questions": [
                ("tortuosity", "Vessel tortuosity (pulmonary: very low)?", "low"),
                ("branch_angle_range", "Branch angle range (pulmonary: 20-45 typical)?", "20-45"),
                ("segment_length_range", "Segment length (airway-aligned: 1-5mm)?", "1-5"),
                ("tapering", "Radius tapering (Horsfield model or Murray)?", "murray"),
            ],
        },
    },
    "heart": {
        "description": "Coronary vascular network",
        "C": {
            "name": "Coronary Inlets/Outlets",
            "questions": [
                ("num_inlets", "How many coronary artery ostia (typically 2: LCA, RCA)?", "2"),
                ("num_outlets", "How many coronary sinus drainage points?", "1"),
                ("inlet_outlet_location", "Ostia at aortic root, drainage to right atrium?", "aortic root"),
                ("inlet_radius", "Left main coronary artery radius (mm)?", "2.0"),
                ("outlet_radius", "Right coronary artery radius (mm)?", "1.5"),
                ("direction_vectors", "Follow epicardial surface? (epicardial/transmural)", "epicardial"),
                ("lad_lcx_rca", "Include LAD, LCx, and RCA territories?", "yes"),
            ],
        },
        "D": {
            "name": "Coronary Topology",
            "questions": [
                ("strict_tree", "Strict tree (no coronary collaterals)?", "yes"),
                ("target_terminals", "Target myocardial terminal count?", "300"),
                ("max_depth", "Maximum depth (epicardial->intramural->capillary)?", "6"),
                ("branching_style", "Branching: territory-based or uniform?", "territory-based"),
                ("perfusion_zones", "Perfusion zones (LV/RV/septum distribution)?", "LV-dominant"),
            ],
        },
        "E": {
            "name": "Coronary Geometry",
            "questions": [
                ("tortuosity", "Vessel tortuosity (coronary: medium, follows surface)?", "medium"),
                ("branch_angle_range", "Branch angle range (coronary: 40-80)?", "40-80"),
                ("segment_length_range", "Segment length (myocardial scale: 0.5-3mm)?", "0.5-3"),
                ("tapering", "Radius tapering profile?", "murray"),
            ],
        },
    },
    "generic": {
        "description": "Generic tubular/vascular network",
        "C": {
            "name": "Inlets/Outlets (Hard Requirements)",
            "questions": [
                ("num_inlets", "How many inlets?", "1"),
                ("num_outlets", "How many outlets?", "0"),
                ("inlet_outlet_location", "Location (face/region)? same face or opposite?", "same face"),
                ("inlet_radius", "Inlet radii (comma-separated if multiple)?", "0.002"),
                ("outlet_radius", "Outlet radii (comma-separated if multiple)?", "0.001"),
                ("direction_vectors", "Direction vectors explicit or infer from face? (explicit/infer)", "infer"),
                ("same_x_face", "Inlets/outlets on same X face?", "yes"),
            ],
        },
        "D": {
            "name": "Topology (Tree Structure)",
            "questions": [
                ("strict_tree", "Strict tree (no loops)?", "yes"),
                ("target_terminals", "Target terminal tip count?", None),
                ("max_depth", "Maximum depth (levels)?", None),
                ("branching_style", "Balanced or aggressive early branching?", "balanced"),
                ("perfusion_zones", "Perfusion zones needing more density? (describe or 'none')", "none"),
            ],
        },
        "E": {
            "name": "Geometry Character (Path Style)",
            "questions": [
                ("tortuosity", "Straight vs tortuous? (low/med/high)", "low"),
                ("branch_angle_range", "Branch angle range? (e.g., 30-90)", "30-90"),
                ("segment_length_range", "Segment length range? (e.g., 0.5-5mm)", "0.5-5"),
                ("tapering", "Tapering? (murray/linear/fixed)", "murray"),
            ],
        },
    },
}


def detect_organ_type(intent: str) -> str:
    """
    Detect the organ type from the user's intent description.
    
    Returns the organ key (liver, kidney, lung, heart, generic) based on
    keywords found in the intent string.
    """
    intent_lower = intent.lower()
    
    organ_keywords = {
        "liver": ["liver", "hepatic", "hepato", "lobule", "sinusoid", "portal"],
        "kidney": ["kidney", "renal", "nephron", "glomerul", "cortex", "medulla", "ureter"],
        "lung": ["lung", "pulmonary", "bronch", "alveol", "respiratory", "airway"],
        "heart": ["heart", "coronary", "cardiac", "myocard", "ventricle", "atrium", "aortic"],
    }
    
    for organ, keywords in organ_keywords.items():
        for keyword in keywords:
            if keyword in intent_lower:
                return organ
    
    return "generic"


def detect_topology_kind(intent: str) -> str:
    """
    Detect the topology kind from the user's intent description.
    
    Returns the topology kind based on keywords found in the intent string:
    - "path": Simple inlet→outlet channel, no branching
    - "tree": Full tree structure with branching
    - "loop": Looping/recirculating structure
    - "multi_tree": Multiple independent trees
    
    This determines which questions are relevant to ask.
    """
    intent_lower = intent.lower()
    
    # Check for path/channel indicators (simple inlet→outlet)
    path_keywords = [
        "channel", "tube", "pipe", "conduit", "duct", "passage",
        "single path", "straight", "direct", "simple", "inlet to outlet",
        "inlet-outlet", "one inlet", "one outlet", "single inlet", "single outlet"
    ]
    
    # Check for loop/recirculation indicators
    loop_keywords = [
        "loop", "recircul", "cycle", "circular", "closed", "return",
        "anastomo", "connect back", "reconnect"
    ]
    
    # Check for multi-tree indicators
    multi_tree_keywords = [
        "multiple tree", "multi-tree", "several tree", "independent tree",
        "parallel tree", "separate tree", "dual tree", "twin tree"
    ]
    
    # Check for explicit tree indicators
    tree_keywords = [
        "tree", "branch", "bifurcat", "terminal", "tip", "dendritic",
        "hierarchical", "fractal", "vascular", "network", "distribute"
    ]
    
    # Priority order: path > loop > multi_tree > tree (default)
    for keyword in path_keywords:
        if keyword in intent_lower:
            return "path"
    
    for keyword in loop_keywords:
        if keyword in intent_lower:
            return "loop"
    
    for keyword in multi_tree_keywords:
        if keyword in intent_lower:
            return "multi_tree"
    
    # Default to tree for most vascular/organ structures
    return "tree"


# =============================================================================
# Topology-Specific Question Sets (Minimum Viable Spec + Adaptive Modules)
# =============================================================================
# 
# These question sets implement the "Minimum Viable Spec" approach:
# - PATH: Simple inlet→outlet channel (5-7 questions total)
# - TREE: Branching network with terminals (5-8 questions total)
# - LOOP: Looping/recirculating structure
# - MULTI_TREE: Multiple independent trees
#
# Key principles:
# 1. Topology confirmation is FIRST (gates all subsequent questions)
# 2. Domain dimensions are explicit (not hidden behind "use defaults?")
# 3. Ports are first-class (inlet AND outlet for path topology)
# 4. Embedding questions only when needed (with print-resolution proxy)
# 5. "Sufficient-to-generate" threshold per topology

TOPOLOGY_QUESTION_VARIANTS = {
    "path": {
        # PATH topology: Simple inlet→outlet channel
        # Required: topology, domain, inlet, outlet, route
        # Optional: clearance, embedding
        "A": {
            "name": "1. Topology Confirmation",
            "questions": [
                ("topology_kind", "Is this a simple channel connecting inlet to outlet (path), or a branching network (tree)?", "path"),
            ],
        },
        "B": {
            "name": "2. Domain Dimensions",
            "questions": [
                ("domain_dims", "What are the box dimensions (X Y Z) and units? Example: '20 60 30 mm'", "20 60 30 mm"),
                ("boundary_margin", "Boundary margin - how far should channels stay from walls? (mm)", "1"),
            ],
        },
        "C": {
            "name": "3. Inlet Definition",
            "questions": [
                ("inlet_face", "Which face is the inlet on? (x_min/x_max/y_min/y_max/z_min/z_max)", "x_min"),
                ("inlet_position", "Inlet position on face: center, or specify offsets (u,v in mm)?", "center"),
                ("inlet_radius", "Inlet radius (mm)?", "2.0"),
            ],
        },
        "C2": {
            "name": "4. Outlet Definition",
            "questions": [
                ("outlet_face", "Which face is the outlet on? (x_min/x_max/y_min/y_max/z_min/z_max)", "x_max"),
                ("outlet_position", "Outlet position on face: center, or specify offsets (u,v in mm)?", "center"),
                ("outlet_radius", "Outlet radius (mm)? (default = inlet radius)", None),
            ],
        },
        "D": {
            "name": "5. Path Routing",
            "questions": [
                ("route_type", "How should the path route? (straight/single_bend/s_curve/via_points)", "straight"),
                ("bend_radius", "If curved: minimum bend radius (mm)? (or 'auto')", "auto"),
            ],
        },
        "F": {
            "name": "6. Constraints (Optional)",
            "questions": [
                ("min_clearance", "Minimum clearance from other structures (mm)?", "1"),
                ("wall_thickness", "Wall thickness for hollow channels (mm)? (or 'solid')", "solid"),
            ],
        },
        "H": {
            "name": "7. Embedding (Optional)",
            "questions": [
                ("embed_requested", "Do you want to embed this into a voxel domain for 3D printing? (yes/no)", "no"),
                ("min_feature_size", "If embedding: minimum printable feature size / nozzle diameter (mm)?", "0.4"),
                ("voxel_pitch_confirm", "Suggested voxel pitch based on feature size. Accept? (yes/custom)", "yes"),
            ],
        },
        "G": {
            "name": "8. Acceptance Criteria",
            "questions": [
                ("watertight_required", "Require watertight mesh?", "yes"),
            ],
        },
    },
    "tree": {
        # TREE topology: Branching network with terminals
        # Required: topology, domain, inlet, terminal mode, branching
        # Optional: outlet, embedding
        "A": {
            "name": "1. Topology Confirmation",
            "questions": [
                ("topology_kind", "Is this a branching tree network? (tree/path)", "tree"),
            ],
        },
        "B": {
            "name": "2. Domain Dimensions",
            "questions": [
                ("domain_dims", "What are the box dimensions (X Y Z) and units? Example: '20 60 30 mm'", "20 60 30 mm"),
                ("boundary_margin", "Boundary margin - how far should vessels stay from walls? (mm)", "1"),
            ],
        },
        "C": {
            "name": "3. Inlet Definition",
            "questions": [
                ("inlet_face", "Which face is the inlet on? (x_min/x_max/y_min/y_max/z_min/z_max)", "x_min"),
                ("inlet_position", "Inlet position on face: center, or specify offsets (u,v in mm)?", "center"),
                ("inlet_radius", "Inlet radius (mm)?", "2.0"),
            ],
        },
        "C2": {
            "name": "4. Outlet/Terminal Mode",
            "questions": [
                ("terminal_mode", "Is this: (a) inlet→many terminals (no outlet port), or (b) inlet→outlet with branches?", "a"),
                ("outlet_face", "If outlet port: which face? (x_min/x_max/y_min/y_max/z_min/z_max, or 'none')", "none"),
                ("outlet_radius", "If outlet port: outlet radius (mm)?", None),
            ],
        },
        "D": {
            "name": "5. Terminal/Branching Configuration",
            "questions": [
                ("target_terminals", "Target terminal count? (or 'sparse'/'medium'/'dense')", "medium"),
                ("max_depth", "Maximum branching depth (levels)?", "6"),
                ("branching_style", "Branching style: balanced, aggressive_early, or space_filling?", "balanced"),
            ],
        },
        "E": {
            "name": "6. Geometry",
            "questions": [
                ("tapering", "Radius tapering profile? (murray/linear/fixed)", "murray"),
                ("min_radius", "Minimum terminal radius (mm)?", "0.1"),
            ],
        },
        "F": {
            "name": "7. Constraints",
            "questions": [
                ("ban_self_intersection", "Hard ban on self-intersections?", "yes"),
                ("min_clearance", "Minimum clearance between vessels (mm)?", "0.5"),
            ],
        },
        "H": {
            "name": "8. Embedding (Optional)",
            "questions": [
                ("embed_requested", "Do you want to embed this into a voxel domain for 3D printing? (yes/no)", "no"),
                ("min_feature_size", "If embedding: minimum printable feature size / nozzle diameter (mm)?", "0.4"),
                ("voxel_pitch_confirm", "Suggested voxel pitch based on feature size. Accept? (yes/custom)", "yes"),
            ],
        },
        "G": {
            "name": "9. Acceptance Criteria",
            "questions": [
                ("terminal_range", "Acceptable terminal count range? (e.g., 50-100, or 'any')", "any"),
                ("watertight_required", "Require watertight mesh?", "yes"),
            ],
        },
    },
    "loop": {
        # LOOP topology: Looping/recirculating structure
        "A": {
            "name": "1. Topology Confirmation",
            "questions": [
                ("topology_kind", "Is this a looping/recirculating network?", "loop"),
            ],
        },
        "B": {
            "name": "2. Domain Dimensions",
            "questions": [
                ("domain_dims", "What are the box dimensions (X Y Z) and units? Example: '20 60 30 mm'", "20 60 30 mm"),
                ("boundary_margin", "Boundary margin (mm)?", "1"),
            ],
        },
        "C": {
            "name": "3. Ports",
            "questions": [
                ("inlet_face", "Inlet face? (x_min/x_max/y_min/y_max/z_min/z_max)", "x_min"),
                ("inlet_radius", "Inlet radius (mm)?", "2.0"),
                ("outlet_face", "Outlet face?", "x_max"),
                ("outlet_radius", "Outlet radius (mm)?", "2.0"),
            ],
        },
        "D": {
            "name": "4. Loop Configuration",
            "questions": [
                ("loop_count", "Number of loops/anastomoses?", "1"),
                ("loop_style", "Loop style: single_loop, mesh, or ladder?", "single_loop"),
                ("recirculation", "Allow recirculation paths?", "yes"),
            ],
        },
        "G": {
            "name": "5. Acceptance Criteria",
            "questions": [
                ("watertight_required", "Require watertight mesh?", "yes"),
            ],
        },
    },
    "multi_tree": {
        # MULTI_TREE topology: Multiple independent trees
        "A": {
            "name": "1. Topology Confirmation",
            "questions": [
                ("topology_kind", "Multiple independent branching trees?", "multi_tree"),
            ],
        },
        "B": {
            "name": "2. Domain Dimensions",
            "questions": [
                ("domain_dims", "What are the box dimensions (X Y Z) and units? Example: '20 60 30 mm'", "20 60 30 mm"),
                ("boundary_margin", "Boundary margin (mm)?", "1"),
            ],
        },
        "C": {
            "name": "3. Tree Configuration",
            "questions": [
                ("tree_count", "Number of independent trees?", "2"),
                ("shared_inlet", "Do trees share an inlet?", "no"),
            ],
        },
        "D": {
            "name": "4. Per-Tree Settings",
            "questions": [
                ("target_terminals_per_tree", "Target terminals per tree?", "50"),
                ("max_depth", "Maximum depth per tree?", "5"),
                ("inlet_radius", "Inlet radius per tree (mm)?", "2.0"),
            ],
        },
        "G": {
            "name": "5. Acceptance Criteria",
            "questions": [
                ("watertight_required", "Require watertight mesh?", "yes"),
            ],
        },
    },
}


def get_tailored_questions(intent: str, base_questions: dict = None) -> dict:
    """
    Get question groups tailored to the user's described object.
    
    This function implements topology-first gating:
    1. Detect topology kind from intent (path/tree/loop/multi_tree)
    2. Use the complete topology-specific question set
    3. Only fall back to base questions if no topology detected
    
    The topology-specific question sets are designed as "Minimum Viable Spec":
    - PATH: 5-7 questions (topology, domain, inlet, outlet, route, embed?)
    - TREE: 5-8 questions (topology, domain, inlet, terminals, branching, embed?)
    
    This ensures:
    - Simple "channel" requests don't get asked about branching/terminals
    - "Vascular tree" requests get full tree-related questions
    - Domain dimensions are always asked explicitly (not hidden behind defaults)
    - Embedding questions only appear when embedding is requested
    
    Parameters
    ----------
    intent : str
        The user's description of the object they want to generate
    base_questions : dict, optional
        Base question groups to use as fallback. If None, uses QUESTION_GROUPS.
        
    Returns
    -------
    dict
        Topology-specific question groups, or base questions if no topology detected
    """
    # Detect topology kind first - this gates all subsequent questions
    topology_kind = detect_topology_kind(intent)
    
    # If we have a topology-specific question set, use it entirely
    # This implements "topology-first gating" - the topology determines the question flow
    if topology_kind in TOPOLOGY_QUESTION_VARIANTS:
        # Return the complete topology-specific question set
        # These are designed as "Minimum Viable Spec" for each topology
        return TOPOLOGY_QUESTION_VARIANTS[topology_kind].copy()
    
    # Fallback to base questions (only if no topology detected)
    if base_questions is None:
        base_questions = QUESTION_GROUPS.copy()
    else:
        base_questions = base_questions.copy()
    
    # Apply organ-specific variants only for fallback case
    organ_type = detect_organ_type(intent)
    if organ_type in ORGAN_QUESTION_VARIANTS:
        organ_variants = ORGAN_QUESTION_VARIANTS[organ_type]
        for group_key in ["C", "D", "E"]:
            if group_key in organ_variants:
                base_questions[group_key] = organ_variants[group_key]
    
    return base_questions


def get_topology_completeness_rules(topology_kind: str) -> dict:
    """
    Get the "sufficient-to-generate" completeness rules for a topology.
    
    Each topology has different required fields:
    - PATH: requires inlet + outlet definitions
    - TREE: requires inlet + terminal mode
    - LOOP: requires inlet + outlet + loop config
    - MULTI_TREE: requires tree count + per-tree settings
    
    Returns a dict with:
    - required_fields: list of field names that must be filled
    - optional_fields: list of field names that have sensible defaults
    """
    rules = {
        "path": {
            "required_fields": [
                "topology_kind",
                "domain_dims",
                "inlet_face",
                "inlet_radius",
                "outlet_face",
            ],
            "optional_fields": [
                "inlet_position",
                "outlet_position",
                "outlet_radius",
                "route_type",
                "bend_radius",
                "boundary_margin",
                "min_clearance",
                "wall_thickness",
                "embed_requested",
            ],
        },
        "tree": {
            "required_fields": [
                "topology_kind",
                "domain_dims",
                "inlet_face",
                "inlet_radius",
                "target_terminals",
            ],
            "optional_fields": [
                "inlet_position",
                "terminal_mode",
                "outlet_face",
                "outlet_radius",
                "max_depth",
                "branching_style",
                "tapering",
                "min_radius",
                "boundary_margin",
                "min_clearance",
                "embed_requested",
            ],
        },
        "loop": {
            "required_fields": [
                "topology_kind",
                "domain_dims",
                "inlet_face",
                "inlet_radius",
                "outlet_face",
                "outlet_radius",
                "loop_count",
            ],
            "optional_fields": [
                "boundary_margin",
                "loop_style",
                "recirculation",
            ],
        },
        "multi_tree": {
            "required_fields": [
                "topology_kind",
                "domain_dims",
                "tree_count",
                "inlet_radius",
            ],
            "optional_fields": [
                "shared_inlet",
                "target_terminals_per_tree",
                "max_depth",
                "boundary_margin",
            ],
        },
    }
    return rules.get(topology_kind, rules["tree"])


# =============================================================================
# Question Groups for Requirements Capture (Base/Generic)
# =============================================================================

QUESTION_GROUPS = {
    # Base/fallback question groups - used when no topology-specific set matches
    # Note: Topology-specific question sets (TOPOLOGY_QUESTION_VARIANTS) are preferred
    # and will completely replace these when a topology is detected from intent.
    "A": {
        "name": "1. Topology and Domain",
        "questions": [
            ("topology_kind", "Is this a simple path/channel (path) or branching network (tree)?", "tree"),
            ("domain_dims", "What are the box dimensions (X Y Z) and units? Example: '20 60 30 mm'", "20 60 30 mm"),
            ("boundary_margin", "Boundary margin - how far should structures stay from walls? (mm)", "1"),
        ],
    },
    "B": {
        "name": "2. Print Constraints",
        "questions": [
            ("min_channel_radius", "Minimum printable channel radius (mm)?", "0.5"),
            ("min_clearance", "Minimum spacing (clearance) between channels (mm)?", "0.5"),
            ("nozzle_size", "Nozzle size or minimum feature size (mm)?", "0.4"),
        ],
    },
    "C": {
        "name": "3. Inlets/Outlets",
        "questions": [
            ("inlet_face", "Which face is the inlet on? (x_min/x_max/y_min/y_max/z_min/z_max)", "x_min"),
            ("inlet_position", "Inlet position on face: center, or specify offsets (u,v in mm)?", "center"),
            ("inlet_radius", "Inlet radius (mm)?", "2.0"),
            ("outlet_face", "Which face is the outlet on? (x_min/x_max/y_min/y_max/z_min/z_max, or 'none' for tree)", "none"),
            ("outlet_radius", "Outlet radius (mm)? (if applicable)", None),
        ],
    },
    "D": {
        "name": "4. Topology Configuration",
        "questions": [
            ("target_terminals", "Target terminal count? (for tree topology, or 'n/a' for path)", None),
            ("max_depth", "Maximum branching depth (levels)?", "6"),
            ("branching_style", "Branching style: balanced, aggressive_early, or space_filling?", "balanced"),
        ],
    },
    "E": {
        "name": "5. Geometry",
        "questions": [
            ("tapering", "Radius tapering profile? (murray/linear/fixed)", "murray"),
            ("min_radius", "Minimum terminal/tip radius (mm)?", "0.1"),
            ("branch_angle_range", "Branch angle range? (e.g., 30-90 degrees)", "30-90"),
        ],
    },
    "F": {
        "name": "6. Constraints",
        "questions": [
            ("ban_self_intersection", "Hard ban on self-intersections?", "yes"),
            ("enforce_clearance", "Hard min clearance enforcement?", "yes"),
            ("keepout_regions", "Keep-out regions? (describe or 'none')", "none"),
        ],
    },
    "G": {
        "name": "7. Acceptance Criteria",
        "questions": [
            ("terminal_range", "Acceptable terminal count range? (e.g., 50-100, or 'any')", "any"),
            ("watertight_required", "Require watertight mesh?", "yes"),
        ],
    },
    "H": {
        "name": "8. Embedding (Optional)",
        "questions": [
            ("embed_requested", "Do you want to embed this into a voxel domain for 3D printing? (yes/no)", "no"),
            ("min_feature_size", "If embedding: minimum printable feature size / nozzle diameter (mm)?", "0.4"),
            ("voxel_pitch_confirm", "Suggested voxel pitch based on feature size. Accept? (yes/custom)", "yes"),
        ],
    },
}


# =============================================================================
# Rule Engine for Adaptive Requirements Capture
# =============================================================================

@dataclass
class RuleFlag:
    """A flag raised by a rule indicating an issue to address."""
    rule_id: str
    family: str  # "A" (completeness), "B" (ambiguity), "C" (conflict)
    field: str
    message: str
    severity: str = "required"  # "required", "warning", "info"
    rework_cost: str = "high"  # "high", "medium", "low"


@dataclass
class ProposedDefault:
    """A proposed default value for a field."""
    field: str
    value: Any
    reason: str
    accepted: bool = False


@dataclass
class RuleEvaluationResult:
    """Result of evaluating all rules against requirements."""
    missing_fields: List[RuleFlag] = field(default_factory=list)
    ambiguity_flags: List[RuleFlag] = field(default_factory=list)
    conflict_flags: List[RuleFlag] = field(default_factory=list)
    proposed_defaults: List[ProposedDefault] = field(default_factory=list)
    is_generation_ready: bool = False
    is_finalization_ready: bool = False
    
    def all_flags(self) -> List[RuleFlag]:
        """Return all flags sorted by rework cost."""
        all_f = self.missing_fields + self.ambiguity_flags + self.conflict_flags
        cost_order = {"high": 0, "medium": 1, "low": 2}
        return sorted(all_f, key=lambda f: cost_order.get(f.rework_cost, 3))
    
    def has_blockers(self) -> bool:
        """Return True if there are any required flags."""
        return any(f.severity == "required" for f in self.all_flags())


@dataclass
class PlannedQuestion:
    """A question planned to be asked."""
    field: str
    question_text: str
    default_value: Optional[str]
    rework_cost: str
    reason: str


class IntentParser:
    """Parses user intent to extract explicit values and detect ambiguities."""
    
    SPATIAL_TERMS = ["left", "right", "top", "bottom", "front", "back", "same side", "opposite"]
    VAGUE_QUANTIFIERS = ["dense", "thin", "big", "small", "large", "highly", "smooth", "tortuous", "branched"]
    IO_TERMS = ["inlet", "outlet", "flow", "perfusion", "supply", "drain"]
    SYMMETRY_TERMS = ["symmetric", "symmetry", "mirror"]
    
    NUMERIC_PATTERNS = {
        "box_size": r"box\s+(\d+(?:\.\d+)?)\s*(?:x|by)\s*(\d+(?:\.\d+)?)\s*(?:x|by)\s*(\d+(?:\.\d+)?)\s*(cm|mm|m)?",
        "diameter": r"(?:diameter|channel)\s*(?:of\s+)?(\d+(?:\.\d+)?)\s*(mm|cm|um)?",
        "radius": r"radius\s*(?:of\s+)?(\d+(?:\.\d+)?)\s*(mm|cm|um)?",
        "count": r"(\d+)\s*(?:inlets?|outlets?|terminals?|branches?)",
    }
    
    def __init__(self, intent: str):
        self.intent = intent
        self.intent_lower = intent.lower()
        self.extracted_values: Dict[str, Any] = {}
        self.detected_ambiguities: List[str] = []
        self._parse()
    
    def _parse(self) -> None:
        """Parse the intent string to extract values and detect ambiguities."""
        import re
        
        for term in self.SPATIAL_TERMS:
            if term in self.intent_lower:
                self.detected_ambiguities.append(f"spatial:{term}")
        
        for term in self.VAGUE_QUANTIFIERS:
            if term in self.intent_lower:
                self.detected_ambiguities.append(f"vague:{term}")
        
        has_io_mention = any(term in self.intent_lower for term in self.IO_TERMS)
        if has_io_mention:
            has_io_count = bool(re.search(r"\d+\s*(?:inlets?|outlets?)", self.intent_lower))
            if not has_io_count:
                self.detected_ambiguities.append("implicit_io")
        
        for term in self.SYMMETRY_TERMS:
            if term in self.intent_lower:
                has_axis = any(ax in self.intent_lower for ax in ["x", "y", "z", "axis", "plane"])
                if not has_axis:
                    self.detected_ambiguities.append(f"symmetry:{term}")
        
        for pattern_name, pattern in self.NUMERIC_PATTERNS.items():
            match = re.search(pattern, self.intent_lower)
            if match:
                self.extracted_values[pattern_name] = match.groups()
    
    def has_spatial_ambiguity(self) -> bool:
        return any(a.startswith("spatial:") for a in self.detected_ambiguities)
    
    def has_vague_quantifiers(self) -> bool:
        return any(a.startswith("vague:") for a in self.detected_ambiguities)
    
    def has_implicit_io(self) -> bool:
        return "implicit_io" in self.detected_ambiguities
    
    def has_symmetry_ambiguity(self) -> bool:
        return any(a.startswith("symmetry:") for a in self.detected_ambiguities)
    
    def get_vague_terms(self) -> List[str]:
        return [a.split(":")[1] for a in self.detected_ambiguities if a.startswith("vague:")]


class RuleEngine:
    """
    Evaluates requirements against three rule families to determine what's needed.
    
    Family A: Completeness rules (required fields)
    Family B: Ambiguity rules (unclear user language)
    Family C: Conflict/feasibility rules (specs that won't work)
    """
    
    GENERATION_REQUIRED_FIELDS = [
        ("domain", "Domain specification"),
        ("inlets_outlets.inlets", "At least one inlet"),
        ("constraints.min_radius_m", "Minimum radius"),
        ("constraints.min_clearance_m", "Minimum clearance"),
    ]
    
    FINALIZATION_REQUIRED_FIELDS = [
        ("embedding_export.voxel_pitch_m", "Voxel pitch for embedding"),
        ("embedding_export.stl_units", "Export units"),
    ]
    
    DEFAULT_VALUES = {
        "domain.type": ("box", "Standard box domain"),
        "domain.size_m": ((0.02, 0.06, 0.03), "Default 2x6x3 cm box"),
        "constraints.min_radius_m": (0.0001, "100 micron minimum radius"),
        "constraints.min_clearance_m": (0.0002, "200 micron minimum clearance"),
        "topology.target_terminals": (200, "Moderate complexity"),
        "embedding_export.voxel_pitch_m": (0.0003, "300 micron voxel pitch"),
        "embedding_export.stl_units": ("mm", "Millimeters for export"),
        "geometry.tortuosity": (0.1, "Low tortuosity"),
        "geometry.radius_profile": ("murray", "Murray's law tapering"),
    }
    
    VAGUE_TO_NUMERIC = {
        "dense": {"topology.target_terminals": (500, 1000)},
        "sparse": {"topology.target_terminals": (50, 100)},
        "thin": {"constraints.min_radius_m": (0.00005, 0.0001)},
        "thick": {"constraints.min_radius_m": (0.0002, 0.0005)},
        "big": {"domain.size_m": ((0.04, 0.12, 0.06), None)},
        "small": {"domain.size_m": ((0.01, 0.03, 0.015), None)},
        "tortuous": {"geometry.tortuosity": (0.7, 0.9)},
        "smooth": {"geometry.tortuosity": (0.0, 0.2)},
        "highly": {"topology.target_terminals": (500, 1000)},
        "branched": {"topology.target_terminals": (300, 500)},
    }
    
    def __init__(self, organ_type: str = "generic"):
        self.organ_type = organ_type
    
    def evaluate(
        self,
        requirements: ObjectRequirements,
        intent: str,
        check_finalization: bool = False
    ) -> RuleEvaluationResult:
        """
        Evaluate requirements against all rule families.
        
        Returns a RuleEvaluationResult with flags and proposed defaults.
        """
        result = RuleEvaluationResult()
        parser = IntentParser(intent)
        
        self._evaluate_completeness_rules(requirements, result, check_finalization)
        self._evaluate_ambiguity_rules(requirements, intent, parser, result)
        self._evaluate_conflict_rules(requirements, result)
        self._propose_defaults(requirements, result)
        
        result.is_generation_ready = not any(
            f.severity == "required" 
            for f in result.missing_fields + result.ambiguity_flags + result.conflict_flags
        )
        
        if check_finalization:
            result.is_finalization_ready = result.is_generation_ready and not any(
                f.field.startswith("embedding_export") for f in result.missing_fields
            )
        
        return result
    
    def _evaluate_completeness_rules(
        self,
        req: ObjectRequirements,
        result: RuleEvaluationResult,
        check_finalization: bool
    ) -> None:
        """Family A: Check for missing required fields."""
        
        if not req.inlets_outlets.inlets:
            result.missing_fields.append(RuleFlag(
                rule_id="A1_inlets",
                family="A",
                field="inlets_outlets.inlets",
                message="At least one inlet is required for generation",
                severity="required",
                rework_cost="high"
            ))
        
        if req.constraints.min_radius_m is None or req.constraints.min_radius_m <= 0:
            result.missing_fields.append(RuleFlag(
                rule_id="A1_min_radius",
                family="A",
                field="constraints.min_radius_m",
                message="Minimum channel radius is required",
                severity="required",
                rework_cost="high"
            ))
        
        if req.constraints.min_clearance_m is None or req.constraints.min_clearance_m <= 0:
            result.missing_fields.append(RuleFlag(
                rule_id="A1_min_clearance",
                family="A",
                field="constraints.min_clearance_m",
                message="Minimum clearance between channels is required",
                severity="required",
                rework_cost="high"
            ))
        
        if req.topology.target_terminals is None:
            result.missing_fields.append(RuleFlag(
                rule_id="A1_complexity",
                family="A",
                field="topology.target_terminals",
                message="Target complexity (terminal count) is needed",
                severity="required",
                rework_cost="medium"
            ))
        
        if req.inlets_outlets.inlets and not req.inlets_outlets.placement_rule:
            result.missing_fields.append(RuleFlag(
                rule_id="A1_inlet_placement",
                family="A",
                field="inlets_outlets.placement_rule",
                message="Inlet placement rule is required for deterministic generation",
                severity="required",
                rework_cost="high"
            ))
        
        if check_finalization:
            if req.embedding_export.voxel_pitch_m is None:
                result.missing_fields.append(RuleFlag(
                    rule_id="A3_voxel_pitch",
                    family="A",
                    field="embedding_export.voxel_pitch_m",
                    message="Voxel pitch is required for embedding",
                    severity="required",
                    rework_cost="medium"
                ))
    
    def _evaluate_ambiguity_rules(
        self,
        req: ObjectRequirements,
        intent: str,
        parser: IntentParser,
        result: RuleEvaluationResult
    ) -> None:
        """Family B: Check for ambiguous user language."""
        
        if parser.has_spatial_ambiguity() and not req.frame_of_reference.confirmed:
            result.ambiguity_flags.append(RuleFlag(
                rule_id="B1_spatial",
                family="B",
                field="frame_of_reference",
                message="Spatial terms used without coordinate convention",
                severity="required",
                rework_cost="high"
            ))
        
        vague_terms = parser.get_vague_terms()
        for term in vague_terms:
            if term in self.VAGUE_TO_NUMERIC:
                mappings = self.VAGUE_TO_NUMERIC[term]
                for field_name in mappings:
                    result.ambiguity_flags.append(RuleFlag(
                        rule_id=f"B2_vague_{term}",
                        family="B",
                        field=field_name,
                        message=f"'{term}' needs numeric mapping for {field_name}",
                        severity="warning",
                        rework_cost="medium"
                    ))
        
        if parser.has_implicit_io() and not req.inlets_outlets.inlets:
            result.ambiguity_flags.append(RuleFlag(
                rule_id="B3_implicit_io",
                family="B",
                field="inlets_outlets",
                message="Perfusion/flow mentioned but I/O geometry not specified",
                severity="required",
                rework_cost="high"
            ))
        
        if parser.has_symmetry_ambiguity():
            result.ambiguity_flags.append(RuleFlag(
                rule_id="B4_symmetry",
                family="B",
                field="geometry.symmetry_axis",
                message="Symmetry mentioned without specifying axis/plane",
                severity="warning",
                rework_cost="low"
            ))
    
    def _evaluate_conflict_rules(
        self,
        req: ObjectRequirements,
        result: RuleEvaluationResult
    ) -> None:
        """Family C: Check for conflicts and feasibility issues."""
        
        if (req.constraints.min_radius_m and req.embedding_export.voxel_pitch_m and
            req.constraints.min_radius_m < req.embedding_export.voxel_pitch_m):
            result.conflict_flags.append(RuleFlag(
                rule_id="C1_printability",
                family="C",
                field="constraints.min_radius_m",
                message=f"Min radius ({req.constraints.min_radius_m*1000:.2f}mm) smaller than voxel pitch ({req.embedding_export.voxel_pitch_m*1000:.2f}mm)",
                severity="required",
                rework_cost="high"
            ))
        
        if req.topology.target_terminals and req.constraints.min_clearance_m:
            domain_volume = req.domain.size_m[0] * req.domain.size_m[1] * req.domain.size_m[2]
            clearance_volume = (req.constraints.min_clearance_m ** 3) * req.topology.target_terminals
            if clearance_volume > domain_volume * 0.5:
                result.conflict_flags.append(RuleFlag(
                    rule_id="C2_clearance",
                    family="C",
                    field="topology.target_terminals",
                    message=f"Target terminals ({req.topology.target_terminals}) may be too dense for clearance constraint",
                    severity="warning",
                    rework_cost="medium"
                ))
        
        if req.topology.target_terminals and req.topology.target_terminals > 2000:
            result.conflict_flags.append(RuleFlag(
                rule_id="C3_complexity",
                family="C",
                field="topology.target_terminals",
                message=f"High terminal count ({req.topology.target_terminals}) may cause long runtime",
                severity="warning",
                rework_cost="medium"
            ))
        
        if req.inlets_outlets.inlets:
            for inlet in req.inlets_outlets.inlets:
                domain_min = min(req.domain.size_m) / 2
                if inlet.radius_m > domain_min * 0.5:
                    result.conflict_flags.append(RuleFlag(
                        rule_id="C4_io_feasibility",
                        family="C",
                        field="inlets_outlets.inlets",
                        message=f"Inlet radius ({inlet.radius_m*1000:.2f}mm) too large for domain",
                        severity="required",
                        rework_cost="high"
                    ))
    
    def _propose_defaults(
        self,
        req: ObjectRequirements,
        result: RuleEvaluationResult
    ) -> None:
        """Propose defaults for missing fields."""
        
        missing_field_names = {f.field for f in result.missing_fields}
        
        for field_path, (default_val, reason) in self.DEFAULT_VALUES.items():
            if field_path in missing_field_names or self._field_is_empty(req, field_path):
                result.proposed_defaults.append(ProposedDefault(
                    field=field_path,
                    value=default_val,
                    reason=reason
                ))
    
    def _field_is_empty(self, req: ObjectRequirements, field_path: str) -> bool:
        """Check if a field is empty/None."""
        parts = field_path.split(".")
        obj = req
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return True
        return obj is None or obj == "" or obj == []


class QuestionPlanner:
    """
    Plans minimal questions based on rule evaluation results.
    
    Questions are ranked by rework cost (high → medium → low) and
    only the necessary questions are asked.
    """
    
    FIELD_TO_QUESTION = {
        "frame_of_reference": ("coordinate_frame", "Which axis is length/height/width? (x=length, y=depth, z=height)", None),
        "inlets_outlets.inlets": ("num_inlets", "How many inlets?", "1"),
        "inlets_outlets.outlets": ("num_outlets", "How many outlets?", "0"),
        "inlets_outlets.placement_rule": ("inlet_placement", "Inlet placement? (x_min_face/x_max_face/y_min_face/y_max_face/z_min_face/z_max_face/center/custom)", "x_min_face"),
        "inlets_outlets.outlet_placement": ("outlet_placement", "Outlet placement? (x_min_face/x_max_face/y_min_face/y_max_face/z_min_face/z_max_face/opposite_inlet/same_face/none)", "opposite_inlet"),
        "constraints.min_radius_m": ("min_channel_radius", "Minimum channel radius (mm)?", "0.1"),
        "constraints.min_clearance_m": ("min_clearance", "Minimum spacing between channels (mm)?", "0.2"),
        "topology.target_terminals": ("target_terminals", "Target complexity: ~200, ~500, or ~1000 terminals?", "200"),
        "domain.size_m": ("domain_size", "Use default box 0.02x0.06x0.03 m? (yes/custom)", "yes"),
        "embedding_export.voxel_pitch_m": ("voxel_pitch", "Voxel pitch for embedding (mm)?", "0.3"),
        "embedding_export.stl_units": ("export_units", "Export units? (mm/cm)", "mm"),
        "geometry.tortuosity": ("tortuosity", "Vessel tortuosity? (low/medium/high)", "low"),
        "geometry.symmetry_axis": ("symmetry_axis", "Symmetric around which axis? (x/y/z/none)", "none"),
    }
    
    def plan(
        self,
        eval_result: RuleEvaluationResult,
        max_questions: int = 4
    ) -> List[PlannedQuestion]:
        """
        Plan minimal questions based on evaluation result.
        
        Returns at most max_questions, prioritized by rework cost.
        """
        questions = []
        seen_fields = set()
        
        all_flags = eval_result.all_flags()
        
        for flag in all_flags:
            if flag.field in seen_fields:
                continue
            if len(questions) >= max_questions:
                break
            
            if flag.field in self.FIELD_TO_QUESTION:
                q_key, q_text, q_default = self.FIELD_TO_QUESTION[flag.field]
                questions.append(PlannedQuestion(
                    field=flag.field,
                    question_text=q_text,
                    default_value=q_default,
                    rework_cost=flag.rework_cost,
                    reason=flag.message
                ))
                seen_fields.add(flag.field)
        
        for proposed in eval_result.proposed_defaults:
            if proposed.field in seen_fields:
                continue
            if len(questions) >= max_questions:
                break
            
            if proposed.field in self.FIELD_TO_QUESTION:
                q_key, q_text, q_default = self.FIELD_TO_QUESTION[proposed.field]
                questions.append(PlannedQuestion(
                    field=proposed.field,
                    question_text=f"Use default {proposed.value}? ({proposed.reason})",
                    default_value="yes",
                    rework_cost="low",
                    reason=proposed.reason
                ))
                seen_fields.add(proposed.field)
        
        return questions
    
    def format_turn_output(
        self,
        requirements: ObjectRequirements,
        eval_result: RuleEvaluationResult,
        questions: List[PlannedQuestion]
    ) -> str:
        """
        Format the agent's turn output with:
        1. Current requirements snapshot
        2. Missing/ambiguous/conflict flags
        3. Proposed defaults
        4. Questions to ask
        """
        lines = []
        
        lines.append("=== Current Requirements Snapshot ===")
        lines.append(f"Domain: {requirements.domain.type} {requirements.domain.size_m}")
        lines.append(f"Inlets: {len(requirements.inlets_outlets.inlets)}")
        lines.append(f"Outlets: {len(requirements.inlets_outlets.outlets)}")
        lines.append(f"Min radius: {requirements.constraints.min_radius_m}")
        lines.append(f"Min clearance: {requirements.constraints.min_clearance_m}")
        lines.append(f"Target terminals: {requirements.topology.target_terminals}")
        lines.append("")
        
        if eval_result.missing_fields or eval_result.ambiguity_flags or eval_result.conflict_flags:
            lines.append("=== Issues to Address ===")
            for flag in eval_result.all_flags():
                prefix = {"required": "[!]", "warning": "[?]", "info": "[i]"}.get(flag.severity, "[-]")
                lines.append(f"{prefix} {flag.message}")
            lines.append("")
        
        if eval_result.proposed_defaults:
            lines.append("=== Proposed Defaults ===")
            for prop in eval_result.proposed_defaults:
                lines.append(f"  {prop.field}: {prop.value} ({prop.reason})")
            lines.append("")
        
        if questions:
            lines.append("=== Questions ===")
            for i, q in enumerate(questions, 1):
                default_str = f" [{q.default_value}]" if q.default_value else ""
                lines.append(f"{i}. {q.question_text}{default_str}")
            lines.append("")
            lines.append("(Say 'use defaults' to accept all proposed defaults)")
        
        return "\n".join(lines)


FIELD_VALIDATORS = {
    "num_inlets": ("integer", "Please enter a whole number (e.g., 1, 2, 3)"),
    "num_outlets": ("integer", "Please enter a whole number (e.g., 0, 1, 2)"),
    "min_channel_radius": ("float", "Please enter a number in mm (e.g., 0.1, 0.5)"),
    "min_clearance": ("float", "Please enter a number in mm (e.g., 0.2, 0.5)"),
    "target_terminals": ("integer", "Please enter a whole number (e.g., 200, 500, 1000)"),
    "voxel_pitch": ("float", "Please enter a number in mm (e.g., 0.3, 0.5)"),
}


def _validate_answer(field: str, answer: str) -> Tuple[bool, Optional[str]]:
    """
    Validate an answer for a specific field.
    
    Returns (is_valid, error_message).
    If valid, error_message is None.
    """
    if field not in FIELD_VALIDATORS:
        return True, None
    
    validator_type, error_hint = FIELD_VALIDATORS[field]
    
    answer = answer.strip()
    if not answer:
        return True, None
    
    import re
    numeric_match = re.match(r'^([0-9]*\.?[0-9]+)', answer)
    if numeric_match:
        answer = numeric_match.group(1)
    
    if validator_type == "integer":
        try:
            int(float(answer))
            return True, None
        except ValueError:
            return False, f"Invalid input: '{answer}' is not a valid integer. {error_hint}"
    
    elif validator_type == "float":
        try:
            float(answer)
            return True, None
        except ValueError:
            return False, f"Invalid input: '{answer}' is not a valid number. {error_hint}"
    
    return True, None


def _get_field_key_from_question(question: 'PlannedQuestion') -> str:
    """Get the field key used for validation from a PlannedQuestion."""
    field_to_key = {
        "inlets_outlets.inlets": "num_inlets",
        "inlets_outlets.outlets": "num_outlets",
        "constraints.min_radius_m": "min_channel_radius",
        "constraints.min_clearance_m": "min_clearance",
        "topology.target_terminals": "target_terminals",
        "embedding_export.voxel_pitch_m": "voxel_pitch",
    }
    return field_to_key.get(question.field, question.field)


def run_rule_based_capture(
    requirements: ObjectRequirements,
    intent: str,
    organ_type: str = "generic",
    verbose: bool = True
) -> Tuple[ObjectRequirements, Dict[str, Any]]:
    """
    Run the rule-based requirements capture loop.
    
    This replaces the fixed question groups with an adaptive loop that:
    1. Parses user intent to extract explicit values
    2. Runs validators (missing, ambiguity, conflicts)
    3. Generates minimal questions
    4. Repeats until generation-ready
    
    The loop includes safeguards against infinite loops:
    - Tracks previously asked question sets to detect repetition
    - Automatically applies defaults if same questions asked 2+ times
    - Maximum iteration limit as final safeguard
    
    Returns the updated requirements and collected answers.
    """
    engine = RuleEngine(organ_type)
    planner = QuestionPlanner()
    parser = IntentParser(intent)
    collected_answers: Dict[str, Any] = {}
    
    for field_name, values in parser.extracted_values.items():
        if field_name == "box_size" and values:
            try:
                size = (float(values[0]), float(values[1]), float(values[2]))
                unit = values[3] if len(values) > 3 else "mm"
                scale = {"mm": 0.001, "cm": 0.01, "m": 1.0}.get(unit, 0.001)
                requirements.domain.size_m = tuple(s * scale for s in size)
                collected_answers["domain_size"] = f"{size[0]}x{size[1]}x{size[2]} {unit}"
            except (ValueError, IndexError):
                pass
        elif field_name == "diameter" and values:
            try:
                val = float(values[0])
                unit = values[1] if len(values) > 1 else "mm"
                scale = {"mm": 0.001, "cm": 0.01, "um": 0.000001}.get(unit, 0.001)
                requirements.constraints.min_radius_m = (val / 2) * scale
                collected_answers["min_channel_radius"] = f"{val/2} {unit}"
            except (ValueError, IndexError):
                pass
        elif field_name == "count" and values:
            try:
                count = int(values[0])
                if "inlet" in intent.lower():
                    collected_answers["num_inlets"] = str(count)
                elif "outlet" in intent.lower():
                    collected_answers["num_outlets"] = str(count)
            except (ValueError, IndexError):
                pass
    
    max_iterations = 10
    iteration = 0
    
    previous_question_sets: List[frozenset] = []
    repeat_count = 0
    max_repeats = 2
    
    while iteration < max_iterations:
        iteration += 1
        
        eval_result = engine.evaluate(requirements, intent)
        
        if eval_result.is_generation_ready:
            if verbose:
                print("\nRequirements are complete. Ready for generation.")
            break
        
        questions = planner.plan(eval_result, max_questions=4)
        
        if not questions:
            if verbose:
                print("\nNo more questions needed. Proceeding with defaults.")
            for prop in eval_result.proposed_defaults:
                _apply_default_to_requirements(requirements, prop)
            break
        
        current_question_set = frozenset(q.field for q in questions)
        
        if current_question_set in previous_question_sets:
            repeat_count += 1
            if verbose:
                print(f"\n[Warning] Same questions detected (repeat #{repeat_count})")
            
            if repeat_count >= max_repeats:
                if verbose:
                    print("\n[Info] Breaking loop - applying defaults to avoid infinite loop.")
                for prop in eval_result.proposed_defaults:
                    _apply_default_to_requirements(requirements, prop)
                    collected_answers[prop.field] = prop.value
                
                for q in questions:
                    if q.default_value and q.field not in collected_answers:
                        collected_answers[q.field] = q.default_value
                        _apply_answer_to_requirements(requirements, q.field, q.default_value)
                break
        else:
            repeat_count = 0
            previous_question_sets.append(current_question_set)
        
        if verbose:
            output = planner.format_turn_output(requirements, eval_result, questions)
            print(output)
        
        for q in questions:
            default_str = f" [{q.default_value}]" if q.default_value else ""
            field_key = _get_field_key_from_question(q)
            
            max_validation_attempts = 3
            for attempt in range(max_validation_attempts):
                answer = input(f"  {q.question_text}{default_str}: ").strip()
                
                if answer.lower() in ("quit", "exit"):
                    return requirements, collected_answers
                
                if answer.lower() == "use defaults":
                    for prop in eval_result.proposed_defaults:
                        _apply_default_to_requirements(requirements, prop)
                        collected_answers[prop.field] = prop.value
                    break
                
                if not answer and q.default_value:
                    answer = q.default_value
                
                is_valid, error_msg = _validate_answer(field_key, answer)
                if is_valid:
                    break
                else:
                    print(f"  [Error] {error_msg}")
                    if attempt < max_validation_attempts - 1:
                        print(f"  Please try again ({max_validation_attempts - attempt - 1} attempts remaining).")
                    else:
                        print(f"  Using default value: {q.default_value}")
                        answer = q.default_value if q.default_value else answer
            else:
                if answer.lower() == "use defaults":
                    break
            
            if answer.lower() == "use defaults":
                break
            
            collected_answers[q.field] = answer
            _apply_answer_to_requirements(requirements, q.field, answer)
    
    if iteration >= max_iterations:
        if verbose:
            print(f"\n[Warning] Reached max iterations ({max_iterations}). Applying remaining defaults.")
        eval_result = engine.evaluate(requirements, intent)
        for prop in eval_result.proposed_defaults:
            if prop.field not in collected_answers:
                _apply_default_to_requirements(requirements, prop)
                collected_answers[prop.field] = prop.value
    
    return requirements, collected_answers


def _parse_domain_size_string(value: str) -> Optional[Tuple[float, float, float]]:
    """
    Parse a domain size string into a tuple of floats in meters.
    
    Supports formats:
    - "0.02,0.06,0.03 m" or "0.02, 0.06, 0.03m"
    - "20,60,30 mm" or "20, 60, 30mm"
    - "2x6x3 cm" or "2 x 6 x 3 cm"
    - "2 by 6 by 3 cm"
    
    Returns None if parsing fails.
    """
    import re
    
    value = value.strip().lower()
    
    unit_multipliers = {
        "m": 1.0,
        "cm": 0.01,
        "mm": 0.001,
    }
    
    unit = "m"
    for u in ["mm", "cm", "m"]:
        if value.endswith(u):
            unit = u
            value = value[:-len(u)].strip()
            break
    
    multiplier = unit_multipliers.get(unit, 1.0)
    
    separators = [r"\s*x\s*", r"\s*by\s*", r"\s*,\s*"]
    
    for sep in separators:
        parts = re.split(sep, value)
        if len(parts) == 3:
            try:
                dims = [float(p.strip()) * multiplier for p in parts]
                return (dims[0], dims[1], dims[2])
            except ValueError:
                continue
    
    return None


def _apply_default_to_requirements(req: ObjectRequirements, prop: ProposedDefault) -> None:
    """Apply a proposed default to requirements."""
    _apply_answer_to_requirements(req, prop.field, prop.value)


def _apply_answer_to_requirements(req: ObjectRequirements, field: str, value: Any) -> None:
    """Apply an answer to the appropriate requirements field."""
    if field == "constraints.min_radius_m":
        try:
            if isinstance(value, str):
                req.constraints.min_radius_m = float(value) / 1000
            else:
                req.constraints.min_radius_m = float(value)
        except (ValueError, TypeError):
            pass
    
    elif field == "constraints.min_clearance_m":
        try:
            if isinstance(value, str):
                req.constraints.min_clearance_m = float(value) / 1000
            else:
                req.constraints.min_clearance_m = float(value)
        except (ValueError, TypeError):
            pass
    
    elif field == "topology.target_terminals":
        try:
            if isinstance(value, str):
                req.topology.target_terminals = int(value)
            else:
                req.topology.target_terminals = int(value)
        except (ValueError, TypeError):
            pass
    
    elif field == "domain.size_m":
        if isinstance(value, tuple):
            req.domain.size_m = value
        elif isinstance(value, str):
            val_lower = value.lower().strip()
            if val_lower in ("yes", "y", "default"):
                pass
            else:
                parsed = _parse_domain_size_string(val_lower)
                if parsed:
                    req.domain.size_m = parsed
    
    elif field == "embedding_export.voxel_pitch_m":
        try:
            if isinstance(value, str):
                req.embedding_export.voxel_pitch_m = float(value) / 1000
            else:
                req.embedding_export.voxel_pitch_m = float(value)
        except (ValueError, TypeError):
            pass
    
    elif field == "embedding_export.stl_units":
        if isinstance(value, str):
            req.embedding_export.stl_units = value
    
    elif field == "geometry.tortuosity":
        tortuosity_map = {"low": 0.1, "medium": 0.5, "high": 0.9}
        if isinstance(value, str):
            req.geometry.tortuosity = tortuosity_map.get(value.lower(), 0.1)
        else:
            req.geometry.tortuosity = float(value)
    
    elif field == "inlets_outlets.inlets":
        try:
            if isinstance(value, str):
                num = int(value)
            else:
                num = int(value)
            if num > 0 and not req.inlets_outlets.inlets:
                for i in range(num):
                    req.inlets_outlets.inlets.append(PortSpec(
                        name=f"inlet_{i+1}",
                        radius_m=0.002
                    ))
        except (ValueError, TypeError):
            pass
    
    elif field == "inlets_outlets.outlets":
        try:
            if isinstance(value, str):
                num = int(value)
            else:
                num = int(value)
            if num > 0 and not req.inlets_outlets.outlets:
                for i in range(num):
                    req.inlets_outlets.outlets.append(PortSpec(
                        name=f"outlet_{i+1}",
                        radius_m=0.001
                    ))
        except (ValueError, TypeError):
            pass
    
    elif field == "inlets_outlets.placement_rule":
        if isinstance(value, str):
            placement = value.lower().strip()
            valid_placements = [
                "x_min_face", "x_max_face", "y_min_face", "y_max_face",
                "z_min_face", "z_max_face", "center", "custom"
            ]
            if placement in valid_placements:
                req.inlets_outlets.placement_rule = placement
                _apply_placement_to_inlets(req, placement)
            else:
                req.inlets_outlets.placement_rule = placement
    
    elif field == "inlets_outlets.outlet_placement":
        if isinstance(value, str):
            placement = value.lower().strip()
            _apply_placement_to_outlets(req, placement)
    
    # =========================================================================
    # New topology-specific question field mappings
    # =========================================================================
    
    elif field == "topology_kind":
        # Set topology kind (path, tree, loop, multi_tree)
        if isinstance(value, str):
            req.topology.topology_kind = value.lower().strip()
    
    elif field == "domain_dims":
        # Parse domain dimensions string like "20 60 30 mm"
        if isinstance(value, str):
            parsed = _parse_domain_size_string(value)
            if parsed:
                req.domain.size_m = parsed
    
    elif field == "inlet_face":
        # Set inlet face (x_min, x_max, y_min, y_max, z_min, z_max)
        if isinstance(value, str):
            face = value.lower().strip()
            # Create inlet if it doesn't exist
            if not req.inlets_outlets.inlets:
                req.inlets_outlets.inlets.append(PortSpec(name="inlet_1", radius_m=0.002))
            # Store face in placement_rule for now
            req.inlets_outlets.placement_rule = f"{face}_face"
            _apply_placement_to_inlets(req, f"{face}_face")
    
    elif field == "inlet_position":
        # Set inlet position (center or u,v offsets)
        if isinstance(value, str):
            pos = value.lower().strip()
            if pos != "center" and "," in pos:
                # Parse u,v offsets
                try:
                    parts = pos.split(",")
                    u = float(parts[0].strip()) / 1000  # mm to m
                    v = float(parts[1].strip()) / 1000
                    if req.inlets_outlets.inlets:
                        req.inlets_outlets.inlets[0].position_m = (u, v, 0)
                except (ValueError, IndexError):
                    pass
    
    elif field == "inlet_radius":
        # Set inlet radius in mm
        if isinstance(value, str):
            try:
                radius_mm = float(value.strip())
                if not req.inlets_outlets.inlets:
                    req.inlets_outlets.inlets.append(PortSpec(name="inlet_1", radius_m=radius_mm / 1000))
                else:
                    req.inlets_outlets.inlets[0].radius_m = radius_mm / 1000
            except ValueError:
                pass
    
    elif field == "outlet_face":
        # Set outlet face (x_min, x_max, y_min, y_max, z_min, z_max, or 'none')
        if isinstance(value, str):
            face = value.lower().strip()
            if face != "none":
                # Create outlet if it doesn't exist
                if not req.inlets_outlets.outlets:
                    req.inlets_outlets.outlets.append(PortSpec(name="outlet_1", radius_m=0.002))
                _apply_placement_to_outlets(req, f"{face}_face")
    
    elif field == "outlet_position":
        # Set outlet position (center or u,v offsets)
        if isinstance(value, str):
            pos = value.lower().strip()
            if pos != "center" and "," in pos:
                try:
                    parts = pos.split(",")
                    u = float(parts[0].strip()) / 1000
                    v = float(parts[1].strip()) / 1000
                    if req.inlets_outlets.outlets:
                        req.inlets_outlets.outlets[0].position_m = (u, v, 0)
                except (ValueError, IndexError):
                    pass
    
    elif field == "outlet_radius":
        # Set outlet radius in mm
        if isinstance(value, str):
            try:
                radius_mm = float(value.strip())
                if req.inlets_outlets.outlets:
                    req.inlets_outlets.outlets[0].radius_m = radius_mm / 1000
            except ValueError:
                pass
    
    elif field == "route_type":
        # Set path route type (straight, single_bend, s_curve, via_points)
        if isinstance(value, str):
            req.geometry.route_type = value.lower().strip()
    
    elif field == "bend_radius":
        # Set minimum bend radius in mm
        if isinstance(value, str):
            val = value.lower().strip()
            if val != "auto":
                try:
                    req.geometry.bend_radius_m = float(val) / 1000
                except ValueError:
                    pass
    
    elif field == "terminal_mode":
        # Set terminal mode (a = inlet→terminals, b = inlet→outlet with branches)
        if isinstance(value, str):
            mode = value.lower().strip()
            req.topology.terminal_mode = mode
    
    elif field == "target_terminals":
        # Set target terminal count (can be number or 'sparse'/'medium'/'dense')
        if isinstance(value, str):
            val = value.lower().strip()
            density_map = {"sparse": 20, "medium": 50, "dense": 100}
            if val in density_map:
                req.topology.target_terminals = density_map[val]
            elif val not in ("any", "n/a", ""):
                try:
                    req.topology.target_terminals = int(val)
                except ValueError:
                    pass
    
    elif field == "max_depth":
        # Set maximum branching depth
        if isinstance(value, str):
            try:
                req.topology.max_depth = int(value.strip())
            except ValueError:
                pass
    
    elif field == "branching_style":
        # Set branching style (balanced, aggressive_early, space_filling)
        if isinstance(value, str):
            req.topology.style = value.lower().strip()
    
    elif field == "tapering":
        # Set radius tapering profile (murray, linear, fixed)
        if isinstance(value, str):
            req.geometry.tapering = value.lower().strip()
    
    elif field == "min_radius":
        # Set minimum terminal radius in mm
        if isinstance(value, str):
            try:
                req.constraints.min_radius_m = float(value.strip()) / 1000
            except ValueError:
                pass
    
    elif field == "min_clearance":
        # Set minimum clearance in mm
        if isinstance(value, str):
            try:
                req.constraints.min_clearance_m = float(value.strip()) / 1000
            except ValueError:
                pass
    
    elif field == "boundary_margin":
        # Set boundary margin in mm
        if isinstance(value, str):
            try:
                req.constraints.boundary_margin_m = float(value.strip()) / 1000
            except ValueError:
                pass
    
    elif field == "embed_requested":
        # Set whether embedding is requested
        if isinstance(value, str):
            val = value.lower().strip()
            req.embedding_export.embed_requested = val in ("yes", "y", "true", "1")
    
    elif field == "min_feature_size":
        # Set minimum feature size / nozzle diameter in mm
        # This is used to compute suggested voxel pitch
        if isinstance(value, str):
            try:
                feature_mm = float(value.strip())
                # Suggested voxel pitch = feature_size / 3 (for good resolution)
                req.embedding_export.min_feature_size_m = feature_mm / 1000
                # Auto-compute suggested voxel pitch
                suggested_pitch = feature_mm / 3 / 1000  # 3 voxels per feature
                if req.embedding_export.voxel_pitch_m is None:
                    req.embedding_export.voxel_pitch_m = suggested_pitch
            except ValueError:
                pass
    
    elif field == "voxel_pitch_confirm":
        # Confirm or customize voxel pitch
        if isinstance(value, str):
            val = value.lower().strip()
            if val not in ("yes", "y"):
                # User wants custom pitch - parse it
                try:
                    custom_pitch_mm = float(val)
                    req.embedding_export.voxel_pitch_m = custom_pitch_mm / 1000
                except ValueError:
                    pass
    
    elif field == "watertight_required":
        # Set whether watertight mesh is required
        if isinstance(value, str):
            val = value.lower().strip()
            req.acceptance_criteria.watertight_required = val in ("yes", "y", "true", "1")
    
    elif field == "terminal_range":
        # Set acceptable terminal count range
        if isinstance(value, str):
            val = value.lower().strip()
            if val not in ("any", ""):
                req.acceptance_criteria.terminal_range = val
    
    elif field == "wall_thickness":
        # Set wall thickness for hollow channels
        if isinstance(value, str):
            val = value.lower().strip()
            if val != "solid":
                try:
                    req.geometry.wall_thickness_m = float(val) / 1000
                except ValueError:
                    pass
    
    elif field == "ban_self_intersection":
        # Set whether self-intersections are banned
        if isinstance(value, str):
            val = value.lower().strip()
            req.constraints.ban_self_intersection = val in ("yes", "y", "true", "1")
    
    elif field == "enforce_clearance":
        # Set whether clearance is enforced
        if isinstance(value, str):
            val = value.lower().strip()
            req.constraints.enforce_clearance = val in ("yes", "y", "true", "1")
    
    elif field == "loop_count":
        # Set number of loops for loop topology
        if isinstance(value, str):
            try:
                req.topology.loop_count = int(value.strip())
            except ValueError:
                pass
    
    elif field == "loop_style":
        # Set loop style (single_loop, mesh, ladder)
        if isinstance(value, str):
            req.topology.loop_style = value.lower().strip()
    
    elif field == "tree_count":
        # Set number of trees for multi_tree topology
        if isinstance(value, str):
            try:
                req.topology.tree_count = int(value.strip())
            except ValueError:
                pass
    
    elif field == "shared_inlet":
        # Set whether trees share an inlet
        if isinstance(value, str):
            val = value.lower().strip()
            req.topology.shared_inlet = val in ("yes", "y", "true", "1")


def _apply_placement_to_inlets(req: ObjectRequirements, placement: str) -> None:
    """Apply placement rule to inlet positions based on domain size."""
    if not req.inlets_outlets.inlets:
        return
    
    domain_size = req.domain.size_m or (0.02, 0.06, 0.03)
    x_size, y_size, z_size = domain_size
    
    placement_positions = {
        "x_min_face": (0.0, y_size / 2, z_size / 2),
        "x_max_face": (x_size, y_size / 2, z_size / 2),
        "y_min_face": (x_size / 2, 0.0, z_size / 2),
        "y_max_face": (x_size / 2, y_size, z_size / 2),
        "z_min_face": (x_size / 2, y_size / 2, 0.0),
        "z_max_face": (x_size / 2, y_size / 2, z_size),
        "center": (x_size / 2, y_size / 2, z_size / 2),
    }
    
    if placement in placement_positions:
        base_pos = placement_positions[placement]
        for i, inlet in enumerate(req.inlets_outlets.inlets):
            offset = i * 0.002
            inlet.position_m = (base_pos[0], base_pos[1] + offset, base_pos[2])


def _apply_placement_to_outlets(req: ObjectRequirements, placement: str) -> None:
    """Apply placement rule to outlet positions based on domain size and inlet placement."""
    if not req.inlets_outlets.outlets:
        return
    
    domain_size = req.domain.size_m or (0.02, 0.06, 0.03)
    x_size, y_size, z_size = domain_size
    
    inlet_placement = req.inlets_outlets.placement_rule or "x_min_face"
    
    opposite_map = {
        "x_min_face": "x_max_face",
        "x_max_face": "x_min_face",
        "y_min_face": "y_max_face",
        "y_max_face": "y_min_face",
        "z_min_face": "z_max_face",
        "z_max_face": "z_min_face",
        "center": "x_max_face",
    }
    
    placement_positions = {
        "x_min_face": (0.0, y_size / 2, z_size / 2),
        "x_max_face": (x_size, y_size / 2, z_size / 2),
        "y_min_face": (x_size / 2, 0.0, z_size / 2),
        "y_max_face": (x_size / 2, y_size, z_size / 2),
        "z_min_face": (x_size / 2, y_size / 2, 0.0),
        "z_max_face": (x_size / 2, y_size / 2, z_size),
    }
    
    if placement == "opposite_inlet":
        placement = opposite_map.get(inlet_placement, "x_max_face")
    elif placement == "same_face":
        placement = inlet_placement
    elif placement == "none":
        return
    
    if placement in placement_positions:
        base_pos = placement_positions[placement]
        for i, outlet in enumerate(req.inlets_outlets.outlets):
            offset = i * 0.002
            outlet.position_m = (base_pos[0], base_pos[1] + offset, base_pos[2])


# =============================================================================
# Main Workflow Class
# =============================================================================

class SingleAgentOrganGeneratorV2:
    """
    Single Agent Organ Generator V2 - Stateful workflow for organ structure generation.
    
    This workflow implements an interactive, LLM-driven process for generating
    organ vascular structures with per-object folder structure, schema-gated
    requirements capture, and ability to pull context from previous attempts.
    
    V2 Features:
    - Agent dialogue system (Understand -> Plan -> Ask)
    - Dynamic schema with activatable modules
    - LLM healthcheck and circuit breaker
    - Iteration feedback integration
    """
    
    WORKFLOW_NAME = "Single Agent Organ Generator V2"
    WORKFLOW_VERSION = "2.0.0"
    
    def __init__(
        self,
        agent: AgentRunner,
        base_output_dir: str = "./outputs",
        verbose: bool = True,
        execution_mode: ExecutionMode = DEFAULT_EXECUTION_MODE,
        timeout_seconds: float = 300.0,
        skip_healthcheck: bool = False,
    ):
        self.agent = agent
        self.base_output_dir = base_output_dir
        self.verbose = verbose
        self.execution_mode = execution_mode
        self.timeout_seconds = timeout_seconds
        self.state = WorkflowState.PROJECT_INIT
        self.context = ProjectContext()
        
        self.current_question_group: Optional[str] = None
        self.current_question_index: int = 0
        self.collected_answers: Dict[str, Any] = {}
        
        self.schema_manager = SchemaManager()
        self._assert_schema_manager_interface()
        
        self.current_understanding: Optional[UnderstandingReport] = None
        self.current_plans: Optional[List[PlanOption]] = None
        self.chosen_plan: Optional[PlanOption] = None
        
        self.prompt_session = ReactivePromptSession()
        
        # Progress watchdog: track (state, question_id) repetition counts
        # to detect infinite loops and provide clear diagnostics
        self._state_repetition_counts: Dict[Tuple[str, Optional[str]], int] = {}
        self._max_state_repetitions: int = 5  # Stop if same state/question repeated > N times
        self._last_state_key: Optional[Tuple[str, Optional[str]]] = None
        
        os.makedirs(base_output_dir, exist_ok=True)
    
    def _assert_schema_manager_interface(self) -> None:
        """
        Verify SchemaManager has the expected interface.
        
        Raises RuntimeError if required methods are missing, preventing
        cryptic errors later in the workflow.
        """
        required_methods = [
            "activate_module",
            "deactivate_module",
            "is_module_active",
            "get_active_modules",
            "missing_required_fields",
            "update_from_results",
            "plan_questions",
            "get_spec_summary",
        ]
        
        missing = []
        for method in required_methods:
            if not hasattr(self.schema_manager, method) or not callable(getattr(self.schema_manager, method)):
                missing.append(method)
        
        if missing:
            raise RuntimeError(
                f"SchemaManager is missing required methods: {', '.join(missing)}. "
                f"Please ensure SchemaManager implements the expected interface."
            )
    
    def run(self) -> ProjectContext:
        """Run the complete workflow interactively.
        
        Raises
        ------
        MissingCredentialsError
            If LLM API credentials are missing
        ProviderMisconfiguredError
            If LLM provider is misconfigured
        FatalLLMError
            If LLM ping fails fatally
        WorkflowStuckError
            If the workflow is stuck in an infinite loop
        """
        self._run_llm_preflight_check()
        self._print_header()
        
        while self.state != WorkflowState.COMPLETE:
            # Progress watchdog: check for infinite loops
            self._check_progress_watchdog()
            self._run_state()
        
        self._print_completion()
        return self.context
    
    def _check_progress_watchdog(self) -> None:
        """
        Check if the workflow is stuck in an infinite loop.
        
        Tracks (state, question_id) combinations and raises WorkflowStuckError
        if the same combination is repeated too many times without progress.
        
        Raises
        ------
        WorkflowStuckError
            If the workflow is stuck (same state/question repeated > max times)
        """
        # Build current state key: (state_name, current_question_id)
        question_id = None
        if self.current_question_group:
            question_id = f"{self.current_question_group}:{self.current_question_index}"
        
        state_key = (self.state.value, question_id)
        
        # Check if this is the same state as last iteration
        if state_key == self._last_state_key:
            # Increment repetition count
            self._state_repetition_counts[state_key] = (
                self._state_repetition_counts.get(state_key, 0) + 1
            )
            
            repetition_count = self._state_repetition_counts[state_key]
            
            if repetition_count > self._max_state_repetitions:
                # Diagnose the issue
                diagnosis = self._diagnose_stuck_workflow(state_key)
                
                # Print diagnostic message
                print()
                print("=" * 60)
                print("  WORKFLOW STUCK - INFINITE LOOP DETECTED")
                print("=" * 60)
                print()
                print(f"State: {self.state.value}")
                print(f"Question: {question_id or 'N/A'}")
                print(f"Repetitions: {repetition_count}")
                print()
                print("Diagnosis:")
                print(f"  {diagnosis}")
                print()
                
                raise WorkflowStuckError(
                    state=self.state.value,
                    question_id=question_id,
                    repetition_count=repetition_count,
                    diagnosis=diagnosis,
                )
        else:
            # State changed, reset tracking for new state
            self._last_state_key = state_key
            # Keep history but don't reset - we want to detect oscillations too
    
    def _diagnose_stuck_workflow(self, state_key: Tuple[str, Optional[str]]) -> str:
        """
        Diagnose why the workflow is stuck.
        
        Returns a human-readable diagnosis string.
        """
        state_name, question_id = state_key
        
        # Check for common issues
        diagnostics = []
        
        # Check if LLM is reachable
        if hasattr(self.agent, 'client') and self.agent.client is not None:
            diagnostics.append("LLM client is configured")
        else:
            diagnostics.append("LLM client may not be properly configured")
        
        # Check for missing required fields
        if hasattr(self, 'schema_manager'):
            try:
                missing = self.schema_manager.missing_required_fields()
                if missing:
                    diagnostics.append(f"Missing required fields: {', '.join(missing)}")
            except Exception:
                pass
        
        # Check current object state
        current_obj = self.context.get_current_object()
        if current_obj:
            diagnostics.append(f"Current object: {current_obj.name} (status: {current_obj.status})")
        else:
            diagnostics.append("No current object selected")
        
        # Check for parse failures (indicated by repeated questions)
        if question_id:
            diagnostics.append(
                f"Question '{question_id}' is being repeated - "
                "likely due to invalid input parsing or validation failure"
            )
        
        # Build diagnosis message
        if not diagnostics:
            return "Unknown cause - workflow is stuck without clear reason"
        
        return "; ".join(diagnostics)
    
    def _run_llm_preflight_check(self) -> None:
        """Run LLM preflight check before starting the workflow.
        
        This prevents infinite loops when the LLM isn't properly configured
        by performing hard readiness checks before any workflow starts.
        
        Raises
        ------
        MissingCredentialsError
            If LLM API credentials are missing
        ProviderMisconfiguredError
            If LLM provider is misconfigured
        FatalLLMError
            If LLM ping fails fatally
        """
        if hasattr(self.agent, 'client') and self.agent.client is not None:
            client = self.agent.client
            if hasattr(client, 'config'):
                config = client.config
                provider = getattr(config, 'provider', None)
                api_key = getattr(config, 'api_key', None)
                api_base = getattr(config, 'api_base', None)
                model = getattr(config, 'model', None)
                
                if provider:
                    if self.verbose:
                        print()
                        print("=" * 60)
                        print("  LLM Preflight Check")
                        print("=" * 60)
                    
                    try:
                        assert_llm_ready(
                            provider=provider,
                            api_key=api_key,
                            api_base=api_base,
                            model=model,
                            skip_ping=False,
                            verbose=self.verbose
                        )
                        if self.verbose:
                            print()
                    except (MissingCredentialsError, ProviderMisconfiguredError, FatalLLMError) as e:
                        print()
                        print("=" * 60)
                        print("  LLM CONFIGURATION ERROR")
                        print("=" * 60)
                        print()
                        print(f"Error: {e}")
                        print()
                        print("The workflow cannot proceed without a working LLM connection.")
                        print("Please fix the configuration and try again.")
                        print()
                        raise
    
    def step(self, user_input: str) -> Tuple[WorkflowState, str]:
        """Process user input and advance workflow by one step."""
        return self._process_input(user_input)
    
    def get_state(self) -> WorkflowState:
        """Get current workflow state."""
        return self.state
    
    def get_context(self) -> ProjectContext:
        """Get current project context."""
        return self.context
    
    def save_state(self, filepath: str) -> None:
        """Save workflow state to file for later resumption."""
        state_data = {
            "workflow_name": self.WORKFLOW_NAME,
            "workflow_version": self.WORKFLOW_VERSION,
            "state": self.state.value,
            "context": self.context.to_dict(),
            "current_question_group": self.current_question_group,
            "current_question_index": self.current_question_index,
            "collected_answers": self.collected_answers,
            "timestamp": time.time(),
        }
        with open(filepath, 'w') as f:
            json.dump(state_data, f, indent=2)
        if self.verbose:
            print(f"State saved to: {filepath}")
    
    def load_state(self, filepath: str) -> None:
        """Load workflow state from file."""
        with open(filepath, 'r') as f:
            state_data = json.load(f)
        
        self.state = WorkflowState(state_data["state"])
        self.context = ProjectContext.from_dict(state_data["context"])
        self.current_question_group = state_data.get("current_question_group")
        self.current_question_index = state_data.get("current_question_index", 0)
        self.collected_answers = state_data.get("collected_answers", {})
        if self.verbose:
            print(f"State loaded from: {filepath}")
            print(f"Resuming at state: {self.state.value}")
    
    def _print_header(self) -> None:
        """Print workflow header."""
        print("=" * 60)
        print(f"  {self.WORKFLOW_NAME}")
        print(f"  Version: {self.WORKFLOW_VERSION}")
        print("=" * 60)
        print()
        print("This workflow will guide you through generating one or more")
        print("3D structures based on your description. Type 'quit' at any time to exit.")
        print()
        print("Core Principles:")
        print("  1. Spec-first, deterministic execution")
        print("  2. Schema-gated clarity")
        print("  3. Fixed frame of reference")
        print("  4. Focus on your actual requirements")
        print("  5. Iteration loop is per object")
        print()
    
    def _print_completion(self) -> None:
        """Print completion message."""
        print()
        print("=" * 60)
        print("  Project Complete!")
        print("=" * 60)
        print()
        print(f"Project: {self.context.project_name}")
        print(f"Output directory: {self.context.output_dir}")
        print()
        print("Objects generated:")
        for obj in self.context.objects:
            print(f"  - {obj.name} ({obj.slug}): {obj.status}")
            if obj.final_void_stl:
                print(f"    Final void STL: {obj.final_void_stl}")
            if obj.final_manifest:
                print(f"    Manifest: {obj.final_manifest}")
        print()
        print("Thank you for using the 3D Structure Generator!")
    
    def _run_state(self) -> None:
        """Run the current state's logic."""
        state_handlers = {
            WorkflowState.PROJECT_INIT: self._run_project_init,
            WorkflowState.OBJECT_PLANNING: self._run_object_planning,
            WorkflowState.FRAME_OF_REFERENCE: self._run_frame_of_reference,
            WorkflowState.REQUIREMENTS_CAPTURE: self._run_requirements_capture,
            WorkflowState.SPEC_COMPILATION: self._run_spec_compilation,
            WorkflowState.GENERATION: self._run_generation,
            WorkflowState.ANALYSIS_VALIDATION: self._run_analysis_validation,
            WorkflowState.ITERATION: self._run_iteration,
            WorkflowState.FINALIZATION: self._run_finalization,
        }
        
        handler = state_handlers.get(self.state)
        if handler:
            handler()
    
    def _run_project_init(self) -> None:
        """Run PROJECT_INIT state: Ask for project name and global defaults."""
        print("-" * 40)
        print("Step 0: Project Setup")
        print("-" * 40)
        print()
        
        project_name_help = create_question_help(
            question_id="project_name",
            question_text="Project name",
            default_value=f"project_{int(time.time())}",
            why_asking="The project name is used to create a folder for all outputs and to identify this generation session.",
            examples=["liver_vasculature", "kidney_network", "test_run_001"],
        )
        project_name, intent = self.prompt_session.ask_until_answer(
            question_id="project_name",
            question_text="Project name",
            default_value=f"project_{int(time.time())}",
            help_info=project_name_help,
            show_context=False,
        )
        if intent == TurnIntent.CANCEL:
            self.state = WorkflowState.COMPLETE
            return
        
        if not project_name:
            project_name = f"project_{int(time.time())}"
            print(f"Using default project name: {project_name}")
        
        project_slug = project_name.lower().replace(" ", "_").replace("-", "_")
        
        self.context.project_name = project_name
        self.context.project_slug = project_slug
        self.context.output_dir = os.path.join(self.base_output_dir, project_slug)
        
        os.makedirs(self.context.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.context.output_dir, "objects"), exist_ok=True)
        
        output_dir_help = create_question_help(
            question_id="output_dir",
            question_text="Output directory",
            default_value=self.context.output_dir,
            why_asking="The output directory is where all generated files, meshes, and reports will be saved.",
        )
        output_dir_input, intent = self.prompt_session.ask_until_answer(
            question_id="output_dir",
            question_text=f"Output directory",
            default_value=self.context.output_dir,
            help_info=output_dir_help,
            show_context=False,
        )
        if intent == TurnIntent.CANCEL:
            self.state = WorkflowState.COMPLETE
            return
        if output_dir_input and output_dir_input != self.context.output_dir:
            self.context.output_dir = output_dir_input
            os.makedirs(self.context.output_dir, exist_ok=True)
            os.makedirs(os.path.join(self.context.output_dir, "objects"), exist_ok=True)
        
        print()
        print("Global defaults:")
        print(f"  - Internal units: {self.context.units_internal}")
        print(f"  - Export units: {self.context.units_export}")
        domain = self.context.default_embed_domain
        print(f"  - Default embed domain: {domain[0]*1000:.0f}mm x {domain[1]*1000:.0f}mm x {domain[2]*1000:.0f}mm")
        print(f"  - Flow solver: {'ON' if self.context.flow_solver_enabled else 'OFF'}")
        print()
        print(GlobalDefaults.get_defaults_text())
        
        defaults_help = create_question_help(
            question_id="change_defaults",
            question_text="Change any defaults?",
            default_value="no",
            options=["yes", "no"],
            why_asking="Global defaults provide sensible starting values for voxel pitch, clearances, and other parameters. You can customize them if needed.",
            impact="Choosing 'yes' will prompt you for each parameter individually.",
        )
        
        change_answer, intent = self.prompt_session.ask_yes_no(
            question_id="change_defaults",
            question_text="Change any defaults?",
            default=False,
            help_info=defaults_help,
        )
        
        if intent == TurnIntent.CANCEL:
            self.state = WorkflowState.COMPLETE
            return
        
        if change_answer:
            units_help = create_question_help(
                question_id="export_units",
                question_text="Export units",
                default_value=self.context.units_export,
                options=["mm", "cm", "m", "um"],
                why_asking="Export units determine the scale of the output STL files.",
            )
            units_answer, _ = self.prompt_session.ask_until_answer(
                question_id="export_units",
                question_text="Export units (mm/cm/m/um)",
                default_value=self.context.units_export,
                allowed_values=["mm", "cm", "m", "um"],
                help_info=units_help,
                show_context=False,
            )
            if units_answer in ("mm", "cm", "m", "um"):
                self.context.units_export = units_answer
                self.prompt_session.show_change_summary("units_export", None, units_answer)
            
            def validate_domain_input(value: str) -> Tuple[bool, str]:
                """Validate domain input format (L,W,H in meters)."""
                if not value:
                    return True, ""  # Empty is OK, keeps default
                try:
                    parts = [float(x.strip()) for x in value.split(",")]
                    if len(parts) != 3:
                        return False, "Expected 3 values (L,W,H) separated by commas"
                    if any(p <= 0 for p in parts):
                        return False, "All dimensions must be positive"
                    return True, ""
                except ValueError:
                    return False, "Invalid number format. Use comma-separated decimals (e.g., 0.02,0.06,0.03)"
            
            domain_help = create_question_help(
                question_id="default_embed_domain",
                question_text="Default embed domain (L,W,H in meters)",
                default_value=f"{self.context.default_embed_domain[0]},{self.context.default_embed_domain[1]},{self.context.default_embed_domain[2]}",
                why_asking="The embed domain defines the bounding box for the generated structure.",
                examples=["0.02,0.06,0.03", "0.05,0.05,0.05"],
            )
            domain_input, _ = self.prompt_session.ask_until_answer(
                question_id="default_embed_domain",
                question_text="Default embed domain (L,W,H in meters)",
                default_value=f"{self.context.default_embed_domain[0]},{self.context.default_embed_domain[1]},{self.context.default_embed_domain[2]}",
                help_info=domain_help,
                validator=validate_domain_input,
                show_context=False,
            )
            if domain_input:
                parts = [float(x.strip()) for x in domain_input.split(",")]
                if len(parts) == 3:
                    self.context.default_embed_domain = tuple(parts)
                    self.prompt_session.show_change_summary("default_embed_domain", None, tuple(parts))
            
            flow_help = create_question_help(
                question_id="flow_solver",
                question_text="Enable flow solver?",
                default_value="no",
                options=["yes", "no"],
                why_asking="The flow solver validates that the generated network has plausible fluid dynamics.",
            )
            flow_answer, _ = self.prompt_session.ask_yes_no(
                question_id="flow_solver",
                question_text="Enable flow solver?",
                default=False,
                help_info=flow_help,
            )
            if flow_answer:
                self.context.flow_solver_enabled = True
                self.prompt_session.show_change_summary("flow_solver_enabled", False, True)
        
        self.context.save_project_json()
        
        print()
        print(f"Project '{project_name}' created at: {self.context.output_dir}")
        
        self.state = WorkflowState.OBJECT_PLANNING
    
    def _run_object_planning(self) -> None:
        """Run OBJECT_PLANNING state: Ask how many objects and create folders."""
        print()
        print("-" * 40)
        print("Step 1: Object Planning")
        print("-" * 40)
        print()
        
        def validate_num_objects(value: str) -> Tuple[bool, str]:
            """Validate number of objects input."""
            if not value:
                return True, ""  # Empty is OK, defaults to 1
            try:
                n = int(value)
                if n < 1:
                    return False, "Number of objects must be at least 1"
                if n > 100:
                    return False, "Maximum 100 objects per project"
                return True, ""
            except ValueError:
                return False, "Please enter a valid integer"
        
        num_objects_help = create_question_help(
            question_id="num_objects",
            question_text="How many objects in this project?",
            default_value="1",
            why_asking="Each object represents a separate 3D structure to generate. Multiple objects can be variants of the same design or completely different structures.",
        )
        num_objects_input, intent = self.prompt_session.ask_until_answer(
            question_id="num_objects",
            question_text="How many objects in this project?",
            default_value="1",
            help_info=num_objects_help,
            validator=validate_num_objects,
            show_context=False,
        )
        if intent == TurnIntent.CANCEL:
            self.state = WorkflowState.COMPLETE
            return
        
        num_objects = int(num_objects_input) if num_objects_input else 1
        if num_objects < 1:
            num_objects = 1
        
        if num_objects > 1:
            variant_help = create_question_help(
                question_id="variant_mode",
                question_text="Variants of same concept or different objects?",
                default_value="different",
                options=["variants", "different"],
                why_asking="Variants share the same base design with parameter variations. Different objects are independent designs.",
            )
            variant_input, intent = self.prompt_session.ask_until_answer(
                question_id="variant_mode",
                question_text="Variants of same concept or different objects? (variants/different)",
                default_value="different",
                allowed_values=["variants", "different", "v", "d"],
                help_info=variant_help,
                show_context=False,
            )
            if intent == TurnIntent.CANCEL:
                self.state = WorkflowState.COMPLETE
                return
            self.context.variant_mode = variant_input.lower().startswith("v")
        
        print()
        print("Give each object a short name (or press Enter for auto-generated slugs):")
        
        for i in range(num_objects):
            default_name = f"object_{i+1:03d}"
            name_help = create_question_help(
                question_id=f"object_name_{i+1}",
                question_text=f"Object {i+1} name",
                default_value=default_name,
                why_asking="The object name is used to identify this structure and create its output folder.",
                examples=["liver_arterial", "kidney_network", "test_bifurcation"],
            )
            name_input, intent = self.prompt_session.ask_until_answer(
                question_id=f"object_name_{i+1}",
                question_text=f"  Object {i+1} name",
                default_value=default_name,
                help_info=name_help,
                show_context=False,
            )
            if intent == TurnIntent.CANCEL:
                self.state = WorkflowState.COMPLETE
                return
            
            name = name_input if name_input else default_name
            slug = name.lower().replace(" ", "_").replace("-", "_")
            
            obj = self.context.add_object(name, slug)
            print(f"    Created: {obj.object_dir}")
        
        self.context.save_project_json()
        
        print()
        print(f"Created {len(self.context.objects)} object(s).")
        
        self.context.current_object_index = 0
        self.state = WorkflowState.FRAME_OF_REFERENCE
    
    def _collect_initial_intent(self, obj: ObjectContext) -> bool:
        """Collect initial intent for an object. Returns False if user quits.
        
        V2: After collecting raw intent, runs the agent dialogue:
        1. understand_object() - produces UnderstandingReport
        2. propose_plans() - produces list of PlanOptions
        3. Shows understanding and plans to user
        4. User picks a plan
        """
        print()
        print(f"--- Object: {obj.name} ---")
        print()
        print("Describe the object you want to generate (purpose, shape, where")
        print("inlets/outlets should be, any constraints):")
        print()
        
        description_help = get_question_help("object_description")
        if description_help is None:
            description_help = create_question_help(
                question_id="object_description",
                question_text="Describe the 3D structure you want to generate",
                why_asking="Your description helps the agent understand what kind of structure to create, including topology, density, and any special requirements.",
                impact="A detailed description leads to better initial understanding and fewer follow-up questions.",
                examples=[
                    "A branching tree structure with 3 inlets and 50 terminals",
                    "A simple Y-shaped bifurcation for testing",
                    "A channel connecting two points with smooth curves",
                ],
            )
        
        intent, intent_type = self.prompt_session.ask_until_answer(
            question_id="object_description",
            question_text="Description",
            help_info=description_help,
            show_context=False,
        )
        
        if intent_type == TurnIntent.CANCEL:
            return False
        
        if not intent:
            print("Error: Description is required.")
            return self._collect_initial_intent(obj)
        
        obj.raw_intent = intent
        
        intent_path = os.path.join(obj.intent_dir, "intent.txt")
        with open(intent_path, 'w') as f:
            f.write(intent)
        
        self.schema_manager.reset()
        self.schema_manager.update_from_user_turn(intent)
        
        context = {
            "project_name": self.context.project_name,
            "object_name": obj.name,
            "previous_objects": [o.name for o in self.context.objects if o != obj],
        }
        
        print()
        print("-" * 40)
        print("Understanding your request...")
        print("-" * 40)
        
        llm_client = None
        if hasattr(self.agent, 'client') and self.agent.client is not None:
            llm_client = self.agent.client
        
        self.current_understanding = understand_object(intent, context, llm_client=llm_client)
        
        print()
        print("Here's what I understood:")
        print()
        print(f"  {self.current_understanding.summary}")
        print()
        
        if self.current_understanding.assumptions:
            print("Assumptions I'm making:")
            for assumption in self.current_understanding.assumptions:
                print(f"  - {assumption.reason} (confidence: {assumption.confidence})")
            print()
        
        if self.current_understanding.ambiguities:
            print("Ambiguities I noticed:")
            for ambiguity in self.current_understanding.ambiguities:
                print(f"  - {ambiguity.description}")
            print()
        
        if self.current_understanding.risks:
            print("Potential risks:")
            for risk in self.current_understanding.risks:
                print(f"  - [{risk.severity}] {risk.description}")
            print()
        
        self.current_plans = propose_plans(self.current_understanding, context)
        
        print("-" * 40)
        print("Generation Strategies")
        print("-" * 40)
        print()
        
        for i, plan in enumerate(self.current_plans):
            letter = chr(ord('A') + i)
            print(f"Option {letter}: {plan.name}")
            print(f"  Approach: {plan.description}")
            print(f"  Utilities: {', '.join(plan.utilities)}")
            if plan.tunable_knobs:
                print(f"  Tunable: {', '.join(plan.tunable_knobs)}")
            print()
        
        plan_options = [chr(ord('A') + i) for i in range(len(self.current_plans))]
        plan_help = create_question_help(
            question_id="plan_selection",
            question_text="Which generation plan would you like to use?",
            options=plan_options,
            why_asking="Different plans use different generation strategies optimized for different network types.",
            impact="The chosen plan determines which algorithms and parameters are used for generation.",
        )
        
        choice, choice_intent = self.prompt_session.ask_until_answer(
            question_id="plan_selection",
            question_text=f"Choose a plan ({'/'.join(plan_options)})",
            allowed_values=plan_options + [o.lower() for o in plan_options],
            help_info=plan_help,
            show_context=False,
        )
        
        if choice_intent in (TurnIntent.CANCEL, TurnIntent.BACK):
            return False
        
        choice = choice.upper()
        if choice not in plan_options:
            print(f"Invalid choice '{choice}'. Please choose from: {', '.join(plan_options)}")
            return False
        
        if choice in plan_options:
            plan_index = ord(choice) - ord('A')
            self.chosen_plan = self.current_plans[plan_index]
            print(f"\nSelected: Option {choice} - {self.chosen_plan.name}")
            self.prompt_session.show_change_summary("generation_plan", None, self.chosen_plan.name)
            
            for module_name in self.chosen_plan.required_modules:
                self.schema_manager.activate_module(module_name)
        
        understanding_path = os.path.join(obj.intent_dir, "understanding.json")
        with open(understanding_path, 'w') as f:
            json.dump({
                "summary": self.current_understanding.summary,
                "assumptions": [{"field": a.field, "reason": a.reason, "confidence": a.confidence} for a in self.current_understanding.assumptions],
                "ambiguities": [{"description": a.description, "field": a.field} for a in self.current_understanding.ambiguities],
                "risks": [{"description": r.description, "severity": r.severity} for r in self.current_understanding.risks],
                "chosen_plan": {
                    "name": self.chosen_plan.name,
                    "description": self.chosen_plan.description,
                    "strategy": self.chosen_plan.strategy.value,
                    "utilities": self.chosen_plan.utilities,
                    "modules": self.chosen_plan.required_modules,
                }
            }, f, indent=2)
        
        return True
    
    def _run_frame_of_reference(self) -> None:
        """Run FRAME_OF_REFERENCE state: Establish coordinate conventions."""
        obj = self.context.get_current_object()
        if not obj:
            self.state = WorkflowState.COMPLETE
            return
        
        if not obj.raw_intent:
            if not self._collect_initial_intent(obj):
                self.state = WorkflowState.COMPLETE
                return
        
        print()
        print("-" * 40)
        print(f"Step 2.2: Frame of Reference ({obj.name})")
        print("-" * 40)
        print()
        print("Before discussing spatial placement, we need to establish coordinate conventions.")
        print()
        
        if obj.requirements is None:
            obj.requirements = ObjectRequirements()
            obj.requirements.identity.object_name = obj.name
            obj.requirements.identity.object_slug = obj.slug
        
        frame = obj.requirements.frame_of_reference
        
        origin_help = create_question_help(
            question_id="use_domain_center",
            question_text="Use domain center as (0,0,0)?",
            default_value="yes",
            options=["yes", "no"],
            why_asking="The origin defines where (0,0,0) is located. Domain center is the most common choice.",
        )
        origin_answer, intent = self.prompt_session.ask_yes_no(
            question_id="use_domain_center",
            question_text="Use domain center as (0,0,0)?",
            default=True,
            help_info=origin_help,
        )
        if intent == TurnIntent.CANCEL:
            self.state = WorkflowState.COMPLETE
            return
        if not origin_answer:
            frame.origin = "custom"
            def validate_origin(value: str) -> Tuple[bool, str]:
                if not value:
                    return True, ""
                try:
                    parts = [float(x.strip()) for x in value.split(",")]
                    if len(parts) != 3:
                        return False, "Expected 3 values (x,y,z) separated by commas"
                    return True, ""
                except ValueError:
                    return False, "Invalid number format"
            
            custom_origin_help = create_question_help(
                question_id="custom_origin",
                question_text="Specify origin (x,y,z in meters)",
                why_asking="Custom origin allows you to define where (0,0,0) is located.",
                examples=["0,0,0", "0.01,0.03,0.015"],
            )
            custom_origin, _ = self.prompt_session.ask_until_answer(
                question_id="custom_origin",
                question_text="Specify origin (x,y,z in meters)",
                help_info=custom_origin_help,
                validator=validate_origin,
                show_context=False,
            )
            if custom_origin:
                frame.origin = custom_origin
        
        print()
        print("Which axis is which for you?")
        print("  Default: X=width (left-right), Y=depth (front-back), Z=height (up-down)")
        change_axes_help = create_question_help(
            question_id="change_axes",
            question_text="Change axis mapping?",
            default_value="no",
            options=["yes", "no"],
            why_asking="Axis mapping defines how X, Y, Z correspond to width, depth, and height.",
        )
        change_axes_answer, intent = self.prompt_session.ask_yes_no(
            question_id="change_axes",
            question_text="Change axis mapping?",
            default=False,
            help_info=change_axes_help,
        )
        if intent == TurnIntent.CANCEL:
            self.state = WorkflowState.COMPLETE
            return
        if change_axes_answer:
            x_axis_help = create_question_help(
                question_id="x_axis",
                question_text="X axis represents",
                default_value="width",
                options=["width", "depth", "height"],
            )
            x_axis, _ = self.prompt_session.ask_until_answer(
                question_id="x_axis",
                question_text="X axis represents (width/depth/height)",
                default_value="width",
                allowed_values=["width", "depth", "height"],
                help_info=x_axis_help,
                show_context=False,
            )
            y_axis, _ = self.prompt_session.ask_until_answer(
                question_id="y_axis",
                question_text="Y axis represents (width/depth/height)",
                default_value="depth",
                allowed_values=["width", "depth", "height"],
                show_context=False,
            )
            z_axis, _ = self.prompt_session.ask_until_answer(
                question_id="z_axis",
                question_text="Z axis represents (width/depth/height)",
                default_value="height",
                allowed_values=["width", "depth", "height"],
                show_context=False,
            )
            frame.axes = {"x": x_axis or "width", "y": y_axis or "depth", "z": z_axis or "height"}
        
        print()
        viewpoint_help = create_question_help(
            question_id="viewpoint",
            question_text="Viewpoint for left/right?",
            default_value="front",
            options=["front", "top", "side"],
            why_asking="The viewpoint determines how 'left' and 'right' are interpreted.",
        )
        viewpoint, intent = self.prompt_session.ask_until_answer(
            question_id="viewpoint",
            question_text="Viewpoint for left/right? (front/top/side)",
            default_value="front",
            allowed_values=["front", "top", "side"],
            help_info=viewpoint_help,
            show_context=False,
        )
        if intent == TurnIntent.CANCEL:
            self.state = WorkflowState.COMPLETE
            return
        if viewpoint in ("front", "top", "side"):
            frame.viewpoint = viewpoint
        
        obj.frame_locked = True
        frame.confirmed = True
        
        print()
        print("Frame of reference locked:")
        print(f"  Origin: {frame.origin}")
        print(f"  Axes: X={frame.axes['x']}, Y={frame.axes['y']}, Z={frame.axes['z']}")
        print(f"  Viewpoint: {frame.viewpoint}")
        print()
        print("Spatial references will now be interpreted according to this frame.")
        
        self.current_question_group = "A"
        self.current_question_index = 0
        self.collected_answers = {}
        
        self.state = WorkflowState.REQUIREMENTS_CAPTURE
    
    def _run_requirements_capture(self) -> None:
        """Run REQUIREMENTS_CAPTURE state: Rule-based adaptive requirements gathering.
        
        V2: Uses SchemaManager for dynamic schema with activatable modules.
        Questions are ranked by rework cost (frame, scale, I/O first).
        
        This method uses the RuleEngine to determine what questions to ask based on:
        - Missing required fields (Family A - Completeness)
        - Ambiguous user language (Family B - Ambiguity)
        - Conflicts/feasibility issues (Family C - Conflict)
        
        The attempt strategy is:
        1. Infer from user text (high confidence only)
        2. Propose concrete defaults (mark as assumed)
        3. Ask targeted questions (only if needed)
        """
        obj = self.context.get_current_object()
        if not obj:
            self.state = WorkflowState.COMPLETE
            return
        
        print()
        print("-" * 40)
        print(f"Step 3: Requirements Capture ({obj.name})")
        print("-" * 40)
        
        # Only detect organ type if user explicitly mentions organ-specific terms
        # Otherwise, use generic mode to focus on what the user actually described
        organ_type = detect_organ_type(obj.raw_intent)
        
        # Be more conservative about organ detection - only mention if very explicit
        explicit_organ_keywords = ["liver", "kidney", "lung", "heart", "hepatic", "renal", "pulmonary", "coronary"]
        intent_lower = obj.raw_intent.lower()
        is_explicit_organ = any(kw in intent_lower for kw in explicit_organ_keywords)
        
        if is_explicit_organ and organ_type != "generic":
            print(f"\nDetected structure type: {organ_type}")
            print("Questions will be tailored to this structure type.")
        else:
            organ_type = "generic"  # Force generic if not explicitly organ-related
            print("\nUsing generic structure mode based on your description.")
        
        print("\nUsing adaptive rule-based requirements capture.")
        print("The system will ask only the questions needed based on your description.")
        print("Say 'use defaults' at any time to accept all proposed defaults.")
        print()
        
        missing_fields = self.schema_manager.missing_required_fields()
        ambiguities = []
        if self.current_understanding and self.current_understanding.ambiguities:
            # Convert Ambiguity objects to dicts for plan_questions
            ambiguities = [
                {
                    "field": a.field,
                    "description": a.description,
                    "options": a.options,
                    "impact": a.impact,
                }
                for a in self.current_understanding.ambiguities
            ]
        
        questions = self.schema_manager.plan_questions(
            missing=missing_fields,
            ambiguities=ambiguities,
            conflicts=[],
            max_questions=5
        )
        
        if questions:
            print(f"I need to ask {len(questions)} question(s) to complete the specification:")
            print()
            for i, q in enumerate(questions, 1):
                print(f"  {i}. {q.question_text}")
                print(f"     Why: {q.reason} (rework cost: {q.rework_cost})")
            print()
        
        updated_req, collected = run_rule_based_capture(
            requirements=obj.requirements,
            intent=obj.raw_intent,
            organ_type=organ_type,
            verbose=True
        )
        
        obj.requirements = updated_req
        self.collected_answers = collected
        
        for field, value in collected.items():
            self.schema_manager.set_field_value(field, value)
        
        self._apply_answers_to_requirements(obj)
        
        req_path = obj.get_versioned_path(obj.intent_dir, "requirements", "json")
        obj.requirements.to_json(req_path)
        
        print()
        print("-" * 40)
        print("Living Spec Summary")
        print("-" * 40)
        spec_summary = self.schema_manager.get_spec_summary()
        print(spec_summary)
        
        object_summary = generate_requirements_summary(obj.requirements)
        print(object_summary)
        
        print()
        print("Requirements captured and saved.")
        print(f"  File: {req_path}")
        
        print()
        print("Please confirm these requirements are accurate before we generate.")
        confirm_help = create_question_help(
            question_id="proceed_generation",
            question_text="Proceed with generation?",
            default_value="yes",
            options=["yes", "no"],
            why_asking="Confirming requirements before generation ensures the design matches your intent.",
        )
        confirm_answer, intent = self.prompt_session.ask_yes_no(
            question_id="proceed_generation",
            question_text="Proceed with generation?",
            default=True,
            help_info=confirm_help,
        )
        if intent == TurnIntent.CANCEL:
            self.state = WorkflowState.COMPLETE
            return
        if not confirm_answer:
            print("\nRestarting requirements capture...")
            obj.requirements = ObjectRequirements()
            obj.requirements.identity.object_name = obj.name
            obj.requirements.identity.object_slug = obj.slug
            self.schema_manager.reset()
            return
        
        self.state = WorkflowState.SPEC_COMPILATION
    
    def _apply_answers_to_requirements(self, obj: ObjectContext) -> None:
        """Apply collected answers to the requirements schema."""
        req = obj.requirements
        answers = self.collected_answers
        
        if answers.get("scale"):
            req.frame_of_reference.units_export = answers["scale"]
        if answers.get("min_channel_radius"):
            try:
                req.constraints.min_radius_m = float(answers["min_channel_radius"]) / 1000
            except ValueError:
                pass
        if answers.get("min_clearance"):
            try:
                req.constraints.min_clearance_m = float(answers["min_clearance"]) / 1000
            except ValueError:
                pass
        
        if answers.get("boundary_margin"):
            try:
                req.constraints.boundary_buffer_m = float(answers["boundary_margin"])
            except ValueError:
                pass
        
        try:
            num_inlets = int(answers.get("num_inlets", "1"))
            num_outlets = int(answers.get("num_outlets", "0"))
        except ValueError:
            num_inlets = 1
            num_outlets = 0
        
        inlet_radii = answers.get("inlet_radius", "0.002").split(",")
        outlet_radii = answers.get("outlet_radius", "0.001").split(",")
        
        req.inlets_outlets.inlets = []
        for i in range(num_inlets):
            radius = float(inlet_radii[i % len(inlet_radii)].strip()) / 1000 if inlet_radii else 0.002
            req.inlets_outlets.inlets.append(PortSpec(
                name=f"inlet_{i+1}",
                radius_m=radius,
            ))
        
        req.inlets_outlets.outlets = []
        for i in range(num_outlets):
            radius = float(outlet_radii[i % len(outlet_radii)].strip()) / 1000 if outlet_radii else 0.001
            req.inlets_outlets.outlets.append(PortSpec(
                name=f"outlet_{i+1}",
                radius_m=radius,
            ))
        
        req.inlets_outlets.placement_rule = answers.get("inlet_outlet_location", "")
        
        if answers.get("target_terminals"):
            try:
                req.topology.target_terminals = int(answers["target_terminals"])
            except ValueError:
                pass
        if answers.get("max_depth"):
            try:
                req.topology.max_depth = int(answers["max_depth"])
            except ValueError:
                pass
        
        tortuosity_map = {"low": 0.1, "med": 0.5, "high": 0.9}
        req.geometry.tortuosity = tortuosity_map.get(answers.get("tortuosity", "low"), 0.1)
        
        if answers.get("branch_angle_range"):
            try:
                parts = answers["branch_angle_range"].split("-")
                if len(parts) == 2:
                    req.geometry.branch_angle_deg = {
                        "min": float(parts[0]),
                        "max": float(parts[1]),
                    }
            except ValueError:
                pass
        
        if answers.get("segment_length_range"):
            try:
                parts = answers["segment_length_range"].split("-")
                if len(parts) == 2:
                    req.geometry.segment_length_m = {
                        "min": float(parts[0]) / 1000,
                        "max": float(parts[1]) / 1000,
                    }
            except ValueError:
                pass
        
        req.geometry.radius_profile = answers.get("tapering", "murray")
        
        req.constraints.avoid_self_intersection = answers.get("ban_self_intersection", "yes").lower() in ("yes", "y")
        
        if answers.get("terminal_range"):
            try:
                parts = answers["terminal_range"].split("-")
                if len(parts) == 2:
                    req.acceptance_criteria.terminals_range = (int(parts[0]), int(parts[1]))
            except ValueError:
                pass
        
        req.acceptance_criteria.mesh_watertight_required = answers.get("watertight_required", "yes").lower() in ("yes", "y")
    
    def _compile_requirements_to_spec(self, requirements: ObjectRequirements, topology_kind: str) -> DesignSpec:
        """
        Deterministically compile ObjectRequirements to DesignSpec.
        
        This function builds the DesignSpec directly in Python without LLM involvement,
        ensuring reliable and reproducible spec generation.
        
        Parameters
        ----------
        requirements : ObjectRequirements
            The collected requirements from the user
        topology_kind : str
            The topology kind: "path", "tree", "loop", or "multi_tree"
            
        Returns
        -------
        DesignSpec
            The compiled design specification
        """
        req = requirements
        
        # Build domain spec
        domain_shape = req.domain.shape or "box"
        domain_size = req.domain.size_m or (0.02, 0.06, 0.03)  # Default 20x60x30 mm
        
        if domain_shape == "ellipsoid":
            # For ellipsoid, size represents semi-axes
            domain_spec = EllipsoidSpec(
                center=(0.0, 0.0, 0.0),
                semi_axes=tuple(s / 2 for s in domain_size),  # Convert full size to semi-axes
            )
        else:
            # Default to box
            domain_spec = BoxSpec(
                center=(0.0, 0.0, 0.0),
                size=domain_size,
            )
        
        # Build inlet specs from requirements
        inlets = []
        for i, port in enumerate(req.inlets_outlets.inlets):
            position = port.position_m or (0.0, 0.0, 0.0)
            radius = port.radius_m or 0.002  # Default 2mm
            inlets.append(InletSpec(
                position=position,
                radius=radius,
                vessel_type="arterial",
            ))
        
        # If no inlets defined, create a default based on topology
        if not inlets:
            # Default inlet at x_min face center
            inlet_x = -domain_size[0] / 2 + 0.001  # Slightly inside domain
            inlets.append(InletSpec(
                position=(inlet_x, 0.0, 0.0),
                radius=0.002,
                vessel_type="arterial",
            ))
        
        # Build outlet specs from requirements
        outlets = []
        for i, port in enumerate(req.inlets_outlets.outlets):
            position = port.position_m or (0.0, 0.0, 0.0)
            radius = port.radius_m or 0.002  # Default 2mm
            outlets.append(OutletSpec(
                position=position,
                radius=radius,
                vessel_type="venous",
            ))
        
        # For path topology, ensure we have an outlet
        if topology_kind == "path" and not outlets:
            # Default outlet at x_max face center
            outlet_x = domain_size[0] / 2 - 0.001  # Slightly inside domain
            outlets.append(OutletSpec(
                position=(outlet_x, 0.0, 0.0),
                radius=inlets[0].radius if inlets else 0.002,
                vessel_type="venous",
            ))
        
        # Build colonization spec from requirements
        min_radius = req.constraints.min_radius_m or 0.0001  # Default 0.1mm
        
        # Adjust colonization parameters based on topology
        if topology_kind == "path":
            # For path: minimal colonization, just connect inlet to outlet
            colonization = ColonizationSpec(
                influence_radius=0.015,
                kill_radius=0.002,
                step_size=0.001,
                max_steps=100,  # Fewer steps for simple path
                initial_radius=inlets[0].radius if inlets else 0.002,
                min_radius=min_radius,
                radius_decay=1.0,  # No decay for path
                encourage_bifurcation=False,  # No branching for path
                max_children_per_node=1,  # Single path, no branching
            )
        else:
            # For tree/loop/multi_tree: full colonization
            target_terminals = req.topology.target_terminals or 50
            colonization = ColonizationSpec(
                influence_radius=0.015,
                kill_radius=0.002,
                step_size=0.001,
                max_steps=500,
                initial_radius=inlets[0].radius if inlets else 0.002,
                min_radius=min_radius,
                radius_decay=0.95,
                encourage_bifurcation=True,
                max_children_per_node=2,
            )
        
        # Build tree spec with topology_kind
        tree_spec = TreeSpec(
            inlets=inlets,
            outlets=outlets,
            colonization=colonization,
            topology_kind=topology_kind,
            terminal_count=req.topology.target_terminals if topology_kind == "tree" else None,
        )
        
        # Build final DesignSpec
        design_spec = DesignSpec(
            domain=domain_spec,
            tree=tree_spec,
            seed=42,  # Default seed for reproducibility
            metadata={
                "source": "deterministic_compilation",
                "topology_kind": topology_kind,
                "user_intent": req.identity.description or "",
            },
            output_units="mm",
        )
        
        return design_spec
    
    def _validate_spec(self, spec_path: str, topology_kind: str) -> Tuple[bool, str]:
        """
        Validate that a spec file exists, loads correctly, and has required fields.
        
        Parameters
        ----------
        spec_path : str
            Path to the spec.json file
        topology_kind : str
            Expected topology kind
            
        Returns
        -------
        Tuple[bool, str]
            (is_valid, error_message)
        """
        # Check file exists
        if not os.path.exists(spec_path):
            return False, f"Spec file does not exist: {spec_path}"
        
        # Check JSON loads
        try:
            with open(spec_path, 'r') as f:
                spec_data = json.load(f)
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON in spec file: {e}"
        except Exception as e:
            return False, f"Error reading spec file: {e}"
        
        # Check DesignSpec.from_dict works
        try:
            spec = DesignSpec.from_dict(spec_data)
        except Exception as e:
            return False, f"Failed to parse DesignSpec: {e}"
        
        # Check required fields for topology
        if topology_kind == "path":
            # Path requires exactly 1 inlet and 1 outlet
            if spec.tree:
                if len(spec.tree.inlets) != 1:
                    return False, f"Path topology requires exactly 1 inlet, got {len(spec.tree.inlets)}"
                if len(spec.tree.outlets) != 1:
                    return False, f"Path topology requires exactly 1 outlet, got {len(spec.tree.outlets)}"
        elif topology_kind == "tree":
            # Tree requires at least 1 inlet
            if spec.tree and len(spec.tree.inlets) == 0:
                return False, "Tree topology requires at least 1 inlet"
        
        # Check domain exists
        if not spec.domain:
            return False, "Spec missing domain"
        
        # Check units are valid
        if spec.output_units not in ("m", "mm", "cm", "um"):
            return False, f"Invalid output_units: {spec.output_units}"
        
        return True, ""
    
    def _run_spec_compilation(self) -> None:
        """Run SPEC_COMPILATION state: Compile requirements to DesignSpec.
        
        This step uses DETERMINISTIC compilation - no LLM involvement.
        The DesignSpec is built directly from requirements in Python,
        ensuring reliable and reproducible spec generation.
        
        Key improvements over LLM-based compilation:
        1. Deterministic: Same requirements always produce same spec
        2. Reliable: No dependency on LLM file writing
        3. Topology-aware: Different compilation for path vs tree
        4. Validated: Post-compilation validation ensures spec is usable
        """
        obj = self.context.get_current_object()
        if not obj:
            self.state = WorkflowState.COMPLETE
            return
        
        print()
        print("-" * 40)
        print(f"Step 4: Spec Compilation ({obj.name})")
        print("-" * 40)
        print()
        
        req = obj.requirements
        req_dict = req.to_dict()
        
        # Detect topology kind from requirements
        topology_kind = req.topology.topology_kind or detect_topology_kind(obj.raw_intent)
        
        # Show transparency: display what we're compiling
        print("Compiling requirements to DesignSpec (deterministic)...")
        print()
        print("Input Requirements Summary:")
        print("-" * 30)
        
        # Show domain info
        if req_dict.get("domain"):
            domain = req_dict["domain"]
            print(f"  Domain: {domain.get('shape', 'box')}")
            if domain.get("size_m"):
                size = domain["size_m"]
                print(f"    Size: {size[0]*1000:.1f} x {size[1]*1000:.1f} x {size[2]*1000:.1f} mm")
        
        # Show I/O info
        if req_dict.get("inlets_outlets"):
            io = req_dict["inlets_outlets"]
            print(f"  Inlets: {len(io.get('inlets', []))}")
            print(f"  Outlets: {len(io.get('outlets', []))}")
        
        # Show topology info
        print(f"  Topology kind: {topology_kind}")
        if req_dict.get("topology"):
            topo = req_dict["topology"]
            if topo.get("target_terminals"):
                print(f"    Target terminals: {topo['target_terminals']}")
        
        # Show constraints
        if req_dict.get("constraints"):
            cons = req_dict["constraints"]
            if cons.get("min_radius_m"):
                print(f"  Min radius: {cons['min_radius_m']*1000:.3f} mm")
        
        print("-" * 30)
        print()
        print(f"User intent: {obj.raw_intent[:100]}..." if len(obj.raw_intent) > 100 else f"User intent: {obj.raw_intent}")
        print()
        
        # Deterministic compilation - no LLM involved
        print("Building DesignSpec from requirements...")
        
        try:
            design_spec = self._compile_requirements_to_spec(req, topology_kind)
            
            # Save spec to JSON
            spec_path = obj.get_versioned_path(obj.spec_dir, "spec", "json")
            os.makedirs(os.path.dirname(spec_path), exist_ok=True)
            design_spec.to_json(spec_path)
            
            obj.spec_path = spec_path
            
            print()
            print("Spec compilation complete!")
            print(f"  Spec saved to: {obj.spec_path}")
            
            # Validate the spec
            is_valid, error_msg = self._validate_spec(spec_path, topology_kind)
            if not is_valid:
                print()
                print(f"WARNING: Spec validation failed: {error_msg}")
                print("Attempting to continue anyway...")
            else:
                print("  Spec validation: PASSED")
            
            # Show what was generated for transparency
            print()
            print("Generated Spec Summary:")
            print("-" * 30)
            print(f"  Domain type: {design_spec.domain.type}")
            if isinstance(design_spec.domain, BoxSpec):
                size = design_spec.domain.size
                print(f"  Domain size: {size[0]*1000:.1f} x {size[1]*1000:.1f} x {size[2]*1000:.1f} mm")
            print(f"  Topology kind: {topology_kind}")
            if design_spec.tree:
                print(f"  Inlets: {len(design_spec.tree.inlets)}")
                print(f"  Outlets: {len(design_spec.tree.outlets)}")
                if design_spec.tree.terminal_count:
                    print(f"  Target terminals: {design_spec.tree.terminal_count}")
            print(f"  Output units: {design_spec.output_units}")
            print("-" * 30)
            
            self.state = WorkflowState.GENERATION
            
        except Exception as e:
            print()
            print(f"Spec compilation failed: {e}")
            print()
            print("Troubleshooting:")
            print("  1. Check that your requirements are complete")
            print("  2. Ensure domain size and I/O are specified")
            print("  3. For path topology: inlet and outlet are required")
            print("  4. For tree topology: inlet and terminal count are required")
            print()
            print("Returning to requirements capture...")
            self.state = WorkflowState.REQUIREMENTS_CAPTURE
    
    def _run_generation(self) -> None:
        """Run GENERATION state: Execute generation within object folder.
        
        Supports three execution modes:
        - WRITE_ONLY: Generate script, don't run
        - REVIEW_THEN_RUN: Generate script, pause for review, then run
        - AUTO_RUN: Generate script and run automatically
        """
        obj = self.context.get_current_object()
        if not obj:
            self.state = WorkflowState.COMPLETE
            return
        
        print()
        print("-" * 40)
        print(f"Step 5: Generation ({obj.name}, v{obj.version})")
        print(f"Execution mode: {self.execution_mode.value}")
        print("-" * 40)
        print()
        
        obj.status = "generating"
        
        previous_context = self._get_previous_attempts_context(obj)
        feedback_context = ""
        if obj.feedback_history:
            feedback_context = "\n\nPrevious feedback from user:\n"
            for i, fb in enumerate(obj.feedback_history, 1):
                feedback_context += f"{i}. {fb}\n"
        
        network_path = obj.get_versioned_path(obj.outputs_dir, "network", "json")
        mesh_path = obj.get_versioned_path(obj.mesh_dir, "mesh_network", "stl")
        
        task = f"""Generate a Python script for 3D structure generation based on the spec.

Spec file: {obj.spec_path}

IMPORTANT: Focus on the user's actual requirements from the spec. Do not assume organ-specific
structures unless explicitly specified. Generate exactly what the user described.

The script MUST follow these requirements:
1. Read the output directory from environment variable ORGAN_AGENT_OUTPUT_DIR
2. Define a main() function as the entry point
3. Save the network to: network.json (relative to OUTPUT_DIR)
4. Export STL mesh to: mesh_network.stl (relative to OUTPUT_DIR)
5. Use output_units="{self.context.units_export}" for exports
6. Print ARTIFACTS_JSON: {{"files": [...], "metrics": {{...}}, "status": "success"}} at the end
7. Do NOT use subprocess, os.system, eval, exec, or pip install
8. Do NOT delete files or write outside OUTPUT_DIR

Use generation.api.design_from_spec() or appropriate generation functions.

{feedback_context}

If generation fails, report the error and suggest parameter adjustments.

Provide complete, runnable Python code in a ```python code block."""

        if previous_context:
            task += f"\n\nContext from previous attempts (use workable parts):\n{previous_context}"
        
        print("Requesting generation script from LLM...")
        
        result = self.agent.run_task(
            task=task,
            context={
                "object_name": obj.name,
                "object_dir": obj.object_dir,
                "version": obj.version,
                "output_units": self.context.units_export,
            }
        )
        
        if result.status != TaskStatus.COMPLETED:
            print(f"\nLLM failed to generate script: {result.error}")
            obj.status = "failed"
            
            retry_help = create_question_help(
                question_id="retry_generation",
                question_text="Retry with adjusted parameters?",
                default_value="yes",
                options=["yes", "no"],
                why_asking="Retrying allows you to adjust parameters and try again.",
            )
            retry_answer, intent = self.prompt_session.ask_yes_no(
                question_id="retry_generation",
                question_text="Retry with adjusted parameters?",
                default=True,
                help_info=retry_help,
            )
            if intent == TurnIntent.CANCEL or not retry_answer:
                self._move_to_next_object_or_complete()
            else:
                self.state = WorkflowState.REQUIREMENTS_CAPTURE
            return
        
        if should_write_script(self.execution_mode):
            print("Writing script to disk...")
            write_result = write_script(
                llm_response=result.output,
                output_dir=obj.code_dir,
                version=obj.version,
                object_name=obj.name,
                add_header=True,
                add_footer=True,
                ensure_main=True,
            )
            
            if not write_result.success:
                print(f"\nFailed to write script: {write_result.error}")
                obj.status = "failed"
                self.state = WorkflowState.REQUIREMENTS_CAPTURE
                return
            
            obj.code_path = write_result.script_path
            print(f"  Script: {write_result.script_path}")
            print(f"  Response: {write_result.response_path}")
            
            if write_result.warnings:
                print("\nWarnings:")
                for w in write_result.warnings:
                    print(f"  - {w}")
            
            run_command = get_run_command(write_result.script_path)
            
            if self.execution_mode == ExecutionMode.WRITE_ONLY:
                print("\nScript written (write_only mode - not executing)")
                print(f"To run manually: {run_command}")
                obj.network_path = network_path
                obj.mesh_path = mesh_path
                self.state = WorkflowState.ANALYSIS_VALIDATION
                return
            
            if should_pause_for_review(self.execution_mode):
                review_result = run_review_gate(
                    script_path=write_result.script_path,
                    run_command=run_command,
                    warnings=write_result.warnings,
                    object_name=obj.name,
                    version=obj.version,
                    interactive=True,
                )
                
                if review_result.action == ReviewAction.CANCEL:
                    print("\nGeneration cancelled by user.")
                    obj.status = "cancelled"
                    self._move_to_next_object_or_complete()
                    return
                
                if review_result.action == ReviewAction.DONE:
                    print("\nSkipping execution (user indicated script was run manually)")
                    obj.network_path = network_path
                    obj.mesh_path = mesh_path
                    self.state = WorkflowState.ANALYSIS_VALIDATION
                    return
            
            if should_execute(self.execution_mode):
                print("\nExecuting script...")
                run_result = run_script(
                    script_path=write_result.script_path,
                    object_dir=obj.outputs_dir,
                    version=obj.version,
                    timeout_seconds=self.timeout_seconds,
                )
                
                print_run_summary(run_result)
                
                if not run_result.success:
                    print(f"\nScript execution failed")
                    obj.status = "failed"
                    
                    retry_help = create_question_help(
                        question_id="retry_execution",
                        question_text="Retry with adjusted parameters?",
                        default_value="yes",
                        options=["yes", "no"],
                        why_asking="Retrying allows you to adjust parameters and try again.",
                    )
                    retry_answer, intent = self.prompt_session.ask_yes_no(
                        question_id="retry_execution",
                        question_text="Retry with adjusted parameters?",
                        default=True,
                        help_info=retry_help,
                    )
                    if intent == TurnIntent.CANCEL or not retry_answer:
                        self._move_to_next_object_or_complete()
                    else:
                        self.state = WorkflowState.REQUIREMENTS_CAPTURE
                    return
                
                print("\nVerifying artifacts...")
                verification = verify_generation_stage(
                    object_dir=obj.outputs_dir,
                    version=obj.version,
                    script_output=run_result.stdout,
                    spec_path=obj.spec_path,
                )
                
                print_verification_summary(verification)
                
                if verification.manifest:
                    manifest_path = save_manifest(
                        verification.manifest,
                        obj.outputs_dir,
                        obj.version,
                    )
                    print(f"  Manifest: {manifest_path}")
                
                if not verification.success:
                    print("\nArtifact verification failed")
                    obj.status = "failed"
                    
                    retry_help = create_question_help(
                        question_id="retry_verification",
                        question_text="Retry with adjusted parameters?",
                        default_value="yes",
                        options=["yes", "no"],
                        why_asking="Retrying allows you to adjust parameters and try again.",
                    )
                    retry_answer, intent = self.prompt_session.ask_yes_no(
                        question_id="retry_verification",
                        question_text="Retry with adjusted parameters?",
                        default=True,
                        help_info=retry_help,
                    )
                    if intent == TurnIntent.CANCEL or not retry_answer:
                        self._move_to_next_object_or_complete()
                    else:
                        self.state = WorkflowState.REQUIREMENTS_CAPTURE
                    return
        
        else:
            result = self.agent.run_task(
                task=task,
                context={
                    "object_name": obj.name,
                    "object_dir": obj.object_dir,
                    "version": obj.version,
                    "output_units": self.context.units_export,
                }
            )
        
        obj.network_path = network_path
        obj.mesh_path = mesh_path
        
        print("\nGeneration complete!")
        print(f"  Network: {obj.network_path}")
        print(f"  Mesh: {obj.mesh_path}")
        
        self.state = WorkflowState.ANALYSIS_VALIDATION
    
    def _run_analysis_validation(self) -> None:
        """Run ANALYSIS_VALIDATION state: Analyze and validate generated structure."""
        obj = self.context.get_current_object()
        if not obj:
            self.state = WorkflowState.COMPLETE
            return
        
        print()
        print("-" * 40)
        print(f"Step 6: Analysis & Validation ({obj.name}, v{obj.version})")
        print("-" * 40)
        print()
        print("Analyzing and validating generated structure...")
        
        task = f"""Analyze and validate the generated 3D structure.

Network file: {obj.network_path}
Mesh file: {obj.mesh_path}

Requirements:
1. Load the network and mesh
2. Compute analysis metrics:
   - Node count, segment count, terminal count
   - Radius statistics (min, max, mean)
   - Clearance statistics
   - Coverage metrics if applicable
3. Run validation checks:
   - Mesh integrity (watertight, manifold)
   - Murray's law compliance
   - Collision detection
   - Constraint satisfaction
4. Save analysis to: {obj.get_versioned_path(obj.analysis_dir, "analysis", "json")}
5. Save human-readable analysis to: {obj.get_versioned_path(obj.analysis_dir, "analysis", "txt")}
6. Save validation to: {obj.get_versioned_path(obj.validation_dir, "validation", "json")}

Report:
- Counts (nodes/segments/terminals)
- Radius/clearance minima
- Any failing checks with suggested knob adjustments

Flow solver is {'ON' if self.context.flow_solver_enabled else 'OFF'}.

Provide complete analysis results."""

        result = self.agent.run_task(
            task=task,
            context={
                "object_name": obj.name,
                "object_dir": obj.object_dir,
                "version": obj.version,
            }
        )
        
        if result.status == TaskStatus.COMPLETED:
            obj.analysis_path = obj.get_versioned_path(obj.analysis_dir, "analysis", "json")
            obj.validation_path = obj.get_versioned_path(obj.validation_dir, "validation", "json")
            
            print("\nAnalysis complete!")
            print(f"\nAgent response:\n{result.output[:2000]}...")
            
            self.state = WorkflowState.ITERATION
        else:
            print(f"\nAnalysis failed: {result.error}")
            self.state = WorkflowState.ITERATION
    
    def _run_iteration(self) -> None:
        """Run ITERATION state: Accept user critique and iterate.
        
        V2: Uses SchemaManager.update_from_results() to activate modules based on:
        - Collisions -> activate ComplexityBudget + stricter clearance fields
        - Mesh too heavy -> activate MeshQuality module
        - Embedding memory failure -> require voxel_pitch + memory_policy fields
        - User asks "optimize flow" -> activate FlowPhysics module
        """
        obj = self.context.get_current_object()
        if not obj:
            self.state = WorkflowState.COMPLETE
            return
        
        print()
        print("-" * 40)
        print(f"Step 7: Review & Iterate ({obj.name}, v{obj.version})")
        print("-" * 40)
        print()
        
        print("Generated files:")
        mesh_exists = obj.mesh_path and os.path.exists(obj.mesh_path)
        analysis_exists = obj.analysis_path and os.path.exists(obj.analysis_path)
        
        if mesh_exists:
            print(f"  [OK] Mesh: {obj.mesh_path}")
        else:
            print(f"  [--] Mesh: Not generated")
        if analysis_exists:
            print(f"  [OK] Analysis: {obj.analysis_path}")
        else:
            print(f"  [--] Analysis: Not generated")
        
        results = {
            "mesh_generated": mesh_exists,
            "analysis_generated": analysis_exists,
            "version": obj.version,
        }
        
        if analysis_exists:
            try:
                with open(obj.analysis_path, 'r') as f:
                    analysis_data = json.load(f)
                results["collision_count"] = analysis_data.get("collision_count", 0)
                results["mesh_faces"] = analysis_data.get("mesh_faces", 0)
                results["embedding_failed"] = analysis_data.get("embedding_failed", False)
            except (json.JSONDecodeError, IOError):
                pass
        
        modules_activated = self.schema_manager.update_from_results(results)
        if modules_activated:
            print()
            print("Based on results, activated additional schema modules:")
            for module in modules_activated:
                print(f"  - {module}")
        
        print()
        print("To visualize, open the STL file in a 3D viewer (MeshLab, Blender, etc.)")
        print()
        
        accept_help = create_question_help(
            question_id="accept_structure",
            question_text="Is this structure acceptable?",
            options=["yes", "no"],
            why_asking="If acceptable, the workflow will finalize the structure. If not, you can provide feedback for improvements.",
        )
        accept_answer, intent = self.prompt_session.ask_yes_no(
            question_id="accept_structure",
            question_text="Is this structure acceptable?",
            default=None,  # No default - user must explicitly choose
            help_info=accept_help,
        )
        
        if intent == TurnIntent.CANCEL:
            self.state = WorkflowState.COMPLETE
            return
        
        if accept_answer:
            obj.status = "accepted"
            self.state = WorkflowState.FINALIZATION
        else:
            print()
            print("Please describe what's wrong and what changes you'd like:")
            print("Examples: 'denser near top', 'avoid crossing', 'bigger outlet', 'more symmetry'")
            
            feedback_help = create_question_help(
                question_id="iteration_feedback",
                question_text="Feedback",
                why_asking="Your feedback helps the system understand what to improve in the next iteration.",
                examples=["denser near top", "avoid crossing", "bigger outlet", "more symmetry"],
            )
            feedback, intent = self.prompt_session.ask_until_answer(
                question_id="iteration_feedback",
                question_text="Feedback",
                help_info=feedback_help,
                show_context=False,
            )
            
            if intent == TurnIntent.CANCEL:
                self.state = WorkflowState.COMPLETE
                return
            
            if not feedback:
                print("Error: Feedback is required to improve the structure.")
                return
            
            obj.feedback_history.append(feedback)
            
            self.schema_manager.update_from_user_turn(feedback)
            
            if "flow" in feedback.lower() or "optimize" in feedback.lower():
                self.schema_manager.activate_module("FlowPhysicsModule")
                print("  -> Activated FlowPhysicsModule based on feedback")
            
            obj.increment_version()
            
            print(f"\nThank you for your feedback. Regenerating (v{obj.version})...")
            
            self.state = WorkflowState.SPEC_COMPILATION
    
    def _run_finalization(self) -> None:
        """Run FINALIZATION state: Embed and produce final outputs."""
        obj = self.context.get_current_object()
        if not obj:
            self.state = WorkflowState.COMPLETE
            return
        
        print()
        print("-" * 40)
        print(f"Step 8: Finalization ({obj.name})")
        print("-" * 40)
        print()
        print("Generating final outputs (embedded structure, final STL, manifest)...")
        
        embed_domain = self.context.default_embed_domain
        
        task = f"""Finalize the 3D structure for object '{obj.name}'.

Network file: {obj.network_path}
Mesh file: {obj.mesh_path}

Requirements:
1. Embed the structure into the domain as negative space:
   - Domain: box {embed_domain[0]*1000:.0f}mm x {embed_domain[1]*1000:.0f}mm x {embed_domain[2]*1000:.0f}mm
   - Use embed_tree_as_negative_space() or equivalent
   - Save void STL to: {os.path.join(obj.final_dir, "embed", "void.stl")}
   - Optionally save domain STL to: {os.path.join(obj.final_dir, "embed", "domain.stl")}

2. Create embed_report.json with embedding parameters

3. Create final_description.txt with:
   - Human narrative of the structure
   - Constraints applied
   - What was accepted

4. Create final_analysis.json with accepted metrics

5. Create manifest.json with:
   - Complete file list
   - All paths to artifacts
   - Version information

6. Create run_metadata.json with:
   - Seed used
   - Units
   - Library versions
   - Git hash if available
   - Environment snapshot

Use output_units="{self.context.units_export}" for all exports."""

        result = self.agent.run_task(
            task=task,
            context={
                "object_name": obj.name,
                "object_dir": obj.object_dir,
                "version": obj.version,
                "embed_domain": embed_domain,
                "output_units": self.context.units_export,
            }
        )
        
        if result.status == TaskStatus.COMPLETED:
            obj.final_void_stl = os.path.join(obj.final_dir, "embed", "void.stl")
            obj.final_domain_stl = os.path.join(obj.final_dir, "embed", "domain.stl")
            obj.final_manifest = os.path.join(obj.final_dir, "manifest.json")
            obj.status = "finalized"
            
            print("\nFinalization complete!")
            print(f"  Void STL: {obj.final_void_stl}")
            print(f"  Manifest: {obj.final_manifest}")
        else:
            print(f"\nFinalization had issues: {result.error}")
            print("Some final outputs may not have been generated.")
            obj.status = "finalized_with_errors"
        
        obj_json_path = os.path.join(obj.object_dir, "object.json")
        with open(obj_json_path, 'w') as f:
            json.dump(obj.to_dict(), f, indent=2)
        
        self._move_to_next_object_or_complete()
    
    def _move_to_next_object_or_complete(self) -> None:
        """Move to the next object or complete the workflow."""
        self.context.current_object_index += 1
        
        if self.context.current_object_index < len(self.context.objects):
            self.current_question_group = None
            self.current_question_index = 0
            self.collected_answers = {}
            self.state = WorkflowState.FRAME_OF_REFERENCE
        else:
            self._save_project_summary()
            self.state = WorkflowState.COMPLETE
    
    def _save_project_summary(self) -> None:
        """Save project summary JSON."""
        summary = {
            "workflow_name": self.WORKFLOW_NAME,
            "workflow_version": self.WORKFLOW_VERSION,
            "project_name": self.context.project_name,
            "project_slug": self.context.project_slug,
            "units_internal": self.context.units_internal,
            "units_export": self.context.units_export,
            "flow_solver_enabled": self.context.flow_solver_enabled,
            "variant_mode": self.context.variant_mode,
            "objects": [
                {
                    "name": obj.name,
                    "slug": obj.slug,
                    "status": obj.status,
                    "version": obj.version,
                    "final_void_stl": obj.final_void_stl,
                    "final_manifest": obj.final_manifest,
                }
                for obj in self.context.objects
            ],
            "timestamp": time.time(),
        }
        
        summary_path = os.path.join(self.context.output_dir, "project_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        if self.verbose:
            print(f"\nProject summary saved to: {summary_path}")
    
    def _get_previous_attempts_context(self, obj: ObjectContext) -> str:
        """
        Get context from previous attempts for the current object.
        
        This allows the agent to pull code and context from previous iterations
        to keep workable parts and learn from past attempts.
        """
        context_parts = []
        
        for v in range(1, obj.version):
            prev_spec = os.path.join(obj.spec_dir, f"spec_v{v:03d}.json")
            if os.path.exists(prev_spec):
                try:
                    with open(prev_spec, 'r') as f:
                        spec_content = f.read()
                    context_parts.append(f"=== Previous Spec v{v} ===\n{spec_content[:1000]}")
                except Exception:
                    pass
            
            prev_code = os.path.join(obj.code_dir, f"generate_v{v:03d}.py")
            if os.path.exists(prev_code):
                try:
                    with open(prev_code, 'r') as f:
                        code_content = f.read()
                    context_parts.append(f"=== Previous Code v{v} ===\n{code_content[:2000]}")
                except Exception:
                    pass
            
            prev_analysis = os.path.join(obj.analysis_dir, f"analysis_v{v:03d}.txt")
            if os.path.exists(prev_analysis):
                try:
                    with open(prev_analysis, 'r') as f:
                        analysis_content = f.read()
                    context_parts.append(f"=== Previous Analysis v{v} ===\n{analysis_content[:1000]}")
                except Exception:
                    pass
        
        if obj.feedback_history:
            feedback_str = "\n".join(f"  {i+1}. {fb}" for i, fb in enumerate(obj.feedback_history))
            context_parts.append(f"=== Feedback History ===\n{feedback_str}")
        
        return "\n\n".join(context_parts) if context_parts else ""
    
    def _process_input(self, user_input: str) -> Tuple[WorkflowState, str]:
        """Process user input for programmatic workflow control."""
        if user_input.lower() in ("quit", "exit"):
            self.state = WorkflowState.COMPLETE
            return self.state, "Workflow terminated by user."
        
        if self.state == WorkflowState.PROJECT_INIT:
            self.context.project_name = user_input
            self.context.project_slug = user_input.lower().replace(" ", "_")
            self.context.output_dir = os.path.join(self.base_output_dir, self.context.project_slug)
            os.makedirs(self.context.output_dir, exist_ok=True)
            os.makedirs(os.path.join(self.context.output_dir, "objects"), exist_ok=True)
            self.context.save_project_json()
            self.state = WorkflowState.OBJECT_PLANNING
            return self.state, f"Project '{user_input}' created. How many objects?"
        
        elif self.state == WorkflowState.OBJECT_PLANNING:
            try:
                num_objects = int(user_input)
            except ValueError:
                num_objects = 1
            
            for i in range(num_objects):
                name = f"object_{i+1:03d}"
                self.context.add_object(name, name)
            
            self.context.save_project_json()
            self.state = WorkflowState.FRAME_OF_REFERENCE
            return self.state, f"Created {num_objects} object(s). Starting requirements capture."
        
        elif self.state == WorkflowState.ITERATION:
            if user_input.lower() in ("yes", "y"):
                obj = self.context.get_current_object()
                if obj:
                    obj.status = "accepted"
                self.state = WorkflowState.FINALIZATION
                return self.state, "Great! Finalizing..."
            elif user_input.lower() in ("no", "n"):
                return self.state, "Please provide feedback on what to change."
            else:
                obj = self.context.get_current_object()
                if obj:
                    obj.feedback_history.append(user_input)
                    obj.increment_version()
                self.state = WorkflowState.SPEC_COMPILATION
                return self.state, "Feedback received. Regenerating..."
        
        return self.state, "Unexpected state. Please continue with the workflow."


def run_single_agent_workflow(
    provider: str = "openai",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_output_dir: str = "./outputs",
    **kwargs,
) -> ProjectContext:
    """
    Convenience function to run the Single Agent Organ Generator V2 workflow.
    
    Parameters
    ----------
    provider : str
        LLM provider ("openai", "anthropic", "local")
    api_key : str, optional
        API key (or set via environment variable)
    model : str, optional
        Model name
    base_output_dir : str
        Base directory for project outputs
    **kwargs
        Additional arguments for AgentConfig
        
    Returns
    -------
    ProjectContext
        Final project context with all generated artifacts
    """
    from .agent_runner import create_agent
    
    agent = create_agent(
        provider=provider,
        api_key=api_key,
        model=model,
        **kwargs,
    )
    
    workflow = SingleAgentOrganGeneratorV2(
        agent=agent,
        base_output_dir=base_output_dir,
    )
    
    return workflow.run()


SingleAgentOrganGeneratorV1 = SingleAgentOrganGeneratorV2
