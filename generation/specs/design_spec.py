"""
Design specifications for LLM-driven vascular network design.

This module provides dataclasses for specifying vascular network designs
in a JSON-serializable format suitable for LLM agents.

UNIT CONVENTIONS
----------------
**Spec units**: All geometric values in spec classes (positions, radii, sizes) are in
METERS. This is the internal unit system used throughout the codebase. For example:
- EllipsoidSpec(semi_axes=(0.05, 0.045, 0.035)) represents 50mm x 45mm x 35mm
- InletSpec(radius=0.002) represents a 2mm radius inlet
- ColonizationSpec(step_size=0.001) represents a 1mm step size

**Runtime units**: When specs are compiled via compile_domain(), the resulting runtime
domain objects also use METERS internally. All geometric operations operate in meters.

**Output units**: At export time (STL, JSON), values are converted from internal meters
to the user-specified output_units (default "mm") via UnitContext.

COORDINATE FRAME
----------------
The default coordinate frame is:
- Origin at domain center (0, 0, 0)
- X-axis: left-right (width)
- Y-axis: front-back (depth)
- Z-axis: bottom-top (height)

For organ-specific coordinate frames (e.g., anatomical orientation), use the
transform parameter in compile_domain() to apply a rotation/translation.

See generation/specs/compile.py for the compile_domain() function that converts
spec classes to runtime domain objects.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple, Dict, Any, Literal
import json


@dataclass
class DomainSpec:
    """Base class for domain specifications."""
    
    type: str  # "ellipsoid" or "box"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'DomainSpec':
        """Create DomainSpec from dictionary."""
        domain_type = d.get("type")
        if domain_type == "ellipsoid":
            return EllipsoidSpec.from_dict(d)
        elif domain_type == "box":
            return BoxSpec.from_dict(d)
        else:
            raise ValueError(f"Unknown domain type: {domain_type}")


@dataclass
class EllipsoidSpec(DomainSpec):
    """Ellipsoid domain specification (spec units: METERS).
    
    Use compile_domain() to convert to runtime EllipsoidDomain.
    
    Parameters
    ----------
    center : Tuple[float, float, float]
        Center point (x, y, z) in METERS. Default: origin (0, 0, 0).
    semi_axes : Tuple[float, float, float]
        Semi-axes lengths (a, b, c) in METERS. Default: (0.05, 0.045, 0.035)
        which represents 50mm x 45mm x 35mm (typical liver dimensions).
    """
    
    type: str = "ellipsoid"
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    semi_axes: Tuple[float, float, float] = (0.05, 0.045, 0.035)
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'EllipsoidSpec':
        """Create EllipsoidSpec from dictionary."""
        return EllipsoidSpec(
            center=tuple(d.get("center", [0.0, 0.0, 0.0])),
            semi_axes=tuple(d.get("semi_axes", [0.05, 0.045, 0.035])),
        )


@dataclass
class BoxSpec(DomainSpec):
    """Box domain specification (spec units: METERS).
    
    Use compile_domain() to convert to runtime BoxDomain.
    
    Parameters
    ----------
    center : Tuple[float, float, float]
        Center point (x, y, z) in METERS. Default: origin (0, 0, 0).
    size : Tuple[float, float, float]
        Box dimensions (width, height, depth) in METERS. Default: (0.10, 0.09, 0.07)
        which represents 100mm x 90mm x 70mm.
    """
    
    type: str = "box"
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    size: Tuple[float, float, float] = (0.10, 0.09, 0.07)
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'BoxSpec':
        """Create BoxSpec from dictionary."""
        return BoxSpec(
            center=tuple(d.get("center", [0.0, 0.0, 0.0])),
            size=tuple(d.get("size", [0.10, 0.09, 0.07])),
        )


@dataclass
class InletSpec:
    """Inlet specification (spec units: METERS).
    
    Parameters
    ----------
    position : Tuple[float, float, float]
        Inlet position (x, y, z) in METERS.
    radius : float
        Inlet radius in METERS. Example: 0.002 = 2mm.
    vessel_type : str
        Either "arterial" or "venous". Default: "arterial".
    """
    
    position: Tuple[float, float, float]
    radius: float
    vessel_type: Literal["arterial", "venous"] = "arterial"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'InletSpec':
        return InletSpec(
            position=tuple(d["position"]),
            radius=d["radius"],
            vessel_type=d.get("vessel_type", "arterial"),
        )


@dataclass
class OutletSpec:
    """Outlet specification (spec units: METERS).
    
    Parameters
    ----------
    position : Tuple[float, float, float]
        Outlet position (x, y, z) in METERS.
    radius : float
        Outlet radius in METERS. Example: 0.002 = 2mm.
    vessel_type : str
        Either "arterial" or "venous". Default: "venous".
    """
    
    position: Tuple[float, float, float]
    radius: float
    vessel_type: Literal["arterial", "venous"] = "venous"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'OutletSpec':
        return OutletSpec(
            position=tuple(d["position"]),
            radius=d["radius"],
            vessel_type=d.get("vessel_type", "venous"),
        )


@dataclass
class ColonizationSpec:
    """Space colonization parameters specification."""
    
    tissue_points: Optional[List[List[float]]] = None
    influence_radius: float = 0.015
    kill_radius: float = 0.002
    step_size: float = 0.001
    max_steps: int = 500
    initial_radius: float = 0.0005
    min_radius: float = 0.0001
    radius_decay: float = 0.95
    preferred_direction: Optional[Tuple[float, float, float]] = None
    directional_bias: float = 0.0
    max_deviation_deg: float = 180.0
    smoothing_weight: float = 0.3
    encourage_bifurcation: bool = False
    min_attractions_for_bifurcation: int = 3
    max_children_per_node: int = 2
    bifurcation_angle_threshold_deg: float = 40.0
    bifurcation_probability: float = 0.7
    max_curvature_deg: Optional[float] = None
    min_clearance: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'ColonizationSpec':
        return ColonizationSpec(
            tissue_points=d.get("tissue_points"),
            influence_radius=d.get("influence_radius", 0.015),
            kill_radius=d.get("kill_radius", 0.002),
            step_size=d.get("step_size", 0.001),
            max_steps=d.get("max_steps", 500),
            initial_radius=d.get("initial_radius", 0.0005),
            min_radius=d.get("min_radius", 0.0001),
            radius_decay=d.get("radius_decay", 0.95),
            preferred_direction=tuple(d["preferred_direction"]) if d.get("preferred_direction") else None,
            directional_bias=d.get("directional_bias", 0.0),
            max_deviation_deg=d.get("max_deviation_deg", 180.0),
            smoothing_weight=d.get("smoothing_weight", 0.3),
            encourage_bifurcation=d.get("encourage_bifurcation", False),
            min_attractions_for_bifurcation=d.get("min_attractions_for_bifurcation", 3),
            max_children_per_node=d.get("max_children_per_node", 2),
            bifurcation_angle_threshold_deg=d.get("bifurcation_angle_threshold_deg", 40.0),
            bifurcation_probability=d.get("bifurcation_probability", 0.7),
            max_curvature_deg=d.get("max_curvature_deg"),
            min_clearance=d.get("min_clearance"),
        )


# Topology kind type for explicit topology modes (Priority 4)
TopologyKind = Literal["path", "tree", "loop", "multi_tree"]


@dataclass
class TreeSpec:
    """Single tree specification.
    
    Parameters
    ----------
    inlets : List[InletSpec]
        List of inlet specifications
    outlets : List[OutletSpec]
        List of outlet specifications
    colonization : ColonizationSpec
        Space colonization parameters
    topology_kind : TopologyKind, optional
        Explicit topology mode. Determines required fields and validation rules:
        - "path": Simple inlet->outlet channel. Requires exactly 1 inlet and 1 outlet.
                  No branching, terminals=0.
        - "tree": Branching tree structure. Requires inlet(s), terminal targets or
                  density target, and branching constraints.
        - "loop": Closed loop structure. Requires at least 2 connection points.
        - "multi_tree": Multiple independent trees. Requires multiple inlets.
        Default: None (auto-inferred from inlets/outlets configuration)
    terminal_count : int, optional
        Target number of terminal nodes (for "tree" topology)
    terminal_density : float, optional
        Target terminal density per unit volume (for "tree" topology)
    """
    
    inlets: List[InletSpec]
    outlets: List[OutletSpec]
    colonization: Optional[ColonizationSpec] = None
    topology_kind: Optional[TopologyKind] = None
    terminal_count: Optional[int] = None
    terminal_density: Optional[float] = None
    
    def __post_init__(self):
        """Validate topology-specific requirements."""
        if self.topology_kind == "path":
            if len(self.inlets) != 1:
                raise ValueError("'path' topology requires exactly 1 inlet")
            if len(self.outlets) != 1:
                raise ValueError("'path' topology requires exactly 1 outlet")
        elif self.topology_kind == "tree":
            if len(self.inlets) == 0:
                raise ValueError("'tree' topology requires at least 1 inlet")
            if self.terminal_count is None and self.terminal_density is None:
                # Warning only - don't raise error for backward compatibility
                pass
        elif self.topology_kind == "loop":
            total_ports = len(self.inlets) + len(self.outlets)
            if total_ports < 2:
                raise ValueError("'loop' topology requires at least 2 connection points")
        elif self.topology_kind == "multi_tree":
            if len(self.inlets) < 2:
                raise ValueError("'multi_tree' topology requires at least 2 inlets")
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "inlets": [inlet.to_dict() for inlet in self.inlets],
            "outlets": [outlet.to_dict() for outlet in self.outlets],
        }
        if self.colonization is not None:
            result["colonization"] = self.colonization.to_dict()
        if self.topology_kind is not None:
            result["topology_kind"] = self.topology_kind
        if self.terminal_count is not None:
            result["terminal_count"] = self.terminal_count
        if self.terminal_density is not None:
            result["terminal_density"] = self.terminal_density
        return result
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'TreeSpec':
        return TreeSpec(
            inlets=[InletSpec.from_dict(i) for i in d["inlets"]],
            outlets=[OutletSpec.from_dict(o) for o in d["outlets"]],
            colonization=ColonizationSpec.from_dict(d["colonization"]) if "colonization" in d and d["colonization"] is not None else None,
            topology_kind=d.get("topology_kind"),
            terminal_count=d.get("terminal_count"),
            terminal_density=d.get("terminal_density"),
        )
    
    @staticmethod
    def single_inlet(inlet_position: Tuple[float, float, float], 
                     inlet_radius: float,
                     colonization: ColonizationSpec,
                     vessel_type: Literal["arterial", "venous"] = "arterial") -> 'TreeSpec':
        """Convenience constructor for single inlet tree (no outlets)."""
        return TreeSpec(
            inlets=[InletSpec(position=inlet_position, radius=inlet_radius, vessel_type=vessel_type)],
            outlets=[],
            colonization=colonization,
        )


@dataclass
class DualTreeSpec:
    """Dual tree specification (arterial + venous)."""
    
    arterial_inlets: List[InletSpec]
    venous_outlets: List[OutletSpec]
    arterial_colonization: ColonizationSpec
    venous_colonization: ColonizationSpec
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "arterial_inlets": [inlet.to_dict() for inlet in self.arterial_inlets],
            "venous_outlets": [outlet.to_dict() for outlet in self.venous_outlets],
            "arterial_colonization": self.arterial_colonization.to_dict(),
            "venous_colonization": self.venous_colonization.to_dict(),
        }
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'DualTreeSpec':
        return DualTreeSpec(
            arterial_inlets=[InletSpec.from_dict(i) for i in d["arterial_inlets"]],
            venous_outlets=[OutletSpec.from_dict(o) for o in d["venous_outlets"]],
            arterial_colonization=ColonizationSpec.from_dict(d["arterial_colonization"]),
            venous_colonization=ColonizationSpec.from_dict(d["venous_colonization"]),
        )
    
    @staticmethod
    def single_inlet_outlet(arterial_inlet_position: Tuple[float, float, float],
                           arterial_inlet_radius: float,
                           venous_outlet_position: Tuple[float, float, float],
                           venous_outlet_radius: float,
                           arterial_colonization: ColonizationSpec,
                           venous_colonization: ColonizationSpec) -> 'DualTreeSpec':
        """Convenience constructor for single arterial inlet and single venous outlet."""
        return DualTreeSpec(
            arterial_inlets=[InletSpec(position=arterial_inlet_position, radius=arterial_inlet_radius, vessel_type="arterial")],
            venous_outlets=[OutletSpec(position=venous_outlet_position, radius=venous_outlet_radius, vessel_type="venous")],
            arterial_colonization=arterial_colonization,
            venous_colonization=venous_colonization,
        )


@dataclass
class DesignSpec:
    """Top-level design specification for vascular networks.
    
    INTERNAL UNITS: The library uses METER-SCALE values internally.
    All coordinates, lengths, and radii are stored in meters throughout the codebase.
    For example, EllipsoidSpec defaults to semi_axes=(0.05, 0.045, 0.035) which 
    represents 50mm, 45mm, 35mm in meters.
    
    OUTPUT UNITS: At export time (STL, JSON, etc.), internal meter values are 
    converted to the user-specified output_units:
    - output_units="mm" (default): internal 0.05 becomes 50.0 in output
    - output_units="m": internal 0.05 stays 0.05 in output
    
    Attributes
    ----------
    domain : DomainSpec
        Domain specification (ellipsoid or box)
    tree : TreeSpec, optional
        Single tree specification
    dual_tree : DualTreeSpec, optional
        Dual tree specification (arterial + venous)
    seed : int, optional
        Random seed for reproducibility
    metadata : dict
        Additional metadata
    output_units : str
        Units for exported files (STL, JSON, etc.). Default: "mm"
        Supported: "m", "mm", "cm", "um"
        Internal meter values are scaled to this unit on export.
    """
    
    domain: DomainSpec
    tree: Optional[TreeSpec] = None
    dual_tree: Optional[DualTreeSpec] = None
    seed: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    output_units: str = "mm"
    
    def __post_init__(self):
        if self.tree is None and self.dual_tree is None:
            raise ValueError("Must specify either 'tree' or 'dual_tree'")
        if self.tree is not None and self.dual_tree is not None:
            raise ValueError("Cannot specify both 'tree' and 'dual_tree'")
        if self.output_units not in ("m", "mm", "cm", "um"):
            raise ValueError(f"Unknown output_units '{self.output_units}'. Supported: m, mm, cm, um")
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "domain": self.domain.to_dict(),
            "seed": self.seed,
            "metadata": self.metadata,
            "output_units": self.output_units,
        }
        if self.tree is not None:
            result["tree"] = self.tree.to_dict()
        if self.dual_tree is not None:
            result["dual_tree"] = self.dual_tree.to_dict()
        return result
    
    def to_json(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'DesignSpec':
        return DesignSpec(
            domain=DomainSpec.from_dict(d["domain"]),
            tree=TreeSpec.from_dict(d["tree"]) if "tree" in d else None,
            dual_tree=DualTreeSpec.from_dict(d["dual_tree"]) if "dual_tree" in d else None,
            seed=d.get("seed"),
            metadata=d.get("metadata", {}),
            output_units=d.get("output_units", "mm"),
        )
    
    @staticmethod
    def from_json(path: str) -> 'DesignSpec':
        with open(path, 'r') as f:
            d = json.load(f)
        return DesignSpec.from_dict(d)
