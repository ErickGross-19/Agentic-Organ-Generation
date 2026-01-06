"""Design specifications for LLM-driven vascular network design.

This module provides dataclasses for specifying vascular network designs
in a JSON-serializable format suitable for LLM agents.
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
    """Ellipsoid domain specification.
    
    Parameters
    ----------
    center : Tuple[float, float, float]
        Center point (x, y, z)
    semi_axes : Tuple[float, float, float]
        Semi-axes lengths (a, b, c)
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
    """Box domain specification.
    
    Parameters
    ----------
    center : Tuple[float, float, float]
        Center point (x, y, z)
    size : Tuple[float, float, float]
        Box dimensions (width, height, depth)
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
    """Inlet specification."""
    
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
    """Outlet specification."""
    
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


@dataclass
class TreeSpec:
    """Single tree specification."""
    
    inlets: List[InletSpec]
    outlets: List[OutletSpec]
    colonization: ColonizationSpec
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "inlets": [inlet.to_dict() for inlet in self.inlets],
            "outlets": [outlet.to_dict() for outlet in self.outlets],
            "colonization": self.colonization.to_dict(),
        }
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'TreeSpec':
        return TreeSpec(
            inlets=[InletSpec.from_dict(i) for i in d["inlets"]],
            outlets=[OutletSpec.from_dict(o) for o in d["outlets"]],
            colonization=ColonizationSpec.from_dict(d["colonization"]),
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
    """Top-level design specification for vascular networks."""
    
    domain: DomainSpec
    tree: Optional[TreeSpec] = None
    dual_tree: Optional[DualTreeSpec] = None
    seed: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.tree is None and self.dual_tree is None:
            raise ValueError("Must specify either 'tree' or 'dual_tree'")
        if self.tree is not None and self.dual_tree is not None:
            raise ValueError("Cannot specify both 'tree' and 'dual_tree'")
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "domain": self.domain.to_dict(),
            "seed": self.seed,
            "metadata": self.metadata,
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
        )
    
    @staticmethod
    def from_json(path: str) -> 'DesignSpec':
        with open(path, 'r') as f:
            d = json.load(f)
        return DesignSpec.from_dict(d)
