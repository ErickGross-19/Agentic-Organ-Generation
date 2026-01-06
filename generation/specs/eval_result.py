"""Evaluation results for vascular network quality assessment."""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
import json


@dataclass
class CoverageMetrics:
    """Coverage and perfusion metrics."""
    coverage_fraction: float  # 0-1: fraction of tissue points perfused
    unperfused_points: int  # Number of tissue points not reached
    perfusion_uniformity: float  # 0-1: uniformity of perfusion (1=perfect)
    mean_distance_to_vessel: float  # Average distance from tissue to nearest vessel
    max_distance_to_vessel: float  # Maximum distance from tissue to nearest vessel


@dataclass
class FlowMetrics:
    """Hemodynamic flow metrics."""
    total_flow_arterial: float  # Total flow through arterial tree (m^3/s)
    total_flow_venous: float  # Total flow through venous tree (m^3/s)
    flow_balance_error: float  # 0-1: |arterial-venous|/arterial (0=perfect balance)
    min_pressure: float  # Minimum pressure in network (Pa)
    mean_pressure: float  # Mean pressure (Pa)
    max_pressure: float  # Maximum pressure (Pa)
    turbulent_fraction: float  # 0-1: fraction of segments with Re > 2300
    max_reynolds: float  # Maximum Reynolds number in network
    pressure_drop_arterial: float  # Pressure drop across arterial tree (Pa)
    pressure_drop_venous: float  # Pressure drop across venous tree (Pa)


@dataclass
class StructureMetrics:
    """Structural and topological metrics."""
    total_length: float  # Total centerline length (m)
    num_nodes: int  # Total number of nodes
    num_segments: int  # Total number of segments
    num_terminals: int  # Number of terminal nodes
    mean_branch_order: float  # Average branch order (generation)
    median_branch_order: float  # Median branch order
    max_branch_order: int  # Maximum branch order
    degree_histogram: Dict[int, int]  # {degree: count}
    mean_branching_angle: float  # Average angle at bifurcations (degrees)
    murray_deviation: float  # 0-1: deviation from Murray's law (0=perfect)
    collision_count: int  # Number of segment collisions detected
    min_clearance: float  # Minimum distance between non-connected segments (m)


@dataclass
class ValidityMetrics:
    """Validity and quality checks."""
    is_watertight: bool  # Whether mesh is watertight (if applicable)
    has_self_intersections: bool  # Whether network has self-intersections
    parameter_warnings: List[str]  # List of parameter validation warnings
    error_codes: List[str]  # List of error codes encountered


@dataclass
class EvalScores:
    """Normalized quality scores (0-1, higher is better)."""
    coverage_score: float  # Based on coverage_fraction and uniformity
    flow_score: float  # Based on flow balance and turbulence
    structure_score: float  # Based on Murray deviation and clearance
    overall_score: float  # Weighted combination of above scores


@dataclass
class EvalResult:
    """Complete evaluation result for a vascular network."""
    
    coverage: CoverageMetrics
    flow: FlowMetrics
    structure: StructureMetrics
    validity: ValidityMetrics
    scores: EvalScores
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "coverage": asdict(self.coverage),
            "flow": asdict(self.flow),
            "structure": asdict(self.structure),
            "validity": asdict(self.validity),
            "scores": asdict(self.scores),
            "metadata": self.metadata,
        }
    
    def to_json(self, path: str) -> None:
        """Save to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'EvalResult':
        """Create EvalResult from dictionary."""
        return EvalResult(
            coverage=CoverageMetrics(**d["coverage"]),
            flow=FlowMetrics(**d["flow"]),
            structure=StructureMetrics(**d["structure"]),
            validity=ValidityMetrics(**d["validity"]),
            scores=EvalScores(**d["scores"]),
            metadata=d.get("metadata", {}),
        )
    
    @staticmethod
    def from_json(path: str) -> 'EvalResult':
        """Load from JSON file."""
        with open(path, 'r') as f:
            d = json.load(f)
        return EvalResult.from_dict(d)
