"""
Structured run report for generation and validation.

This module provides a standardized JSON-serializable report structure
for tracking all aspects of a generation run.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class RunReport:
    """
    Comprehensive run report for generation and validation.
    
    This report captures all inputs, intermediate states, and outputs
    from a generation run in a JSON-serializable format.
    
    JSON Schema:
    {
        "inputs": {
            "domain": {},
            "policies": {},
            "seeds": []
        },
        "placement": {
            "effective_radius": float,
            "clamps": int,
            "projection_distances": {}
        },
        "generation": {
            "terminal_counts": {},
            "growth_stats": {}
        },
        "merge": {
            "mode": str,
            "pitch_adjustments": [],
            "failures": []
        },
        "synthesis": {
            "caps_added": bool,
            "spheres_added": bool,
            "radius_clamps": {}
        },
        "embedding": {
            "pitch_adjustments": [],
            "shrink_warnings": []
        },
        "validity": {
            "watertight": bool,
            "drift_metrics": {},
            "repairs_applied": []
        },
        "outputs": {
            "paths": {}
        }
    }
    """
    # Metadata
    run_id: str = ""
    timestamp: str = ""
    duration_seconds: float = 0.0
    
    # Inputs
    inputs: Dict[str, Any] = field(default_factory=dict)
    
    # Placement
    placement: Dict[str, Any] = field(default_factory=dict)
    
    # Generation
    generation: Dict[str, Any] = field(default_factory=dict)
    
    # Merge
    merge: Dict[str, Any] = field(default_factory=dict)
    
    # Synthesis
    synthesis: Dict[str, Any] = field(default_factory=dict)
    
    # Embedding
    embedding: Dict[str, Any] = field(default_factory=dict)
    
    # Validity
    validity: Dict[str, Any] = field(default_factory=dict)
    
    # Outputs
    outputs: Dict[str, Any] = field(default_factory=dict)
    
    # Warnings and errors
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RunReport":
        """Create from dictionary."""
        return RunReport(**{k: v for k, v in d.items() if k in RunReport.__dataclass_fields__})
    
    @staticmethod
    def from_json(s: str) -> "RunReport":
        """Create from JSON string."""
        return RunReport.from_dict(json.loads(s))


def create_run_report(
    run_id: Optional[str] = None,
) -> RunReport:
    """
    Create a new run report with initialized metadata.
    
    Parameters
    ----------
    run_id : str, optional
        Unique identifier for the run (generated if not provided)
        
    Returns
    -------
    RunReport
        New run report with metadata initialized
    """
    if run_id is None:
        run_id = f"run_{int(time.time())}"
    
    return RunReport(
        run_id=run_id,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def save_run_report(
    report: RunReport,
    output_dir: Path,
    filename: str = "run_report.json",
) -> Path:
    """
    Save a run report to a JSON file.
    
    Parameters
    ----------
    report : RunReport
        Report to save
    output_dir : Path
        Directory to save to
    filename : str
        Filename for the report
        
    Returns
    -------
    Path
        Path to the saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / filename
    
    with open(output_path, 'w') as f:
        f.write(report.to_json())
    
    logger.info(f"Saved run report to {output_path}")
    
    return output_path


def update_report_section(
    report: RunReport,
    section: str,
    data: Dict[str, Any],
) -> RunReport:
    """
    Update a section of the run report.
    
    Parameters
    ----------
    report : RunReport
        Report to update
    section : str
        Section name (inputs, placement, generation, etc.)
    data : dict
        Data to merge into the section
        
    Returns
    -------
    RunReport
        Updated report
    """
    if hasattr(report, section):
        current = getattr(report, section)
        if isinstance(current, dict):
            current.update(data)
        else:
            setattr(report, section, data)
    
    return report


__all__ = [
    "RunReport",
    "create_run_report",
    "save_run_report",
    "update_report_section",
]
