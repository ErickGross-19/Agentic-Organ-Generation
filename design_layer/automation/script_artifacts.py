"""
Script Artifacts

Defines the artifact contract for LLM-generated scripts, including expected files
per stage and manifest schema for tracking generated outputs.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import time


@dataclass
class ArtifactProfile:
    """
    Profile defining expected artifacts for a workflow stage.
    
    Attributes
    ----------
    name : str
        Profile name (e.g., "generation", "final")
    required_files : List[str]
        List of required file patterns (relative to object folder)
    optional_files : List[str]
        List of optional file patterns
    description : str
        Human-readable description of this profile
    """
    name: str
    required_files: List[str]
    optional_files: List[str] = field(default_factory=list)
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def generation_expected_files(version: int) -> ArtifactProfile:
    """
    Get expected files for the generation stage.
    
    Parameters
    ----------
    version : int
        Version number for versioned file names
        
    Returns
    -------
    ArtifactProfile
        Profile with required and optional files for generation stage
    """
    v = f"v{version:03d}"
    return ArtifactProfile(
        name="generation",
        required_files=[
            f"04_outputs/network_{v}.json",
            f"05_mesh/mesh_network_{v}.stl",
        ],
        optional_files=[
            f"06_analysis/analysis_{v}.json",
            f"06_analysis/analysis_{v}.txt",
        ],
        description=f"Generation stage artifacts for version {version}",
    )


def analysis_expected_files(version: int) -> ArtifactProfile:
    """
    Get expected files for the analysis/validation stage.
    
    Parameters
    ----------
    version : int
        Version number for versioned file names
        
    Returns
    -------
    ArtifactProfile
        Profile with required and optional files for analysis stage
    """
    v = f"v{version:03d}"
    return ArtifactProfile(
        name="analysis",
        required_files=[
            f"06_analysis/analysis_{v}.json",
        ],
        optional_files=[
            f"06_analysis/analysis_{v}.txt",
            f"07_validation/validation_{v}.json",
        ],
        description=f"Analysis stage artifacts for version {version}",
    )


def final_expected_files() -> ArtifactProfile:
    """
    Get expected files for the finalization stage.
    
    Returns
    -------
    ArtifactProfile
        Profile with required and optional files for final stage
    """
    return ArtifactProfile(
        name="final",
        required_files=[
            "09_final/embed/void.stl",
            "09_final/manifest.json",
        ],
        optional_files=[
            "09_final/embed/domain.stl",
            "09_final/embed_report.json",
            "09_final/final_description.txt",
            "09_final/final_analysis.json",
            "09_final/run_metadata.json",
        ],
        description="Final stage artifacts after embedding and finalization",
    )


@dataclass
class ArtifactManifest:
    """
    Manifest tracking all artifacts generated during a workflow iteration.
    
    Attributes
    ----------
    version : int
        Iteration version number
    created_files : List[str]
        List of file paths created during this iteration
    spec_path : str
        Path to the design spec used for generation
    seed : int or None
        Random seed used for reproducibility
    timestamps : Dict[str, float]
        Timestamps for various stages (start, end, etc.)
    key_metrics : Dict[str, Any]
        Key metrics from generation/analysis (optional)
    status : str
        Status of this iteration (pending, completed, failed)
    errors : List[str]
        List of error messages if any
    """
    version: int
    created_files: List[str] = field(default_factory=list)
    spec_path: str = ""
    seed: Optional[int] = None
    timestamps: Dict[str, float] = field(default_factory=dict)
    key_metrics: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    errors: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if "created" not in self.timestamps:
            self.timestamps["created"] = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ArtifactManifest":
        return cls(
            version=d.get("version", 1),
            created_files=d.get("created_files", []),
            spec_path=d.get("spec_path", ""),
            seed=d.get("seed"),
            timestamps=d.get("timestamps", {}),
            key_metrics=d.get("key_metrics", {}),
            status=d.get("status", "pending"),
            errors=d.get("errors", []),
        )
    
    def add_file(self, file_path: str) -> None:
        """Add a file to the created files list."""
        if file_path not in self.created_files:
            self.created_files.append(file_path)
    
    def add_metric(self, key: str, value: Any) -> None:
        """Add a key metric."""
        self.key_metrics[key] = value
    
    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
        self.status = "failed"
    
    def mark_completed(self) -> None:
        """Mark the manifest as completed."""
        self.status = "completed"
        self.timestamps["completed"] = time.time()
    
    def mark_failed(self, error: Optional[str] = None) -> None:
        """Mark the manifest as failed."""
        self.status = "failed"
        self.timestamps["failed"] = time.time()
        if error:
            self.add_error(error)
    
    def save(self, path: str) -> None:
        """Save manifest to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "ArtifactManifest":
        """Load manifest from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


@dataclass
class ArtifactsJson:
    """
    Structure for the ARTIFACTS_JSON footer that scripts should print.
    
    This is parsed from script stdout to verify what was created.
    
    Attributes
    ----------
    files : List[str]
        List of file paths created by the script
    metrics : Dict[str, Any]
        Key metrics from the script execution
    status : str
        Execution status (success, partial, failed)
    """
    files: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    status: str = "success"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json_line(self) -> str:
        """Format as the expected footer line."""
        return f"ARTIFACTS_JSON: {json.dumps(self.to_dict())}"
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ArtifactsJson":
        return cls(
            files=d.get("files", []),
            metrics=d.get("metrics", {}),
            status=d.get("status", "success"),
        )
    
    @classmethod
    def parse_from_output(cls, output: str) -> Optional["ArtifactsJson"]:
        """
        Parse ARTIFACTS_JSON from script output.
        
        Parameters
        ----------
        output : str
            Script stdout/stderr output
            
        Returns
        -------
        ArtifactsJson or None
            Parsed artifacts info, or None if not found
        """
        for line in output.split('\n'):
            line = line.strip()
            if line.startswith("ARTIFACTS_JSON:"):
                try:
                    json_str = line[len("ARTIFACTS_JSON:"):].strip()
                    data = json.loads(json_str)
                    return cls.from_dict(data)
                except (json.JSONDecodeError, ValueError):
                    continue
        return None


def get_artifact_profile(stage: str, version: int = 1) -> ArtifactProfile:
    """
    Get the artifact profile for a given stage.
    
    Parameters
    ----------
    stage : str
        Stage name: "generation", "analysis", or "final"
    version : int
        Version number for versioned files
        
    Returns
    -------
    ArtifactProfile
        The artifact profile for the stage
        
    Raises
    ------
    ValueError
        If stage is not recognized
    """
    stage = stage.lower()
    if stage == "generation":
        return generation_expected_files(version)
    elif stage == "analysis":
        return analysis_expected_files(version)
    elif stage == "final":
        return final_expected_files()
    else:
        raise ValueError(f"Unknown stage: {stage}. Valid stages: generation, analysis, final")
