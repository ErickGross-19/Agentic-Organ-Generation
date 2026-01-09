"""
MOGS Data Models

Data classes for the MultiAgentOrgan Generation System (MOGS).
Defines the core data structures for objects, specs, versions, and manifests.
"""

import uuid
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from enum import Enum


class ApprovalStatus(Enum):
    """Status of an approval gate."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"


class VersionStatus(Enum):
    """Status of a spec version."""
    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


class AgentType(Enum):
    """Types of agents in the MOGS workflow."""
    CSA = "csa"  # Concept & Spec Agent
    CBA = "cba"  # Coding & Build Agent
    VQA = "vqa"  # Validation & QA Agent


class GateType(Enum):
    """Types of approval gates."""
    SPEC_APPROVAL = "spec_approval"      # Gate A: CSA -> user
    CODE_APPROVAL = "code_approval"      # Gate B: CBA -> user
    RESULTS_APPROVAL = "results_approval"  # Gate C: VQA -> user


@dataclass
class RetentionPolicy:
    """Retention policy for version management."""
    keep_last_versions: int = 5
    pinned_versions: List[int] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "keep_last_versions": self.keep_last_versions,
            "pinned_versions": self.pinned_versions,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RetentionPolicy":
        return cls(
            keep_last_versions=d.get("keep_last_versions", 5),
            pinned_versions=d.get("pinned_versions", []),
        )


@dataclass
class ObjectManifest:
    """
    Manifest for an object in the MOGS system.
    
    Stored at: objects/<object_uuid>/project/00_admin/object_manifest.json
    """
    object_uuid: str
    object_name: str
    created_at: str
    active_spec_version: int = 0
    retention_policy: RetentionPolicy = field(default_factory=RetentionPolicy)
    notes: str = ""
    last_modified_at: str = ""
    total_versions: int = 0
    accepted_versions: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.last_modified_at:
            self.last_modified_at = self.created_at
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_uuid": self.object_uuid,
            "object_name": self.object_name,
            "created_at": self.created_at,
            "last_modified_at": self.last_modified_at,
            "active_spec_version": self.active_spec_version,
            "retention_policy": self.retention_policy.to_dict(),
            "notes": self.notes,
            "total_versions": self.total_versions,
            "accepted_versions": self.accepted_versions,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ObjectManifest":
        return cls(
            object_uuid=d["object_uuid"],
            object_name=d["object_name"],
            created_at=d["created_at"],
            last_modified_at=d.get("last_modified_at", d["created_at"]),
            active_spec_version=d.get("active_spec_version", 0),
            retention_policy=RetentionPolicy.from_dict(d.get("retention_policy", {})),
            notes=d.get("notes", ""),
            total_versions=d.get("total_versions", 0),
            accepted_versions=d.get("accepted_versions", []),
        )
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_json(cls, json_str: str) -> "ObjectManifest":
        return cls.from_dict(json.loads(json_str))
    
    def save(self, path: str) -> None:
        """Save manifest to file."""
        with open(path, 'w') as f:
            f.write(self.to_json())
    
    @classmethod
    def load(cls, path: str) -> "ObjectManifest":
        """Load manifest from file."""
        with open(path, 'r') as f:
            return cls.from_json(f.read())


@dataclass
class RiskFlag:
    """A risk flag identified during spec creation."""
    id: str
    severity: str  # "low", "medium", "high", "critical"
    category: str  # e.g., "geometry", "topology", "manufacturing", "performance"
    description: str
    mitigation: str
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RiskFlag":
        return cls(**d)


@dataclass
class SpecVersion:
    """
    A versioned specification in the MOGS system.
    
    The spec is the canonical source of truth for what should be generated.
    Only CSA can create new spec versions.
    """
    version: int
    spec_data: Dict[str, Any]  # The actual specification content
    status: VersionStatus = VersionStatus.DRAFT
    created_at: str = ""
    created_by: str = "CSA"
    summary: str = ""
    risk_flags: List[RiskFlag] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    changelog_entry: str = ""
    parent_version: Optional[int] = None
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat() + "Z"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "spec_data": self.spec_data,
            "status": self.status.value,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "summary": self.summary,
            "risk_flags": [rf.to_dict() for rf in self.risk_flags],
            "assumptions": self.assumptions,
            "changelog_entry": self.changelog_entry,
            "parent_version": self.parent_version,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SpecVersion":
        return cls(
            version=d["version"],
            spec_data=d["spec_data"],
            status=VersionStatus(d.get("status", "draft")),
            created_at=d.get("created_at", ""),
            created_by=d.get("created_by", "CSA"),
            summary=d.get("summary", ""),
            risk_flags=[RiskFlag.from_dict(rf) for rf in d.get("risk_flags", [])],
            assumptions=d.get("assumptions", []),
            changelog_entry=d.get("changelog_entry", ""),
            parent_version=d.get("parent_version"),
        )


@dataclass
class ExpectedArtifact:
    """An expected artifact from script execution."""
    filename: str
    description: str
    required: bool = True
    validation_type: str = "exists"  # "exists", "json", "stl", "custom"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExpectedArtifact":
        return cls(**d)


@dataclass
class ScriptManifest:
    """
    Manifest for a set of scripts generated by CBA.
    
    Stored at: 04_scripts/scripts_v###/run_manifest.json
    """
    spec_version: int
    scripts: List[str]  # ["01_generate.py", "02_analyze.py", "03_finalize.py"]
    expected_artifacts: List[ExpectedArtifact]
    created_at: str = ""
    build_plan_path: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat() + "Z"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "scripts": self.scripts,
            "expected_artifacts": [ea.to_dict() for ea in self.expected_artifacts],
            "created_at": self.created_at,
            "build_plan_path": self.build_plan_path,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ScriptManifest":
        return cls(
            spec_version=d["spec_version"],
            scripts=d["scripts"],
            expected_artifacts=[ExpectedArtifact.from_dict(ea) for ea in d.get("expected_artifacts", [])],
            created_at=d.get("created_at", ""),
            build_plan_path=d.get("build_plan_path", ""),
        )


@dataclass
class RunEntry:
    """
    Entry in the run index tracking a script execution.
    
    Stored in: 00_admin/run_index.json
    """
    run_id: str
    spec_version: int
    timestamp: str
    scripts_path: str
    outputs_path: str
    validation_path: str
    status: str  # "running", "completed", "failed"
    accepted: bool = False
    pinned: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunEntry":
        return cls(**d)


@dataclass
class RunIndex:
    """
    Index of all runs for an object.
    
    Stored at: objects/<object_uuid>/project/00_admin/run_index.json
    """
    object_uuid: str
    runs: List[RunEntry] = field(default_factory=list)
    last_updated: str = ""
    
    def __post_init__(self):
        if not self.last_updated:
            self.last_updated = datetime.utcnow().isoformat() + "Z"
    
    def add_run(self, entry: RunEntry) -> None:
        """Add a new run entry."""
        self.runs.append(entry)
        self.last_updated = datetime.utcnow().isoformat() + "Z"
    
    def get_versions_to_delete(self, retention_policy: RetentionPolicy) -> List[int]:
        """
        Get list of versions that should be deleted based on retention policy.
        
        Returns versions to delete (oldest first), respecting:
        - Keep last N versions
        - Never delete pinned versions
        - Never delete most recent accepted version
        """
        # Get all unique versions
        all_versions = sorted(set(r.spec_version for r in self.runs))
        
        if len(all_versions) <= retention_policy.keep_last_versions:
            return []
        
        # Find accepted versions
        accepted_versions = [r.spec_version for r in self.runs if r.accepted]
        most_recent_accepted = max(accepted_versions) if accepted_versions else None
        
        # Determine which versions to keep
        versions_to_keep = set()
        
        # Keep last N versions
        versions_to_keep.update(all_versions[-retention_policy.keep_last_versions:])
        
        # Keep pinned versions
        versions_to_keep.update(retention_policy.pinned_versions)
        
        # Keep most recent accepted version
        if most_recent_accepted is not None:
            versions_to_keep.add(most_recent_accepted)
        
        # Return versions to delete (oldest first)
        versions_to_delete = [v for v in all_versions if v not in versions_to_keep]
        return sorted(versions_to_delete)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_uuid": self.object_uuid,
            "runs": [r.to_dict() for r in self.runs],
            "last_updated": self.last_updated,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunIndex":
        return cls(
            object_uuid=d["object_uuid"],
            runs=[RunEntry.from_dict(r) for r in d.get("runs", [])],
            last_updated=d.get("last_updated", ""),
        )
    
    def save(self, path: str) -> None:
        """Save run index to file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "RunIndex":
        """Load run index from file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


@dataclass
class ValidationCheck:
    """A single validation check result."""
    check_id: str
    name: str
    passed: bool
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ValidationCheck":
        return cls(**d)


@dataclass
class ValidationReport:
    """
    Validation report generated by VQA.
    
    Stored at: 07_validation/v###/validation.json
    """
    spec_version: int
    timestamp: str
    overall_passed: bool
    checks: List[ValidationCheck]
    metrics: Dict[str, Any] = field(default_factory=dict)
    suggested_refinements: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "timestamp": self.timestamp,
            "overall_passed": self.overall_passed,
            "checks": [c.to_dict() for c in self.checks],
            "metrics": self.metrics,
            "suggested_refinements": self.suggested_refinements,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ValidationReport":
        return cls(
            spec_version=d["spec_version"],
            timestamp=d["timestamp"],
            overall_passed=d["overall_passed"],
            checks=[ValidationCheck.from_dict(c) for c in d.get("checks", [])],
            metrics=d.get("metrics", {}),
            suggested_refinements=d.get("suggested_refinements", []),
        )
    
    def save(self, path: str) -> None:
        """Save validation report to file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "ValidationReport":
        """Load validation report from file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


@dataclass
class ApprovalRecord:
    """Record of an approval decision."""
    gate_type: GateType
    spec_version: int
    status: ApprovalStatus
    timestamp: str
    reviewer: str = "user"
    comments: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_type": self.gate_type.value,
            "spec_version": self.spec_version,
            "status": self.status.value,
            "timestamp": self.timestamp,
            "reviewer": self.reviewer,
            "comments": self.comments,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ApprovalRecord":
        return cls(
            gate_type=GateType(d["gate_type"]),
            spec_version=d["spec_version"],
            status=ApprovalStatus(d["status"]),
            timestamp=d["timestamp"],
            reviewer=d.get("reviewer", "user"),
            comments=d.get("comments", ""),
        )


@dataclass
class FinalManifest:
    """
    Final manifest for completed outputs.
    
    Stored at: 06_outputs/v###/final/final_manifest.json
    """
    spec_version: int
    void_stl_path: str
    scaffold_stl_path: str
    units: str = "mm"
    created_at: str = ""
    generation_metrics: Dict[str, Any] = field(default_factory=dict)
    validation_summary: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat() + "Z"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "void_stl_path": self.void_stl_path,
            "scaffold_stl_path": self.scaffold_stl_path,
            "units": self.units,
            "created_at": self.created_at,
            "generation_metrics": self.generation_metrics,
            "validation_summary": self.validation_summary,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FinalManifest":
        return cls(
            spec_version=d["spec_version"],
            void_stl_path=d["void_stl_path"],
            scaffold_stl_path=d["scaffold_stl_path"],
            units=d.get("units", "mm"),
            created_at=d.get("created_at", ""),
            generation_metrics=d.get("generation_metrics", {}),
            validation_summary=d.get("validation_summary", {}),
        )
    
    def save(self, path: str) -> None:
        """Save final manifest to file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "FinalManifest":
        """Load final manifest from file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


def generate_object_uuid() -> str:
    """Generate a new UUID for an object."""
    return str(uuid.uuid4())


def get_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.utcnow().isoformat() + "Z"
