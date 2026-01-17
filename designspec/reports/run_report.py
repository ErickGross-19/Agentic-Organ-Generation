"""
RunReport schema for DesignSpec execution reproducibility.

This module provides the RunReport dataclass that captures full
reproducibility state from a DesignSpec run.

CONTENTS
--------
- meta: seed, input_units, timestamps
- env: python version, package versions, platform
- hashes: spec_hash, repo_commit if available
- stages: list of stage reports (requested/effective policy + warnings + metrics)
- artifacts: name → path + hash + basic mesh stats
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
import platform
import sys
import subprocess
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetaInfo:
    """Metadata about the run."""
    seed: Optional[int] = None
    input_units: str = "m"
    spec_name: str = ""
    spec_description: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_s: float = 0.0
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "input_units": self.input_units,
            "spec_name": self.spec_name,
            "spec_description": self.spec_description,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_s": self.duration_s,
            "tags": self.tags,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MetaInfo":
        return cls(
            seed=d.get("seed"),
            input_units=d.get("input_units", "m"),
            spec_name=d.get("spec_name", ""),
            spec_description=d.get("spec_description", ""),
            started_at=d.get("started_at"),
            completed_at=d.get("completed_at"),
            duration_s=d.get("duration_s", 0.0),
            tags=d.get("tags", []),
        )


@dataclass
class EnvInfo:
    """Environment information for reproducibility."""
    python_version: str = ""
    platform_system: str = ""
    platform_release: str = ""
    platform_machine: str = ""
    package_versions: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "python_version": self.python_version,
            "platform_system": self.platform_system,
            "platform_release": self.platform_release,
            "platform_machine": self.platform_machine,
            "package_versions": self.package_versions,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EnvInfo":
        return cls(
            python_version=d.get("python_version", ""),
            platform_system=d.get("platform_system", ""),
            platform_release=d.get("platform_release", ""),
            platform_machine=d.get("platform_machine", ""),
            package_versions=d.get("package_versions", {}),
        )
    
    @classmethod
    def capture(cls) -> "EnvInfo":
        """Capture current environment information."""
        package_versions = {}
        
        try:
            import numpy
            package_versions["numpy"] = numpy.__version__
        except ImportError:
            pass
        
        try:
            import trimesh
            package_versions["trimesh"] = trimesh.__version__
        except ImportError:
            pass
        
        try:
            import scipy
            package_versions["scipy"] = scipy.__version__
        except ImportError:
            pass
        
        try:
            import networkx
            package_versions["networkx"] = networkx.__version__
        except ImportError:
            pass
        
        return cls(
            python_version=sys.version,
            platform_system=platform.system(),
            platform_release=platform.release(),
            platform_machine=platform.machine(),
            package_versions=package_versions,
        )


@dataclass
class HashInfo:
    """Hash information for reproducibility."""
    spec_hash: str = ""
    repo_commit: Optional[str] = None
    repo_branch: Optional[str] = None
    repo_dirty: Optional[bool] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec_hash": self.spec_hash,
            "repo_commit": self.repo_commit,
            "repo_branch": self.repo_branch,
            "repo_dirty": self.repo_dirty,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HashInfo":
        return cls(
            spec_hash=d.get("spec_hash", ""),
            repo_commit=d.get("repo_commit"),
            repo_branch=d.get("repo_branch"),
            repo_dirty=d.get("repo_dirty"),
        )
    
    @classmethod
    def capture(cls, spec_hash: str) -> "HashInfo":
        """Capture current hash information."""
        repo_commit = None
        repo_branch = None
        repo_dirty = None
        
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                repo_commit = result.stdout.strip()
        except Exception:
            pass
        
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                repo_branch = result.stdout.strip()
        except Exception:
            pass
        
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                repo_dirty = len(result.stdout.strip()) > 0
        except Exception:
            pass
        
        return cls(
            spec_hash=spec_hash,
            repo_commit=repo_commit,
            repo_branch=repo_branch,
            repo_dirty=repo_dirty,
        )


@dataclass
class StageReportEntry:
    """Entry for a single stage in the run report."""
    stage: str
    success: bool
    duration_s: float = 0.0
    requested_policy: Optional[Dict[str, Any]] = None
    effective_policy: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "success": self.success,
            "duration_s": self.duration_s,
            "requested_policy": self.requested_policy,
            "effective_policy": self.effective_policy,
            "warnings": self.warnings,
            "errors": self.errors,
            "metrics": self.metrics,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StageReportEntry":
        return cls(
            stage=d.get("stage", ""),
            success=d.get("success", False),
            duration_s=d.get("duration_s", 0.0),
            requested_policy=d.get("requested_policy"),
            effective_policy=d.get("effective_policy"),
            warnings=d.get("warnings", []),
            errors=d.get("errors", []),
            metrics=d.get("metrics", {}),
        )


@dataclass
class ArtifactEntry:
    """Entry for a single artifact in the run report."""
    name: str
    path: Optional[str] = None
    content_hash: Optional[str] = None
    stage: str = ""
    mesh_stats: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "content_hash": self.content_hash,
            "stage": self.stage,
            "mesh_stats": self.mesh_stats,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ArtifactEntry":
        return cls(
            name=d.get("name", ""),
            path=d.get("path"),
            content_hash=d.get("content_hash"),
            stage=d.get("stage", ""),
            mesh_stats=d.get("mesh_stats"),
        )


@dataclass
class RunReport:
    """
    Comprehensive run report for DesignSpec execution.
    
    Captures full reproducibility state including:
    - meta: seed, input_units, timestamps
    - env: python version, package versions, platform
    - hashes: spec_hash, repo_commit if available
    - stages: list of stage reports
    - artifacts: name → path + hash + stats
    """
    success: bool
    meta: MetaInfo = field(default_factory=MetaInfo)
    env: EnvInfo = field(default_factory=EnvInfo)
    hashes: HashInfo = field(default_factory=HashInfo)
    stages: List[StageReportEntry] = field(default_factory=list)
    artifacts: Dict[str, ArtifactEntry] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        from .serializers import make_json_safe
        
        return make_json_safe({
            "success": self.success,
            "meta": self.meta.to_dict(),
            "env": self.env.to_dict(),
            "hashes": self.hashes.to_dict(),
            "stages": [s.to_dict() for s in self.stages],
            "artifacts": {k: v.to_dict() for k, v in self.artifacts.items()},
            "warnings": self.warnings,
            "errors": self.errors,
        })
    
    def to_json(self, indent: int = 2) -> str:
        import json
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, path: str) -> None:
        """Save report to JSON file."""
        from pathlib import Path
        
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        
        with open(p, "w") as f:
            f.write(self.to_json())
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunReport":
        return cls(
            success=d.get("success", False),
            meta=MetaInfo.from_dict(d.get("meta", {})),
            env=EnvInfo.from_dict(d.get("env", {})),
            hashes=HashInfo.from_dict(d.get("hashes", {})),
            stages=[StageReportEntry.from_dict(s) for s in d.get("stages", [])],
            artifacts={
                k: ArtifactEntry.from_dict(v)
                for k, v in d.get("artifacts", {}).items()
            },
            warnings=d.get("warnings", []),
            errors=d.get("errors", []),
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> "RunReport":
        import json
        return cls.from_dict(json.loads(json_str))
    
    @classmethod
    def load(cls, path: str) -> "RunReport":
        """Load report from JSON file."""
        with open(path, "r") as f:
            return cls.from_json(f.read())
    
    @classmethod
    def from_runner_result(
        cls,
        result: Any,
        spec: Any,
    ) -> "RunReport":
        """
        Create a RunReport from a RunnerResult and DesignSpec.
        
        Parameters
        ----------
        result : RunnerResult
            Result from DesignSpecRunner.run()
        spec : DesignSpec
            The spec that was executed
            
        Returns
        -------
        RunReport
            Comprehensive run report
        """
        meta_dict = spec.meta
        
        meta = MetaInfo(
            seed=meta_dict.get("seed"),
            input_units=meta_dict.get("input_units", "m"),
            spec_name=meta_dict.get("name", ""),
            spec_description=meta_dict.get("description", ""),
            duration_s=result.total_duration_s,
            tags=meta_dict.get("tags", []),
        )
        
        env = EnvInfo.capture()
        hashes = HashInfo.capture(spec.spec_hash)
        
        stages = []
        for stage_report in result.stage_reports:
            stages.append(StageReportEntry(
                stage=stage_report.stage,
                success=stage_report.success,
                duration_s=stage_report.duration_s,
                requested_policy=stage_report.requested_policy,
                effective_policy=stage_report.effective_policy,
                warnings=stage_report.warnings,
                errors=stage_report.errors,
                metrics=stage_report.metadata,
            ))
        
        artifacts = {}
        for name, artifact_dict in result.artifacts.items():
            artifacts[name] = ArtifactEntry(
                name=name,
                path=artifact_dict.get("path"),
                content_hash=artifact_dict.get("content_hash"),
                stage=artifact_dict.get("stage", ""),
                mesh_stats=artifact_dict.get("metadata"),
            )
        
        return cls(
            success=result.success,
            meta=meta,
            env=env,
            hashes=hashes,
            stages=stages,
            artifacts=artifacts,
            warnings=result.warnings,
            errors=result.errors,
        )


__all__ = [
    "RunReport",
    "MetaInfo",
    "EnvInfo",
    "HashInfo",
    "StageReportEntry",
    "ArtifactEntry",
]
