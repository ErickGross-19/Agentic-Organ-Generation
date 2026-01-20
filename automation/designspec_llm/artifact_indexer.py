"""
Artifact Indexer for DesignSpec LLM Agent

This module provides principled access to previous run artifacts by:
1. Building an index of all runs in the project
2. Extracting key metrics from each run
3. Selecting the most relevant runs for context

The index is stored at project_dir/reports/artifact_index.json and
is updated after each run.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class RunArtifactEntry:
    """Index entry for a single run."""
    run_id: str
    timestamp: str
    timestamp_unix: float = 0.0
    success: bool = False
    
    # Stage outcomes
    stages_completed: List[str] = field(default_factory=list)
    stages_failed: List[str] = field(default_factory=list)
    last_stage: str = ""
    
    # File inventory
    files: Dict[str, int] = field(default_factory=dict)  # filename -> size in bytes
    total_size_bytes: int = 0
    
    # Extracted metrics
    mesh_faces: Optional[int] = None
    mesh_vertices: Optional[int] = None
    mesh_watertight: Optional[bool] = None
    mesh_bbox: Optional[List[float]] = None
    mesh_volume: Optional[float] = None
    
    network_nodes: Optional[int] = None
    network_segments: Optional[int] = None
    network_bbox: Optional[List[float]] = None
    
    validity_passed: Optional[bool] = None
    validity_errors: List[str] = field(default_factory=list)
    
    # Error summary
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "timestamp_unix": self.timestamp_unix,
            "success": self.success,
            "stages_completed": self.stages_completed,
            "stages_failed": self.stages_failed,
            "last_stage": self.last_stage,
            "files": self.files,
            "total_size_bytes": self.total_size_bytes,
            "mesh_metrics": {
                "faces": self.mesh_faces,
                "vertices": self.mesh_vertices,
                "watertight": self.mesh_watertight,
                "bbox": self.mesh_bbox,
                "volume": self.mesh_volume,
            } if self.mesh_faces is not None else None,
            "network_metrics": {
                "nodes": self.network_nodes,
                "segments": self.network_segments,
                "bbox": self.network_bbox,
            } if self.network_nodes is not None else None,
            "validity": {
                "passed": self.validity_passed,
                "errors": self.validity_errors[:5],  # Limit errors
            } if self.validity_passed is not None else None,
            "errors": self.errors[:5],  # Limit errors
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunArtifactEntry":
        entry = cls(
            run_id=d.get("run_id", ""),
            timestamp=d.get("timestamp", ""),
            timestamp_unix=d.get("timestamp_unix", 0.0),
            success=d.get("success", False),
            stages_completed=d.get("stages_completed", []),
            stages_failed=d.get("stages_failed", []),
            last_stage=d.get("last_stage", ""),
            files=d.get("files", {}),
            total_size_bytes=d.get("total_size_bytes", 0),
            errors=d.get("errors", []),
        )
        
        mesh = d.get("mesh_metrics")
        if mesh:
            entry.mesh_faces = mesh.get("faces")
            entry.mesh_vertices = mesh.get("vertices")
            entry.mesh_watertight = mesh.get("watertight")
            entry.mesh_bbox = mesh.get("bbox")
            entry.mesh_volume = mesh.get("volume")
        
        network = d.get("network_metrics")
        if network:
            entry.network_nodes = network.get("nodes")
            entry.network_segments = network.get("segments")
            entry.network_bbox = network.get("bbox")
        
        validity = d.get("validity")
        if validity:
            entry.validity_passed = validity.get("passed")
            entry.validity_errors = validity.get("errors", [])
        
        return entry


@dataclass
class ArtifactIndex:
    """Index of all run artifacts in a project."""
    project_dir: str
    last_updated: str = ""
    last_updated_unix: float = 0.0
    runs: List[RunArtifactEntry] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_dir": self.project_dir,
            "last_updated": self.last_updated,
            "last_updated_unix": self.last_updated_unix,
            "run_count": len(self.runs),
            "runs": [r.to_dict() for r in self.runs],
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ArtifactIndex":
        runs = [RunArtifactEntry.from_dict(r) for r in d.get("runs", [])]
        return cls(
            project_dir=d.get("project_dir", ""),
            last_updated=d.get("last_updated", ""),
            last_updated_unix=d.get("last_updated_unix", 0.0),
            runs=runs,
        )
    
    def get_last_run(self) -> Optional[RunArtifactEntry]:
        """Get the most recent run."""
        if not self.runs:
            return None
        return max(self.runs, key=lambda r: r.timestamp_unix)
    
    def get_last_successful_run(self) -> Optional[RunArtifactEntry]:
        """Get the most recent successful run."""
        successful = [r for r in self.runs if r.success]
        if not successful:
            return None
        return max(successful, key=lambda r: r.timestamp_unix)
    
    def get_recent_runs(self, limit: int = 3) -> List[RunArtifactEntry]:
        """Get the most recent runs."""
        sorted_runs = sorted(self.runs, key=lambda r: r.timestamp_unix, reverse=True)
        return sorted_runs[:limit]
    
    def get_runs_with_stage(self, stage: str) -> List[RunArtifactEntry]:
        """Get runs that completed a specific stage."""
        return [r for r in self.runs if stage in r.stages_completed]


class ArtifactIndexer:
    """
    Builds and maintains an index of run artifacts.
    
    The indexer scans the project's artifacts directory and extracts
    key metrics from each run for efficient context building.
    """
    
    def __init__(self, project_dir: Path):
        """
        Initialize the indexer.
        
        Parameters
        ----------
        project_dir : Path
            The project directory
        """
        self.project_dir = Path(project_dir)
        self.artifacts_dir = self.project_dir / "artifacts"
        self.reports_dir = self.project_dir / "reports"
        self.index_path = self.reports_dir / "artifact_index.json"
        
        self._index: Optional[ArtifactIndex] = None
    
    def load_index(self) -> ArtifactIndex:
        """
        Load the artifact index from disk.
        
        Returns
        -------
        ArtifactIndex
            The loaded or empty index
        """
        if self.index_path.exists():
            try:
                with open(self.index_path) as f:
                    data = json.load(f)
                self._index = ArtifactIndex.from_dict(data)
                return self._index
            except Exception as e:
                logger.warning(f"Failed to load artifact index: {e}")
        
        # Return empty index
        self._index = ArtifactIndex(project_dir=str(self.project_dir))
        return self._index
    
    def save_index(self) -> None:
        """Save the artifact index to disk."""
        if self._index is None:
            return
        
        self.reports_dir.mkdir(exist_ok=True)
        
        try:
            with open(self.index_path, "w") as f:
                json.dump(self._index.to_dict(), f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save artifact index: {e}")
    
    def rebuild_index(self) -> ArtifactIndex:
        """
        Rebuild the entire artifact index from disk.
        
        Returns
        -------
        ArtifactIndex
            The rebuilt index
        """
        self._index = ArtifactIndex(project_dir=str(self.project_dir))
        
        if not self.artifacts_dir.exists():
            return self._index
        
        # Find all run directories
        run_dirs = [
            d for d in self.artifacts_dir.iterdir()
            if d.is_dir() and d.name.startswith("run_")
        ]
        
        for run_dir in run_dirs:
            try:
                entry = self._index_run_directory(run_dir)
                if entry:
                    self._index.runs.append(entry)
            except Exception as e:
                logger.warning(f"Failed to index run directory {run_dir}: {e}")
        
        # Update timestamp
        now = datetime.now()
        self._index.last_updated = now.isoformat()
        self._index.last_updated_unix = now.timestamp()
        
        self.save_index()
        return self._index
    
    def update_index(self) -> ArtifactIndex:
        """
        Update the index with any new runs.
        
        Returns
        -------
        ArtifactIndex
            The updated index
        """
        if self._index is None:
            self.load_index()
        
        if not self.artifacts_dir.exists():
            return self._index
        
        # Get existing run IDs
        existing_ids = {r.run_id for r in self._index.runs}
        
        # Find new run directories
        run_dirs = [
            d for d in self.artifacts_dir.iterdir()
            if d.is_dir() and d.name.startswith("run_") and d.name not in existing_ids
        ]
        
        for run_dir in run_dirs:
            try:
                entry = self._index_run_directory(run_dir)
                if entry:
                    self._index.runs.append(entry)
            except Exception as e:
                logger.warning(f"Failed to index run directory {run_dir}: {e}")
        
        # Update timestamp
        now = datetime.now()
        self._index.last_updated = now.isoformat()
        self._index.last_updated_unix = now.timestamp()
        
        self.save_index()
        return self._index
    
    def _index_run_directory(self, run_dir: Path) -> Optional[RunArtifactEntry]:
        """
        Index a single run directory.
        
        Parameters
        ----------
        run_dir : Path
            The run directory
            
        Returns
        -------
        RunArtifactEntry or None
            The index entry, or None if indexing failed
        """
        run_id = run_dir.name
        
        # Get directory modification time
        try:
            mtime = run_dir.stat().st_mtime
            timestamp = datetime.fromtimestamp(mtime).isoformat()
        except Exception:
            mtime = 0.0
            timestamp = ""
        
        entry = RunArtifactEntry(
            run_id=run_id,
            timestamp=timestamp,
            timestamp_unix=mtime,
        )
        
        # Inventory files
        entry.files, entry.total_size_bytes = self._inventory_files(run_dir)
        
        # Load run report if available
        run_report_path = run_dir / "run_report.json"
        if run_report_path.exists():
            self._extract_run_report_metrics(run_report_path, entry)
        
        # Load validity report if available
        validity_report_path = run_dir / "validity_report.json"
        if validity_report_path.exists():
            self._extract_validity_metrics(validity_report_path, entry)
        
        # Load mesh stats if available
        mesh_stats_path = run_dir / "mesh_stats.json"
        if mesh_stats_path.exists():
            self._extract_mesh_metrics(mesh_stats_path, entry)
        
        # Load network stats if available
        network_stats_path = run_dir / "network_stats.json"
        if network_stats_path.exists():
            self._extract_network_metrics(network_stats_path, entry)
        
        return entry
    
    def _inventory_files(self, run_dir: Path) -> Tuple[Dict[str, int], int]:
        """
        Inventory files in a run directory.
        
        Parameters
        ----------
        run_dir : Path
            The run directory
            
        Returns
        -------
        tuple
            (files dict, total size)
        """
        files = {}
        total_size = 0
        
        try:
            for item in run_dir.iterdir():
                if item.is_file():
                    size = item.stat().st_size
                    files[item.name] = size
                    total_size += size
        except Exception as e:
            logger.warning(f"Failed to inventory files in {run_dir}: {e}")
        
        return files, total_size
    
    def _extract_run_report_metrics(
        self,
        report_path: Path,
        entry: RunArtifactEntry,
    ) -> None:
        """Extract metrics from a run report."""
        try:
            with open(report_path) as f:
                report = json.load(f)
            
            entry.success = report.get("success", False)
            entry.errors = report.get("errors", [])
            
            # Extract stage information
            stage_reports = report.get("stage_reports", [])
            for stage_report in stage_reports:
                if isinstance(stage_report, dict):
                    stage_name = stage_report.get("stage", "")
                    if stage_report.get("success", True):
                        entry.stages_completed.append(stage_name)
                        entry.last_stage = stage_name
                    else:
                        entry.stages_failed.append(stage_name)
            
            # Extract mesh stats if embedded
            mesh_stats = report.get("mesh_stats", {})
            if mesh_stats:
                entry.mesh_faces = mesh_stats.get("faces")
                entry.mesh_vertices = mesh_stats.get("vertices")
                entry.mesh_watertight = mesh_stats.get("watertight")
                entry.mesh_bbox = mesh_stats.get("bbox")
                entry.mesh_volume = mesh_stats.get("volume")
            
            # Extract network stats if embedded
            network_stats = report.get("network_stats", {})
            if network_stats:
                entry.network_nodes = network_stats.get("nodes")
                entry.network_segments = network_stats.get("segments")
                entry.network_bbox = network_stats.get("bbox")
            
        except Exception as e:
            logger.warning(f"Failed to extract run report metrics: {e}")
    
    def _extract_validity_metrics(
        self,
        report_path: Path,
        entry: RunArtifactEntry,
    ) -> None:
        """Extract metrics from a validity report."""
        try:
            with open(report_path) as f:
                report = json.load(f)
            
            entry.validity_passed = report.get("valid", report.get("passed", False))
            entry.validity_errors = report.get("errors", [])
            
        except Exception as e:
            logger.warning(f"Failed to extract validity metrics: {e}")
    
    def _extract_mesh_metrics(
        self,
        stats_path: Path,
        entry: RunArtifactEntry,
    ) -> None:
        """Extract metrics from a mesh stats file."""
        try:
            with open(stats_path) as f:
                stats = json.load(f)
            
            entry.mesh_faces = stats.get("faces", stats.get("num_faces"))
            entry.mesh_vertices = stats.get("vertices", stats.get("num_vertices"))
            entry.mesh_watertight = stats.get("watertight", stats.get("is_watertight"))
            entry.mesh_bbox = stats.get("bbox", stats.get("bounding_box"))
            entry.mesh_volume = stats.get("volume")
            
        except Exception as e:
            logger.warning(f"Failed to extract mesh metrics: {e}")
    
    def _extract_network_metrics(
        self,
        stats_path: Path,
        entry: RunArtifactEntry,
    ) -> None:
        """Extract metrics from a network stats file."""
        try:
            with open(stats_path) as f:
                stats = json.load(f)
            
            entry.network_nodes = stats.get("nodes", stats.get("num_nodes"))
            entry.network_segments = stats.get("segments", stats.get("num_segments"))
            entry.network_bbox = stats.get("bbox", stats.get("bounding_box"))
            
        except Exception as e:
            logger.warning(f"Failed to extract network metrics: {e}")
    
    def get_index(self) -> ArtifactIndex:
        """
        Get the current index, loading if necessary.
        
        Returns
        -------
        ArtifactIndex
            The artifact index
        """
        if self._index is None:
            self.load_index()
        return self._index
    
    def select_relevant_runs(
        self,
        limit: int = 3,
        include_last_successful: bool = True,
    ) -> List[RunArtifactEntry]:
        """
        Select the most relevant runs for context.
        
        Parameters
        ----------
        limit : int
            Maximum number of recent runs to include
        include_last_successful : bool
            Whether to always include the last successful run
            
        Returns
        -------
        list of RunArtifactEntry
            Selected runs
        """
        index = self.get_index()
        
        # Get recent runs
        recent = index.get_recent_runs(limit)
        
        if not include_last_successful:
            return recent
        
        # Check if last successful is already included
        last_successful = index.get_last_successful_run()
        if last_successful is None:
            return recent
        
        recent_ids = {r.run_id for r in recent}
        if last_successful.run_id in recent_ids:
            return recent
        
        # Add last successful, keeping within limit
        return recent[:limit - 1] + [last_successful]
