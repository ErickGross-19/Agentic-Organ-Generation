"""
Context Builder for DesignSpec LLM Agent

This module provides context packaging for the LLM agent, including:
- Current spec summary and full spec
- Recent run artifacts and summaries
- Validation and compile reports
- Patch history

The context builder produces two versions:
1. context_compact: Default, token-efficient summary
2. context_full: Complete context with full spec and artifacts
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..designspec_session import DesignSpecSession

logger = logging.getLogger(__name__)


@dataclass
class SpecSummary:
    """Summary of the current DesignSpec."""
    has_domains: bool = False
    domain_count: int = 0
    domain_names: List[str] = field(default_factory=list)
    domain_types: Dict[str, str] = field(default_factory=dict)
    
    has_components: bool = False
    component_count: int = 0
    component_ids: List[str] = field(default_factory=list)
    components_with_inlets: int = 0
    components_with_outlets: int = 0
    
    has_policies: bool = False
    policy_names: List[str] = field(default_factory=list)
    
    has_features: bool = False
    feature_types: List[str] = field(default_factory=list)
    
    meta_seed: Optional[int] = None
    meta_input_units: str = "mm"
    meta_name: str = ""
    
    schema_version: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "domains": {
                "present": self.has_domains,
                "count": self.domain_count,
                "names": self.domain_names,
                "types": self.domain_types,
            },
            "components": {
                "present": self.has_components,
                "count": self.component_count,
                "ids": self.component_ids,
                "with_inlets": self.components_with_inlets,
                "with_outlets": self.components_with_outlets,
            },
            "policies": {
                "present": self.has_policies,
                "names": self.policy_names,
            },
            "features": {
                "present": self.has_features,
                "types": self.feature_types,
            },
            "meta": {
                "seed": self.meta_seed,
                "input_units": self.meta_input_units,
                "name": self.meta_name,
            },
            "schema_version": self.schema_version,
        }


@dataclass
class RunSummary:
    """Summary of a pipeline run."""
    run_id: str = ""
    timestamp: str = ""
    success: bool = False
    stages_completed: List[str] = field(default_factory=list)
    stages_failed: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    output_dir: str = ""
    
    # Mesh stats if available
    mesh_faces: Optional[int] = None
    mesh_vertices: Optional[int] = None
    mesh_watertight: Optional[bool] = None
    mesh_bbox: Optional[List[float]] = None
    
    # Network stats if available
    network_nodes: Optional[int] = None
    network_segments: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "success": self.success,
            "stages_completed": self.stages_completed,
            "stages_failed": self.stages_failed,
            "errors": self.errors[:5] if self.errors else [],  # Limit errors
            "warnings": self.warnings[:3] if self.warnings else [],  # Limit warnings
            "output_dir": self.output_dir,
        }
        
        if self.mesh_faces is not None:
            result["mesh_stats"] = {
                "faces": self.mesh_faces,
                "vertices": self.mesh_vertices,
                "watertight": self.mesh_watertight,
                "bbox": self.mesh_bbox,
            }
        
        if self.network_nodes is not None:
            result["network_stats"] = {
                "nodes": self.network_nodes,
                "segments": self.network_segments,
            }
        
        return result


@dataclass
class ValidationSummary:
    """Summary of validation results."""
    valid: bool = False
    error_count: int = 0
    warning_count: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    spec_hash: str = ""
    
    # Enhanced validity info for compact context
    failed_checks: int = 0
    failure_names: List[str] = field(default_factory=list)
    failure_reasons: List[str] = field(default_factory=list)
    key_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "errors": self.errors[:5],
            "warnings": self.warnings[:3],
            "spec_hash": self.spec_hash,
            "failed_checks": self.failed_checks,
            "failure_names": self.failure_names[:5],
            "failure_reasons": self.failure_reasons[:5],
            "key_metrics": self.key_metrics,
        }


@dataclass
class PatchHistoryEntry:
    """A single patch history entry."""
    patch_id: str = ""
    timestamp: str = ""
    author: str = ""
    operation_count: int = 0
    paths_modified: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "patch_id": self.patch_id,
            "timestamp": self.timestamp,
            "author": self.author,
            "operation_count": self.operation_count,
            "paths_modified": self.paths_modified[:5],  # Limit paths
        }


@dataclass
class MeshStats:
    """Summary of mesh statistics for compact context."""
    component_void_mesh: Optional[Dict[str, Any]] = None
    union_void_mesh: Optional[Dict[str, Any]] = None
    domain_with_void_mesh: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_void": self.component_void_mesh,
            "union_void": self.union_void_mesh,
            "domain_with_void": self.domain_with_void_mesh,
        }


@dataclass
class NetworkStats:
    """Summary of network statistics for compact context."""
    node_count: int = 0
    edge_count: int = 0
    bbox: Optional[Dict[str, float]] = None
    radius_stats: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "bbox": self.bbox,
            "radius_stats": self.radius_stats,
        }


@dataclass
class ContextPack:
    """
    Complete context pack for the LLM agent.
    
    Contains all information the agent needs to make decisions.
    """
    # Spec information
    spec_summary: Optional[SpecSummary] = None
    full_spec: Optional[Dict[str, Any]] = None
    
    # Validation state
    validation_summary: Optional[ValidationSummary] = None
    
    # Run history
    last_run: Optional[RunSummary] = None
    recent_runs: List[RunSummary] = field(default_factory=list)
    last_successful_run: Optional[RunSummary] = None
    
    # Patch history
    recent_patches: List[PatchHistoryEntry] = field(default_factory=list)
    
    # Compile state
    compile_success: bool = False
    compile_errors: List[str] = field(default_factory=list)
    
    # Additional artifacts
    artifact_index: Optional[Dict[str, Any]] = None
    
    # Mesh and network stats for compact context
    mesh_stats: Optional[MeshStats] = None
    network_stats: Optional[NetworkStats] = None
    
    # Context mode
    is_compact: bool = True
    is_debug_compact: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "spec_summary": self.spec_summary.to_dict() if self.spec_summary else None,
            "validation": self.validation_summary.to_dict() if self.validation_summary else None,
            "last_run": self.last_run.to_dict() if self.last_run else None,
            "compile_success": self.compile_success,
            "is_compact": self.is_compact,
            "is_debug_compact": self.is_debug_compact,
        }
        
        # Include mesh and network stats in compact context
        if self.mesh_stats:
            result["mesh_stats"] = self.mesh_stats.to_dict()
        if self.network_stats:
            result["network_stats"] = self.network_stats.to_dict()
        
        if not self.is_compact:
            result["full_spec"] = self.full_spec
            result["recent_runs"] = [r.to_dict() for r in self.recent_runs]
            result["last_successful_run"] = (
                self.last_successful_run.to_dict() if self.last_successful_run else None
            )
            result["recent_patches"] = [p.to_dict() for p in self.recent_patches]
            result["compile_errors"] = self.compile_errors
            result["artifact_index"] = self.artifact_index
        
        return result
    
    def to_prompt_text(self) -> str:
        """Convert context pack to text suitable for LLM prompt."""
        lines = []
        
        # Spec summary
        if self.spec_summary:
            lines.append("## Current Spec Summary")
            ss = self.spec_summary
            lines.append(f"- Name: {ss.meta_name or 'Unnamed'}")
            lines.append(f"- Units: {ss.meta_input_units}")
            lines.append(f"- Seed: {ss.meta_seed}")
            
            if ss.has_domains:
                domain_info = ", ".join(
                    f"{name} ({ss.domain_types.get(name, 'unknown')})"
                    for name in ss.domain_names
                )
                lines.append(f"- Domains ({ss.domain_count}): {domain_info}")
            else:
                lines.append("- Domains: NONE DEFINED")
            
            if ss.has_components:
                lines.append(f"- Components ({ss.component_count}): {', '.join(ss.component_ids)}")
                lines.append(f"  - With inlets: {ss.components_with_inlets}")
                lines.append(f"  - With outlets: {ss.components_with_outlets}")
            else:
                lines.append("- Components: NONE DEFINED")
            
            if ss.has_policies:
                lines.append(f"- Policies: {', '.join(ss.policy_names)}")
            
            if ss.has_features:
                lines.append(f"- Features: {', '.join(ss.feature_types)}")
            
            lines.append("")
        
        # Validation state
        if self.validation_summary:
            vs = self.validation_summary
            status = "VALID" if vs.valid else "INVALID"
            lines.append(f"## Validation Status: {status}")
            if vs.errors:
                lines.append("Errors:")
                for err in vs.errors[:5]:
                    lines.append(f"  - {err}")
            if vs.warnings:
                lines.append("Warnings:")
                for warn in vs.warnings[:3]:
                    lines.append(f"  - {warn}")
            lines.append("")
        
        # Last run
        if self.last_run:
            lr = self.last_run
            status = "SUCCESS" if lr.success else "FAILED"
            lines.append(f"## Last Run: {status}")
            lines.append(f"- Run ID: {lr.run_id}")
            if lr.stages_completed:
                lines.append(f"- Completed stages: {', '.join(lr.stages_completed)}")
            if lr.stages_failed:
                lines.append(f"- Failed stages: {', '.join(lr.stages_failed)}")
            if lr.errors:
                lines.append("- Errors:")
                for err in lr.errors[:3]:
                    lines.append(f"    {err}")
            if lr.mesh_faces is not None:
                lines.append(f"- Mesh: {lr.mesh_faces} faces, {lr.mesh_vertices} vertices")
                if lr.mesh_watertight is not None:
                    wt = "yes" if lr.mesh_watertight else "no"
                    lines.append(f"- Watertight: {wt}")
            if lr.network_nodes is not None:
                lines.append(f"- Network: {lr.network_nodes} nodes, {lr.network_segments} segments")
            lines.append("")
        
        # Compile state
        lines.append(f"## Compile Status: {'OK' if self.compile_success else 'FAILED'}")
        if self.compile_errors:
            for err in self.compile_errors[:3]:
                lines.append(f"  - {err}")
        lines.append("")
        
        # Full spec (if not compact)
        if not self.is_compact and self.full_spec:
            lines.append("## Full Spec JSON")
            lines.append("```json")
            spec_json = json.dumps(self.full_spec, indent=2)
            # Truncate if too long
            if len(spec_json) > 4000:
                spec_json = spec_json[:4000] + "\n... (truncated)"
            lines.append(spec_json)
            lines.append("```")
            lines.append("")
        
        return "\n".join(lines)


class ContextBuilder:
    """
    Builds context packs for the LLM agent.
    
    Pulls information from:
    - DesignSpecSession (spec, validation, compile reports)
    - Project directory (artifacts, run history, patch history)
    """
    
    def __init__(self, session: "DesignSpecSession"):
        """
        Initialize the context builder.
        
        Parameters
        ----------
        session : DesignSpecSession
            The active design spec session
        """
        self.session = session
        self.project_dir = session.project_dir
    
    def build_spec_summary(self, spec: Dict[str, Any]) -> SpecSummary:
        """
        Build a summary of the spec.
        
        Parameters
        ----------
        spec : dict
            The spec dictionary
            
        Returns
        -------
        SpecSummary
            Summary of the spec
        """
        summary = SpecSummary()
        
        # Schema
        schema = spec.get("schema", {})
        summary.schema_version = schema.get("version", "")
        
        # Meta
        meta = spec.get("meta", {})
        summary.meta_seed = meta.get("seed")
        summary.meta_input_units = meta.get("input_units", "mm")
        summary.meta_name = meta.get("name", "")
        
        # Domains
        domains = spec.get("domains", {})
        if domains:
            summary.has_domains = True
            summary.domain_count = len(domains)
            summary.domain_names = list(domains.keys())
            summary.domain_types = {
                name: d.get("type", "unknown")
                for name, d in domains.items()
            }
        
        # Components
        components = spec.get("components", [])
        if components:
            summary.has_components = True
            summary.component_count = len(components)
            for comp in components:
                comp_id = comp.get("id", "unnamed")
                summary.component_ids.append(comp_id)
                
                ports = comp.get("ports", {})
                if ports.get("inlets"):
                    summary.components_with_inlets += 1
                if ports.get("outlets"):
                    summary.components_with_outlets += 1
        
        # Policies
        policies = spec.get("policies", {})
        if policies:
            summary.has_policies = True
            summary.policy_names = list(policies.keys())
        
        # Features
        features = spec.get("features", {})
        if features:
            summary.has_features = True
            for feature_type in features.keys():
                summary.feature_types.append(feature_type)
        
        return summary
    
    def build_validation_summary(self) -> Optional[ValidationSummary]:
        """
        Build a summary of the current validation state.
        
        Returns
        -------
        ValidationSummary or None
            Validation summary if available
        """
        try:
            validation = self.session.validate_spec()
            return ValidationSummary(
                valid=validation.valid,
                error_count=len(validation.errors),
                warning_count=len(validation.warnings),
                errors=validation.errors,
                warnings=validation.warnings,
                spec_hash=validation.spec_hash or "",
            )
        except Exception as e:
            logger.warning(f"Failed to get validation: {e}")
            return None
    
    def build_run_summary(self, run_result: Dict[str, Any]) -> RunSummary:
        """
        Build a summary from a run result.
        
        Parameters
        ----------
        run_result : dict
            Run result dictionary
            
        Returns
        -------
        RunSummary
            Summary of the run
        """
        summary = RunSummary(
            run_id=run_result.get("run_id", ""),
            timestamp=run_result.get("timestamp", ""),
            success=run_result.get("success", False),
            errors=run_result.get("errors", []),
            warnings=run_result.get("warnings", []),
            output_dir=run_result.get("output_dir", ""),
        )
        
        # Extract stage information
        stage_reports = run_result.get("stage_reports", [])
        for report in stage_reports:
            if isinstance(report, dict):
                stage_name = report.get("stage", "")
                if report.get("success", True):
                    summary.stages_completed.append(stage_name)
                else:
                    summary.stages_failed.append(stage_name)
        
        # Extract mesh stats if available
        mesh_stats = run_result.get("mesh_stats", {})
        if mesh_stats:
            summary.mesh_faces = mesh_stats.get("faces")
            summary.mesh_vertices = mesh_stats.get("vertices")
            summary.mesh_watertight = mesh_stats.get("watertight")
            summary.mesh_bbox = mesh_stats.get("bbox")
        
        # Extract network stats if available
        network_stats = run_result.get("network_stats", {})
        if network_stats:
            summary.network_nodes = network_stats.get("nodes")
            summary.network_segments = network_stats.get("segments")
        
        return summary
    
    def get_last_run_summary(self) -> Optional[RunSummary]:
        """
        Get summary of the last pipeline run.
        
        Returns
        -------
        RunSummary or None
            Summary if a run exists
        """
        run_result = self.session.get_last_runner_result()
        if run_result:
            return self.build_run_summary(run_result)
        return None
    
    def get_recent_runs(self, limit: int = 3) -> List[RunSummary]:
        """
        Get summaries of recent runs.
        
        Parameters
        ----------
        limit : int
            Maximum number of runs to return
            
        Returns
        -------
        list of RunSummary
            Recent run summaries
        """
        runs = []
        artifacts_dir = self.project_dir / "artifacts"
        
        if not artifacts_dir.exists():
            return runs
        
        # Find run directories
        run_dirs = sorted(
            [d for d in artifacts_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        
        for run_dir in run_dirs[:limit]:
            run_report_path = run_dir / "run_report.json"
            if run_report_path.exists():
                try:
                    with open(run_report_path) as f:
                        run_result = json.load(f)
                    runs.append(self.build_run_summary(run_result))
                except Exception as e:
                    logger.warning(f"Failed to load run report {run_report_path}: {e}")
        
        return runs
    
    def get_last_successful_run(self) -> Optional[RunSummary]:
        """
        Get the last successful run.
        
        Returns
        -------
        RunSummary or None
            Last successful run if any
        """
        recent = self.get_recent_runs(limit=10)
        for run in recent:
            if run.success:
                return run
        return None
    
    def get_recent_patches(self, limit: int = 5) -> List[PatchHistoryEntry]:
        """
        Get recent patch history entries.
        
        Parameters
        ----------
        limit : int
            Maximum number of patches to return
            
        Returns
        -------
        list of PatchHistoryEntry
            Recent patch entries
        """
        patches = []
        patches_dir = self.project_dir / "patches"
        
        if not patches_dir.exists():
            return patches
        
        # Find patch files
        patch_files = sorted(
            [f for f in patches_dir.iterdir() if f.suffix == ".json"],
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        
        for patch_file in patch_files[:limit]:
            try:
                with open(patch_file) as f:
                    patch_data = json.load(f)
                
                # Extract paths modified
                paths_modified = []
                for op in patch_data.get("patches", []):
                    if "path" in op:
                        paths_modified.append(op["path"])
                
                patches.append(PatchHistoryEntry(
                    patch_id=patch_data.get("patch_id", patch_file.stem),
                    timestamp=patch_data.get("timestamp", ""),
                    author=patch_data.get("author", ""),
                    operation_count=len(patch_data.get("patches", [])),
                    paths_modified=paths_modified,
                ))
            except Exception as e:
                logger.warning(f"Failed to load patch {patch_file}: {e}")
        
        return patches
    
    def get_compile_state(self) -> tuple:
        """
        Get the current compile state.
        
        Returns
        -------
        tuple
            (success: bool, errors: list)
        """
        compile_report = self.session.get_last_compile_report()
        if compile_report:
            return (
                compile_report.get("success", False),
                compile_report.get("errors", []),
            )
        return (False, ["No compile report available"])
    
    def build_compact(self) -> ContextPack:
        """
        Build a compact context pack (token-efficient).
        
        Includes auto-escalation to "debug compact" mode if:
        - Last run failed
        - Validation has failed checks
        
        Returns
        -------
        ContextPack
            Compact context pack
        """
        spec = self.session.get_spec()
        compile_success, compile_errors = self.get_compile_state()
        validation_summary = self.build_validation_summary()
        last_run = self.get_last_run_summary()
        
        # Check if we need to auto-escalate to debug compact mode
        needs_debug = False
        if last_run and not last_run.success:
            needs_debug = True
        if validation_summary and not validation_summary.valid:
            needs_debug = True
        
        # Build mesh and network stats for compact context
        mesh_stats = self._build_mesh_stats()
        network_stats = self._build_network_stats()
        
        return ContextPack(
            spec_summary=self.build_spec_summary(spec),
            validation_summary=validation_summary,
            last_run=last_run,
            compile_success=compile_success,
            compile_errors=compile_errors,
            mesh_stats=mesh_stats,
            network_stats=network_stats,
            is_compact=True,
            is_debug_compact=needs_debug,
        )
    
    def _build_mesh_stats(self) -> Optional[MeshStats]:
        """
        Build mesh statistics from the last run.
        
        Returns
        -------
        MeshStats or None
            Mesh statistics if available
        """
        last_run = self.session.get_last_runner_result()
        if not last_run or not last_run.get("output_dir"):
            return None
        
        output_dir = Path(last_run["output_dir"])
        mesh_stats = MeshStats()
        
        # Try to get component void mesh stats
        component_void_file = output_dir / "component_void.stl"
        if component_void_file.exists():
            mesh_stats.component_void_mesh = self._get_stl_stats(component_void_file)
        
        # Try to get union void mesh stats
        union_void_file = output_dir / "union_void.stl"
        if union_void_file.exists():
            mesh_stats.union_void_mesh = self._get_stl_stats(union_void_file)
        
        # Try to get domain_with_void mesh stats
        domain_with_void_file = output_dir / "domain_with_void.stl"
        if domain_with_void_file.exists():
            mesh_stats.domain_with_void_mesh = self._get_stl_stats(domain_with_void_file)
        
        return mesh_stats
    
    def _get_stl_stats(self, stl_path: Path) -> Dict[str, Any]:
        """
        Get basic statistics from an STL file.
        
        Parameters
        ----------
        stl_path : Path
            Path to STL file
            
        Returns
        -------
        dict
            Statistics including face count and file size
        """
        try:
            stats = {
                "path": str(stl_path),
                "size_bytes": stl_path.stat().st_size,
            }
            
            # Try to load with trimesh for more detailed stats
            try:
                import trimesh
                mesh = trimesh.load(stl_path)
                stats["faces"] = len(mesh.faces)
                stats["vertices"] = len(mesh.vertices)
                stats["watertight"] = mesh.is_watertight
                stats["volume"] = float(mesh.volume) if mesh.is_watertight else None
            except ImportError:
                pass
            except Exception as e:
                logger.debug(f"Could not load mesh for stats: {e}")
            
            return stats
        except Exception as e:
            logger.warning(f"Failed to get STL stats for {stl_path}: {e}")
            return {"path": str(stl_path), "error": str(e)}
    
    def _build_network_stats(self) -> Optional[NetworkStats]:
        """
        Build network statistics from the last run.
        
        Returns
        -------
        NetworkStats or None
            Network statistics if available
        """
        last_run = self.session.get_last_runner_result()
        if not last_run or not last_run.get("output_dir"):
            return None
        
        output_dir = Path(last_run["output_dir"])
        network_file = output_dir / "network.json"
        
        if not network_file.exists():
            return None
        
        try:
            with open(network_file) as f:
                network_data = json.load(f)
            
            nodes = network_data.get("nodes", [])
            segments = network_data.get("segments", [])
            
            # Calculate bbox from nodes
            bbox = None
            if nodes:
                positions = [n.get("position", [0, 0, 0]) for n in nodes]
                if positions:
                    xs = [p[0] for p in positions]
                    ys = [p[1] for p in positions]
                    zs = [p[2] for p in positions]
                    bbox = {
                        "min_x": min(xs),
                        "max_x": max(xs),
                        "min_y": min(ys),
                        "max_y": max(ys),
                        "min_z": min(zs),
                        "max_z": max(zs),
                    }
            
            # Calculate radius stats from segments
            radius_stats = None
            radii = [s.get("radius", 0) for s in segments if s.get("radius")]
            if radii:
                radius_stats = {
                    "min": min(radii),
                    "max": max(radii),
                    "mean": sum(radii) / len(radii),
                }
            
            return NetworkStats(
                node_count=len(nodes),
                edge_count=len(segments),
                bbox=bbox,
                radius_stats=radius_stats,
            )
        except Exception as e:
            logger.warning(f"Failed to build network stats: {e}")
            return None
    
    def build_full(self) -> ContextPack:
        """
        Build a full context pack (complete information).
        
        Returns
        -------
        ContextPack
            Full context pack
        """
        spec = self.session.get_spec()
        
        compile_success, compile_errors = self.get_compile_state()
        
        return ContextPack(
            spec_summary=self.build_spec_summary(spec),
            full_spec=spec,
            validation_summary=self.build_validation_summary(),
            last_run=self.get_last_run_summary(),
            recent_runs=self.get_recent_runs(limit=3),
            last_successful_run=self.get_last_successful_run(),
            recent_patches=self.get_recent_patches(limit=5),
            compile_success=compile_success,
            compile_errors=compile_errors,
            is_compact=False,
        )
    
    def build(self, full: bool = False) -> ContextPack:
        """
        Build a context pack.
        
        Parameters
        ----------
        full : bool
            Whether to build full context (default: compact)
            
        Returns
        -------
        ContextPack
            The context pack
        """
        if full:
            return self.build_full()
        return self.build_compact()
    
    def add_validation_details(
        self,
        context_pack: ContextPack,
        validation_report: Any,
    ) -> ContextPack:
        """
        Add detailed validation information to a context pack.
        
        Parameters
        ----------
        context_pack : ContextPack
            The base context pack
        validation_report : ValidationReport
            The validation report to add
            
        Returns
        -------
        ContextPack
            Updated context pack with validation details
        """
        if validation_report is None:
            return context_pack
        
        # Build detailed validation summary
        validation_summary = self.build_validation_summary()
        
        # Add more detailed failure information if available
        if hasattr(validation_report, 'failures') and validation_report.failures:
            validation_summary.failed_checks = len(validation_report.failures)
            validation_summary.failure_names = [
                f.name if hasattr(f, 'name') else str(f)
                for f in validation_report.failures[:10]
            ]
        
        context_pack.validation_summary = validation_summary
        return context_pack
    
    def add_run_details(self, context_pack: ContextPack) -> ContextPack:
        """
        Add detailed run information to a context pack.
        
        Parameters
        ----------
        context_pack : ContextPack
            The base context pack
            
        Returns
        -------
        ContextPack
            Updated context pack with run details
        """
        # Add recent runs
        context_pack.recent_runs = self.get_recent_runs(limit=5)
        context_pack.last_successful_run = self.get_last_successful_run()
        
        return context_pack
    
    def add_network_artifact(self, context_pack: ContextPack) -> ContextPack:
        """
        Add network artifact information to a context pack.
        
        Parameters
        ----------
        context_pack : ContextPack
            The base context pack
            
        Returns
        -------
        ContextPack
            Updated context pack with network artifact
        """
        # Try to load network artifact from last run
        last_run = self.session.get_last_runner_result()
        if last_run and last_run.get("output_dir"):
            output_dir = Path(last_run["output_dir"])
            network_file = output_dir / "network.json"
            
            if network_file.exists():
                try:
                    with open(network_file) as f:
                        network_data = json.load(f)
                    
                    # Add network summary to artifact index
                    if context_pack.artifact_index is None:
                        context_pack.artifact_index = {}
                    
                    context_pack.artifact_index["network"] = {
                        "path": str(network_file),
                        "node_count": len(network_data.get("nodes", [])),
                        "segment_count": len(network_data.get("segments", [])),
                    }
                except Exception as e:
                    logger.warning(f"Failed to load network artifact: {e}")
        
        return context_pack
    
    def add_specific_files(
        self,
        context_pack: ContextPack,
        file_paths: List[str],
    ) -> ContextPack:
        """
        Add specific file contents to a context pack.
        
        Parameters
        ----------
        context_pack : ContextPack
            The base context pack
        file_paths : list of str
            Paths to files to include
            
        Returns
        -------
        ContextPack
            Updated context pack with file contents
        """
        if context_pack.artifact_index is None:
            context_pack.artifact_index = {}
        
        for file_path in file_paths:
            # Try to resolve relative to project dir
            full_path = self.session.project_dir / file_path
            if not full_path.exists():
                # Try as absolute path
                full_path = Path(file_path)
            
            if full_path.exists():
                try:
                    # Only include text files
                    if full_path.suffix in [".json", ".txt", ".log", ".yaml", ".yml"]:
                        with open(full_path) as f:
                            content = f.read()
                        
                        # Truncate large files
                        if len(content) > 10000:
                            content = content[:10000] + "\n... (truncated)"
                        
                        context_pack.artifact_index[str(file_path)] = {
                            "path": str(full_path),
                            "content": content,
                        }
                    else:
                        context_pack.artifact_index[str(file_path)] = {
                            "path": str(full_path),
                            "exists": True,
                            "size": full_path.stat().st_size,
                        }
                except Exception as e:
                    logger.warning(f"Failed to load file {file_path}: {e}")
            else:
                logger.warning(f"Requested file not found: {file_path}")
        
        return context_pack
    
    def add_extended_history(
        self,
        context_pack: ContextPack,
        conversation_history: List[Dict[str, str]],
    ) -> ContextPack:
        """
        Add extended conversation history to a context pack.
        
        Parameters
        ----------
        context_pack : ContextPack
            The base context pack
        conversation_history : list of dict
            Full conversation history
            
        Returns
        -------
        ContextPack
            Updated context pack with extended history
        """
        # Add more patch history
        context_pack.recent_patches = self.get_recent_patches(limit=10)
        
        # Store conversation history in artifact index for reference
        if context_pack.artifact_index is None:
            context_pack.artifact_index = {}
        
        context_pack.artifact_index["conversation_history"] = {
            "turn_count": len(conversation_history),
            "recent_turns": conversation_history[-10:] if len(conversation_history) > 10 else conversation_history,
        }
        
        return context_pack
