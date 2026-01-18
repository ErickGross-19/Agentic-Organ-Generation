"""
DesignSpec Session Engine for project-based spec management.

This module provides the DesignSpecSession class that manages:
- Project directory structure and persistence
- Spec loading, validation, and normalization
- Patch application with history tracking
- Auto-compile after patch acceptance
- Runner execution with artifact management

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally, consistent with designspec.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from datetime import datetime
import json
import copy
import hashlib
import logging
import shutil

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Report from spec validation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    schema_version: Optional[str] = None
    spec_hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "schema_version": self.schema_version,
            "spec_hash": self.spec_hash,
        }


@dataclass
class CompileReport:
    """Report from compile_policies + compile_domains stages."""
    success: bool
    policies_compiled: List[str] = field(default_factory=list)
    domains_compiled: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    duration_s: float = 0.0
    
    @property
    def stages(self) -> Dict[str, Any]:
        """Get stages summary for compatibility."""
        return {
            "compile_policies": {
                "success": len(self.policies_compiled) > 0 or self.success,
                "compiled": self.policies_compiled,
            },
            "compile_domains": {
                "success": len(self.domains_compiled) > 0 or self.success,
                "compiled": self.domains_compiled,
            },
        }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "policies_compiled": self.policies_compiled,
            "domains_compiled": self.domains_compiled,
            "warnings": self.warnings,
            "errors": self.errors,
            "duration_s": self.duration_s,
            "stages": self.stages,
        }


@dataclass
class PatchProposal:
    """A proposed patch to the spec."""
    explanation: str
    patches: List[Dict[str, Any]]
    confidence: float = 0.8
    requires_confirmation: bool = True
    patch_id: Optional[str] = None
    
    def __post_init__(self):
        if self.patch_id is None:
            self.patch_id = hashlib.sha256(
                json.dumps(self.patches, sort_keys=True).encode()
            ).hexdigest()[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "explanation": self.explanation,
            "patches": self.patches,
            "confidence": self.confidence,
            "requires_confirmation": self.requires_confirmation,
            "patch_id": self.patch_id,
        }


@dataclass
class OperationReport:
    """Generic operation report following the OperationReport pattern."""
    success: bool
    operation: str
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "operation": self.operation,
            "warnings": self.warnings,
            "errors": self.errors,
            "metadata": self.metadata,
        }


def _apply_json_patch(doc: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply a single JSON Patch operation (RFC 6902) to a document.
    
    Supported operations:
    - add: Add a value at the specified path
    - remove: Remove the value at the specified path
    - replace: Replace the value at the specified path
    - copy: Copy a value from one path to another
    - move: Move a value from one path to another
    - test: Test that a value at the specified path equals the given value
    
    Parameters
    ----------
    doc : dict
        The document to patch (will be modified in place)
    patch : dict
        A single patch operation with 'op', 'path', and optionally 'value'/'from'
        
    Returns
    -------
    dict
        The patched document
        
    Raises
    ------
    ValueError
        If the patch operation is invalid or cannot be applied
    """
    op = patch.get("op")
    path = patch.get("path", "")
    
    if not op:
        raise ValueError("Patch operation missing 'op' field")
    
    parts = [p for p in path.split("/") if p]
    
    def get_parent_and_key(d: Any, parts: List[str]):
        """Navigate to parent and return (parent, key)."""
        if not parts:
            return None, None
        
        current = d
        for part in parts[:-1]:
            if isinstance(current, dict):
                if part not in current:
                    if op == "add":
                        current[part] = {}
                    else:
                        raise ValueError(f"Path not found: {path}")
                current = current[part]
            elif isinstance(current, list):
                try:
                    idx = int(part)
                    current = current[idx]
                except (ValueError, IndexError):
                    raise ValueError(f"Invalid array index in path: {path}")
            else:
                raise ValueError(f"Cannot navigate path: {path}")
        
        return current, parts[-1]
    
    def get_value(d: Any, parts: List[str]) -> Any:
        """Get value at path."""
        current = d
        for part in parts:
            if isinstance(current, dict):
                if part not in current:
                    raise ValueError(f"Path not found: {'/'.join(parts)}")
                current = current[part]
            elif isinstance(current, list):
                try:
                    idx = int(part)
                    current = current[idx]
                except (ValueError, IndexError):
                    raise ValueError(f"Invalid array index: {part}")
            else:
                raise ValueError(f"Cannot navigate to: {'/'.join(parts)}")
        return current
    
    if op == "add":
        value = patch.get("value")
        if not parts:
            return value
        parent, key = get_parent_and_key(doc, parts)
        if isinstance(parent, dict):
            parent[key] = value
        elif isinstance(parent, list):
            if key == "-":
                parent.append(value)
            else:
                try:
                    idx = int(key)
                    parent.insert(idx, value)
                except ValueError:
                    raise ValueError(f"Invalid array index: {key}")
        else:
            raise ValueError(f"Cannot add to path: {path}")
    
    elif op == "remove":
        if not parts:
            raise ValueError("Cannot remove root document")
        parent, key = get_parent_and_key(doc, parts)
        if isinstance(parent, dict):
            if key not in parent:
                raise ValueError(f"Path not found for remove: {path}")
            del parent[key]
        elif isinstance(parent, list):
            try:
                idx = int(key)
                del parent[idx]
            except (ValueError, IndexError):
                raise ValueError(f"Invalid array index for remove: {key}")
        else:
            raise ValueError(f"Cannot remove from path: {path}")
    
    elif op == "replace":
        value = patch.get("value")
        if not parts:
            return value
        parent, key = get_parent_and_key(doc, parts)
        if isinstance(parent, dict):
            if key not in parent:
                raise ValueError(f"Path not found for replace: {path}")
            parent[key] = value
        elif isinstance(parent, list):
            try:
                idx = int(key)
                parent[idx] = value
            except (ValueError, IndexError):
                raise ValueError(f"Invalid array index for replace: {key}")
        else:
            raise ValueError(f"Cannot replace at path: {path}")
    
    elif op == "copy":
        from_path = patch.get("from", "")
        from_parts = [p for p in from_path.split("/") if p]
        value = copy.deepcopy(get_value(doc, from_parts))
        if not parts:
            return value
        parent, key = get_parent_and_key(doc, parts)
        if isinstance(parent, dict):
            parent[key] = value
        elif isinstance(parent, list):
            if key == "-":
                parent.append(value)
            else:
                try:
                    idx = int(key)
                    parent.insert(idx, value)
                except ValueError:
                    raise ValueError(f"Invalid array index: {key}")
    
    elif op == "move":
        from_path = patch.get("from", "")
        from_parts = [p for p in from_path.split("/") if p]
        value = get_value(doc, from_parts)
        from_parent, from_key = get_parent_and_key(doc, from_parts)
        if isinstance(from_parent, dict):
            del from_parent[from_key]
        elif isinstance(from_parent, list):
            try:
                idx = int(from_key)
                del from_parent[idx]
            except (ValueError, IndexError):
                raise ValueError(f"Invalid array index for move: {from_key}")
        if not parts:
            return value
        parent, key = get_parent_and_key(doc, parts)
        if isinstance(parent, dict):
            parent[key] = value
        elif isinstance(parent, list):
            if key == "-":
                parent.append(value)
            else:
                try:
                    idx = int(key)
                    parent.insert(idx, value)
                except ValueError:
                    raise ValueError(f"Invalid array index: {key}")
    
    elif op == "test":
        value = patch.get("value")
        actual = get_value(doc, parts) if parts else doc
        if actual != value:
            raise ValueError(f"Test failed at {path}: expected {value}, got {actual}")
    
    else:
        raise ValueError(f"Unknown patch operation: {op}")
    
    return doc


def apply_json_patches(doc: Dict[str, Any], patches: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Apply a list of JSON Patch operations to a document.
    
    Parameters
    ----------
    doc : dict
        The document to patch
    patches : list of dict
        List of patch operations
        
    Returns
    -------
    dict
        The patched document
    """
    result = copy.deepcopy(doc)
    for patch in patches:
        result = _apply_json_patch(result, patch)
    return result


def validate_json_patches(patches: List[Dict[str, Any]]) -> List[str]:
    """
    Validate JSON Patch operations without applying them.
    
    Parameters
    ----------
    patches : list of dict
        List of patch operations to validate
        
    Returns
    -------
    list of str
        List of validation error messages (empty if valid)
    """
    errors = []
    valid_ops = {"add", "remove", "replace", "copy", "move", "test"}
    
    for i, patch in enumerate(patches):
        if not isinstance(patch, dict):
            errors.append(f"Patch {i}: must be a dict")
            continue
        
        op = patch.get("op")
        if not op:
            errors.append(f"Patch {i}: missing 'op' field")
        elif op not in valid_ops:
            errors.append(f"Patch {i}: unknown operation '{op}'")
        
        path = patch.get("path")
        if path is None:
            errors.append(f"Patch {i}: missing 'path' field")
        elif not isinstance(path, str):
            errors.append(f"Patch {i}: 'path' must be a string")
        
        if op in ("add", "replace", "test") and "value" not in patch:
            errors.append(f"Patch {i}: '{op}' operation requires 'value' field")
        
        if op in ("copy", "move") and "from" not in patch:
            errors.append(f"Patch {i}: '{op}' operation requires 'from' field")
    
    return errors


class DesignSpecSession:
    """
    Session engine for DesignSpec project management.
    
    Manages project directory structure, spec persistence, patch application,
    auto-compile, and runner execution.
    
    Attributes
    ----------
    project_dir : Path
        Root directory of the project
    spec : dict
        Current spec dictionary (normalized to meters)
    """
    
    def __init__(self, project_dir: Path):
        """
        Initialize a session for an existing project.
        
        Use create_project() or load_project() class methods instead.
        """
        self.project_dir = Path(project_dir)
        self._spec: Optional[Dict[str, Any]] = None
        self._spec_hash: Optional[str] = None
        self._loaded: bool = False
    
    @classmethod
    def create_project(
        cls,
        project_root: Union[str, Path],
        name: str,
        template_spec: Optional[Dict[str, Any]] = None,
    ) -> "DesignSpecSession":
        """
        Create a new project with folder structure.
        
        Parameters
        ----------
        project_root : str or Path
            Parent directory where project folder will be created
        name : str
            Project name (used as folder name)
        template_spec : dict, optional
            Initial spec to use (if None, creates minimal valid spec)
            
        Returns
        -------
        DesignSpecSession
            Session for the new project
        """
        project_root = Path(project_root)
        project_dir = project_root / name
        
        if project_dir.exists():
            raise ValueError(f"Project directory already exists: {project_dir}")
        
        project_dir.mkdir(parents=True)
        (project_dir / "spec_history").mkdir()
        (project_dir / "patches").mkdir()
        (project_dir / "reports").mkdir()
        (project_dir / "artifacts").mkdir()
        (project_dir / "logs").mkdir()
        
        if template_spec is None:
            template_spec = cls._create_minimal_spec()
        
        spec_path = project_dir / "spec.json"
        with open(spec_path, "w") as f:
            json.dump(template_spec, f, indent=2)
        
        session = cls(project_dir)
        session._spec = template_spec
        session._spec_hash = session._compute_spec_hash(template_spec)
        session._loaded = True
        
        session._save_spec_history("initial")
        
        logger.info(f"Created new project: {project_dir}")
        return session
    
    @classmethod
    def load_project(cls, project_dir: Union[str, Path]) -> "DesignSpecSession":
        """
        Load an existing project.
        
        Parameters
        ----------
        project_dir : str or Path
            Path to the project directory
            
        Returns
        -------
        DesignSpecSession
            Session for the loaded project
        """
        project_dir = Path(project_dir)
        
        if not project_dir.exists():
            raise ValueError(f"Project directory not found: {project_dir}")
        
        spec_path = project_dir / "spec.json"
        if not spec_path.exists():
            raise ValueError(f"spec.json not found in project: {project_dir}")
        
        session = cls(project_dir)
        
        with open(spec_path, "r") as f:
            session._spec = json.load(f)
        
        session._spec_hash = session._compute_spec_hash(session._spec)
        session._loaded = True
        
        for subdir in ["spec_history", "patches", "reports", "artifacts", "logs"]:
            (project_dir / subdir).mkdir(exist_ok=True)
        
        logger.info(f"Loaded project: {project_dir}")
        return session
    
    @staticmethod
    def _create_minimal_spec() -> Dict[str, Any]:
        """Create a minimal valid DesignSpec."""
        return {
            "schema": {
                "name": "aog_designspec",
                "version": "1.0.0",
            },
            "meta": {
                "name": "New Project",
                "input_units": "mm",
                "seed": 42,
            },
            "policies": {},
            "domains": {},
            "components": [],
        }
    
    @staticmethod
    def _compute_spec_hash(spec: Dict[str, Any]) -> str:
        """Compute a stable hash of the spec."""
        canonical_json = json.dumps(spec, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_json.encode()).hexdigest()[:16]
    
    def _ensure_loaded(self) -> None:
        """Ensure the session is loaded."""
        if not self._loaded:
            raise RuntimeError("Session not loaded. Use create_project() or load_project().")
    
    def get_spec(self) -> Dict[str, Any]:
        """
        Get the current spec dictionary.
        
        Returns
        -------
        dict
            Current spec (deep copy to prevent accidental modification)
        """
        self._ensure_loaded()
        return copy.deepcopy(self._spec)
    
    def validate_spec(self) -> ValidationReport:
        """
        Validate the current spec.
        
        Returns
        -------
        ValidationReport
            Validation results (JSON-serializable)
        """
        self._ensure_loaded()
        
        errors = []
        warnings = []
        
        schema = self._spec.get("schema", {})
        if not schema:
            errors.append("Missing 'schema' section")
        else:
            if schema.get("name") != "aog_designspec":
                errors.append(f"Invalid schema name: {schema.get('name')}")
            if not schema.get("version"):
                errors.append("Missing schema version")
        
        meta = self._spec.get("meta", {})
        if not meta:
            errors.append("Missing 'meta' section")
        else:
            if "seed" not in meta:
                warnings.append("meta.seed not specified, results may not be reproducible")
            if "input_units" not in meta:
                warnings.append("meta.input_units not specified, assuming meters")
        
        if "policies" not in self._spec:
            errors.append("Missing 'policies' section")
        
        if "domains" not in self._spec:
            errors.append("Missing 'domains' section")
        
        if "components" not in self._spec:
            errors.append("Missing 'components' section")
        elif not isinstance(self._spec.get("components"), list):
            errors.append("'components' must be a list")
        
        report = ValidationReport(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            schema_version=schema.get("version") if schema else None,
            spec_hash=self._spec_hash,
        )
        
        report_path = self.project_dir / "reports" / "last_validation.json"
        with open(report_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        
        return report
    
    def compile(self) -> CompileReport:
        """
        Run compile_policies and compile_domains stages.
        
        Returns
        -------
        CompileReport
            Compilation results (JSON-serializable)
        """
        self._ensure_loaded()
        
        import time
        start_time = time.time()
        
        warnings = []
        errors = []
        policies_compiled = []
        domains_compiled = []
        
        try:
            from designspec import DesignSpec, DesignSpecRunner, ExecutionPlan
            
            spec_obj = DesignSpec.from_dict(self._spec)
            warnings.extend(spec_obj.warnings)
            
            plan = ExecutionPlan(run_until="compile_domains")
            runner = DesignSpecRunner(
                spec_obj,
                plan=plan,
                output_dir=self.project_dir / "artifacts",
            )
            
            result = runner.run()
            
            for stage_report in result.stage_reports:
                if stage_report.stage == "compile_policies":
                    if stage_report.success:
                        policies_compiled = list(stage_report.metadata.keys())
                    warnings.extend(stage_report.warnings)
                    errors.extend(stage_report.errors)
                elif stage_report.stage == "compile_domains":
                    if stage_report.success:
                        domains_compiled = list(stage_report.metadata.keys())
                    warnings.extend(stage_report.warnings)
                    errors.extend(stage_report.errors)
            
            success = result.success
            
        except Exception as e:
            logger.exception("Compile failed")
            errors.append(f"Compile failed: {str(e)}")
            success = False
        
        duration = time.time() - start_time
        
        report = CompileReport(
            success=success,
            policies_compiled=policies_compiled,
            domains_compiled=domains_compiled,
            warnings=warnings,
            errors=errors,
            duration_s=duration,
        )
        
        report_path = self.project_dir / "reports" / "compile_report.json"
        with open(report_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        
        return report
    
    def apply_patch(
        self,
        patches: List[Dict[str, Any]],
        author: str = "user",
        auto_compile: bool = True,
    ) -> OperationReport:
        """
        Apply JSON Patch operations to the spec.
        
        Parameters
        ----------
        patches : list of dict
            JSON Patch operations (RFC 6902)
        author : str
            Author of the patch ("agent" or "user")
        auto_compile : bool
            Whether to auto-run compile after successful patch
            
        Returns
        -------
        OperationReport
            Operation results (JSON-serializable)
        """
        self._ensure_loaded()
        
        validation_errors = validate_json_patches(patches)
        if validation_errors:
            return OperationReport(
                success=False,
                operation="apply_patch",
                errors=validation_errors,
            )
        
        try:
            new_spec = apply_json_patches(self._spec, patches)
        except ValueError as e:
            return OperationReport(
                success=False,
                operation="apply_patch",
                errors=[str(e)],
            )
        
        old_spec = self._spec
        old_hash = self._spec_hash
        
        self._spec = new_spec
        self._spec_hash = self._compute_spec_hash(new_spec)
        
        validation = self.validate_spec()
        if not validation.valid:
            self._spec = old_spec
            self._spec_hash = old_hash
            return OperationReport(
                success=False,
                operation="apply_patch",
                errors=["Patch results in invalid spec"] + validation.errors,
                warnings=validation.warnings,
            )
        
        self._save_spec()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        patch_record = {
            "timestamp": timestamp,
            "author": author,
            "patches": patches,
            "old_hash": old_hash,
            "new_hash": self._spec_hash,
        }
        patch_path = self.project_dir / "patches" / f"{timestamp}_{author}.json"
        with open(patch_path, "w") as f:
            json.dump(patch_record, f, indent=2)
        
        self._save_spec_history(f"after_patch_{timestamp}")
        
        metadata = {
            "old_hash": old_hash,
            "new_hash": self._spec_hash,
            "patch_count": len(patches),
        }
        warnings = validation.warnings
        
        if auto_compile:
            compile_report = self.compile()
            metadata["compile_report"] = compile_report.to_dict()
            if not compile_report.success:
                warnings.append("Compile failed after patch - see compile_report for details")
            warnings.extend(compile_report.warnings)
        
        return OperationReport(
            success=True,
            operation="apply_patch",
            warnings=warnings,
            metadata=metadata,
        )
    
    def _save_spec(self) -> None:
        """Save current spec to spec.json."""
        spec_path = self.project_dir / "spec.json"
        with open(spec_path, "w") as f:
            json.dump(self._spec, f, indent=2)
    
    def _save_spec_history(self, label: str) -> None:
        """Save a timestamped spec snapshot."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_path = self.project_dir / "spec_history" / f"{timestamp}_{label}.json"
        with open(history_path, "w") as f:
            json.dump(self._spec, f, indent=2)
    
    def run(
        self,
        plan: Optional[Dict[str, Any]] = None,
        run_until: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the DesignSpecRunner with the current spec.
        
        Parameters
        ----------
        plan : dict, optional
            ExecutionPlan configuration dict
        run_until : str, optional
            Shorthand for plan with run_until stage
            
        Returns
        -------
        dict
            RunnerResult as dict (JSON-serializable)
        """
        self._ensure_loaded()
        
        from designspec import DesignSpec, DesignSpecRunner, ExecutionPlan
        import time
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"run_{timestamp}"
        output_dir = self.project_dir / "artifacts" / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            spec_obj = DesignSpec.from_dict(self._spec)
            
            if plan:
                exec_plan = ExecutionPlan.from_dict(plan)
            elif run_until:
                exec_plan = ExecutionPlan(run_until=run_until)
            else:
                exec_plan = ExecutionPlan()
            
            runner = DesignSpecRunner(
                spec_obj,
                plan=exec_plan,
                output_dir=output_dir,
            )
            
            result = runner.run()
            result_dict = result.to_dict()
            result_dict["run_id"] = run_id
            result_dict["output_dir"] = str(output_dir)
            
        except Exception as e:
            logger.exception("Run failed")
            result_dict = {
                "success": False,
                "run_id": run_id,
                "output_dir": str(output_dir),
                "errors": [f"Run failed: {str(e)}"],
                "stages_completed": [],
                "stage_reports": [],
            }
        
        report_path = self.project_dir / "reports" / "last_runner_result.json"
        with open(report_path, "w") as f:
            json.dump(result_dict, f, indent=2)
        
        run_report_path = output_dir / "run_report.json"
        with open(run_report_path, "w") as f:
            json.dump(result_dict, f, indent=2)
        
        return result_dict
    
    def get_patch_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of applied patches.
        
        Returns
        -------
        list of dict
            List of patch records in chronological order
        """
        self._ensure_loaded()
        
        patches_dir = self.project_dir / "patches"
        patch_files = sorted(patches_dir.glob("*.json"))
        
        history = []
        for patch_file in patch_files:
            with open(patch_file, "r") as f:
                history.append(json.load(f))
        
        return history
    
    def get_spec_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of spec snapshots.
        
        Returns
        -------
        list of dict
            List of spec snapshot metadata in chronological order
        """
        self._ensure_loaded()
        
        history_dir = self.project_dir / "spec_history"
        history_files = sorted(history_dir.glob("*.json"))
        
        history = []
        for history_file in history_files:
            history.append({
                "filename": history_file.name,
                "timestamp": history_file.name.split("_")[0],
                "path": str(history_file),
            })
        
        return history
    
    def get_artifacts(self) -> List[Dict[str, Any]]:
        """
        Get list of generated artifacts.
        
        Returns
        -------
        list of dict
            List of artifact metadata
        """
        self._ensure_loaded()
        
        artifacts_dir = self.project_dir / "artifacts"
        artifacts = []
        
        for run_dir in sorted(artifacts_dir.iterdir()):
            if run_dir.is_dir() and run_dir.name.startswith("run_"):
                run_info = {
                    "run_id": run_dir.name,
                    "path": str(run_dir),
                    "files": [],
                }
                
                for artifact_file in run_dir.iterdir():
                    if artifact_file.is_file():
                        run_info["files"].append({
                            "name": artifact_file.name,
                            "path": str(artifact_file),
                            "size_bytes": artifact_file.stat().st_size,
                        })
                
                artifacts.append(run_info)
        
        return artifacts
    
    def get_last_compile_report(self) -> Optional[Dict[str, Any]]:
        """Get the last compile report if available."""
        report_path = self.project_dir / "reports" / "compile_report.json"
        if report_path.exists():
            with open(report_path, "r") as f:
                return json.load(f)
        return None
    
    def get_last_runner_result(self) -> Optional[Dict[str, Any]]:
        """Get the last runner result if available."""
        report_path = self.project_dir / "reports" / "last_runner_result.json"
        if report_path.exists():
            with open(report_path, "r") as f:
                return json.load(f)
        return None


__all__ = [
    "DesignSpecSession",
    "ValidationReport",
    "CompileReport",
    "PatchProposal",
    "OperationReport",
    "apply_json_patches",
    "validate_json_patches",
]
