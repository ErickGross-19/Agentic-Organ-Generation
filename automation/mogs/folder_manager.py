"""
MOGS Folder Manager

Manages the canonical folder structure for MOGS objects.
Ensures all outputs are written within the object's project folder.
"""

import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from .models import (
    ObjectManifest,
    RunIndex,
    RetentionPolicy,
    generate_object_uuid,
    get_timestamp,
)


# Canonical folder structure
FOLDER_STRUCTURE = {
    "00_admin": {
        "files": ["object_manifest.json", "run_index.json", "README.md", "retention_log.md"],
        "subdirs": [],
    },
    "01_specs": {
        "files": ["spec_changelog.md"],
        "subdirs": [],
    },
    "02_agent_docs": {
        "files": [],
        "subdirs": ["CSA", "CBA", "VQA"],
    },
    "03_prompts": {
        "files": [],
        "subdirs": [],
    },
    "04_scripts": {
        "files": [],
        "subdirs": [],
    },
    "05_runs": {
        "files": [],
        "subdirs": [],
    },
    "06_outputs": {
        "files": [],
        "subdirs": [],
    },
    "07_validation": {
        "files": [],
        "subdirs": [],
    },
    "99_logs": {
        "files": ["system_events.log", "warnings.log"],
        "subdirs": [],
    },
}


class FolderManager:
    """
    Manages the folder structure for a MOGS object.
    
    Ensures all file operations are within the object's project folder
    and maintains the canonical folder layout.
    """
    
    def __init__(self, objects_base_dir: str, object_uuid: str):
        """
        Initialize the folder manager.
        
        Parameters
        ----------
        objects_base_dir : str
            Base directory for all objects (e.g., "./objects")
        object_uuid : str
            UUID of the object
        """
        self.objects_base_dir = os.path.abspath(objects_base_dir)
        self.object_uuid = object_uuid
        self.object_dir = os.path.join(self.objects_base_dir, object_uuid)
        self.project_dir = os.path.join(self.object_dir, "project")
    
    @property
    def admin_dir(self) -> str:
        return os.path.join(self.project_dir, "00_admin")
    
    @property
    def specs_dir(self) -> str:
        return os.path.join(self.project_dir, "01_specs")
    
    @property
    def agent_docs_dir(self) -> str:
        return os.path.join(self.project_dir, "02_agent_docs")
    
    @property
    def prompts_dir(self) -> str:
        return os.path.join(self.project_dir, "03_prompts")
    
    @property
    def scripts_dir(self) -> str:
        return os.path.join(self.project_dir, "04_scripts")
    
    @property
    def runs_dir(self) -> str:
        return os.path.join(self.project_dir, "05_runs")
    
    @property
    def outputs_dir(self) -> str:
        return os.path.join(self.project_dir, "06_outputs")
    
    @property
    def validation_dir(self) -> str:
        return os.path.join(self.project_dir, "07_validation")
    
    @property
    def logs_dir(self) -> str:
        return os.path.join(self.project_dir, "99_logs")
    
    @property
    def manifest_path(self) -> str:
        return os.path.join(self.admin_dir, "object_manifest.json")
    
    @property
    def run_index_path(self) -> str:
        return os.path.join(self.admin_dir, "run_index.json")
    
    @property
    def retention_log_path(self) -> str:
        return os.path.join(self.admin_dir, "retention_log.md")
    
    @property
    def spec_changelog_path(self) -> str:
        return os.path.join(self.specs_dir, "spec_changelog.md")
    
    def is_path_within_project(self, path: str) -> bool:
        """
        Check if a path is within the project folder.
        
        Parameters
        ----------
        path : str
            Path to check
            
        Returns
        -------
        bool
            True if path is within project folder
        """
        abs_path = os.path.abspath(path)
        return abs_path.startswith(self.project_dir + os.sep) or abs_path == self.project_dir
    
    def validate_write_path(self, path: str) -> None:
        """
        Validate that a path is safe to write to.
        
        Parameters
        ----------
        path : str
            Path to validate
            
        Raises
        ------
        PermissionError
            If path is outside project folder
        """
        if not self.is_path_within_project(path):
            raise PermissionError(
                f"Cannot write to '{path}'. "
                f"All writes must be within project folder: {self.project_dir}"
            )
    
    def create_folder_structure(self, object_name: str) -> ObjectManifest:
        """
        Create the canonical folder structure for a new object.
        
        Parameters
        ----------
        object_name : str
            Human-readable name for the object
            
        Returns
        -------
        ObjectManifest
            The created object manifest
        """
        # Create base directories
        os.makedirs(self.project_dir, exist_ok=True)
        
        # Create all folders from the canonical structure
        for folder_name, config in FOLDER_STRUCTURE.items():
            folder_path = os.path.join(self.project_dir, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            
            # Create subdirectories
            for subdir in config["subdirs"]:
                subdir_path = os.path.join(folder_path, subdir)
                os.makedirs(subdir_path, exist_ok=True)
        
        # Create object manifest
        manifest = ObjectManifest(
            object_uuid=self.object_uuid,
            object_name=object_name,
            created_at=get_timestamp(),
        )
        manifest.save(self.manifest_path)
        
        # Create run index
        run_index = RunIndex(object_uuid=self.object_uuid)
        run_index.save(self.run_index_path)
        
        # Create README
        self._create_readme(object_name)
        
        # Create retention log
        self._create_retention_log()
        
        # Create spec changelog
        self._create_spec_changelog()
        
        # Create log files
        self._create_log_files()
        
        return manifest
    
    def _create_readme(self, object_name: str) -> None:
        """Create the README file for the object."""
        readme_content = f"""# {object_name}

## Object Information

- **UUID**: `{self.object_uuid}`
- **Name**: {object_name}
- **Created**: {get_timestamp()}

## Folder Structure

```
project/
  00_admin/       # Administrative files (manifest, run index, logs)
  01_specs/       # Versioned specifications
  02_agent_docs/  # Agent session documents and decisions
  03_prompts/     # Prompt packs for each agent
  04_scripts/     # Generated scripts (versioned)
  05_runs/        # Execution logs and metadata
  06_outputs/     # Generated outputs (versioned)
  07_validation/  # Validation reports (versioned)
  99_logs/        # System logs
```

## Version History

See `01_specs/spec_changelog.md` for version history.

## Notes

(Add notes here)
"""
        readme_path = os.path.join(self.admin_dir, "README.md")
        with open(readme_path, 'w') as f:
            f.write(readme_content)
    
    def _create_retention_log(self) -> None:
        """Create the retention log file."""
        log_content = f"""# Retention Log

This log tracks version cleanup operations.

## Policy

- Keep last 5 versions by default
- Pinned versions are never deleted
- Most recent accepted version is always kept

## Cleanup History

"""
        with open(self.retention_log_path, 'w') as f:
            f.write(log_content)
    
    def _create_spec_changelog(self) -> None:
        """Create the spec changelog file."""
        changelog_content = f"""# Specification Changelog

This file tracks changes to the specification across versions.

## Version History

"""
        with open(self.spec_changelog_path, 'w') as f:
            f.write(changelog_content)
    
    def _create_log_files(self) -> None:
        """Create empty log files."""
        for log_file in ["system_events.log", "warnings.log"]:
            log_path = os.path.join(self.logs_dir, log_file)
            with open(log_path, 'w') as f:
                f.write(f"# {log_file}\n# Created: {get_timestamp()}\n\n")
    
    def get_spec_path(self, version: int) -> str:
        """Get the path for a spec version file."""
        return os.path.join(self.specs_dir, f"spec_v{version:03d}.json")
    
    def get_spec_summary_path(self, version: int) -> str:
        """Get the path for a spec summary file."""
        return os.path.join(self.specs_dir, f"spec_v{version:03d}_summary.md")
    
    def get_spec_risk_flags_path(self, version: int) -> str:
        """Get the path for a spec risk flags file."""
        return os.path.join(self.specs_dir, f"spec_v{version:03d}_risk_flags.json")
    
    def get_scripts_version_dir(self, version: int) -> str:
        """Get the directory for scripts of a specific version."""
        return os.path.join(self.scripts_dir, f"scripts_v{version:03d}")
    
    def get_outputs_version_dir(self, version: int) -> str:
        """Get the directory for outputs of a specific version."""
        return os.path.join(self.outputs_dir, f"v{version:03d}")
    
    def get_validation_version_dir(self, version: int) -> str:
        """Get the directory for validation of a specific version."""
        return os.path.join(self.validation_dir, f"v{version:03d}")
    
    def get_run_dir(self, timestamp: str, spec_version: int) -> str:
        """Get the directory for a specific run."""
        return os.path.join(self.runs_dir, f"run_{timestamp}_spec_v{spec_version:03d}")
    
    def get_agent_docs_dir(self, agent_type: str) -> str:
        """Get the directory for agent documents."""
        return os.path.join(self.agent_docs_dir, agent_type.upper())
    
    def create_version_directories(self, version: int) -> Dict[str, str]:
        """
        Create directories for a new version.
        
        Parameters
        ----------
        version : int
            Version number
            
        Returns
        -------
        Dict[str, str]
            Dictionary of created directory paths
        """
        dirs = {
            "scripts": self.get_scripts_version_dir(version),
            "outputs": self.get_outputs_version_dir(version),
            "validation": self.get_validation_version_dir(version),
        }
        
        for dir_path in dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        # Create output subdirectories
        for subdir in ["generation", "analysis", "final"]:
            os.makedirs(os.path.join(dirs["outputs"], subdir), exist_ok=True)
        
        return dirs
    
    def load_manifest(self) -> ObjectManifest:
        """Load the object manifest."""
        return ObjectManifest.load(self.manifest_path)
    
    def save_manifest(self, manifest: ObjectManifest) -> None:
        """Save the object manifest."""
        manifest.last_modified_at = get_timestamp()
        manifest.save(self.manifest_path)
    
    def load_run_index(self) -> RunIndex:
        """Load the run index."""
        return RunIndex.load(self.run_index_path)
    
    def save_run_index(self, run_index: RunIndex) -> None:
        """Save the run index."""
        run_index.save(self.run_index_path)
    
    def log_event(self, message: str, level: str = "INFO") -> None:
        """
        Log an event to the system events log.
        
        Parameters
        ----------
        message : str
            Log message
        level : str
            Log level (INFO, WARNING, ERROR)
        """
        log_path = os.path.join(self.logs_dir, "system_events.log")
        timestamp = get_timestamp()
        log_line = f"[{timestamp}] [{level}] {message}\n"
        with open(log_path, 'a') as f:
            f.write(log_line)
    
    def log_warning(self, message: str) -> None:
        """Log a warning to the warnings log."""
        log_path = os.path.join(self.logs_dir, "warnings.log")
        timestamp = get_timestamp()
        log_line = f"[{timestamp}] {message}\n"
        with open(log_path, 'a') as f:
            f.write(log_line)
    
    def log_retention_action(self, action: str, version: int, details: str = "") -> None:
        """
        Log a retention action to the retention log.
        
        Parameters
        ----------
        action : str
            Action taken (e.g., "DELETED", "PINNED", "UNPINNED")
        version : int
            Version affected
        details : str
            Additional details
        """
        timestamp = get_timestamp()
        log_entry = f"\n### {timestamp}\n\n- **Action**: {action}\n- **Version**: v{version:03d}\n"
        if details:
            log_entry += f"- **Details**: {details}\n"
        
        with open(self.retention_log_path, 'a') as f:
            f.write(log_entry)
    
    def append_to_spec_changelog(self, version: int, summary: str, changes: List[str]) -> None:
        """
        Append an entry to the spec changelog.
        
        Parameters
        ----------
        version : int
            Version number
        summary : str
            Summary of changes
        changes : List[str]
            List of specific changes
        """
        timestamp = get_timestamp()
        entry = f"\n### Version {version:03d} ({timestamp})\n\n{summary}\n\n"
        if changes:
            entry += "Changes:\n"
            for change in changes:
                entry += f"- {change}\n"
        
        with open(self.spec_changelog_path, 'a') as f:
            f.write(entry)
    
    def get_all_version_paths(self, version: int) -> Dict[str, str]:
        """
        Get all paths associated with a version.
        
        Parameters
        ----------
        version : int
            Version number
            
        Returns
        -------
        Dict[str, str]
            Dictionary of path names to paths
        """
        return {
            "spec": self.get_spec_path(version),
            "spec_summary": self.get_spec_summary_path(version),
            "spec_risk_flags": self.get_spec_risk_flags_path(version),
            "scripts_dir": self.get_scripts_version_dir(version),
            "outputs_dir": self.get_outputs_version_dir(version),
            "validation_dir": self.get_validation_version_dir(version),
        }
    
    def version_exists(self, version: int) -> bool:
        """Check if a version exists (has a spec file)."""
        return os.path.exists(self.get_spec_path(version))
    
    def get_existing_versions(self) -> List[int]:
        """Get list of existing version numbers."""
        versions = []
        if os.path.exists(self.specs_dir):
            for filename in os.listdir(self.specs_dir):
                if filename.startswith("spec_v") and filename.endswith(".json"):
                    try:
                        version = int(filename[6:9])
                        versions.append(version)
                    except ValueError:
                        pass
        return sorted(versions)


def create_new_object(
    objects_base_dir: str,
    object_name: str,
    object_uuid: Optional[str] = None,
) -> FolderManager:
    """
    Create a new MOGS object with full folder structure.
    
    Parameters
    ----------
    objects_base_dir : str
        Base directory for all objects
    object_name : str
        Human-readable name for the object
    object_uuid : str, optional
        UUID for the object (generated if not provided)
        
    Returns
    -------
    FolderManager
        Folder manager for the new object
    """
    if object_uuid is None:
        object_uuid = generate_object_uuid()
    
    manager = FolderManager(objects_base_dir, object_uuid)
    manager.create_folder_structure(object_name)
    manager.log_event(f"Object created: {object_name} ({object_uuid})")
    
    return manager


def load_object(objects_base_dir: str, object_uuid: str) -> FolderManager:
    """
    Load an existing MOGS object.
    
    Parameters
    ----------
    objects_base_dir : str
        Base directory for all objects
    object_uuid : str
        UUID of the object to load
        
    Returns
    -------
    FolderManager
        Folder manager for the object
        
    Raises
    ------
    FileNotFoundError
        If object does not exist
    """
    manager = FolderManager(objects_base_dir, object_uuid)
    
    if not os.path.exists(manager.manifest_path):
        raise FileNotFoundError(f"Object not found: {object_uuid}")
    
    return manager
