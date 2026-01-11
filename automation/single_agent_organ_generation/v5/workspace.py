"""
Workspace Manager - Agent Workspace for LLM-First V5

Manages the agent workspace where the master script and tools live and evolve.
The workspace is the heart of the LLM-first architecture, containing:
- master.py: The primary file the LLM iterates on
- tools/: Optional LLM-created tool modules
- tool_registry.json: Catalog of repo tools + generated tools
- spec.json: Structured living spec snapshot
- runs/: Run logs and artifacts
- history/: Master script snapshots and LLM thought traces
"""

import os
import json
import hashlib
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path


@dataclass
class ToolRegistryEntry:
    """An entry in the tool registry."""
    name: str
    origin: str  # "repo" or "generated"
    module: str  # e.g., "generation.ops.space_colonization" or "tools.my_tool"
    entrypoints: List[str]  # e.g., ["grow_network", "validate_network"]
    description: str
    created_in_version: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "origin": self.origin,
            "module": self.module,
            "entrypoints": self.entrypoints,
            "description": self.description,
            "created_in_version": self.created_in_version,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ToolRegistryEntry":
        return cls(
            name=d["name"],
            origin=d["origin"],
            module=d["module"],
            entrypoints=d.get("entrypoints", []),
            description=d.get("description", ""),
            created_in_version=d.get("created_in_version"),
        )


@dataclass
class FileInfo:
    """P1 #9: File info with hash and modified time."""
    path: str
    exists: bool
    hash: Optional[str]  # SHA256 first 16 chars
    modified_time: Optional[str]  # ISO format
    size_bytes: Optional[int]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "exists": self.exists,
            "hash": self.hash,
            "modified_time": self.modified_time,
            "size_bytes": self.size_bytes,
        }


@dataclass
class WorkspaceSummary:
    """Summary of the workspace state for LLM context."""
    workspace_path: str
    master_script_exists: bool
    master_script_hash: Optional[str]
    master_script_lines: int
    master_script_modified: Optional[str]  # P1 #9: modified time
    tool_count: int
    tool_names: List[str]
    tool_files: List[FileInfo]  # P1 #9: file info for each tool
    run_count: int
    last_run_version: Optional[int]
    last_run_status: Optional[str]
    spec_exists: bool
    spec_keys: List[str]
    spec_hash: Optional[str]  # P1 #9: spec hash
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "workspace_path": self.workspace_path,
            "master_script_exists": self.master_script_exists,
            "master_script_hash": self.master_script_hash,
            "master_script_lines": self.master_script_lines,
            "master_script_modified": self.master_script_modified,
            "tool_count": self.tool_count,
            "tool_names": self.tool_names,
            "tool_files": [f.to_dict() for f in self.tool_files],
            "run_count": self.run_count,
            "last_run_version": self.last_run_version,
            "last_run_status": self.last_run_status,
            "spec_exists": self.spec_exists,
            "spec_keys": self.spec_keys,
            "spec_hash": self.spec_hash,
        }


@dataclass
class RunRecord:
    """Record of a single run execution."""
    version: int
    timestamp: str
    status: str  # "success", "failed", "timeout"
    elapsed_seconds: float
    stdout_path: str
    stderr_path: str
    artifacts_json_path: Optional[str]
    verification_passed: bool
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "timestamp": self.timestamp,
            "status": self.status,
            "elapsed_seconds": self.elapsed_seconds,
            "stdout_path": self.stdout_path,
            "stderr_path": self.stderr_path,
            "artifacts_json_path": self.artifacts_json_path,
            "verification_passed": self.verification_passed,
            "error_message": self.error_message,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunRecord":
        return cls(
            version=d["version"],
            timestamp=d["timestamp"],
            status=d["status"],
            elapsed_seconds=d.get("elapsed_seconds", 0.0),
            stdout_path=d.get("stdout_path", ""),
            stderr_path=d.get("stderr_path", ""),
            artifacts_json_path=d.get("artifacts_json_path"),
            verification_passed=d.get("verification_passed", False),
            error_message=d.get("error_message"),
        )


# Default repo tools that are always available
# NOTE: These must match actual module paths and entrypoints in the codebase!
# Validated against generation/ops/__init__.py and generation/core/__init__.py
DEFAULT_REPO_TOOLS: List[ToolRegistryEntry] = [
    ToolRegistryEntry(
        name="ops",
        origin="repo",
        module="generation.ops",
        entrypoints=[
            "create_network", "add_inlet", "add_outlet",  # from build.py
            "space_colonization_step", "SpaceColonizationParams",  # from space_colonization.py
            "grow_branch", "grow_to_point", "bifurcate",  # from growth.py
            "embed_tree_as_negative_space",  # from embedding.py
        ],
        description="Core operations for building and modifying vascular networks",
    ),
    ToolRegistryEntry(
        name="core",
        origin="repo",
        module="generation.core",
        entrypoints=[
            "Point3D", "Direction3D", "TubeGeometry",  # from types.py
            "Node", "VesselSegment", "VascularNetwork",  # from network.py
            "DomainSpec", "EllipsoidDomain", "BoxDomain", "MeshDomain",  # from domain.py
            "OperationResult", "Delta",  # from result.py
        ],
        description="Core data structures: network, domain, types",
    ),
    ToolRegistryEntry(
        name="adapters",
        origin="repo",
        module="generation.adapters",
        entrypoints=[
            "to_networkx_graph", "from_networkx_graph",  # from networkx_adapter.py
            "to_trimesh", "to_hollow_tube_mesh", "export_hollow_tube_stl",  # from mesh_adapter.py
            "make_full_report",  # from report_adapter.py
        ],
        description="Adapters for mesh export, NetworkX conversion, and reporting",
    ),
    ToolRegistryEntry(
        name="collision",
        origin="repo",
        module="generation.ops",
        entrypoints=["get_collisions", "avoid_collisions"],
        description="Collision detection and avoidance for vascular networks",
    ),
    ToolRegistryEntry(
        name="anastomosis",
        origin="repo",
        module="generation.ops",
        entrypoints=["create_anastomosis", "check_tree_interactions"],
        description="Anastomosis creation between vascular trees",
    ),
    ToolRegistryEntry(
        name="pathfinding",
        origin="repo",
        module="generation.ops",
        entrypoints=["grow_toward_targets", "CostWeights"],
        description="Pathfinding and targeted growth operations",
    ),
]


# P1 #12: Tool discovery capability - scan packages for available functions
def discover_tools_from_package(package_name: str) -> List[ToolRegistryEntry]:
    """
    Discover tools from a package by scanning its modules.
    
    Parameters
    ----------
    package_name : str
        Name of the package to scan (e.g., "generation")
        
    Returns
    -------
    List[ToolRegistryEntry]
        Discovered tool entries with functions, docstrings, and signatures
    """
    import importlib
    import inspect
    import logging
    
    logger = logging.getLogger(__name__)
    discovered = []
    
    try:
        package = importlib.import_module(package_name)
        package_path = getattr(package, '__path__', None)
        
        if not package_path:
            return discovered
        
        # Get all exported names from __all__ if available
        all_names = getattr(package, '__all__', [])
        
        if all_names:
            # Collect functions and classes from __all__
            entrypoints = []
            for name in all_names:
                obj = getattr(package, name, None)
                if obj and (inspect.isfunction(obj) or inspect.isclass(obj)):
                    entrypoints.append(name)
            
            if entrypoints:
                discovered.append(ToolRegistryEntry(
                    name=package_name.split('.')[-1],
                    origin="discovered",
                    module=package_name,
                    entrypoints=entrypoints,
                    description=f"Auto-discovered from {package_name}",
                ))
    except ImportError as e:
        logger.warning(f"Failed to discover tools from {package_name}: {e}")
    
    return discovered


def validate_tool_registry(tools: List[ToolRegistryEntry]) -> List[ToolRegistryEntry]:
    """
    Validate tool registry entries by attempting to import modules.
    
    Returns only the tools whose modules can be successfully imported.
    Invalid entries are logged as warnings.
    """
    import importlib
    import logging
    
    logger = logging.getLogger(__name__)
    valid_tools = []
    
    for tool in tools:
        try:
            module = importlib.import_module(tool.module)
            # Check that at least one entrypoint exists
            found_entrypoints = []
            for ep in tool.entrypoints:
                if hasattr(module, ep):
                    found_entrypoints.append(ep)
            
            if found_entrypoints:
                # Update tool with only valid entrypoints
                validated_tool = ToolRegistryEntry(
                    name=tool.name,
                    origin=tool.origin,
                    module=tool.module,
                    entrypoints=found_entrypoints,
                    description=tool.description,
                    created_in_version=tool.created_in_version,
                )
                valid_tools.append(validated_tool)
                if len(found_entrypoints) < len(tool.entrypoints):
                    missing = set(tool.entrypoints) - set(found_entrypoints)
                    logger.warning(f"Tool '{tool.name}': some entrypoints not found: {missing}")
            else:
                logger.warning(f"Tool '{tool.name}': no valid entrypoints found in module '{tool.module}'")
        except ImportError as e:
            logger.warning(f"Tool '{tool.name}': failed to import module '{tool.module}': {e}")
    
    return valid_tools


class WorkspaceManager:
    """
    Manages the agent workspace for LLM-first V5.
    
    The workspace is created under the output directory and contains:
    - master.py: The primary script the LLM iterates on
    - tools/: Optional LLM-created tool modules
    - tool_registry.json: Catalog of available tools
    - spec.json: Living spec snapshot exported from world model
    - runs/: Run logs and artifacts per version
    - history/: Master script snapshots
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize the workspace manager.
        
        Parameters
        ----------
        output_dir : str
            Base output directory (e.g., from GUI config)
        """
        self.output_dir = os.path.abspath(output_dir)
        self.workspace_dir = os.path.join(self.output_dir, "agent_workspace")
        self.master_script_path = os.path.join(self.workspace_dir, "master.py")
        self.tools_dir = os.path.join(self.workspace_dir, "tools")
        self.registry_path = os.path.join(self.workspace_dir, "tool_registry.json")
        self.spec_path = os.path.join(self.workspace_dir, "spec.json")
        self.runs_dir = os.path.join(self.workspace_dir, "runs")
        self.history_dir = os.path.join(self.workspace_dir, "history")
        
        self._run_counter = 0
        self._tool_registry: List[ToolRegistryEntry] = []
    
    def initialize(self) -> None:
        """
        Create the workspace directory structure.
        
        Creates:
        - agent_workspace/
        - agent_workspace/tools/
        - agent_workspace/runs/
        - agent_workspace/history/
        - agent_workspace/tool_registry.json (with default repo tools)
        """
        os.makedirs(self.workspace_dir, exist_ok=True)
        os.makedirs(self.tools_dir, exist_ok=True)
        os.makedirs(self.runs_dir, exist_ok=True)
        os.makedirs(self.history_dir, exist_ok=True)
        
        # Initialize tool registry with default repo tools if it doesn't exist
        if not os.path.exists(self.registry_path):
            # Validate default tools at startup to catch any module path issues
            self._tool_registry = validate_tool_registry(DEFAULT_REPO_TOOLS)
            self._save_registry()
        else:
            self._load_registry()
            # Re-validate loaded tools to ensure they still work
            self._tool_registry = validate_tool_registry(self._tool_registry)
        
        # Determine run counter from existing runs
        self._run_counter = self._get_max_run_version()
    
    def _get_max_run_version(self) -> int:
        """Get the maximum run version from existing run directories."""
        if not os.path.exists(self.runs_dir):
            return 0
        
        max_version = 0
        for entry in os.listdir(self.runs_dir):
            if entry.startswith("run_"):
                try:
                    version = int(entry.split("_")[1])
                    max_version = max(max_version, version)
                except (IndexError, ValueError):
                    pass
        return max_version
    
    def _load_registry(self) -> None:
        """Load the tool registry from disk."""
        if os.path.exists(self.registry_path):
            with open(self.registry_path, 'r') as f:
                data = json.load(f)
            self._tool_registry = [
                ToolRegistryEntry.from_dict(entry) for entry in data.get("tools", [])
            ]
        else:
            self._tool_registry = list(DEFAULT_REPO_TOOLS)
    
    def _save_registry(self) -> None:
        """Save the tool registry to disk."""
        data = {
            "tools": [entry.to_dict() for entry in self._tool_registry],
            "updated_at": datetime.now().isoformat(),
        }
        with open(self.registry_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_tool_registry(self) -> List[ToolRegistryEntry]:
        """Get the current tool registry."""
        return list(self._tool_registry)
    
    def add_tool(self, entry: ToolRegistryEntry) -> None:
        """
        Add a tool to the registry.
        
        Parameters
        ----------
        entry : ToolRegistryEntry
            The tool entry to add
        """
        # Remove existing entry with same name if present
        self._tool_registry = [t for t in self._tool_registry if t.name != entry.name]
        self._tool_registry.append(entry)
        self._save_registry()
    
    def remove_tool(self, name: str) -> bool:
        """
        Remove a tool from the registry.
        
        Parameters
        ----------
        name : str
            Name of the tool to remove
            
        Returns
        -------
        bool
            True if tool was removed, False if not found
        """
        original_count = len(self._tool_registry)
        self._tool_registry = [t for t in self._tool_registry if t.name != name]
        if len(self._tool_registry) < original_count:
            self._save_registry()
            return True
        return False
    
    def read_master_script(self) -> Optional[str]:
        """
        Read the current master script content.
        
        Returns
        -------
        str or None
            The master script content, or None if it doesn't exist
        """
        if os.path.exists(self.master_script_path):
            with open(self.master_script_path, 'r') as f:
                return f.read()
        return None
    
    def write_master_script(self, content: str, snapshot: bool = True) -> str:
        """
        Write the master script, optionally snapshotting the previous version.
        
        Parameters
        ----------
        content : str
            The new master script content
        snapshot : bool
            Whether to snapshot the previous version (default: True)
            
        Returns
        -------
        str
            Path to the written master script
        """
        # Snapshot previous version if it exists
        if snapshot and os.path.exists(self.master_script_path):
            self._snapshot_master_script()
        
        # Write new content
        with open(self.master_script_path, 'w') as f:
            f.write(content)
        
        # Make executable
        os.chmod(self.master_script_path, 0o755)
        
        return self.master_script_path
    
    def _snapshot_master_script(self) -> str:
        """
        Snapshot the current master script to history.
        
        Returns
        -------
        str
            Path to the snapshot file
        """
        if not os.path.exists(self.master_script_path):
            return ""
        
        # Find next snapshot number
        existing = [f for f in os.listdir(self.history_dir) if f.startswith("master_prev_")]
        next_num = len(existing) + 1
        
        snapshot_name = f"master_prev_{next_num:04d}.py"
        snapshot_path = os.path.join(self.history_dir, snapshot_name)
        
        shutil.copy2(self.master_script_path, snapshot_path)
        return snapshot_path
    
    def validate_path_in_workspace(self, path: str) -> bool:
        """
        P1 #10: Validate that a path is within the workspace directory.
        
        Prevents writes that escape the workspace (e.g., using ../).
        
        Parameters
        ----------
        path : str
            Path to validate
            
        Returns
        -------
        bool
            True if path is within workspace, False otherwise
        """
        # Resolve to absolute path
        abs_path = os.path.abspath(path)
        workspace_abs = os.path.abspath(self.workspace_dir)
        
        # Check if path is within workspace
        return abs_path.startswith(workspace_abs + os.sep) or abs_path == workspace_abs
    
    def write_file_safe(self, relative_path: str, content: str) -> str:
        """
        P1 #10: Write a file safely, ensuring it's within the workspace.
        
        Parameters
        ----------
        relative_path : str
            Path relative to workspace (e.g., "master.py", "tools/helper.py")
        content : str
            File content to write
            
        Returns
        -------
        str
            Absolute path to the written file
            
        Raises
        ------
        ValueError
            If path would escape the workspace
        """
        # Compute absolute path
        if os.path.isabs(relative_path):
            abs_path = relative_path
        else:
            abs_path = os.path.join(self.workspace_dir, relative_path)
        
        # Validate path is within workspace
        if not self.validate_path_in_workspace(abs_path):
            raise ValueError(f"Path '{relative_path}' escapes workspace directory")
        
        # Ensure parent directory exists
        parent_dir = os.path.dirname(abs_path)
        os.makedirs(parent_dir, exist_ok=True)
        
        # Write file
        with open(abs_path, 'w') as f:
            f.write(content)
        
        return abs_path
    
    def write_tool_module(self, name: str, content: str) -> str:
        """
        Write a tool module to the tools directory.
        
        P1 #10: Uses safe write to prevent path escapes.
        
        Parameters
        ----------
        name : str
            Name of the tool (without .py extension)
        content : str
            The tool module content
            
        Returns
        -------
        str
            Path to the written tool module
            
        Raises
        ------
        ValueError
            If name contains path separators or escapes
        """
        # Validate name doesn't contain path separators
        if os.sep in name or '/' in name or '..' in name:
            raise ValueError(f"Tool name '{name}' contains invalid characters")
        
        tool_path = os.path.join(self.tools_dir, f"{name}.py")
        return self.write_file_safe(tool_path, content)
    
    def read_spec(self) -> Optional[Dict[str, Any]]:
        """
        Read the current spec.json.
        
        Returns
        -------
        dict or None
            The spec data, or None if it doesn't exist
        """
        if os.path.exists(self.spec_path):
            with open(self.spec_path, 'r') as f:
                return json.load(f)
        return None
    
    def write_spec(self, spec_data: Dict[str, Any]) -> str:
        """
        Write the spec.json file.
        
        Parameters
        ----------
        spec_data : dict
            The spec data to write
            
        Returns
        -------
        str
            Path to the written spec file
        """
        with open(self.spec_path, 'w') as f:
            json.dump(spec_data, f, indent=2, default=str)
        return self.spec_path
    
    def export_spec_from_world_model(self, world_model: Any) -> str:
        """
        Export the living spec from world model to spec.json.
        
        Parameters
        ----------
        world_model : WorldModel
            The world model to export from
            
        Returns
        -------
        str
            Path to the written spec file
        """
        spec_data = {
            "facts": {},
            "exported_at": datetime.now().isoformat(),
        }
        
        # Export all facts
        for field, fact in world_model.facts.items():
            spec_data["facts"][field] = {
                "value": fact.value,
                "provenance": fact.provenance.value,
                "confidence": fact.confidence,
            }
        
        return self.write_spec(spec_data)
    
    def get_next_run_version(self) -> int:
        """
        Get the next run version number.
        
        Returns
        -------
        int
            The next run version
        """
        self._run_counter += 1
        return self._run_counter
    
    def create_run_directory(self, version: int) -> str:
        """
        Create a run directory for a specific version.
        
        Parameters
        ----------
        version : int
            The run version number
            
        Returns
        -------
        str
            Path to the created run directory
        """
        run_dir = os.path.join(self.runs_dir, f"run_{version:04d}")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir
    
    def save_run_record(self, record: RunRecord) -> str:
        """
        Save a run record to the run directory.
        
        Parameters
        ----------
        record : RunRecord
            The run record to save
            
        Returns
        -------
        str
            Path to the saved record file
        """
        run_dir = os.path.join(self.runs_dir, f"run_{record.version:04d}")
        os.makedirs(run_dir, exist_ok=True)
        
        record_path = os.path.join(run_dir, "run_record.json")
        with open(record_path, 'w') as f:
            json.dump(record.to_dict(), f, indent=2)
        
        return record_path
    
    def get_last_run_record(self) -> Optional[RunRecord]:
        """
        Get the most recent run record.
        
        Returns
        -------
        RunRecord or None
            The last run record, or None if no runs exist
        """
        if self._run_counter == 0:
            return None
        
        run_dir = os.path.join(self.runs_dir, f"run_{self._run_counter:04d}")
        record_path = os.path.join(run_dir, "run_record.json")
        
        if os.path.exists(record_path):
            with open(record_path, 'r') as f:
                data = json.load(f)
            return RunRecord.from_dict(data)
        
        return None
    
    def compute_master_script_hash(self) -> Optional[str]:
        """
        Compute a hash of the current master script.
        
        Returns
        -------
        str or None
            SHA256 hash (first 16 chars), or None if script doesn't exist
        """
        content = self.read_master_script()
        if content is None:
            return None
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _get_file_info(self, path: str) -> FileInfo:
        """
        P1 #9: Get file info with hash and modified time.
        
        Parameters
        ----------
        path : str
            Path to the file
            
        Returns
        -------
        FileInfo
            File information
        """
        if not os.path.exists(path):
            return FileInfo(
                path=path,
                exists=False,
                hash=None,
                modified_time=None,
                size_bytes=None,
            )
        
        stat = os.stat(path)
        with open(path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()[:16]
        
        return FileInfo(
            path=path,
            exists=True,
            hash=file_hash,
            modified_time=datetime.fromtimestamp(stat.st_mtime).isoformat(),
            size_bytes=stat.st_size,
        )
    
    def get_summary(self) -> WorkspaceSummary:
        """
        Get a summary of the workspace state.
        
        P1 #9: Includes file hashes and modified times.
        
        Returns
        -------
        WorkspaceSummary
            Summary of the workspace for LLM context
        """
        master_content = self.read_master_script()
        master_exists = master_content is not None
        master_lines = len(master_content.split('\n')) if master_content else 0
        
        # P1 #9: Get master script modified time
        master_modified = None
        if os.path.exists(self.master_script_path):
            stat = os.stat(self.master_script_path)
            master_modified = datetime.fromtimestamp(stat.st_mtime).isoformat()
        
        tool_names = [t.name for t in self._tool_registry if t.origin == "generated"]
        
        # P1 #9: Get file info for each generated tool
        tool_files = []
        for tool in self._tool_registry:
            if tool.origin == "generated":
                tool_path = os.path.join(self.tools_dir, f"{tool.name}.py")
                tool_files.append(self._get_file_info(tool_path))
        
        last_record = self.get_last_run_record()
        
        spec_data = self.read_spec()
        spec_keys = list(spec_data.get("facts", {}).keys()) if spec_data else []
        
        # P1 #9: Compute spec hash
        spec_hash = None
        if os.path.exists(self.spec_path):
            with open(self.spec_path, 'rb') as f:
                spec_hash = hashlib.sha256(f.read()).hexdigest()[:16]
        
        return WorkspaceSummary(
            workspace_path=self.workspace_dir,
            master_script_exists=master_exists,
            master_script_hash=self.compute_master_script_hash(),
            master_script_lines=master_lines,
            master_script_modified=master_modified,
            tool_count=len([t for t in self._tool_registry if t.origin == "generated"]),
            tool_names=tool_names,
            tool_files=tool_files,
            run_count=self._run_counter,
            last_run_version=last_record.version if last_record else None,
            last_run_status=last_record.status if last_record else None,
            spec_exists=spec_data is not None,
            spec_keys=spec_keys,
            spec_hash=spec_hash,
        )
    
    def get_master_script_for_prompt(self, max_lines: int = 500) -> str:
        """
        Get the master script content formatted for LLM prompt.
        
        P1 #8: Include full master script context (increased from 200 to 500 lines).
        For very long scripts, provides a digest with function signatures.
        
        Parameters
        ----------
        max_lines : int
            Maximum number of lines to include (default: 500)
            
        Returns
        -------
        str
            The master script content, possibly with digest for long scripts
        """
        content = self.read_master_script()
        if content is None:
            return "[No master script exists yet]"
        
        lines = content.split('\n')
        total_lines = len(lines)
        
        # If script fits within limit, return full content
        if total_lines <= max_lines:
            return content
        
        # For long scripts, provide a digest with function signatures + critical sections
        import re
        
        # Extract function/class definitions
        definitions = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('def ') or stripped.startswith('class ') or stripped.startswith('async def '):
                # Include the definition line and docstring if present
                def_lines = [f"L{i+1}: {line}"]
                # Check for docstring
                if i + 1 < total_lines:
                    next_line = lines[i + 1].strip()
                    if next_line.startswith('"""') or next_line.startswith("'''"):
                        def_lines.append(f"L{i+2}: {lines[i + 1]}")
                definitions.append('\n'.join(def_lines))
        
        # Build digest
        digest_parts = [
            f"[Master script: {total_lines} lines total]",
            "",
            "=== IMPORTS AND SETUP (first 50 lines) ===",
            '\n'.join(lines[:50]),
            "",
            "=== FUNCTION/CLASS DEFINITIONS ===",
            '\n'.join(definitions) if definitions else "(no functions/classes found)",
            "",
            "=== MAIN SECTION (last 100 lines) ===",
            '\n'.join(lines[-100:]),
        ]
        
        return '\n'.join(digest_parts)
    
    def get_tool_registry_for_prompt(self) -> str:
        """
        Get the tool registry formatted for LLM prompt.
        
        Returns
        -------
        str
            Formatted tool registry description
        """
        lines = ["Available tools:"]
        for tool in self._tool_registry:
            origin_tag = "[repo]" if tool.origin == "repo" else "[generated]"
            lines.append(f"  - {tool.name} {origin_tag}: {tool.description}")
            lines.append(f"    Module: {tool.module}")
            lines.append(f"    Entrypoints: {', '.join(tool.entrypoints)}")
        return '\n'.join(lines)
    
    def validate_master_script(self) -> Dict[str, Any]:
        """
        P1 #13: Validate master script before execution.
        P4 #33: Add dangerous import warnings.
        P4 #34: Add output dir only enforcement for writes.
        
        Performs:
        - Syntax check (py_compile)
        - Import smoke test (can imports be resolved?)
        - Dangerous import detection (P4 #33)
        - Write path analysis (P4 #34)
        
        Returns
        -------
        dict
            Validation result with keys:
            - valid: bool
            - syntax_ok: bool
            - syntax_error: str or None
            - import_warnings: List[str]
            - dangerous_imports: List[str] (P4 #33)
            - write_warnings: List[str] (P4 #34)
        """
        import subprocess
        import sys
        
        result = {
            "valid": True,
            "syntax_ok": True,
            "syntax_error": None,
            "import_warnings": [],
            "dangerous_imports": [],
            "write_warnings": [],
        }
        
        if not os.path.exists(self.master_script_path):
            result["valid"] = False
            result["syntax_ok"] = False
            result["syntax_error"] = "Master script does not exist"
            return result
        
        # Syntax check using py_compile
        try:
            compile_result = subprocess.run(
                [sys.executable, "-m", "py_compile", self.master_script_path],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if compile_result.returncode != 0:
                result["valid"] = False
                result["syntax_ok"] = False
                result["syntax_error"] = compile_result.stderr.strip()
                return result
        except subprocess.TimeoutExpired:
            result["valid"] = False
            result["syntax_ok"] = False
            result["syntax_error"] = "Syntax check timed out"
            return result
        except Exception as e:
            result["valid"] = False
            result["syntax_ok"] = False
            result["syntax_error"] = str(e)
            return result
        
        # Read content for further analysis
        content = self.read_master_script()
        if content:
            import re
            
            # Import smoke test - check if imports can be resolved
            import_lines = re.findall(r'^(?:from|import)\s+([^\s]+)', content, re.MULTILINE)
            
            for module_name in import_lines:
                # Extract base module name
                base_module = module_name.split('.')[0]
                
                # Skip standard library and common modules
                if base_module in ('os', 'sys', 'json', 'math', 'datetime', 'typing', 're', 'pathlib'):
                    continue
                
                # Try to import
                try:
                    import importlib
                    importlib.import_module(base_module)
                except ImportError as e:
                    result["import_warnings"].append(f"Cannot import '{module_name}': {e}")
            
            # P4 #33: Dangerous import detection
            dangerous_modules = [
                'requests', 'urllib', 'socket', 'subprocess', 'os.system',
                'shutil.rmtree', 'eval', 'exec', 'compile', '__import__',
                'pickle', 'marshal', 'ctypes', 'multiprocessing',
            ]
            
            for dangerous in dangerous_modules:
                if dangerous in content:
                    result["dangerous_imports"].append(
                        f"Potentially dangerous: '{dangerous}' found in script"
                    )
            
            # P4 #34: Write path analysis - check for writes outside output dir
            write_patterns = [
                r'open\s*\(\s*["\']([^"\']+)["\'].*["\']w',
                r'\.write\s*\(',
                r'shutil\.copy',
                r'shutil\.move',
                r'os\.makedirs',
                r'Path\s*\([^)]+\)\.write',
            ]
            
            for pattern in write_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    if isinstance(match, str) and match:
                        # Check if path is absolute and outside workspace
                        if match.startswith('/') and not match.startswith(self.workspace_path):
                            result["write_warnings"].append(
                                f"Write to path outside workspace: '{match}'"
                            )
                        # Check for parent directory traversal
                        if '..' in match:
                            result["write_warnings"].append(
                                f"Path traversal detected: '{match}'"
                            )
        
        # If there are import warnings, mark as potentially invalid but don't fail
        if result["import_warnings"]:
            result["valid"] = True  # Still valid, just warnings
        
        return result
    
    def get_execution_environment(self) -> Dict[str, str]:
        """
        P4 #32: Get restricted execution environment for sandbox.
        
        Returns environment variables for sandboxed execution.
        """
        import os as os_module
        
        # Start with minimal environment
        env = {
            "PATH": os_module.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": os_module.environ.get("HOME", "/tmp"),
            "PYTHONPATH": str(self.workspace_path),
            "WORKSPACE_PATH": str(self.workspace_path),
            "OUTPUT_DIR": str(self.runs_path),
        }
        
        # Add Python-related vars
        if "VIRTUAL_ENV" in os_module.environ:
            env["VIRTUAL_ENV"] = os_module.environ["VIRTUAL_ENV"]
        
        # Restrict certain vars
        restricted_vars = [
            "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
            "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
            "DATABASE_URL", "REDIS_URL",
        ]
        
        for var in restricted_vars:
            if var in env:
                del env[var]
        
        return env
    
    def get_execution_config(self) -> Dict[str, Any]:
        """
        P4 #32: Get execution configuration for sandbox.
        
        Returns configuration for sandboxed execution.
        """
        return {
            "timeout_seconds": 600,  # 10 minute timeout
            "working_directory": str(self.workspace_path),
            "allowed_write_paths": [
                str(self.runs_path),
                str(self.workspace_path / "tools"),
            ],
            "restricted_imports": [
                "requests", "urllib.request", "socket",
                "subprocess", "os.system", "shutil.rmtree",
            ],
            "max_memory_mb": 4096,
            "max_file_size_mb": 100,
        }
