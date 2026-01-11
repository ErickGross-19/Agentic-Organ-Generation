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
class WorkspaceSummary:
    """Summary of the workspace state for LLM context."""
    workspace_path: str
    master_script_exists: bool
    master_script_hash: Optional[str]
    master_script_lines: int
    tool_count: int
    tool_names: List[str]
    run_count: int
    last_run_version: Optional[int]
    last_run_status: Optional[str]
    spec_exists: bool
    spec_keys: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "workspace_path": self.workspace_path,
            "master_script_exists": self.master_script_exists,
            "master_script_hash": self.master_script_hash,
            "master_script_lines": self.master_script_lines,
            "tool_count": self.tool_count,
            "tool_names": self.tool_names,
            "run_count": self.run_count,
            "last_run_version": self.last_run_version,
            "last_run_status": self.last_run_status,
            "spec_exists": self.spec_exists,
            "spec_keys": self.spec_keys,
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
DEFAULT_REPO_TOOLS: List[ToolRegistryEntry] = [
    ToolRegistryEntry(
        name="space_colonization",
        origin="repo",
        module="generation.ops.space_colonization",
        entrypoints=["grow_network", "SpaceColonizationParams"],
        description="Space colonization algorithm for vascular network generation",
    ),
    ToolRegistryEntry(
        name="network_io",
        origin="repo",
        module="generation.ops.network_io",
        entrypoints=["save_network_json", "load_network_json", "network_to_stl"],
        description="Network I/O operations for saving/loading networks and exporting to STL",
    ),
    ToolRegistryEntry(
        name="domain",
        origin="repo",
        module="generation.domain",
        entrypoints=["BoxDomain", "CylinderDomain", "SphereDomain"],
        description="Domain geometry definitions (box, cylinder, sphere)",
    ),
    ToolRegistryEntry(
        name="validity",
        origin="repo",
        module="validity",
        entrypoints=["validate_network", "check_connectivity", "check_radii"],
        description="Network validation utilities",
    ),
]


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
            self._tool_registry = list(DEFAULT_REPO_TOOLS)
            self._save_registry()
        else:
            self._load_registry()
        
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
    
    def write_tool_module(self, name: str, content: str) -> str:
        """
        Write a tool module to the tools directory.
        
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
        """
        tool_path = os.path.join(self.tools_dir, f"{name}.py")
        with open(tool_path, 'w') as f:
            f.write(content)
        return tool_path
    
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
    
    def get_summary(self) -> WorkspaceSummary:
        """
        Get a summary of the workspace state.
        
        Returns
        -------
        WorkspaceSummary
            Summary of the workspace for LLM context
        """
        master_content = self.read_master_script()
        master_exists = master_content is not None
        master_lines = len(master_content.split('\n')) if master_content else 0
        
        tool_names = [t.name for t in self._tool_registry if t.origin == "generated"]
        
        last_record = self.get_last_run_record()
        
        spec_data = self.read_spec()
        spec_keys = list(spec_data.get("facts", {}).keys()) if spec_data else []
        
        return WorkspaceSummary(
            workspace_path=self.workspace_dir,
            master_script_exists=master_exists,
            master_script_hash=self.compute_master_script_hash(),
            master_script_lines=master_lines,
            tool_count=len([t for t in self._tool_registry if t.origin == "generated"]),
            tool_names=tool_names,
            run_count=self._run_counter,
            last_run_version=last_record.version if last_record else None,
            last_run_status=last_record.status if last_record else None,
            spec_exists=spec_data is not None,
            spec_keys=spec_keys,
        )
    
    def get_master_script_for_prompt(self, max_lines: int = 200) -> str:
        """
        Get the master script content formatted for LLM prompt.
        
        Parameters
        ----------
        max_lines : int
            Maximum number of lines to include (default: 200)
            
        Returns
        -------
        str
            The master script content, possibly truncated
        """
        content = self.read_master_script()
        if content is None:
            return "[No master script exists yet]"
        
        lines = content.split('\n')
        if len(lines) > max_lines:
            return '\n'.join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"
        return content
    
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
