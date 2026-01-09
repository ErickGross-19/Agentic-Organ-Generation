"""
MOGS Safety Constraints

Implements non-negotiable safety constraints for script execution:
- Scripts can only write inside the object project folder
- No network access during execution
- No subprocess spawning
- Resource limits
- Environment variable allowlist only
"""

import os
import sys
import json
import signal
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set
from datetime import datetime
from pathlib import Path

try:
    import resource
    RESOURCE_AVAILABLE = True
except ImportError:
    RESOURCE_AVAILABLE = False

from .models import get_timestamp
from .folder_manager import FolderManager


# Environment variable allowlist
ENV_ALLOWLIST = {
    # Python-related
    "PYTHONPATH",
    "PYTHONHASHSEED",
    "PYTHONDONTWRITEBYTECODE",
    # System
    "PATH",
    "HOME",
    "USER",
    "LANG",
    "LC_ALL",
    "TZ",
    # MOGS-specific
    "ORGAN_AGENT_OUTPUT_DIR",
    "ORGAN_AGENT_SPEC_VERSION",
    "ORGAN_AGENT_OBJECT_UUID",
    "ORGAN_AGENT_PROJECT_DIR",
    # Restrictions (set by safety module)
    "ORGAN_AGENT_NO_NETWORK",
    "ORGAN_AGENT_NO_SUBPROCESS",
    "ORGAN_AGENT_WRITE_RESTRICTED",
}

# Default resource limits
DEFAULT_LIMITS = {
    "max_memory_mb": 4096,  # 4GB
    "max_cpu_time_seconds": 600,  # 10 minutes
    "max_file_size_mb": 1024,  # 1GB
    "max_open_files": 256,
}


@dataclass
class SafetyConfig:
    """
    Configuration for safety constraints.
    """
    allowed_write_dir: str
    max_memory_mb: int = DEFAULT_LIMITS["max_memory_mb"]
    max_cpu_time_seconds: int = DEFAULT_LIMITS["max_cpu_time_seconds"]
    max_file_size_mb: int = DEFAULT_LIMITS["max_file_size_mb"]
    max_open_files: int = DEFAULT_LIMITS["max_open_files"]
    allow_network: bool = False
    allow_subprocess: bool = False
    env_allowlist: Set[str] = field(default_factory=lambda: ENV_ALLOWLIST.copy())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed_write_dir": self.allowed_write_dir,
            "max_memory_mb": self.max_memory_mb,
            "max_cpu_time_seconds": self.max_cpu_time_seconds,
            "max_file_size_mb": self.max_file_size_mb,
            "max_open_files": self.max_open_files,
            "allow_network": self.allow_network,
            "allow_subprocess": self.allow_subprocess,
            "env_allowlist": list(self.env_allowlist),
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SafetyConfig":
        return cls(
            allowed_write_dir=d["allowed_write_dir"],
            max_memory_mb=d.get("max_memory_mb", DEFAULT_LIMITS["max_memory_mb"]),
            max_cpu_time_seconds=d.get("max_cpu_time_seconds", DEFAULT_LIMITS["max_cpu_time_seconds"]),
            max_file_size_mb=d.get("max_file_size_mb", DEFAULT_LIMITS["max_file_size_mb"]),
            max_open_files=d.get("max_open_files", DEFAULT_LIMITS["max_open_files"]),
            allow_network=d.get("allow_network", False),
            allow_subprocess=d.get("allow_subprocess", False),
            env_allowlist=set(d.get("env_allowlist", ENV_ALLOWLIST)),
        )


@dataclass
class SafetyViolation:
    """
    Record of a safety violation.
    """
    violation_type: str
    timestamp: str
    description: str
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "violation_type": self.violation_type,
            "timestamp": self.timestamp,
            "description": self.description,
            "details": self.details,
        }


class SafetyManager:
    """
    Manages safety constraints for MOGS script execution.
    
    Enforces:
    - Write path restrictions
    - Network access restrictions
    - Subprocess restrictions
    - Resource limits
    - Environment sanitization
    """
    
    def __init__(self, folder_manager: FolderManager, config: Optional[SafetyConfig] = None):
        """
        Initialize the safety manager.
        
        Parameters
        ----------
        folder_manager : FolderManager
            Folder manager for the object
        config : SafetyConfig, optional
            Safety configuration (uses defaults if not provided)
        """
        self.folder_manager = folder_manager
        self.config = config or SafetyConfig(
            allowed_write_dir=folder_manager.project_dir
        )
        self._violations: List[SafetyViolation] = []
    
    def build_safe_environment(
        self,
        spec_version: int,
        output_dir: str,
        additional_vars: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """
        Build a sanitized environment for script execution.
        
        Parameters
        ----------
        spec_version : int
            Version of the spec being executed
        output_dir : str
            Output directory for the script
        additional_vars : Dict[str, str], optional
            Additional environment variables to include
            
        Returns
        -------
        Dict[str, str]
            Sanitized environment variables
        """
        env = {}
        
        # Copy allowed variables from current environment
        for var in self.config.env_allowlist:
            if var in os.environ:
                env[var] = os.environ[var]
        
        # Set MOGS-specific variables
        env["ORGAN_AGENT_OUTPUT_DIR"] = output_dir
        env["ORGAN_AGENT_SPEC_VERSION"] = str(spec_version)
        env["ORGAN_AGENT_OBJECT_UUID"] = self.folder_manager.object_uuid
        env["ORGAN_AGENT_PROJECT_DIR"] = self.folder_manager.project_dir
        
        # Set restriction flags
        if not self.config.allow_network:
            env["ORGAN_AGENT_NO_NETWORK"] = "1"
        if not self.config.allow_subprocess:
            env["ORGAN_AGENT_NO_SUBPROCESS"] = "1"
        env["ORGAN_AGENT_WRITE_RESTRICTED"] = self.config.allowed_write_dir
        
        # Add additional variables (if in allowlist)
        if additional_vars:
            for key, value in additional_vars.items():
                if key in self.config.env_allowlist:
                    env[key] = value
        
        return env
    
    def get_resource_limits(self) -> Dict[str, int]:
        """
        Get resource limits for script execution.
        
        Returns
        -------
        Dict[str, int]
            Resource limits
        """
        return {
            "max_memory_bytes": self.config.max_memory_mb * 1024 * 1024,
            "max_cpu_time_seconds": self.config.max_cpu_time_seconds,
            "max_file_size_bytes": self.config.max_file_size_mb * 1024 * 1024,
            "max_open_files": self.config.max_open_files,
        }
    
    def apply_resource_limits(self) -> None:
        """
        Apply resource limits to the current process.
        
        This should be called in a subprocess before executing a script.
        Note: Resource limits are only available on Unix-like systems.
        On Windows, this method is a no-op.
        """
        if not RESOURCE_AVAILABLE:
            return
        
        limits = self.get_resource_limits()
        
        # Memory limit (soft and hard)
        try:
            resource.setrlimit(
                resource.RLIMIT_AS,
                (limits["max_memory_bytes"], limits["max_memory_bytes"])
            )
        except (ValueError, resource.error):
            pass  # May not be supported on all systems
        
        # CPU time limit
        try:
            resource.setrlimit(
                resource.RLIMIT_CPU,
                (limits["max_cpu_time_seconds"], limits["max_cpu_time_seconds"])
            )
        except (ValueError, resource.error):
            pass
        
        # File size limit
        try:
            resource.setrlimit(
                resource.RLIMIT_FSIZE,
                (limits["max_file_size_bytes"], limits["max_file_size_bytes"])
            )
        except (ValueError, resource.error):
            pass
        
        # Open files limit
        try:
            resource.setrlimit(
                resource.RLIMIT_NOFILE,
                (limits["max_open_files"], limits["max_open_files"])
            )
        except (ValueError, resource.error):
            pass
    
    def validate_write_path(self, path: str) -> bool:
        """
        Validate that a path is safe to write to.
        
        Parameters
        ----------
        path : str
            Path to validate
            
        Returns
        -------
        bool
            True if path is safe
        """
        abs_path = os.path.abspath(path)
        allowed_dir = os.path.abspath(self.config.allowed_write_dir)
        
        is_safe = abs_path.startswith(allowed_dir + os.sep) or abs_path == allowed_dir
        
        if not is_safe:
            self._record_violation(
                "write_path",
                f"Attempted write outside allowed directory: {path}",
                {"path": path, "allowed_dir": allowed_dir},
            )
        
        return is_safe
    
    def validate_script_content(self, script_path: str) -> List[str]:
        """
        Validate script content for safety issues.
        
        Parameters
        ----------
        script_path : str
            Path to the script to validate
            
        Returns
        -------
        List[str]
            List of safety warnings
        """
        import re
        
        warnings = []
        
        # Patterns that indicate potential safety issues
        dangerous_patterns = [
            (r'\bsubprocess\b', "Uses subprocess module (blocked)"),
            (r'\bos\.system\s*\(', "Uses os.system() (blocked)"),
            (r'\bos\.popen\s*\(', "Uses os.popen() (blocked)"),
            (r'\bos\.spawn', "Uses os.spawn* (blocked)"),
            (r'\bcommands\b', "Uses commands module (blocked)"),
            (r'\bsocket\b', "Uses socket module (network access)"),
            (r'\burllib\b', "Uses urllib module (network access)"),
            (r'\brequests\b', "Uses requests module (network access)"),
            (r'\bhttplib\b', "Uses httplib module (network access)"),
            (r'\bftplib\b', "Uses ftplib module (network access)"),
            (r'\bsmtplib\b', "Uses smtplib module (network access)"),
            (r'\beval\s*\(', "Uses eval() (potential code injection)"),
            (r'\bexec\s*\(', "Uses exec() (potential code injection)"),
            (r'__import__\s*\(', "Uses __import__() (dynamic import)"),
            (r'\bcompile\s*\(', "Uses compile() (potential code injection)"),
            (r'\bctypes\b', "Uses ctypes module (low-level access)"),
        ]
        
        try:
            with open(script_path, 'r') as f:
                content = f.read()
            
            for pattern, message in dangerous_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    warnings.append(message)
                    
                    # Record violation for blocked patterns
                    if "blocked" in message.lower():
                        self._record_violation(
                            "dangerous_pattern",
                            message,
                            {"pattern": pattern, "script": script_path},
                        )
        except Exception as e:
            warnings.append(f"Could not read script: {e}")
        
        return warnings
    
    def create_execution_metadata(
        self,
        spec_version: int,
        script_name: str,
        output_dir: str,
    ) -> Dict[str, Any]:
        """
        Create execution metadata for logging.
        
        Parameters
        ----------
        spec_version : int
            Version of the spec
        script_name : str
            Name of the script being executed
        output_dir : str
            Output directory
            
        Returns
        -------
        Dict[str, Any]
            Execution metadata
        """
        return {
            "spec_version": spec_version,
            "script_name": script_name,
            "output_dir": output_dir,
            "timestamp": get_timestamp(),
            "safety_config": self.config.to_dict(),
            "resource_limits": self.get_resource_limits(),
            "python_version": sys.version,
            "platform": sys.platform,
        }
    
    def save_execution_metadata(
        self,
        run_dir: str,
        metadata: Dict[str, Any],
    ) -> str:
        """
        Save execution metadata to disk.
        
        Parameters
        ----------
        run_dir : str
            Run directory
        metadata : Dict[str, Any]
            Execution metadata
            
        Returns
        -------
        str
            Path to saved metadata file
        """
        os.makedirs(run_dir, exist_ok=True)
        
        metadata_path = os.path.join(run_dir, "execution_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata_path
    
    def save_sanitized_environment(
        self,
        run_dir: str,
        env: Dict[str, str],
    ) -> str:
        """
        Save sanitized environment summary to disk.
        
        Parameters
        ----------
        run_dir : str
            Run directory
        env : Dict[str, str]
            Sanitized environment
            
        Returns
        -------
        str
            Path to saved environment file
        """
        os.makedirs(run_dir, exist_ok=True)
        
        # Sanitize sensitive values
        sanitized = {}
        sensitive_patterns = ["key", "secret", "password", "token", "auth"]
        
        for key, value in env.items():
            key_lower = key.lower()
            if any(p in key_lower for p in sensitive_patterns):
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = value
        
        env_path = os.path.join(run_dir, "environment_sanitized.json")
        with open(env_path, 'w') as f:
            json.dump(sanitized, f, indent=2)
        
        return env_path
    
    def _record_violation(
        self,
        violation_type: str,
        description: str,
        details: Dict[str, Any],
    ) -> None:
        """Record a safety violation."""
        violation = SafetyViolation(
            violation_type=violation_type,
            timestamp=get_timestamp(),
            description=description,
            details=details,
        )
        self._violations.append(violation)
        
        self.folder_manager.log_warning(f"Safety violation: {description}")
        self.folder_manager.log_event(f"Safety violation: {violation_type} - {description}")
    
    def get_violations(self) -> List[SafetyViolation]:
        """Get all recorded violations."""
        return self._violations
    
    def clear_violations(self) -> None:
        """Clear recorded violations."""
        self._violations = []
    
    def get_preexec_fn(self):
        """
        Get a preexec function for subprocess.
        
        This function will be called in the child process before
        executing the script.
        
        Returns
        -------
        Callable
            Preexec function
        """
        def preexec():
            self.apply_resource_limits()
        
        return preexec


def create_safe_runner_script(
    script_path: str,
    output_dir: str,
    project_dir: str,
) -> str:
    """
    Create a wrapper script that enforces safety constraints.
    
    Parameters
    ----------
    script_path : str
        Path to the script to wrap
    output_dir : str
        Output directory
    project_dir : str
        Project directory (allowed write directory)
        
    Returns
    -------
    str
        Content of the wrapper script
    """
    return f'''#!/usr/bin/env python3
"""
MOGS Safe Runner Wrapper

This wrapper enforces safety constraints on script execution.
"""

import os
import sys
import builtins

# Store original functions
_original_open = builtins.open
_original_import = builtins.__import__

# Allowed write directory
ALLOWED_WRITE_DIR = os.path.abspath("{project_dir}")
OUTPUT_DIR = os.path.abspath("{output_dir}")

# Blocked modules
BLOCKED_MODULES = {{
    "subprocess", "commands", "socket", "urllib", "urllib2", "urllib3",
    "requests", "httplib", "http.client", "ftplib", "smtplib", "telnetlib",
}}


def safe_open(file, mode='r', *args, **kwargs):
    """Safe open that restricts write paths."""
    if 'w' in mode or 'a' in mode or 'x' in mode or '+' in mode:
        abs_path = os.path.abspath(file)
        if not (abs_path.startswith(ALLOWED_WRITE_DIR + os.sep) or 
                abs_path.startswith(OUTPUT_DIR + os.sep)):
            raise PermissionError(
                f"Cannot write to '{{file}}'. "
                f"Writes restricted to project directory."
            )
    return _original_open(file, mode, *args, **kwargs)


def safe_import(name, *args, **kwargs):
    """Safe import that blocks dangerous modules."""
    base_module = name.split('.')[0]
    if base_module in BLOCKED_MODULES:
        raise ImportError(
            f"Module '{{name}}' is blocked for safety reasons. "
            f"Network access and subprocess spawning are not allowed."
        )
    return _original_import(name, *args, **kwargs)


# Install safe functions
builtins.open = safe_open
builtins.__import__ = safe_import

# Set environment
os.environ["ORGAN_AGENT_OUTPUT_DIR"] = OUTPUT_DIR
os.environ["ORGAN_AGENT_WRITE_RESTRICTED"] = ALLOWED_WRITE_DIR

# Execute the actual script
script_path = "{script_path}"
with _original_open(script_path, 'r') as f:
    script_content = f.read()

exec(compile(script_content, script_path, 'exec'))
'''
