"""
Operation result types for structured feedback.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum


class OperationStatus(Enum):
    """Status of an operation."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    WARNING = "warning"


class ErrorCode(Enum):
    """Standard error codes for LLM consumption."""
    OUTSIDE_DOMAIN = "OUTSIDE_DOMAIN"
    COLLISION_BLOCKED = "COLLISION_BLOCKED"
    RADIUS_TOO_SMALL = "RADIUS_TOO_SMALL"
    RADIUS_TOO_LARGE = "RADIUS_TOO_LARGE"
    LENGTH_TOO_SHORT = "LENGTH_TOO_SHORT"
    LENGTH_TOO_LONG = "LENGTH_TOO_LONG"
    ANGLE_TOO_LARGE = "ANGLE_TOO_LARGE"
    NODE_NOT_FOUND = "NODE_NOT_FOUND"
    SEGMENT_NOT_FOUND = "SEGMENT_NOT_FOUND"
    MISSING_DIRECTION = "MISSING_DIRECTION"
    MISSING_RADIUS = "MISSING_RADIUS"
    NO_TERMINAL_NODES = "NO_TERMINAL_NODES"
    DIRICHLET_SINGULAR = "DIRICHLET_SINGULAR"
    FLOW_BALANCE_ERROR = "FLOW_BALANCE_ERROR"
    MESH_EXPORT_FAILED = "MESH_EXPORT_FAILED"
    REPAIR_FAILED = "REPAIR_FAILED"
    TERMINATED_DUE_TO_COLLISIONS = "TERMINATED_DUE_TO_COLLISIONS"
    REROUTE_FAILED = "REROUTE_FAILED"
    SHRINK_FAILED = "SHRINK_FAILED"
    INVALID_PARAMETER = "INVALID_PARAMETER"
    ANASTOMOSIS_NOT_ALLOWED = "ANASTOMOSIS_NOT_ALLOWED"
    ANASTOMOSIS_TOO_LONG = "ANASTOMOSIS_TOO_LONG"
    ANASTOMOSIS_RADIUS_OUT_OF_RANGE = "ANASTOMOSIS_RADIUS_OUT_OF_RANGE"
    INCOMPATIBLE_VESSEL_TYPES = "INCOMPATIBLE_VESSEL_TYPES"
    CLEARANCE_VIOLATION = "CLEARANCE_VIOLATION"
    CROSS_TYPE_TOO_CLOSE = "CROSS_TYPE_TOO_CLOSE"
    BELOW_MIN_TERMINAL_RADIUS = "BELOW_MIN_TERMINAL_RADIUS"
    MAX_GENERATION_EXCEEDED = "MAX_GENERATION_EXCEEDED"


@dataclass
class Delta:
    """
    Record of changes made by an operation.
    
    Used for undo/redo functionality.
    """
    
    created_node_ids: List[int] = field(default_factory=list)
    created_segment_ids: List[int] = field(default_factory=list)
    deleted_node_ids: List[int] = field(default_factory=list)
    deleted_segment_ids: List[int] = field(default_factory=list)
    modified_nodes: Dict[int, dict] = field(default_factory=dict)
    modified_segments: Dict[int, dict] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "created_node_ids": self.created_node_ids,
            "created_segment_ids": self.created_segment_ids,
            "deleted_node_ids": self.deleted_node_ids,
            "deleted_segment_ids": self.deleted_segment_ids,
            "modified_nodes": self.modified_nodes,
            "modified_segments": self.modified_segments,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "Delta":
        """Create from dictionary."""
        return cls(
            created_node_ids=d.get("created_node_ids", []),
            created_segment_ids=d.get("created_segment_ids", []),
            deleted_node_ids=d.get("deleted_node_ids", []),
            deleted_segment_ids=d.get("deleted_segment_ids", []),
            modified_nodes=d.get("modified_nodes", {}),
            modified_segments=d.get("modified_segments", {}),
        )


@dataclass
class OperationResult:
    """
    Structured result from a vascular network operation.
    
    Provides explicit feedback for LLM consumption.
    """
    
    status: OperationStatus
    message: str = ""
    new_ids: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    error_codes: List[str] = field(default_factory=list)
    delta: Optional[Delta] = None
    rng_state: Optional[dict] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_success(self) -> bool:
        """Check if operation was successful."""
        return self.status in (OperationStatus.SUCCESS, OperationStatus.PARTIAL_SUCCESS)
    
    def is_failure(self) -> bool:
        """Check if operation failed."""
        return self.status == OperationStatus.FAILURE
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)
    
    def add_error(self, error: str, code: Optional[ErrorCode] = None) -> None:
        """Add an error message with optional error code."""
        self.errors.append(error)
        if code is not None:
            self.error_codes.append(code.value)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization (JSON-safe)."""
        return {
            "status": self.status.value,
            "message": self.message,
            "new_ids": self.new_ids,
            "warnings": self.warnings,
            "errors": self.errors,
            "error_codes": self.error_codes,
            "delta": self.delta.to_dict() if self.delta else None,
            "rng_state": self.rng_state,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "OperationResult":
        """Create from dictionary."""
        return cls(
            status=OperationStatus(d["status"]),
            message=d.get("message", ""),
            new_ids=d.get("new_ids", {}),
            warnings=d.get("warnings", []),
            errors=d.get("errors", []),
            error_codes=d.get("error_codes", []),
            delta=Delta.from_dict(d["delta"]) if d.get("delta") else None,
            rng_state=d.get("rng_state"),
            metadata=d.get("metadata", {}),
        )
    
    @classmethod
    def success(cls, message: str = "", **kwargs) -> "OperationResult":
        """Create a success result."""
        return cls(status=OperationStatus.SUCCESS, message=message, **kwargs)
    
    @classmethod
    def failure(cls, message: str = "", **kwargs) -> "OperationResult":
        """Create a failure result."""
        return cls(status=OperationStatus.FAILURE, message=message, **kwargs)
    
    @classmethod
    def partial_success(cls, message: str = "", **kwargs) -> "OperationResult":
        """Create a partial success result."""
        return cls(status=OperationStatus.PARTIAL_SUCCESS, message=message, **kwargs)
