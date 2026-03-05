"""
Error Taxonomy and Structured Parsing for DesignSpec

This module provides structured error parsing with a comprehensive taxonomy
of known error patterns. Each pattern includes:
- Regex for matching
- Affected policy/component
- Suggested fix template
- Confidence level
"""

import re
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Pattern

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Types of errors that can occur in DesignSpec."""
    # Mesh-related errors
    PITCH_TOO_LARGE = "pitch_too_large"
    PITCH_TOO_SMALL = "pitch_too_small"
    MESH_DEGENERATE = "mesh_degenerate"
    MESH_NOT_WATERTIGHT = "mesh_not_watertight"

    # Units and scale errors
    UNITS_MISMATCH = "units_mismatch"
    SCALE_INVALID = "scale_invalid"
    VALUE_TOO_SMALL = "value_too_small"
    VALUE_TOO_LARGE = "value_too_large"

    # Component errors
    COMPONENT_OUTSIDE_DOMAIN = "component_outside_domain"
    COMPONENT_OVERLAP = "component_overlap"
    COMPONENT_INVALID_GEOMETRY = "component_invalid_geometry"

    # Port errors
    PORT_DIRECTION_WRONG = "port_direction_wrong"
    PORT_OUTSIDE_DOMAIN = "port_outside_domain"
    PORT_RADIUS_INVALID = "port_radius_invalid"

    # Domain errors
    DOMAIN_TOO_SMALL = "domain_too_small"
    DOMAIN_INVALID = "domain_invalid"
    NO_DOMAIN_DEFINED = "no_domain_defined"

    # Policy errors
    POLICY_MISSING = "policy_missing"
    POLICY_INVALID_VALUE = "policy_invalid_value"
    POLICY_CONFLICT = "policy_conflict"

    # Validation errors
    VALIDATION_FAILED = "validation_failed"
    SCHEMA_INVALID = "schema_invalid"

    # Runtime errors
    MEMORY_ERROR = "memory_error"
    TIMEOUT_ERROR = "timeout_error"
    FILE_NOT_FOUND = "file_not_found"

    # Unknown
    UNKNOWN = "unknown"


@dataclass
class StructuredError:
    """
    A structured representation of an error.

    Attributes
    ----------
    raw_message : str
        The original error message
    error_type : ErrorType
        The classified error type
    affected_component : str, optional
        Component ID if error is component-specific
    affected_policy : str, optional
        Policy name if error is policy-specific
    affected_parameter : str, optional
        Parameter name if error is parameter-specific
    json_path : str, optional
        JSON path to the problematic value (e.g., "/policies/mesh_merge/voxel_pitch")
    current_value : Any, optional
        The current value that's causing the error
    suggested_value : Any, optional
        A suggested replacement value
    suggested_fix : str
        Human-readable fix suggestion
    confidence : float
        Confidence in the classification (0.0 to 1.0)
    """
    raw_message: str
    error_type: ErrorType
    affected_component: Optional[str] = None
    affected_policy: Optional[str] = None
    affected_parameter: Optional[str] = None
    json_path: Optional[str] = None
    current_value: Optional[Any] = None
    suggested_value: Optional[Any] = None
    suggested_fix: str = ""
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_message": self.raw_message,
            "error_type": self.error_type.value,
            "affected_component": self.affected_component,
            "affected_policy": self.affected_policy,
            "affected_parameter": self.affected_parameter,
            "json_path": self.json_path,
            "current_value": self.current_value,
            "suggested_value": self.suggested_value,
            "suggested_fix": self.suggested_fix,
            "confidence": self.confidence,
        }


@dataclass
class ErrorPattern:
    """A pattern for matching and classifying errors."""
    error_type: ErrorType
    regex_pattern: str
    affected_policy: Optional[str] = None
    affected_parameter: Optional[str] = None
    fix_template: str = ""
    confidence: float = 0.9
    _compiled_pattern: Optional[Pattern] = None

    def __post_init__(self):
        """Compile the regex pattern."""
        self._compiled_pattern = re.compile(self.regex_pattern, re.IGNORECASE | re.DOTALL)

    def matches(self, error_message: str) -> Optional[re.Match]:
        """Check if this pattern matches the error message."""
        return self._compiled_pattern.search(error_message)


class ErrorParser:
    """
    Parses error messages into structured errors.

    Uses a database of error patterns to classify and extract
    information from error messages.
    """

    # Error patterns database
    PATTERNS = [
        # === PITCH ERRORS ===
        ErrorPattern(
            error_type=ErrorType.PITCH_TOO_LARGE,
            regex_pattern=r"domain scale is ~?([\d.]+).*voxel_pitch is ([\d.]+)",
            affected_policy="mesh_merge",
            affected_parameter="voxel_pitch",
            fix_template="Set voxel_pitch = domain_scale / 100 (approximately {suggested_value:.6f}m)",
            confidence=0.95,
        ),
        ErrorPattern(
            error_type=ErrorType.PITCH_TOO_LARGE,
            regex_pattern=r"voxel.*pitch.*too.*(?:large|coarse|big)",
            affected_policy="mesh_merge",
            affected_parameter="voxel_pitch",
            fix_template="Reduce voxel_pitch to approximately domain_scale / 100",
            confidence=0.85,
        ),
        ErrorPattern(
            error_type=ErrorType.PITCH_TOO_SMALL,
            regex_pattern=r"voxel.*pitch.*too.*(?:small|fine)",
            affected_policy="mesh_merge",
            affected_parameter="voxel_pitch",
            fix_template="Increase voxel_pitch to reduce memory usage",
            confidence=0.85,
        ),

        # === MESH ERRORS ===
        ErrorPattern(
            error_type=ErrorType.MESH_DEGENERATE,
            regex_pattern=r"(?:degenerate|invalid|malformed).*mesh",
            affected_policy="mesh_merge",
            fix_template="Check component geometry and ensure valid mesh parameters",
            confidence=0.80,
        ),
        ErrorPattern(
            error_type=ErrorType.MESH_NOT_WATERTIGHT,
            regex_pattern=r"not.*watertight|mesh.*(?:open|holes)",
            affected_policy="mesh_merge",
            fix_template="Ensure all mesh geometry is closed and watertight",
            confidence=0.85,
        ),

        # === COMPONENT ERRORS ===
        ErrorPattern(
            error_type=ErrorType.COMPONENT_OUTSIDE_DOMAIN,
            regex_pattern=r"component.*(?:outside|beyond|exceeds).*domain",
            fix_template="Adjust component position or size, or increase domain size",
            confidence=0.90,
        ),
        ErrorPattern(
            error_type=ErrorType.COMPONENT_OUTSIDE_DOMAIN,
            regex_pattern=r"(?:position|center).*outside.*(?:bounds|domain)",
            fix_template="Move component position inside domain bounds",
            confidence=0.85,
        ),
        ErrorPattern(
            error_type=ErrorType.COMPONENT_OVERLAP,
            regex_pattern=r"component.*overlap",
            fix_template="Adjust component positions to avoid overlap",
            confidence=0.85,
        ),

        # === PORT ERRORS ===
        ErrorPattern(
            error_type=ErrorType.PORT_DIRECTION_WRONG,
            regex_pattern=r"(?:port|inlet|outlet).*direction.*(?:invalid|wrong)",
            fix_template="Set port direction to a valid face normal (e.g., [1,0,0], [0,1,0], [0,0,1])",
            confidence=0.90,
        ),
        ErrorPattern(
            error_type=ErrorType.PORT_OUTSIDE_DOMAIN,
            regex_pattern=r"(?:port|inlet|outlet).*outside.*domain",
            fix_template="Adjust port position to be on domain boundary",
            confidence=0.90,
        ),
        ErrorPattern(
            error_type=ErrorType.PORT_RADIUS_INVALID,
            regex_pattern=r"(?:port|inlet|outlet).*radius.*(?:invalid|too|negative)",
            fix_template="Set port radius to a positive value appropriate for domain scale",
            confidence=0.85,
        ),

        # === DOMAIN ERRORS ===
        ErrorPattern(
            error_type=ErrorType.DOMAIN_TOO_SMALL,
            regex_pattern=r"domain.*(?:too small|insufficient|inadequate)",
            fix_template="Increase domain size to accommodate components",
            confidence=0.85,
        ),
        ErrorPattern(
            error_type=ErrorType.NO_DOMAIN_DEFINED,
            regex_pattern=r"no domain|domain.*(?:missing|not defined)",
            affected_policy="domain",
            fix_template="Add a domain to the spec",
            confidence=0.95,
        ),
        ErrorPattern(
            error_type=ErrorType.DOMAIN_INVALID,
            regex_pattern=r"domain.*invalid",
            affected_policy="domain",
            fix_template="Check domain parameters for validity",
            confidence=0.80,
        ),

        # === UNITS ERRORS ===
        ErrorPattern(
            error_type=ErrorType.UNITS_MISMATCH,
            regex_pattern=r"units?.*(?:mismatch|wrong|incorrect)",
            fix_template="Check that all values are in correct units (meters internally, but input_units can be mm)",
            confidence=0.80,
        ),
        ErrorPattern(
            error_type=ErrorType.VALUE_TOO_SMALL,
            regex_pattern=r"value.*too small.*expected.*(?:mm|millimeters)",
            fix_template="Values may be in wrong units - check if values should be in mm instead of meters",
            confidence=0.75,
        ),
        ErrorPattern(
            error_type=ErrorType.VALUE_TOO_LARGE,
            regex_pattern=r"value.*(?:too large|exceeds)",
            fix_template="Reduce the parameter value or check units",
            confidence=0.70,
        ),

        # === POLICY ERRORS ===
        ErrorPattern(
            error_type=ErrorType.POLICY_MISSING,
            regex_pattern=r"(?:policy|parameter).*(?:missing|required|not found)",
            fix_template="Add the required policy to the spec",
            confidence=0.85,
        ),
        ErrorPattern(
            error_type=ErrorType.POLICY_INVALID_VALUE,
            regex_pattern=r"(?:policy|parameter).*(?:invalid|out of range)",
            fix_template="Set parameter to a valid value within expected range",
            confidence=0.80,
        ),

        # === VALIDATION ERRORS ===
        ErrorPattern(
            error_type=ErrorType.SCHEMA_INVALID,
            regex_pattern=r"schema.*(?:invalid|validation failed)",
            fix_template="Check spec structure matches schema requirements",
            confidence=0.85,
        ),

        # === RUNTIME ERRORS ===
        ErrorPattern(
            error_type=ErrorType.MEMORY_ERROR,
            regex_pattern=r"(?:memory|out of memory|oom)",
            fix_template="Reduce mesh resolution (increase voxel_pitch) to decrease memory usage",
            confidence=0.90,
        ),
        ErrorPattern(
            error_type=ErrorType.TIMEOUT_ERROR,
            regex_pattern=r"timeout|timed out",
            fix_template="Simplify geometry or increase timeout threshold",
            confidence=0.85,
        ),
        ErrorPattern(
            error_type=ErrorType.FILE_NOT_FOUND,
            regex_pattern=r"file not found|no such file",
            fix_template="Check file paths and ensure required files exist",
            confidence=0.90,
        ),
    ]

    def __init__(self):
        """Initialize the error parser."""
        pass

    def parse(self, error_message: str, spec: Optional[Dict[str, Any]] = None) -> StructuredError:
        """
        Parse an error message into a structured error.

        Parameters
        ----------
        error_message : str
            The raw error message
        spec : dict, optional
            The current spec (used for context)

        Returns
        -------
        StructuredError
            The structured error with classification and suggestions
        """
        # Try to match against known patterns
        for pattern in self.PATTERNS:
            match = pattern.matches(error_message)
            if match:
                return self._create_structured_error(
                    error_message,
                    pattern,
                    match,
                    spec,
                )

        # No pattern matched - return unknown error
        return StructuredError(
            raw_message=error_message,
            error_type=ErrorType.UNKNOWN,
            suggested_fix="Unable to classify error. Please review the error message manually.",
            confidence=0.1,
        )

    def _create_structured_error(
        self,
        error_message: str,
        pattern: ErrorPattern,
        match: re.Match,
        spec: Optional[Dict[str, Any]],
    ) -> StructuredError:
        """
        Create a structured error from a pattern match.

        Parameters
        ----------
        error_message : str
            The raw error message
        pattern : ErrorPattern
            The matched pattern
        match : re.Match
            The regex match object
        spec : dict, optional
            The current spec

        Returns
        -------
        StructuredError
            The structured error
        """
        error = StructuredError(
            raw_message=error_message,
            error_type=pattern.error_type,
            affected_policy=pattern.affected_policy,
            affected_parameter=pattern.affected_parameter,
            confidence=pattern.confidence,
        )

        # Extract specific information based on error type
        if pattern.error_type == ErrorType.PITCH_TOO_LARGE:
            # Extract domain scale and current pitch from match groups
            groups = match.groups()
            if len(groups) >= 2:
                try:
                    domain_scale = float(groups[0])
                    current_pitch = float(groups[1])
                    suggested_pitch = domain_scale / 100.0

                    error.current_value = current_pitch
                    error.suggested_value = suggested_pitch
                    error.json_path = "/policies/mesh_merge/voxel_pitch"
                    error.suggested_fix = pattern.fix_template.format(
                        suggested_value=suggested_pitch
                    )
                except (ValueError, IndexError):
                    error.suggested_fix = pattern.fix_template
            else:
                error.suggested_fix = pattern.fix_template
        else:
            # Use template as-is
            error.suggested_fix = pattern.fix_template

        return error

    def parse_multiple(
        self,
        error_messages: List[str],
        spec: Optional[Dict[str, Any]] = None,
    ) -> List[StructuredError]:
        """
        Parse multiple error messages.

        Parameters
        ----------
        error_messages : list of str
            List of error messages
        spec : dict, optional
            The current spec

        Returns
        -------
        list of StructuredError
            List of structured errors
        """
        return [self.parse(msg, spec) for msg in error_messages]
