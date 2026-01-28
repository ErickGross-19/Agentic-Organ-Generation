"""
Error Recovery Module for Enhanced Workflow Resilience.

This module provides error recovery mechanisms for the V5 controller,
including:
- Automatic error classification and recovery strategies
- Detailed text-based error message formatting
- Verbose mode with step-by-step explanations
- Text-based validation reports with clear explanations
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import traceback


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors for classification."""
    VALIDATION = "validation"
    GENERATION = "generation"
    CONFIGURATION = "configuration"
    USER_INPUT = "user_input"
    SYSTEM = "system"
    NETWORK = "network"
    FILE_IO = "file_io"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    """Strategies for error recovery."""
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    PROMPT_USER = "prompt_user"
    ROLLBACK = "rollback"
    ABORT = "abort"


@dataclass
class ErrorContext:
    """Context information for an error."""
    operation: str = ""
    phase: str = ""
    iteration: int = 0
    spec_snapshot: Optional[Dict[str, Any]] = None
    stack_trace: str = ""
    related_fields: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operation": self.operation,
            "phase": self.phase,
            "iteration": self.iteration,
            "spec_snapshot": self.spec_snapshot,
            "stack_trace": self.stack_trace,
            "related_fields": self.related_fields,
        }


@dataclass
class RecoveryAction:
    """An action to take for recovery."""
    strategy: RecoveryStrategy
    description: str
    auto_apply: bool = False
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy": self.strategy.value,
            "description": self.description,
            "auto_apply": self.auto_apply,
            "parameters": self.parameters,
        }


@dataclass
class ErrorReport:
    """A detailed error report."""
    error_id: str
    message: str
    severity: ErrorSeverity
    category: ErrorCategory
    context: ErrorContext
    recovery_actions: List[RecoveryAction] = field(default_factory=list)
    user_explanation: str = ""
    technical_details: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "error_id": self.error_id,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "context": self.context.to_dict(),
            "recovery_actions": [a.to_dict() for a in self.recovery_actions],
            "user_explanation": self.user_explanation,
            "technical_details": self.technical_details,
        }
    
    def format_for_user(self, verbose: bool = False) -> str:
        """Format the error report for user display."""
        lines = []
        
        severity_icons = {
            ErrorSeverity.INFO: "[INFO]",
            ErrorSeverity.WARNING: "[WARN]",
            ErrorSeverity.ERROR: "[ERROR]",
            ErrorSeverity.CRITICAL: "[CRITICAL]",
        }
        
        lines.append(f"{severity_icons[self.severity]} {self.message}")
        lines.append("")
        
        if self.user_explanation:
            lines.append("What happened:")
            lines.append(f"  {self.user_explanation}")
            lines.append("")
        
        if self.recovery_actions:
            lines.append("Suggested actions:")
            for i, action in enumerate(self.recovery_actions, 1):
                auto_tag = " (auto)" if action.auto_apply else ""
                lines.append(f"  {i}. {action.description}{auto_tag}")
            lines.append("")
        
        if verbose:
            lines.append("Technical details:")
            lines.append(f"  Category: {self.category.value}")
            lines.append(f"  Operation: {self.context.operation}")
            lines.append(f"  Phase: {self.context.phase}")
            if self.context.related_fields:
                lines.append(f"  Related fields: {', '.join(self.context.related_fields)}")
            if self.technical_details:
                lines.append(f"  Details: {self.technical_details}")
        
        return "\n".join(lines)


class ErrorClassifier:
    """Classify errors into categories and suggest recovery strategies."""
    
    PATTERNS = {
        ErrorCategory.VALIDATION: [
            "validation", "invalid", "constraint", "check failed",
            "out of range", "incompatible", "conflict",
        ],
        ErrorCategory.GENERATION: [
            "generation", "create", "build", "grow", "bifurcate",
            "collision", "space colonization", "embedding",
        ],
        ErrorCategory.CONFIGURATION: [
            "config", "parameter", "setting", "option",
            "missing required", "not configured",
        ],
        ErrorCategory.USER_INPUT: [
            "input", "parse", "format", "expected", "invalid value",
            "unrecognized", "unknown command",
        ],
        ErrorCategory.FILE_IO: [
            "file", "read", "write", "path", "directory",
            "permission", "not found", "exists",
        ],
        ErrorCategory.NETWORK: [
            "network", "connection", "timeout", "api", "request",
        ],
    }
    
    RECOVERY_STRATEGIES = {
        ErrorCategory.VALIDATION: [
            RecoveryAction(
                RecoveryStrategy.PROMPT_USER,
                "Review and adjust the conflicting parameters",
            ),
            RecoveryAction(
                RecoveryStrategy.FALLBACK,
                "Use default values for problematic parameters",
                auto_apply=False,
            ),
        ],
        ErrorCategory.GENERATION: [
            RecoveryAction(
                RecoveryStrategy.RETRY,
                "Retry generation with adjusted parameters",
            ),
            RecoveryAction(
                RecoveryStrategy.ROLLBACK,
                "Rollback to last successful state",
            ),
        ],
        ErrorCategory.CONFIGURATION: [
            RecoveryAction(
                RecoveryStrategy.PROMPT_USER,
                "Provide missing configuration values",
            ),
            RecoveryAction(
                RecoveryStrategy.FALLBACK,
                "Use default configuration",
                auto_apply=True,
            ),
        ],
        ErrorCategory.USER_INPUT: [
            RecoveryAction(
                RecoveryStrategy.PROMPT_USER,
                "Re-enter the value in the correct format",
            ),
        ],
        ErrorCategory.FILE_IO: [
            RecoveryAction(
                RecoveryStrategy.RETRY,
                "Retry file operation",
            ),
            RecoveryAction(
                RecoveryStrategy.SKIP,
                "Skip this file and continue",
            ),
        ],
        ErrorCategory.NETWORK: [
            RecoveryAction(
                RecoveryStrategy.RETRY,
                "Retry network request",
                auto_apply=True,
            ),
        ],
    }
    
    @classmethod
    def classify(cls, error: Exception, context: Optional[ErrorContext] = None) -> ErrorReport:
        """
        Classify an error and generate a report.
        
        Parameters
        ----------
        error : Exception
            The error to classify
        context : ErrorContext, optional
            Additional context about the error
            
        Returns
        -------
        ErrorReport
            Classified error report with recovery suggestions
        """
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        category = ErrorCategory.UNKNOWN
        for cat, patterns in cls.PATTERNS.items():
            if any(p in error_str for p in patterns):
                category = cat
                break
        
        severity = cls._determine_severity(error, category)
        
        recovery_actions = cls.RECOVERY_STRATEGIES.get(category, [
            RecoveryAction(
                RecoveryStrategy.PROMPT_USER,
                "Contact support or try a different approach",
            ),
        ])
        
        user_explanation = cls._generate_user_explanation(error, category)
        
        context = context or ErrorContext()
        context.stack_trace = traceback.format_exc()
        
        return ErrorReport(
            error_id=f"{category.value}_{id(error)}",
            message=str(error),
            severity=severity,
            category=category,
            context=context,
            recovery_actions=list(recovery_actions),
            user_explanation=user_explanation,
            technical_details=f"{error_type}: {error}",
        )
    
    @classmethod
    def _determine_severity(cls, error: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Determine error severity."""
        if category == ErrorCategory.SYSTEM:
            return ErrorSeverity.CRITICAL
        
        if isinstance(error, (ValueError, TypeError)):
            return ErrorSeverity.ERROR
        
        if isinstance(error, (Warning,)):
            return ErrorSeverity.WARNING
        
        if category == ErrorCategory.USER_INPUT:
            return ErrorSeverity.WARNING
        
        return ErrorSeverity.ERROR
    
    @classmethod
    def _generate_user_explanation(cls, error: Exception, category: ErrorCategory) -> str:
        """Generate a user-friendly explanation."""
        explanations = {
            ErrorCategory.VALIDATION: (
                "The design parameters don't meet the required constraints. "
                "This often happens when values are incompatible with each other."
            ),
            ErrorCategory.GENERATION: (
                "There was a problem generating the vascular network. "
                "This can happen if the parameters make generation impossible."
            ),
            ErrorCategory.CONFIGURATION: (
                "Some required settings are missing or incorrect. "
                "Please check the configuration values."
            ),
            ErrorCategory.USER_INPUT: (
                "The input couldn't be understood. "
                "Please check the format and try again."
            ),
            ErrorCategory.FILE_IO: (
                "There was a problem reading or writing a file. "
                "Please check file permissions and paths."
            ),
            ErrorCategory.NETWORK: (
                "There was a network communication problem. "
                "Please check your connection and try again."
            ),
            ErrorCategory.UNKNOWN: (
                "An unexpected error occurred. "
                "Please try again or contact support if the problem persists."
            ),
        }
        return explanations.get(category, explanations[ErrorCategory.UNKNOWN])


class ErrorRecoveryManager:
    """
    Manage error recovery for the workflow.
    
    Provides:
    - Error tracking and history
    - Automatic recovery attempts
    - User-guided recovery
    - State rollback capabilities
    """
    
    def __init__(self, max_retries: int = 3, verbose: bool = False):
        """
        Initialize the error recovery manager.
        
        Parameters
        ----------
        max_retries : int
            Maximum automatic retry attempts
        verbose : bool
            Enable verbose error reporting
        """
        self.max_retries = max_retries
        self.verbose = verbose
        self.error_history: List[ErrorReport] = []
        self.retry_counts: Dict[str, int] = {}
        self.state_snapshots: List[Dict[str, Any]] = []
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[ErrorContext] = None,
        auto_recover: bool = True,
    ) -> Tuple[bool, Optional[RecoveryAction]]:
        """
        Handle an error with optional automatic recovery.
        
        Parameters
        ----------
        error : Exception
            The error to handle
        context : ErrorContext, optional
            Additional context
        auto_recover : bool
            Whether to attempt automatic recovery
            
        Returns
        -------
        Tuple[bool, Optional[RecoveryAction]]
            (recovered, action_taken)
        """
        report = ErrorClassifier.classify(error, context)
        self.error_history.append(report)
        
        if not auto_recover:
            return False, None
        
        for action in report.recovery_actions:
            if action.auto_apply:
                retry_key = f"{report.category.value}_{action.strategy.value}"
                current_retries = self.retry_counts.get(retry_key, 0)
                
                if current_retries < self.max_retries:
                    self.retry_counts[retry_key] = current_retries + 1
                    return True, action
        
        return False, None
    
    def save_state_snapshot(self, state: Dict[str, Any]) -> int:
        """
        Save a state snapshot for potential rollback.
        
        Parameters
        ----------
        state : dict
            State to save
            
        Returns
        -------
        int
            Snapshot index
        """
        self.state_snapshots.append(state.copy())
        return len(self.state_snapshots) - 1
    
    def rollback_to_snapshot(self, index: int) -> Optional[Dict[str, Any]]:
        """
        Rollback to a saved state snapshot.
        
        Parameters
        ----------
        index : int
            Snapshot index
            
        Returns
        -------
        Optional[Dict[str, Any]]
            The restored state, or None if index invalid
        """
        if 0 <= index < len(self.state_snapshots):
            return self.state_snapshots[index].copy()
        return None
    
    def get_error_summary(self) -> str:
        """Get a summary of all errors."""
        if not self.error_history:
            return "No errors recorded."
        
        lines = ["ERROR SUMMARY", "=" * 50, ""]
        
        by_category: Dict[ErrorCategory, List[ErrorReport]] = {}
        for report in self.error_history:
            if report.category not in by_category:
                by_category[report.category] = []
            by_category[report.category].append(report)
        
        for category, reports in by_category.items():
            lines.append(f"{category.value.upper()} ({len(reports)} errors):")
            for report in reports[-3:]:
                lines.append(f"  - {report.message[:60]}...")
            lines.append("")
        
        return "\n".join(lines)
    
    def format_error_for_display(self, report: ErrorReport) -> str:
        """Format an error report for display."""
        return report.format_for_user(verbose=self.verbose)
    
    def clear_history(self) -> None:
        """Clear error history and retry counts."""
        self.error_history.clear()
        self.retry_counts.clear()


class ValidationReportFormatter:
    """Format validation reports as clear text explanations."""
    
    @staticmethod
    def format_validation_report(
        validation_result: Dict[str, Any],
        verbose: bool = False,
    ) -> str:
        """
        Format a validation result as a text report.
        
        Parameters
        ----------
        validation_result : dict
            Validation result dictionary
        verbose : bool
            Include detailed explanations
            
        Returns
        -------
        str
            Formatted text report
        """
        lines = []
        
        passed = validation_result.get("passed", False)
        status = "PASSED" if passed else "FAILED"
        status_icon = "[OK]" if passed else "[FAIL]"
        
        lines.append("=" * 60)
        lines.append(f"VALIDATION REPORT - {status_icon} {status}")
        lines.append("=" * 60)
        lines.append("")
        
        checks = validation_result.get("checks", [])
        if checks:
            lines.append("CHECK RESULTS:")
            lines.append("-" * 40)
            
            for check in checks:
                name = check.get("name", "Unknown")
                check_passed = check.get("passed", False)
                icon = "[OK]" if check_passed else "[FAIL]"
                lines.append(f"  {icon} {name}")
                
                if verbose or not check_passed:
                    message = check.get("message", "")
                    if message:
                        lines.append(f"      {message}")
                    
                    details = check.get("details", {})
                    if details and verbose:
                        for key, value in details.items():
                            lines.append(f"      {key}: {value}")
            
            lines.append("")
        
        errors = validation_result.get("errors", [])
        if errors:
            lines.append("ERRORS:")
            lines.append("-" * 40)
            for error in errors:
                lines.append(f"  [!] {error}")
            lines.append("")
        
        warnings = validation_result.get("warnings", [])
        if warnings:
            lines.append("WARNINGS:")
            lines.append("-" * 40)
            for warning in warnings:
                lines.append(f"  [?] {warning}")
            lines.append("")
        
        if not passed:
            lines.append("SUGGESTED FIXES:")
            lines.append("-" * 40)
            
            for error in errors[:3]:
                suggestion = ValidationReportFormatter._suggest_fix(error)
                if suggestion:
                    lines.append(f"  -> {suggestion}")
            
            lines.append("")
        
        summary = validation_result.get("summary", {})
        if summary:
            lines.append("SUMMARY:")
            lines.append("-" * 40)
            for key, value in summary.items():
                lines.append(f"  {key}: {value}")
            lines.append("")
        
        return "\n".join(lines)
    
    @staticmethod
    def _suggest_fix(error: str) -> str:
        """Suggest a fix for a validation error."""
        error_lower = error.lower()
        
        suggestions = {
            "collision": "Reduce inlet radius or increase domain size",
            "radius": "Adjust radius values to be within valid range",
            "connectivity": "Check inlet/outlet positions are compatible",
            "wall thickness": "Increase minimum wall thickness parameter",
            "overhang": "Adjust geometry to reduce overhang angles",
            "murray": "Check branching radii follow Murray's law",
            "pressure": "Verify inlet/outlet radius ratio",
        }
        
        for keyword, suggestion in suggestions.items():
            if keyword in error_lower:
                return suggestion
        
        return "Review the error details and adjust parameters"


def create_verbose_step_explanation(
    step_name: str,
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    success: bool,
    error: Optional[str] = None,
) -> str:
    """
    Create a verbose step-by-step explanation.
    
    Parameters
    ----------
    step_name : str
        Name of the step
    inputs : dict
        Input parameters
    outputs : dict
        Output results
    success : bool
        Whether step succeeded
    error : str, optional
        Error message if failed
        
    Returns
    -------
    str
        Formatted explanation
    """
    lines = []
    
    status = "[OK]" if success else "[FAIL]"
    lines.append(f"STEP: {step_name} {status}")
    lines.append("-" * 50)
    
    lines.append("Inputs:")
    for key, value in inputs.items():
        value_str = str(value)[:50]
        lines.append(f"  {key}: {value_str}")
    
    lines.append("")
    
    if success:
        lines.append("Outputs:")
        for key, value in outputs.items():
            value_str = str(value)[:50]
            lines.append(f"  {key}: {value_str}")
    else:
        lines.append(f"Error: {error}")
    
    lines.append("")
    
    return "\n".join(lines)
