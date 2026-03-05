"""
DesignSpec Directive - Structured LLM Output Contract

This module defines the DesignSpecDirective dataclass that represents
the structured output from the LLM agent. All LLM responses must conform
to this schema and are validated before use.

The directive controls:
- What message to show the user
- What questions to ask
- What JSON Patch operations to propose
- When to run the pipeline
- What additional context is needed
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class Question:
    """A clarifying question to ask the user."""
    id: str
    question: str
    why_needed: str = ""
    default: Optional[Union[str, int, float, Dict, List]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "why_needed": self.why_needed,
            "default": self.default,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Question":
        return cls(
            id=d.get("id", ""),
            question=d.get("question", ""),
            why_needed=d.get("why_needed", ""),
            default=d.get("default"),
        )


@dataclass
class RunRequest:
    """A request to run the DesignSpec pipeline."""
    run: bool = False
    run_until: Optional[str] = None
    reason: str = ""
    expected_signal: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "run": self.run,
            "run_until": self.run_until,
            "reason": self.reason,
            "expected_signal": self.expected_signal,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunRequest":
        return cls(
            run=d.get("run", False),
            run_until=d.get("run_until"),
            reason=d.get("reason", ""),
            expected_signal=d.get("expected_signal", ""),
        )


@dataclass
class ContextRequest:
    """A request for additional context from the system."""
    need_full_spec: bool = False
    need_last_run_report: bool = False
    need_validity_report: bool = False
    need_network_artifact: bool = False
    need_specific_files: List[str] = field(default_factory=list)
    need_more_history: bool = False
    why: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "need_full_spec": self.need_full_spec,
            "need_last_run_report": self.need_last_run_report,
            "need_validity_report": self.need_validity_report,
            "need_network_artifact": self.need_network_artifact,
            "need_specific_files": self.need_specific_files,
            "need_more_history": self.need_more_history,
            "why": self.why,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ContextRequest":
        return cls(
            need_full_spec=d.get("need_full_spec", False),
            need_last_run_report=d.get("need_last_run_report", False),
            need_validity_report=d.get("need_validity_report", False),
            need_network_artifact=d.get("need_network_artifact", False),
            need_specific_files=d.get("need_specific_files", []),
            need_more_history=d.get("need_more_history", False),
            why=d.get("why", ""),
        )
    
    def has_requests(self) -> bool:
        """Check if any context is being requested."""
        return (
            self.need_full_spec or
            self.need_last_run_report or
            self.need_validity_report or
            self.need_network_artifact or
            bool(self.need_specific_files) or
            self.need_more_history
        )


# Valid RFC 6902 JSON Patch operations
VALID_PATCH_OPS = {"add", "remove", "replace", "move", "copy", "test"}

# Known pipeline stages
PIPELINE_STAGES = [
    "compile_policies",
    "compile_domains",
    "component_ports",
    "component_build",
    "component_mesh",
    "union_voids",
    "mesh_domain",
    "embed",
    "port_recarve",
    "validity",
    "export",
    "full",
]


def validate_json_patch(patch: Dict[str, Any]) -> List[str]:
    """
    Validate a single JSON Patch operation (RFC 6902).
    
    Parameters
    ----------
    patch : dict
        A single patch operation
        
    Returns
    -------
    list of str
        List of validation error messages (empty if valid)
    """
    errors = []
    
    if not isinstance(patch, dict):
        return ["Patch must be a dictionary"]
    
    op = patch.get("op")
    if not op:
        errors.append("Patch missing 'op' field")
    elif op not in VALID_PATCH_OPS:
        errors.append(f"Invalid patch operation: '{op}'. Must be one of: {VALID_PATCH_OPS}")
    
    path = patch.get("path")
    if path is None:
        errors.append("Patch missing 'path' field")
    elif not isinstance(path, str):
        errors.append("Patch 'path' must be a string")
    elif path and not path.startswith("/"):
        errors.append(f"Patch 'path' must start with '/': got '{path}'")
    
    if op in ("add", "replace", "test") and "value" not in patch:
        errors.append(f"Patch operation '{op}' requires 'value' field")
    
    if op in ("copy", "move") and "from" not in patch:
        errors.append(f"Patch operation '{op}' requires 'from' field")
    
    return errors


def validate_patches(patches: List[Dict[str, Any]]) -> List[str]:
    """
    Validate a list of JSON Patch operations.
    
    Parameters
    ----------
    patches : list of dict
        List of patch operations
        
    Returns
    -------
    list of str
        List of validation error messages (empty if all valid)
    """
    errors = []
    
    if not isinstance(patches, list):
        return ["proposed_patches must be a list"]
    
    for i, patch in enumerate(patches):
        patch_errors = validate_json_patch(patch)
        for err in patch_errors:
            errors.append(f"Patch {i}: {err}")
    
    return errors


@dataclass
class DesignSpecDirective:
    """
    Structured output from the LLM agent for DesignSpec workflows.
    
    This is the contract between the LLM and the system. All LLM responses
    must be parsed into this structure and validated before use.
    
    Attributes
    ----------
    assistant_message : str
        The message to display to the user
    questions : list of Question
        Optional clarifying questions to ask
    proposed_patches : list of dict
        Optional RFC 6902 JSON Patch operations to propose
    run_request : RunRequest, optional
        Optional request to run the pipeline
    context_requests : ContextRequest, optional
        Optional request for additional context
    confidence : float
        Confidence level (0-1) in the proposed action
    requires_approval : bool
        Whether user approval is required before applying patches/running
    stop : bool
        Whether the agent believes the workflow is complete
    """
    assistant_message: str = ""
    questions: List[Question] = field(default_factory=list)
    proposed_patches: List[Dict[str, Any]] = field(default_factory=list)
    run_request: Optional[RunRequest] = None
    context_requests: Optional[ContextRequest] = None
    confidence: float = 0.5
    requires_approval: bool = True
    stop: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "assistant_message": self.assistant_message,
            "questions": [q.to_dict() for q in self.questions],
            "proposed_patches": self.proposed_patches,
            "run_request": self.run_request.to_dict() if self.run_request else None,
            "context_requests": self.context_requests.to_dict() if self.context_requests else None,
            "confidence": self.confidence,
            "requires_approval": self.requires_approval,
            "stop": self.stop,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DesignSpecDirective":
        """
        Create a DesignSpecDirective from a dictionary.
        
        Parameters
        ----------
        d : dict
            Dictionary representation of the directive
            
        Returns
        -------
        DesignSpecDirective
            The parsed directive
        """
        questions = [Question.from_dict(q) for q in d.get("questions", [])]
        
        run_request = None
        if d.get("run_request"):
            run_request = RunRequest.from_dict(d["run_request"])
        
        context_requests = None
        if d.get("context_requests"):
            context_requests = ContextRequest.from_dict(d["context_requests"])
        
        return cls(
            assistant_message=d.get("assistant_message", ""),
            questions=questions,
            proposed_patches=d.get("proposed_patches", []),
            run_request=run_request,
            context_requests=context_requests,
            confidence=d.get("confidence", 0.5),
            requires_approval=d.get("requires_approval", True),
            stop=d.get("stop", False),
        )
    
    def has_patches(self) -> bool:
        """Check if the directive proposes any patches."""
        return bool(self.proposed_patches)
    
    def has_questions(self) -> bool:
        """Check if the directive has questions to ask."""
        return bool(self.questions)
    
    def has_run_request(self) -> bool:
        """Check if the directive requests a pipeline run."""
        return self.run_request is not None and self.run_request.run
    
    def needs_more_context(self) -> bool:
        """Check if the directive requests additional context."""
        return self.context_requests is not None and self.context_requests.has_requests()


class DirectiveParseError(Exception):
    """Exception raised when directive parsing fails."""
    
    def __init__(self, message: str, raw_text: str = "", errors: Optional[List[str]] = None):
        super().__init__(message)
        self.raw_text = raw_text
        self.errors = errors or []


def extract_json_from_text(text: str) -> str:
    """
    Extract JSON from LLM output text.
    
    Handles common cases:
    - Pure JSON
    - JSON wrapped in markdown code blocks
    - JSON with leading/trailing text
    
    Parameters
    ----------
    text : str
        Raw LLM output text
        
    Returns
    -------
    str
        Extracted JSON string
        
    Raises
    ------
    DirectiveParseError
        If no valid JSON can be extracted
    """
    text = text.strip()
    
    # Try to parse as-is first
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass
    
    # Try to extract from markdown code blocks
    code_block_patterns = [
        r"```json\s*([\s\S]*?)\s*```",
        r"```\s*([\s\S]*?)\s*```",
    ]
    
    for pattern in code_block_patterns:
        match = re.search(pattern, text)
        if match:
            candidate = match.group(1).strip()
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                continue
    
    # Try to find JSON object boundaries
    start_idx = text.find("{")
    if start_idx != -1:
        # Find matching closing brace
        depth = 0
        in_string = False
        escape_next = False
        
        for i, char in enumerate(text[start_idx:], start_idx):
            if escape_next:
                escape_next = False
                continue
            
            if char == "\\":
                escape_next = True
                continue
            
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            
            if in_string:
                continue
            
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start_idx:i + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        break
    
    raise DirectiveParseError(
        "Could not extract valid JSON from LLM output",
        raw_text=text,
        errors=["No valid JSON object found in response"],
    )


def validate_directive(d: Dict[str, Any]) -> List[str]:
    """
    Validate a directive dictionary.
    
    Parameters
    ----------
    d : dict
        Directive dictionary to validate
        
    Returns
    -------
    list of str
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Required field: assistant_message
    if "assistant_message" not in d:
        errors.append("Missing required field: assistant_message")
    elif not isinstance(d["assistant_message"], str):
        errors.append("assistant_message must be a string")
    
    # Optional field: questions
    if "questions" in d:
        if not isinstance(d["questions"], list):
            errors.append("questions must be a list")
        else:
            for i, q in enumerate(d["questions"]):
                if not isinstance(q, dict):
                    errors.append(f"questions[{i}] must be a dictionary")
                elif "question" not in q:
                    errors.append(f"questions[{i}] missing 'question' field")
    
    # Optional field: proposed_patches
    if "proposed_patches" in d:
        patch_errors = validate_patches(d["proposed_patches"])
        errors.extend(patch_errors)
    
    # Optional field: run_request
    if "run_request" in d and d["run_request"] is not None:
        rr = d["run_request"]
        if not isinstance(rr, dict):
            errors.append("run_request must be a dictionary")
        else:
            if "run_until" in rr and rr["run_until"] is not None:
                if rr["run_until"] not in PIPELINE_STAGES:
                    errors.append(
                        f"Invalid run_until stage: '{rr['run_until']}'. "
                        f"Must be one of: {PIPELINE_STAGES}"
                    )
    
    # Optional field: confidence
    if "confidence" in d:
        conf = d["confidence"]
        if not isinstance(conf, (int, float)):
            errors.append("confidence must be a number")
        elif conf < 0 or conf > 1:
            errors.append("confidence must be between 0 and 1")
    
    # Optional field: requires_approval
    if "requires_approval" in d:
        if not isinstance(d["requires_approval"], bool):
            errors.append("requires_approval must be a boolean")
    
    # Optional field: stop
    if "stop" in d:
        if not isinstance(d["stop"], bool):
            errors.append("stop must be a boolean")
    
    return errors


def from_json(text: str) -> DesignSpecDirective:
    """
    Parse LLM output text into a DesignSpecDirective.
    
    This is the main entry point for parsing LLM responses. It:
    1. Extracts JSON from the text (handling markdown, etc.)
    2. Validates the JSON structure
    3. Validates patch operations
    4. Returns a validated DesignSpecDirective
    
    Parameters
    ----------
    text : str
        Raw LLM output text
        
    Returns
    -------
    DesignSpecDirective
        Validated directive
        
    Raises
    ------
    DirectiveParseError
        If parsing or validation fails
    """
    # Extract JSON
    try:
        json_str = extract_json_from_text(text)
    except DirectiveParseError:
        raise
    
    # Parse JSON
    try:
        d = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise DirectiveParseError(
            f"Invalid JSON: {e}",
            raw_text=text,
            errors=[str(e)],
        )
    
    # Validate structure
    errors = validate_directive(d)
    if errors:
        raise DirectiveParseError(
            f"Directive validation failed: {errors[0]}",
            raw_text=text,
            errors=errors,
        )
    
    # Create directive
    return DesignSpecDirective.from_dict(d)


def create_error_directive(message: str, error_details: str = "") -> DesignSpecDirective:
    """
    Create a directive for error cases.
    
    Parameters
    ----------
    message : str
        Error message to show the user
    error_details : str
        Additional error details (for logging)
        
    Returns
    -------
    DesignSpecDirective
        A directive with the error message
    """
    full_message = message
    if error_details:
        full_message = f"{message}\n\nDetails: {error_details}"
    
    return DesignSpecDirective(
        assistant_message=full_message,
        confidence=0.0,
        requires_approval=False,
        stop=False,
    )


def create_fallback_directive(user_message: str) -> DesignSpecDirective:
    """
    Create a fallback directive when LLM is unavailable.
    
    Parameters
    ----------
    user_message : str
        The user's original message
        
    Returns
    -------
    DesignSpecDirective
        A directive asking for clarification
    """
    return DesignSpecDirective(
        assistant_message=(
            "I'm currently unable to process your request through the LLM. "
            "Please try again or provide more specific instructions."
        ),
        questions=[
            Question(
                id="retry_or_clarify",
                question="Would you like to retry or provide more details?",
                why_needed="LLM processing is temporarily unavailable",
            )
        ],
        confidence=0.0,
        requires_approval=False,
        stop=False,
    )
