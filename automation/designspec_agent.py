"""
DesignSpec Agent for conversation-driven spec editing.

This module provides the DesignSpecAgent class that:
- Analyzes current spec and validation errors
- Generates clarifying questions for missing/ambiguous fields
- Proposes JSON Patch operations to modify the spec
- Requests run operations when spec is ready

IMPORTANT: This agent MUST NOT emit Python code as a solution.
All modifications are expressed as JSON Patch operations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import json
import re
import logging

from .designspec_session import PatchProposal, ValidationReport

logger = logging.getLogger(__name__)


class AgentResponseType(str, Enum):
    """Type of agent response."""
    QUESTION = "question"
    PATCH_PROPOSAL = "patch_proposal"
    RUN_REQUEST = "run_request"
    MESSAGE = "message"
    ERROR = "error"


@dataclass
class Question:
    """A clarifying question to ask the user."""
    field_path: str
    question_text: str
    field_type: str = "string"
    options: Optional[List[str]] = None
    default: Optional[Any] = None
    reason: str = ""
    priority: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_path": self.field_path,
            "question_text": self.question_text,
            "field_type": self.field_type,
            "options": self.options,
            "default": self.default,
            "reason": self.reason,
            "priority": self.priority,
        }


@dataclass
class RunRequest:
    """A request to run the DesignSpecRunner."""
    run_until: Optional[str] = None
    full_run: bool = False
    reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_until": self.run_until,
            "full_run": self.full_run,
            "reason": self.reason,
        }


@dataclass
class AgentResponse:
    """Response from the DesignSpecAgent."""
    response_type: AgentResponseType
    message: str = ""
    questions: List[Question] = field(default_factory=list)
    patch_proposal: Optional[PatchProposal] = None
    run_request: Optional[RunRequest] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "response_type": self.response_type.value,
            "message": self.message,
        }
        if self.questions:
            result["questions"] = [q.to_dict() for q in self.questions]
        if self.patch_proposal:
            result["patch_proposal"] = self.patch_proposal.to_dict()
        if self.run_request:
            result["run_request"] = self.run_request.to_dict()
        return result


REQUIRED_SPEC_FIELDS = {
    "meta.seed": {
        "type": "integer",
        "question": "What seed should be used for reproducibility?",
        "default": 42,
        "reason": "Seed ensures reproducible results across runs",
    },
    "meta.input_units": {
        "type": "string",
        "question": "What units are your measurements in?",
        "options": ["mm", "m", "um"],
        "default": "mm",
        "reason": "Units are needed to correctly interpret all dimensions",
    },
}

DOMAIN_FIELDS = {
    "type": {
        "type": "string",
        "question": "What shape should the domain be?",
        "options": ["box", "ellipsoid", "cylinder"],
        "reason": "Domain type determines the outer boundary shape",
    },
    "center": {
        "type": "array",
        "question": "Where should the domain be centered? (x, y, z coordinates)",
        "default": [0, 0, 0],
        "reason": "Center position for the domain",
    },
    "size": {
        "type": "array",
        "question": "What are the domain dimensions? (width, height, depth)",
        "reason": "Size determines the overall scale of the generated structure",
    },
}

COMPONENT_FIELDS = {
    "id": {
        "type": "string",
        "question": "What identifier should this component have?",
        "reason": "Unique ID for referencing this component",
    },
    "domain_ref": {
        "type": "string",
        "question": "Which domain should this component use?",
        "default": "main_domain",
        "reason": "Links component to its containing domain",
    },
    "build.backend": {
        "type": "string",
        "question": "What generation algorithm should be used?",
        "options": ["space_colonization", "cco_hybrid", "primitive"],
        "default": "space_colonization",
        "reason": "Backend determines the network generation algorithm",
    },
}

RUN_STAGES = [
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
]


class DesignSpecAgent:
    """
    Agent for conversation-driven DesignSpec editing.
    
    Analyzes spec state and user messages to produce:
    - Clarifying questions for missing/ambiguous fields
    - JSON Patch proposals for spec modifications
    - Run requests when spec is ready
    
    IMPORTANT: This agent MUST NOT emit Python code as a solution.
    """
    
    def __init__(self, llm_client: Optional[Any] = None):
        """
        Initialize the agent.
        
        Parameters
        ----------
        llm_client : LLMClient, optional
            LLM client for natural language understanding.
            If None, uses rule-based logic only.
        """
        self.llm_client = llm_client
        self._conversation_history: List[Dict[str, str]] = []
    
    def process_message(
        self,
        user_message: str,
        spec: Dict[str, Any],
        validation_report: Optional[ValidationReport] = None,
        compile_report: Optional[Dict[str, Any]] = None,
    ) -> AgentResponse:
        """
        Process a user message and generate a response.
        
        Parameters
        ----------
        user_message : str
            The user's message
        spec : dict
            Current spec dictionary
        validation_report : ValidationReport, optional
            Latest validation report
        compile_report : dict, optional
            Latest compile report
            
        Returns
        -------
        AgentResponse
            Agent's response (questions, patch proposal, or run request)
        """
        self._conversation_history.append({
            "role": "user",
            "content": user_message,
        })
        
        message_lower = user_message.lower().strip()
        
        if self._is_run_request(message_lower):
            return self._handle_run_request(message_lower, spec, validation_report)
        
        if self._is_question_answer(message_lower, spec):
            return self._handle_question_answer(user_message, spec)
        
        patches = self._extract_patches_from_message(user_message, spec)
        if patches:
            explanation = self._generate_patch_explanation(user_message, patches, spec)
            return AgentResponse(
                response_type=AgentResponseType.PATCH_PROPOSAL,
                message=explanation,
                patch_proposal=PatchProposal(
                    explanation=explanation,
                    patches=patches,
                    confidence=0.8,
                    requires_confirmation=True,
                ),
            )
        
        if validation_report and not validation_report.valid:
            return self._handle_validation_errors(validation_report, spec)
        
        missing_fields = self._find_missing_required_fields(spec)
        if missing_fields:
            questions = self._generate_questions_for_fields(missing_fields, spec)
            if questions:
                checklist = self._build_missing_fields_checklist(spec)
                message = self._generate_conversational_question_prompt(questions, checklist, spec)
                return AgentResponse(
                    response_type=AgentResponseType.QUESTION,
                    message=message,
                    questions=questions,
                )
        
        if self.llm_client:
            return self._llm_process_message(user_message, spec, validation_report)
        
        return self._generate_conversational_fallback(user_message, spec)
    
    def _is_run_request(self, message: str) -> bool:
        """Check if the message is a run request."""
        run_keywords = [
            "run", "execute", "generate", "build", "start",
            "run until", "run to", "run through",
        ]
        return any(kw in message for kw in run_keywords)
    
    def _is_question_answer(self, message: str, spec: Dict[str, Any]) -> bool:
        """Check if the message appears to be an answer to a previous question."""
        if not self._conversation_history:
            return False
        
        for entry in reversed(self._conversation_history[:-1]):
            if entry.get("role") == "assistant":
                content = entry.get("content", "").lower()
                if "?" in content:
                    return True
                break
        
        return False
    
    def _handle_run_request(
        self,
        message: str,
        spec: Dict[str, Any],
        validation_report: Optional[ValidationReport],
    ) -> AgentResponse:
        """Handle a run request from the user."""
        if validation_report and not validation_report.valid:
            return AgentResponse(
                response_type=AgentResponseType.ERROR,
                message="Cannot run: the spec has validation errors that must be fixed first.",
            )
        
        missing = self._find_critical_missing_fields(spec)
        if missing:
            questions = self._generate_questions_for_fields(missing[:3], spec)
            return AgentResponse(
                response_type=AgentResponseType.QUESTION,
                message=f"Before running, I need some required information: {', '.join(missing)}",
                questions=questions,
            )
        
        run_until = None
        full_run = False
        
        for stage in RUN_STAGES:
            stage_name = stage.replace("_", " ")
            if stage_name in message or stage in message:
                run_until = stage
                break
        
        if "full" in message or "complete" in message or "all" in message:
            full_run = True
            run_until = None
        elif run_until is None:
            run_until = "union_voids"
        
        return AgentResponse(
            response_type=AgentResponseType.RUN_REQUEST,
            message=f"Ready to run the pipeline{' until ' + run_until if run_until else ' (full run)'}.",
            run_request=RunRequest(
                run_until=run_until,
                full_run=full_run,
                reason=f"User requested: {message}",
            ),
        )
    
    def _handle_question_answer(
        self,
        message: str,
        spec: Dict[str, Any],
    ) -> AgentResponse:
        """Handle an answer to a previous question."""
        patches = self._extract_patches_from_message(message, spec)
        
        if patches:
            return AgentResponse(
                response_type=AgentResponseType.PATCH_PROPOSAL,
                message="I'll update the spec with your answer.",
                patch_proposal=PatchProposal(
                    explanation=f"Update based on answer: {message}",
                    patches=patches,
                    confidence=0.9,
                    requires_confirmation=True,
                ),
            )
        
        return AgentResponse(
            response_type=AgentResponseType.MESSAGE,
            message="I'm not sure how to apply that answer. Could you be more specific?",
        )
    
    def _handle_validation_errors(
        self,
        validation_report: ValidationReport,
        spec: Dict[str, Any],
    ) -> AgentResponse:
        """Handle validation errors by generating questions or patches."""
        questions = []
        
        for error in validation_report.errors:
            if "missing" in error.lower():
                field_match = re.search(r"'(\w+)'", error)
                if field_match:
                    field_name = field_match.group(1)
                    questions.append(Question(
                        field_path=field_name,
                        question_text=f"The spec is missing '{field_name}'. What value should it have?",
                        reason=error,
                        priority=10,
                    ))
        
        if questions:
            return AgentResponse(
                response_type=AgentResponseType.QUESTION,
                message="The spec has validation errors. Let me help fix them.",
                questions=questions,
            )
        
        return AgentResponse(
            response_type=AgentResponseType.ERROR,
            message=f"Validation errors: {'; '.join(validation_report.errors)}",
        )
    
    def _find_missing_required_fields(self, spec: Dict[str, Any]) -> List[str]:
        """Find required fields that are missing from the spec."""
        missing = []
        
        meta = spec.get("meta", {})
        if "seed" not in meta:
            missing.append("meta.seed")
        
        domains = spec.get("domains", {})
        if not domains:
            missing.append("domains")
        else:
            for domain_name, domain in domains.items():
                if "type" not in domain:
                    missing.append(f"domains.{domain_name}.type")
        
        components = spec.get("components", [])
        if not components:
            missing.append("components")
        else:
            for i, comp in enumerate(components):
                if "id" not in comp:
                    missing.append(f"components.{i}.id")
                if "domain_ref" not in comp:
                    missing.append(f"components.{i}.domain_ref")
        
        return missing
    
    def _find_critical_missing_fields(self, spec: Dict[str, Any]) -> List[str]:
        """Find critical fields that must be present before running."""
        missing = []
        
        domains = spec.get("domains", {})
        if not domains:
            missing.append("domains (at least one domain required)")
        
        components = spec.get("components", [])
        if not components:
            missing.append("components (at least one component required)")
        else:
            for i, comp in enumerate(components):
                ports = comp.get("ports", {})
                inlets = ports.get("inlets", [])
                if not inlets:
                    missing.append(f"components[{i}].ports.inlets")
        
        return missing
    
    def _build_missing_fields_checklist(self, spec: Dict[str, Any]) -> str:
        """Build a checklist of missing fields for user display."""
        lines = []
        
        domains = spec.get("domains", {})
        if domains:
            lines.append("[x] domains - defined")
            for name, domain in domains.items():
                dtype = domain.get("type", "unknown")
                lines.append(f"    [x] {name}: {dtype}")
        else:
            lines.append("[ ] domains - need at least one domain?")
        
        components = spec.get("components", [])
        if components:
            lines.append("[x] components - defined")
            for comp in components:
                comp_id = comp.get("id", "unnamed")
                ports = comp.get("ports", {})
                inlets = ports.get("inlets", [])
                outlets = ports.get("outlets", [])
                if inlets:
                    lines.append(f"    [x] {comp_id}: {len(inlets)} inlet(s)")
                else:
                    lines.append(f"    [ ] {comp_id}: needs inlet ports?")
        else:
            lines.append("[ ] components - need at least one component?")
        
        features = spec.get("features", {})
        ridges = features.get("ridges", [])
        if ridges:
            faces = [r.get("face", "?") for r in ridges]
            lines.append(f"[x] features.ridges - {len(ridges)} ridge(s) on {', '.join(faces)}")
        
        return "\n".join(lines)
    
    def _generate_questions_for_fields(
        self,
        field_paths: List[str],
        spec: Dict[str, Any],
    ) -> List[Question]:
        """Generate questions for missing fields."""
        questions = []
        
        for field_path in field_paths:
            if field_path in REQUIRED_SPEC_FIELDS:
                field_info = REQUIRED_SPEC_FIELDS[field_path]
                questions.append(Question(
                    field_path=field_path,
                    question_text=field_info["question"],
                    field_type=field_info["type"],
                    options=field_info.get("options"),
                    default=field_info.get("default"),
                    reason=field_info.get("reason", ""),
                    priority=5,
                ))
            elif field_path == "domains":
                questions.append(Question(
                    field_path="domains.main_domain",
                    question_text="What shape and size should the domain be? "
                                  "(e.g., 'box 20mm x 60mm x 30mm' or 'cylinder radius 10mm height 20mm')",
                    field_type="domain",
                    reason="A domain defines the outer boundary for generation",
                    priority=10,
                ))
            elif field_path == "components":
                questions.append(Question(
                    field_path="components",
                    question_text="What type of vascular structure do you want to generate? "
                                  "(e.g., 'a branching tree with inlet on top face')",
                    field_type="component",
                    reason="Components define the vascular networks to generate",
                    priority=10,
                ))
            elif "inlets" in field_path:
                questions.append(Question(
                    field_path=field_path,
                    question_text="Where should the inlet be located? "
                                  "(e.g., 'top face center' or 'position [0, 0, 15] with radius 1mm')",
                    field_type="port",
                    reason="Inlets define where fluid enters the network",
                    priority=8,
                ))
            else:
                questions.append(Question(
                    field_path=field_path,
                    question_text=f"What value should '{field_path}' have?",
                    field_type="string",
                    reason=f"Required field: {field_path}",
                    priority=3,
                ))
        
        questions.sort(key=lambda q: -q.priority)
        return questions
    
    def _extract_patches_from_message(
        self,
        message: str,
        spec: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Extract JSON Patch operations from a user message."""
        patches = []
        message_lower = message.lower()
        
        seed_match = re.search(r"seed\s*[=:]\s*(\d+)", message_lower)
        if seed_match:
            seed_value = int(seed_match.group(1))
            if "meta" in spec and "seed" in spec["meta"]:
                patches.append({
                    "op": "replace",
                    "path": "/meta/seed",
                    "value": seed_value,
                })
            else:
                patches.append({
                    "op": "add",
                    "path": "/meta/seed",
                    "value": seed_value,
                })
        
        name_match = re.search(r"name\s*[=:]\s*[\"']?([^\"'\n,]+)[\"']?", message, re.IGNORECASE)
        if name_match:
            name_value = name_match.group(1).strip()
            if "meta" in spec and "name" in spec["meta"]:
                patches.append({
                    "op": "replace",
                    "path": "/meta/name",
                    "value": name_value,
                })
            else:
                patches.append({
                    "op": "add",
                    "path": "/meta/name",
                    "value": name_value,
                })
        
        box_match = re.search(
            r"box\s+(\d+(?:\.\d+)?)\s*(?:mm|m)?\s*[x×]\s*(\d+(?:\.\d+)?)\s*(?:mm|m)?\s*[x×]\s*(\d+(?:\.\d+)?)\s*(?:mm|m)?",
            message_lower,
        )
        cube_patterns = [
            r"(?:box|cube)\s+(?:with\s+)?(\d+(?:\.\d+)?)\s*(?:mm|m)?\s+sides?",
            r"(\d+(?:\.\d+)?)\s*(?:mm|m)?\s+(?:box|cube)",
            r"(?:box|cube)\s+(?:should\s+be\s+|is\s+)?(\d+(?:\.\d+)?)\s*(?:mm|m)?\s+(?:on\s+)?(?:all|each|every)\s+sides?",
            r"(?:box|cube)\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*(?:mm|m)?\s+(?:on\s+)?(?:all|each|every)\s+sides?",
            r"(\d+(?:\.\d+)?)\s*(?:mm|m)?\s+(?:on\s+)?(?:all|each|every)\s+sides?",
        ]
        cube_match = None
        for pattern in cube_patterns:
            cube_match = re.search(pattern, message_lower)
            if cube_match:
                break
        if box_match:
            width = float(box_match.group(1))
            height = float(box_match.group(2))
            depth = float(box_match.group(3))
            
            domain_value = {
                "type": "box",
                "x_min": -width / 2,
                "x_max": width / 2,
                "y_min": -height / 2,
                "y_max": height / 2,
                "z_min": -depth / 2,
                "z_max": depth / 2,
            }
            
            if "domains" not in spec or not spec["domains"]:
                patches.append({
                    "op": "add",
                    "path": "/domains",
                    "value": {"main_domain": domain_value},
                })
            elif "main_domain" not in spec.get("domains", {}):
                patches.append({
                    "op": "add",
                    "path": "/domains/main_domain",
                    "value": domain_value,
                })
            else:
                patches.append({
                    "op": "replace",
                    "path": "/domains/main_domain",
                    "value": domain_value,
                })
        elif cube_match:
            side_length = float(cube_match.group(1))
            half_side = side_length / 2
            
            domain_value = {
                "type": "box",
                "x_min": -half_side,
                "x_max": half_side,
                "y_min": -half_side,
                "y_max": half_side,
                "z_min": -half_side,
                "z_max": half_side,
            }
            
            if "domains" not in spec or not spec["domains"]:
                patches.append({
                    "op": "add",
                    "path": "/domains",
                    "value": {"main_domain": domain_value},
                })
            elif "main_domain" not in spec.get("domains", {}):
                patches.append({
                    "op": "add",
                    "path": "/domains/main_domain",
                    "value": domain_value,
                })
            else:
                patches.append({
                    "op": "replace",
                    "path": "/domains/main_domain",
                    "value": domain_value,
                })
        
        cylinder_match = re.search(
            r"cylinder\s+(?:radius\s+)?(\d+(?:\.\d+)?)\s*(?:mm|m)?\s+(?:height\s+)?(\d+(?:\.\d+)?)\s*(?:mm|m)?",
            message_lower,
        )
        if cylinder_match:
            radius = float(cylinder_match.group(1))
            height = float(cylinder_match.group(2))
            
            domain_value = {
                "type": "cylinder",
                "center": [0, 0, 0],
                "radius": radius,
                "height": height,
            }
            
            if "domains" not in spec or not spec["domains"]:
                patches.append({
                    "op": "add",
                    "path": "/domains",
                    "value": {"main_domain": domain_value},
                })
            elif "main_domain" not in spec.get("domains", {}):
                patches.append({
                    "op": "add",
                    "path": "/domains/main_domain",
                    "value": domain_value,
                })
            else:
                patches.append({
                    "op": "replace",
                    "path": "/domains/main_domain",
                    "value": domain_value,
                })
        
        if "inlet" in message_lower or "outlet" in message_lower:
            port_patches = self._extract_port_patches(message, spec)
            patches.extend(port_patches)
        
        if "tree" in message_lower or "network" in message_lower or "vascular" in message_lower:
            component_patches = self._extract_component_patches(message, spec)
            patches.extend(component_patches)
        
        if "channel" in message_lower and ("straight" in message_lower or "through" in message_lower):
            channel_patches = self._extract_channel_patches(message, spec)
            patches.extend(channel_patches)
        
        if "ridge" in message_lower:
            ridge_patches = self._extract_ridge_patches(message, spec)
            patches.extend(ridge_patches)
        
        return patches
    
    def _extract_port_patches(
        self,
        message: str,
        spec: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Extract port-related patches from a message."""
        patches = []
        message_lower = message.lower()
        
        face_map = {
            "top": "+z",
            "bottom": "-z",
            "front": "-y",
            "back": "+y",
            "left": "-x",
            "right": "+x",
        }
        
        inlet_face = None
        outlet_face = None
        
        for face_name, face_code in face_map.items():
            if f"inlet" in message_lower and face_name in message_lower:
                if f"inlet on {face_name}" in message_lower or f"inlet at {face_name}" in message_lower or f"{face_name} inlet" in message_lower:
                    inlet_face = face_code
            if f"outlet" in message_lower and face_name in message_lower:
                if f"outlet on {face_name}" in message_lower or f"outlet at {face_name}" in message_lower or f"{face_name} outlet" in message_lower:
                    outlet_face = face_code
        
        radius_match = re.search(r"radius\s+(\d+(?:\.\d+)?)\s*(?:mm|m)?", message_lower)
        default_radius = float(radius_match.group(1)) if radius_match else 1.0
        
        components = spec.get("components", [])
        
        if inlet_face or outlet_face:
            if not components:
                new_component = {
                    "id": "net_1",
                    "domain_ref": "main_domain",
                    "build": {
                        "backend": "space_colonization",
                    },
                    "ports": {
                        "inlets": [],
                        "outlets": [],
                    },
                }
                
                if inlet_face:
                    new_component["ports"]["inlets"].append({
                        "face": inlet_face,
                        "radius": default_radius,
                    })
                
                if outlet_face:
                    new_component["ports"]["outlets"].append({
                        "face": outlet_face,
                        "radius": default_radius,
                    })
                
                patches.append({
                    "op": "add",
                    "path": "/components/-",
                    "value": new_component,
                })
            else:
                if inlet_face:
                    inlet_value = {
                        "face": inlet_face,
                        "radius": default_radius,
                    }
                    patches.append({
                        "op": "add",
                        "path": "/components/0/ports/inlets/-",
                        "value": inlet_value,
                    })
                
                if outlet_face:
                    outlet_value = {
                        "face": outlet_face,
                        "radius": default_radius,
                    }
                    patches.append({
                        "op": "add",
                        "path": "/components/0/ports/outlets/-",
                        "value": outlet_value,
                    })
        
        return patches
    
    def _extract_component_patches(
        self,
        message: str,
        spec: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Extract component-related patches from a message."""
        patches = []
        message_lower = message.lower()
        
        components = spec.get("components", [])
        
        if not components and ("tree" in message_lower or "network" in message_lower):
            backend = "space_colonization"
            if "cco" in message_lower:
                backend = "cco_hybrid"
            
            new_component = {
                "id": "net_1",
                "domain_ref": "main_domain",
                "build": {
                    "backend": backend,
                },
                "ports": {
                    "inlets": [],
                    "outlets": [],
                },
            }
            
            patches.append({
                "op": "add",
                "path": "/components/-",
                "value": new_component,
            })
        
        return patches
    
    def _extract_channel_patches(
        self,
        message: str,
        spec: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Extract primitive_channels component patches from a message.
        
        Handles phrases like "straight channel through it".
        If left/right is mentioned, creates a straight channel along x-axis
        with inlet at (-x) and outlet at (+x).
        """
        patches = []
        message_lower = message.lower()
        
        radius_match = re.search(r"radius\s+(\d+(?:\.\d+)?)\s*(?:mm|m)?", message_lower)
        channel_radius = float(radius_match.group(1)) if radius_match else None
        
        components = spec.get("components", [])
        channel_components = [
            c for c in components
            if c.get("build", {}).get("type") == "primitive_channels"
        ]
        
        domains = spec.get("domains", {})
        domain_ref = "main_domain" if "main_domain" in domains else (
            list(domains.keys())[0] if domains else "main_domain"
        )
        
        domain = domains.get(domain_ref, {})
        
        x_min = domain.get("x_min", -10)
        x_max = domain.get("x_max", 10)
        y_min = domain.get("y_min", -10)
        y_max = domain.get("y_max", 10)
        z_min = domain.get("z_min", -10)
        z_max = domain.get("z_max", 10)
        
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        z_center = (z_min + z_max) / 2
        
        policies = spec.get("policies", {})
        resolution = policies.get("resolution", {})
        default_radius = resolution.get("min_channel_radius", None)
        
        if channel_radius is None and default_radius is not None:
            channel_radius = default_radius
        
        mentions_left_right = "left" in message_lower or "right" in message_lower
        mentions_through = "through" in message_lower or "straight" in message_lower
        
        if not channel_components:
            if mentions_left_right or mentions_through:
                inlet_pos = [x_min, y_center, z_center]
                outlet_pos = [x_max, y_center, z_center]
                inlet_dir = [1, 0, 0]
                outlet_dir = [-1, 0, 0]
            else:
                inlet_pos = [x_center, y_center, z_max]
                outlet_pos = [x_center, y_center, z_min]
                inlet_dir = [0, 0, -1]
                outlet_dir = [0, 0, 1]
            
            new_channel = {
                "id": "channel_1",
                "domain_ref": domain_ref,
                "ports": {
                    "inlets": [
                        {
                            "name": "channel_inlet",
                            "position": inlet_pos,
                            "direction": inlet_dir,
                            "vessel_type": "arterial",
                        }
                    ],
                    "outlets": [
                        {
                            "name": "channel_outlet",
                            "position": outlet_pos,
                            "direction": outlet_dir,
                            "vessel_type": "arterial",
                        }
                    ],
                },
                "build": {
                    "type": "primitive_channels",
                },
            }
            
            if channel_radius is not None:
                new_channel["ports"]["inlets"][0]["radius"] = channel_radius
                new_channel["ports"]["outlets"][0]["radius"] = channel_radius
            
            patches.append({
                "op": "add",
                "path": "/components/-",
                "value": new_channel,
            })
            
            if channel_radius is None:
                self._pending_channel_radius_confirmation = True
        
        return patches
    
    def _extract_ridge_patches(
        self,
        message: str,
        spec: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Extract ridge feature patches from a message.
        
        Handles phrases like "ridge on left face", "ridge on right face",
        "ridge on left side", "ridge on the left", "where inlet is".
        """
        patches = []
        message_lower = message.lower()
        
        face_map = {
            "top": "+z",
            "bottom": "-z",
            "front": "-y",
            "back": "+y",
            "left": "-x",
            "right": "+x",
        }
        
        ridge_faces = []
        for face_name, face_code in face_map.items():
            if face_name in message_lower and "ridge" in message_lower:
                if (f"ridge on {face_name}" in message_lower or 
                    f"ridge at {face_name}" in message_lower or
                    f"{face_name} ridge" in message_lower or
                    f"{face_name} face" in message_lower or
                    f"{face_name} side" in message_lower or
                    f"on the {face_name}" in message_lower or
                    f"on {face_name}" in message_lower):
                    ridge_faces.append(face_code)
        
        if "where inlet is" in message_lower or "where the inlet is" in message_lower:
            components = spec.get("components", [])
            for comp in components:
                ports = comp.get("ports", {})
                inlets = ports.get("inlets", [])
                for inlet in inlets:
                    inlet_face = inlet.get("face")
                    if inlet_face and inlet_face not in ridge_faces:
                        ridge_faces.append(inlet_face)
        
        if not ridge_faces:
            return patches
        
        policies = spec.get("policies", {})
        ridge_policy = policies.get("ridge", {})
        
        height = ridge_policy.get("height", 0.001)
        thickness = ridge_policy.get("thickness", 0.001)
        
        height_match = re.search(r"height\s+(\d+(?:\.\d+)?)\s*(?:mm|m)?", message_lower)
        if height_match:
            height = float(height_match.group(1))
            if "mm" in message_lower:
                height = height / 1000.0
        
        thickness_match = re.search(r"thickness\s+(\d+(?:\.\d+)?)\s*(?:mm|m)?", message_lower)
        if thickness_match:
            thickness = float(thickness_match.group(1))
            if "mm" in message_lower:
                thickness = thickness / 1000.0
        
        features = spec.get("features", {})
        ridges = features.get("ridges", [])
        
        new_ridges = list(ridges)
        for face_code in ridge_faces:
            existing = any(r.get("face") == face_code for r in new_ridges)
            if not existing:
                new_ridges.append({
                    "face": face_code,
                    "height": height,
                    "thickness": thickness,
                })
        
        if "features" not in spec:
            patches.append({
                "op": "add",
                "path": "/features",
                "value": {"ridges": new_ridges},
            })
        elif "ridges" not in features:
            patches.append({
                "op": "add",
                "path": "/features/ridges",
                "value": new_ridges,
            })
        else:
            patches.append({
                "op": "replace",
                "path": "/features/ridges",
                "value": new_ridges,
            })
        
        return patches
    
    def _generate_patch_explanation(
        self,
        user_message: str,
        patches: List[Dict[str, Any]],
        spec: Dict[str, Any],
    ) -> str:
        """Generate a conversational explanation for the proposed patches."""
        if self.llm_client:
            try:
                prompt = f"""Generate a brief, friendly explanation (1-2 sentences) for these spec changes.
User request: {user_message}
Changes: {json.dumps(patches, indent=2)[:500]}
Be conversational and helpful. Don't be overly formal."""
                response = self.llm_client.chat(message=prompt)
                return response.content.strip()
            except Exception:
                pass
        
        patch_descriptions = []
        for patch in patches:
            op = patch.get("op", "")
            path = patch.get("path", "")
            
            if "/domains" in path:
                if "main_domain" in str(patch.get("value", {})):
                    value = patch.get("value", {}).get("main_domain", {})
                    dtype = value.get("type", "")
                    if dtype == "box":
                        x_size = value.get("x_max", 0) - value.get("x_min", 0)
                        y_size = value.get("y_max", 0) - value.get("y_min", 0)
                        z_size = value.get("z_max", 0) - value.get("z_min", 0)
                        patch_descriptions.append(f"create a {x_size}x{y_size}x{z_size} box domain")
                    else:
                        patch_descriptions.append(f"create a {dtype} domain")
                else:
                    patch_descriptions.append("update the domain")
            elif "/components" in path:
                value = patch.get("value", {})
                build_type = value.get("build", {}).get("type", "")
                if build_type == "primitive_channels":
                    patch_descriptions.append("add a channel component")
                else:
                    patch_descriptions.append("add a component")
            elif "/features" in path:
                value = patch.get("value", {})
                ridges = value.get("ridges", [])
                if ridges:
                    faces = [r.get("face", "") for r in ridges]
                    patch_descriptions.append(f"add ridges on {', '.join(faces)} faces")
            elif "/meta" in path:
                if "seed" in path:
                    patch_descriptions.append(f"set the seed to {patch.get('value')}")
                elif "name" in path:
                    patch_descriptions.append(f"set the name to '{patch.get('value')}'")
        
        if patch_descriptions:
            if len(patch_descriptions) == 1:
                return f"I'll {patch_descriptions[0]}. Does this look right?"
            else:
                return f"I'll {', '.join(patch_descriptions[:-1])}, and {patch_descriptions[-1]}. Does this look right?"
        
        return "I've prepared some changes based on your request. Please review and approve."
    
    def _generate_conversational_question_prompt(
        self,
        questions: List[Question],
        checklist: str,
        spec: Dict[str, Any],
    ) -> str:
        """Generate a conversational prompt for asking questions."""
        if self.llm_client:
            try:
                questions_text = "\n".join(f"- {q.question_text}" for q in questions)
                prompt = f"""Rephrase these questions in a friendly, conversational way (keep it brief):
{questions_text}

Current spec status:
{checklist}

Be helpful and guide the user. Ask one main question at a time."""
                response = self.llm_client.chat(message=prompt)
                return response.content.strip()
            except Exception:
                pass
        
        if len(questions) == 1:
            q = questions[0]
            if "domain" in q.field_path.lower():
                return f"Let's start with the basics - {q.question_text}"
            elif "component" in q.field_path.lower():
                return f"Now for the vascular structure - {q.question_text}"
            else:
                return q.question_text
        
        main_question = questions[0].question_text
        message = f"I have a few questions to help build your spec. First: {main_question}"
        if checklist:
            message += f"\n\nCurrent progress:\n{checklist}"
        return message
    
    def _generate_conversational_fallback(
        self,
        user_message: str,
        spec: Dict[str, Any],
    ) -> AgentResponse:
        """Generate a conversational fallback response when no specific action is detected."""
        domains = spec.get("domains", {})
        components = spec.get("components", [])
        features = spec.get("features", {})
        
        if not domains:
            return AgentResponse(
                response_type=AgentResponseType.MESSAGE,
                message="I'd love to help! To get started, could you describe the shape and size "
                        "of the structure you want to create? For example: 'Create a 20mm cube' "
                        "or 'Make a cylinder 10mm radius and 30mm tall'.",
            )
        
        if not components:
            domain_names = list(domains.keys())
            return AgentResponse(
                response_type=AgentResponseType.MESSAGE,
                message=f"Great, you have a domain set up ({domain_names[0]}). "
                        "What kind of vascular structure would you like inside it? "
                        "You could say 'add a straight channel through it' or describe "
                        "a branching network.",
            )
        
        if not features.get("ridges"):
            return AgentResponse(
                response_type=AgentResponseType.MESSAGE,
                message="Your spec is looking good! Would you like to add any surface features "
                        "like ridges? You can say 'add ridges on the left and right faces' "
                        "or we can proceed to run the pipeline.",
            )
        
        return AgentResponse(
            response_type=AgentResponseType.MESSAGE,
            message="Your spec looks ready! You can say 'run' to generate the structure, "
                    "or describe any other changes you'd like to make.",
        )
    
    def _llm_process_message(
        self,
        message: str,
        spec: Dict[str, Any],
        validation_report: Optional[ValidationReport],
    ) -> AgentResponse:
        """Use LLM to process the message when rule-based logic is insufficient."""
        system_prompt = """You are a DesignSpec assistant that helps users configure vascular network generation.

Your role is to:
1. Ask clarifying questions when information is missing
2. Propose JSON Patch operations to modify the spec
3. Suggest when to run the pipeline

IMPORTANT: You MUST NOT generate Python code. All modifications must be expressed as JSON Patch operations.

Current spec summary:
- Domains: {domains}
- Components: {components}
- Policies: {policies}

Respond with one of:
- QUESTION: <question text>
- PATCH: <json patch array>
- RUN: <stage to run until>
- MESSAGE: <informational message>
"""
        
        domains_summary = list(spec.get("domains", {}).keys()) or "none"
        components_summary = [c.get("id", "unnamed") for c in spec.get("components", [])] or "none"
        policies_summary = list(spec.get("policies", {}).keys()) or "default"
        
        formatted_prompt = system_prompt.format(
            domains=domains_summary,
            components=components_summary,
            policies=policies_summary,
        )
        
        try:
            response = self.llm_client.chat(
                message=message,
                system_prompt=formatted_prompt,
            )
            
            return self._parse_llm_response(response.content, spec)
            
        except Exception as e:
            logger.exception("LLM processing failed")
            return AgentResponse(
                response_type=AgentResponseType.ERROR,
                message=f"Failed to process with LLM: {str(e)}",
            )
    
    def _parse_llm_response(
        self,
        response: str,
        spec: Dict[str, Any],
    ) -> AgentResponse:
        """Parse LLM response into an AgentResponse."""
        response_upper = response.upper()
        
        if response_upper.startswith("QUESTION:"):
            question_text = response[9:].strip()
            return AgentResponse(
                response_type=AgentResponseType.QUESTION,
                message=question_text,
                questions=[Question(
                    field_path="unknown",
                    question_text=question_text,
                    reason="LLM-generated question",
                )],
            )
        
        if response_upper.startswith("PATCH:"):
            patch_text = response[6:].strip()
            try:
                patches = json.loads(patch_text)
                if isinstance(patches, list):
                    return AgentResponse(
                        response_type=AgentResponseType.PATCH_PROPOSAL,
                        message="I've prepared a patch based on your request.",
                        patch_proposal=PatchProposal(
                            explanation="LLM-generated patch",
                            patches=patches,
                            confidence=0.7,
                            requires_confirmation=True,
                        ),
                    )
            except json.JSONDecodeError:
                pass
        
        if response_upper.startswith("RUN:"):
            stage = response[4:].strip().lower()
            return AgentResponse(
                response_type=AgentResponseType.RUN_REQUEST,
                message=f"Ready to run until {stage}.",
                run_request=RunRequest(
                    run_until=stage if stage in RUN_STAGES else "union_voids",
                    reason="LLM-suggested run",
                ),
            )
        
        return AgentResponse(
            response_type=AgentResponseType.MESSAGE,
            message=response,
        )
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the conversation history."""
        return list(self._conversation_history)
    
    def clear_conversation_history(self) -> None:
        """Clear the conversation history."""
        self._conversation_history.clear()
    
    def add_assistant_message(self, message: str) -> None:
        """Add an assistant message to the conversation history."""
        self._conversation_history.append({
            "role": "assistant",
            "content": message,
        })


__all__ = [
    "DesignSpecAgent",
    "AgentResponse",
    "AgentResponseType",
    "Question",
    "RunRequest",
]
