"""
Contextual Dialogue System

Provides a conversation-driven approach that maintains full context and responds
naturally without rigid state transitions. This replaces the sequential
"Interpret -> Plan -> Ask" pattern with a more fluid, contextual approach.

The system:
1. Maintains full conversation history and context
2. Handles interruptions and topic changes seamlessly
3. Provides real-time feedback on spec changes
4. Responds naturally to meta-questions, corrections, and clarifications
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum
import re
import json

from .agent_dialogue import (
    Assumption,
    Ambiguity,
    Risk,
    UnderstandingReport,
    PlanOption,
    detect_organ_type,
    extract_numeric_values,
    detect_spatial_terms,
    detect_vague_quantifiers,
)
from .reactive_prompt import TurnIntent, interpret_user_turn


class DialogueIntent(Enum):
    """Classification of user dialogue intent."""
    PROVIDE_INFO = "provide_info"
    ASK_QUESTION = "ask_question"
    REQUEST_CHANGE = "request_change"
    CONFIRM = "confirm"
    REJECT = "reject"
    CLARIFY = "clarify"
    META_QUESTION = "meta_question"
    CORRECTION = "correction"
    UNCERTAINTY = "uncertainty"
    TOPIC_CHANGE = "topic_change"
    CANCEL = "cancel"


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    role: str  # "user" or "assistant"
    content: str
    intent: Optional[DialogueIntent] = None
    extracted_values: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass
class DialogueContext:
    """Full context of the ongoing dialogue."""
    conversation_history: List[ConversationTurn] = field(default_factory=list)
    current_topic: str = "general"
    pending_questions: List[str] = field(default_factory=list)
    answered_questions: Dict[str, Any] = field(default_factory=dict)
    assumptions: List[Assumption] = field(default_factory=list)
    ambiguities: List[Ambiguity] = field(default_factory=list)
    risks: List[Risk] = field(default_factory=list)
    spec_values: Dict[str, Any] = field(default_factory=dict)
    user_constraints: List[str] = field(default_factory=list)
    
    def add_turn(self, role: str, content: str, intent: Optional[DialogueIntent] = None,
                 extracted: Optional[Dict[str, Any]] = None) -> None:
        """Add a turn to the conversation history."""
        import time
        turn = ConversationTurn(
            role=role,
            content=content,
            intent=intent,
            extracted_values=extracted or {},
            timestamp=time.time(),
        )
        self.conversation_history.append(turn)
    
    def get_recent_context(self, num_turns: int = 10) -> str:
        """Get recent conversation context as a string."""
        recent = self.conversation_history[-num_turns:]
        lines = []
        for turn in recent:
            prefix = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{prefix}: {turn.content}")
        return "\n".join(lines)
    
    def update_spec_value(self, field: str, value: Any, source: str = "user") -> None:
        """Update a spec value and track the change."""
        self.spec_values[field] = {
            "value": value,
            "source": source,
            "confirmed": source == "user",
        }
    
    def get_spec_summary(self) -> Dict[str, Any]:
        """Get current spec values as a simple dict."""
        return {k: v["value"] for k, v in self.spec_values.items()}


@dataclass
class DialogueResponse:
    """Response from the dialogue system."""
    message: str
    intent: DialogueIntent
    spec_updates: Dict[str, Any] = field(default_factory=dict)
    follow_up_questions: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    is_ready_to_proceed: bool = False
    needs_clarification: bool = False


class ContextualDialogue:
    """
    Maintains conversation context and responds naturally.
    
    This class provides a fluid, conversation-driven approach to requirements
    gathering that handles interruptions, topic changes, and meta-questions
    seamlessly without rigid state transitions.
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        input_func: Optional[Callable[[str], str]] = None,
        print_func: Optional[Callable[[str], None]] = None,
    ):
        self.llm_client = llm_client
        self.input_func = input_func or input
        self.print_func = print_func or print
        self.context = DialogueContext()
        
        # Intent patterns for classification
        self._intent_patterns = {
            DialogueIntent.CONFIRM: [
                r"^\s*(yes|y|ok|okay|sure|correct|right|exactly|yep|yeah)\s*$",
                r"\bthat'?s?\s+(right|correct|good)\b",
                r"\bconfirm(ed)?\b",
                r"\baccept(ed)?\b",
            ],
            DialogueIntent.REJECT: [
                r"^\s*(no|n|nope|nah|wrong|incorrect)\s*$",
                r"\bthat'?s?\s+(wrong|incorrect|not right)\b",
                r"\breject(ed)?\b",
                r"\bdon'?t\s+want\b",
            ],
            DialogueIntent.CORRECTION: [
                r"\bactually\b",
                r"\bi\s+meant\b",
                r"\bchange\s+(that|it|the)\b",
                r"\bcorrect(ion)?\b",
                r"\bwrong\b.*\bshould\s+be\b",
                r"\bnot\s+\d+.*\bbut\s+\d+\b",
            ],
            DialogueIntent.UNCERTAINTY: [
                r"\bi'?m?\s+not\s+sure\b",
                r"\bmaybe\b",
                r"\bperhaps\b",
                r"\bi\s+don'?t\s+know\b",
                r"\bwhat\s+do\s+you\s+(think|suggest|recommend)\b",
                r"\bwhat\s+would\s+you\s+recommend\b",
            ],
            DialogueIntent.META_QUESTION: [
                r"\bwhy\s+(are\s+you\s+)?asking\b",
                r"\bwhat\s+does\s+\w+\s+mean\b",
                r"\bexplain\b",
                r"\bwhat\s+are\s+(the\s+)?options\b",
                r"\bwhat\s+are\s+(the\s+)?defaults?\b",
                r"\bhelp\b",
            ],
            DialogueIntent.TOPIC_CHANGE: [
                r"\blet'?s?\s+(talk|discuss)\s+about\b",
                r"\bwhat\s+about\b",
                r"\bcan\s+we\s+(also|instead)\b",
                r"\bactually,?\s+let'?s?\b",
                r"\bchange\s+topic\b",
            ],
            DialogueIntent.CANCEL: [
                r"^\s*(cancel|quit|exit|stop|abort)\s*$",
                r"\bcancel\s+(this|the)\s+(workflow|process|session)\b",
            ],
        }
    
    def _print(self, text: str) -> None:
        """Print text using configured print function."""
        self.print_func(text)
    
    def _input(self, prompt: str) -> str:
        """Get input using configured input function."""
        return self.input_func(prompt)
    
    def classify_intent(self, text: str) -> DialogueIntent:
        """
        Classify the intent of user input.
        
        Uses pattern matching to detect specific intents, falling back to
        PROVIDE_INFO for general responses.
        """
        text_lower = text.lower().strip()
        
        if not text_lower:
            return DialogueIntent.PROVIDE_INFO
        
        # Check each intent pattern
        for intent, patterns in self._intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    return intent
        
        # Check if it's a question
        if text_lower.endswith("?"):
            return DialogueIntent.ASK_QUESTION
        
        # Default to providing information
        return DialogueIntent.PROVIDE_INFO
    
    def extract_values_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract structured values from user text.
        
        Combines rule-based extraction with context-aware parsing.
        """
        extracted = extract_numeric_values(text)
        
        # Extract topology kind
        topology_patterns = {
            "path": [r"\bpath\b", r"\bchannel\b", r"\bsimple\s+tube\b", r"\bstraight\b"],
            "tree": [r"\btree\b", r"\bbranch(ing|es)?\b", r"\bvascular\b", r"\bnetwork\b"],
            "backbone": [r"\bbackbone\b", r"\bparallel\s+leg\b", r"\b\d+\s*-?\s*leg\b"],
            "loop": [r"\bloop\b", r"\brecircul\b", r"\bclosed\b"],
            "multi_tree": [r"\bmulti(ple)?\s*tree\b", r"\b\d+\s+trees?\b"],
        }
        
        text_lower = text.lower()
        for topology, patterns in topology_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    extracted["topology_kind"] = topology
                    break
            if "topology_kind" in extracted:
                break
        
        # Extract domain shape
        if re.search(r"\bellipsoid\b|\bsphere\b|\boval\b", text_lower):
            extracted["domain_type"] = "ellipsoid"
        elif re.search(r"\bbox\b|\bcube\b|\brectang\b", text_lower):
            extracted["domain_type"] = "box"
        
        # Extract face references
        face_patterns = {
            "x_min": [r"\bleft\b", r"\bx[\s_-]?min\b", r"\b-x\b"],
            "x_max": [r"\bright\b", r"\bx[\s_-]?max\b", r"\b\+x\b"],
            "y_min": [r"\bfront\b", r"\by[\s_-]?min\b", r"\b-y\b"],
            "y_max": [r"\bback\b", r"\by[\s_-]?max\b", r"\b\+y\b"],
            "z_min": [r"\bbottom\b", r"\bz[\s_-]?min\b", r"\b-z\b"],
            "z_max": [r"\btop\b", r"\bz[\s_-]?max\b", r"\b\+z\b"],
        }
        
        for face, patterns in face_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    if "inlet" in text_lower:
                        extracted["inlet_face"] = face
                    if "outlet" in text_lower:
                        extracted["outlet_face"] = face
                    break
        
        return extracted
    
    def process_input(self, user_input: str) -> DialogueResponse:
        """
        Process user input and generate appropriate response.
        
        This is the main entry point for handling user dialogue. It:
        1. Classifies the intent of the input
        2. Extracts any values from the text
        3. Updates the context
        4. Generates an appropriate response
        """
        intent = self.classify_intent(user_input)
        extracted = self.extract_values_from_text(user_input)
        
        # Add to conversation history
        self.context.add_turn("user", user_input, intent, extracted)
        
        # Update spec values from extracted data
        for field, value in extracted.items():
            self.context.update_spec_value(field, value, "user")
        
        # Generate response based on intent
        response = self._generate_response(user_input, intent, extracted)
        
        # Add assistant response to history
        self.context.add_turn("assistant", response.message, response.intent)
        
        return response
    
    def _generate_response(
        self,
        user_input: str,
        intent: DialogueIntent,
        extracted: Dict[str, Any]
    ) -> DialogueResponse:
        """Generate a response based on user intent and extracted values."""
        
        if intent == DialogueIntent.CANCEL:
            return DialogueResponse(
                message="Understood. Cancelling the current workflow.",
                intent=intent,
                is_ready_to_proceed=False,
            )
        
        if intent == DialogueIntent.META_QUESTION:
            return self._handle_meta_question(user_input)
        
        if intent == DialogueIntent.CORRECTION:
            return self._handle_correction(user_input, extracted)
        
        if intent == DialogueIntent.UNCERTAINTY:
            return self._handle_uncertainty(user_input)
        
        if intent == DialogueIntent.CONFIRM:
            return self._handle_confirmation(user_input)
        
        if intent == DialogueIntent.REJECT:
            return self._handle_rejection(user_input)
        
        if intent == DialogueIntent.TOPIC_CHANGE:
            return self._handle_topic_change(user_input, extracted)
        
        # Default: process as information provision
        return self._handle_information(user_input, extracted)
    
    def _handle_meta_question(self, user_input: str) -> DialogueResponse:
        """Handle meta-questions about the process or terminology."""
        text_lower = user_input.lower()
        
        if "default" in text_lower:
            message = self._format_defaults_info()
        elif "option" in text_lower:
            message = self._format_options_info()
        elif "why" in text_lower and "asking" in text_lower:
            message = self._format_why_asking()
        elif "help" in text_lower:
            message = self._format_help()
        else:
            message = self._format_general_explanation(user_input)
        
        return DialogueResponse(
            message=message,
            intent=DialogueIntent.META_QUESTION,
            needs_clarification=False,
        )
    
    def _handle_correction(self, user_input: str, extracted: Dict[str, Any]) -> DialogueResponse:
        """Handle corrections to previously provided information."""
        # Update spec values with corrected data
        for field, value in extracted.items():
            self.context.update_spec_value(field, value, "user_correction")
        
        if extracted:
            fields_updated = ", ".join(extracted.keys())
            message = f"Got it, I've updated: {fields_updated}. Let me know if anything else needs adjustment."
        else:
            message = "I understand you want to make a correction. Could you specify what you'd like to change?"
        
        return DialogueResponse(
            message=message,
            intent=DialogueIntent.CORRECTION,
            spec_updates=extracted,
            needs_clarification=not bool(extracted),
        )
    
    def _handle_uncertainty(self, user_input: str) -> DialogueResponse:
        """Handle user uncertainty by providing suggestions."""
        # Analyze what the user might be uncertain about
        current_topic = self.context.current_topic
        
        suggestions = self._get_suggestions_for_topic(current_topic)
        
        message = "No problem! Here are some suggestions based on common use cases:\n"
        for i, suggestion in enumerate(suggestions[:3], 1):
            message += f"  {i}. {suggestion}\n"
        message += "\nWould any of these work for you, or would you like me to explain more?"
        
        return DialogueResponse(
            message=message,
            intent=DialogueIntent.UNCERTAINTY,
            suggestions=suggestions,
            needs_clarification=True,
        )
    
    def _handle_confirmation(self, user_input: str) -> DialogueResponse:
        """Handle user confirmation."""
        # Mark pending items as confirmed
        for field, data in self.context.spec_values.items():
            if not data.get("confirmed"):
                data["confirmed"] = True
        
        # Check if we have enough to proceed
        is_ready = self._check_readiness()
        
        if is_ready:
            message = "Great! All required information has been confirmed. Ready to proceed with generation."
        else:
            missing = self._get_missing_fields()
            message = f"Confirmed! I still need a few more details: {', '.join(missing[:3])}"
        
        return DialogueResponse(
            message=message,
            intent=DialogueIntent.CONFIRM,
            is_ready_to_proceed=is_ready,
        )
    
    def _handle_rejection(self, user_input: str) -> DialogueResponse:
        """Handle user rejection."""
        message = "Understood. What would you like to change or do differently?"
        
        return DialogueResponse(
            message=message,
            intent=DialogueIntent.REJECT,
            needs_clarification=True,
        )
    
    def _handle_topic_change(self, user_input: str, extracted: Dict[str, Any]) -> DialogueResponse:
        """Handle topic changes in the conversation."""
        # Try to detect the new topic
        text_lower = user_input.lower()
        
        topic_keywords = {
            "domain": ["domain", "size", "dimension", "box", "shape"],
            "ports": ["inlet", "outlet", "port", "entry", "exit"],
            "topology": ["topology", "branch", "tree", "path", "structure"],
            "constraints": ["constraint", "minimum", "clearance", "radius"],
            "embedding": ["embed", "voxel", "print", "export"],
        }
        
        new_topic = "general"
        for topic, keywords in topic_keywords.items():
            if any(kw in text_lower for kw in keywords):
                new_topic = topic
                break
        
        self.context.current_topic = new_topic
        
        message = f"Sure, let's talk about {new_topic}. "
        message += self._get_topic_prompt(new_topic)
        
        return DialogueResponse(
            message=message,
            intent=DialogueIntent.TOPIC_CHANGE,
            spec_updates=extracted,
        )
    
    def _handle_information(self, user_input: str, extracted: Dict[str, Any]) -> DialogueResponse:
        """Handle general information provision."""
        # Update spec with extracted values
        for field, value in extracted.items():
            self.context.update_spec_value(field, value, "user")
        
        # Generate acknowledgment and follow-up
        if extracted:
            fields_captured = ", ".join(extracted.keys())
            message = f"Got it! I've captured: {fields_captured}.\n"
        else:
            message = "I understand. "
        
        # Check what's still needed
        missing = self._get_missing_fields()
        if missing:
            next_field = missing[0]
            message += f"\n{self._get_question_for_field(next_field)}"
            follow_up = [next_field]
        else:
            message += "\nI have all the information I need. Ready to proceed?"
            follow_up = []
        
        return DialogueResponse(
            message=message,
            intent=DialogueIntent.PROVIDE_INFO,
            spec_updates=extracted,
            follow_up_questions=follow_up,
            is_ready_to_proceed=not bool(missing),
        )
    
    def _format_defaults_info(self) -> str:
        """Format information about default values."""
        defaults = {
            "domain_type": "box",
            "domain_size": "20mm x 60mm x 30mm",
            "inlet_radius": "2mm",
            "outlet_radius": "1mm",
            "min_radius": "0.1mm",
            "min_clearance": "0.5mm",
            "topology_kind": "tree",
            "target_terminals": "50",
        }
        
        lines = ["Here are the default values I'll use if not specified:\n"]
        for field, value in defaults.items():
            lines.append(f"  - {field}: {value}")
        lines.append("\nYou can override any of these by specifying your preferred values.")
        
        return "\n".join(lines)
    
    def _format_options_info(self) -> str:
        """Format information about available options."""
        options = {
            "topology_kind": ["path", "tree", "backbone", "loop", "multi_tree"],
            "domain_type": ["box", "ellipsoid"],
            "tapering": ["murray", "linear", "fixed"],
            "branching_style": ["balanced", "aggressive_early", "space_filling"],
        }
        
        lines = ["Here are the available options:\n"]
        for field, opts in options.items():
            lines.append(f"  {field}: {', '.join(opts)}")
        
        return "\n".join(lines)
    
    def _format_why_asking(self) -> str:
        """Format explanation of why certain questions are asked."""
        current_topic = self.context.current_topic
        
        explanations = {
            "domain": "The domain defines the 3D space where your structure will be generated. Getting this right ensures the structure fits your intended use case.",
            "ports": "Inlets and outlets define where fluid enters and exits the structure. Their positions and sizes affect flow characteristics.",
            "topology": "The topology determines the overall structure type - whether it's a simple path, branching tree, or more complex network.",
            "constraints": "Constraints ensure the generated structure is manufacturable and meets your physical requirements.",
            "embedding": "Embedding settings control how the structure is converted to a printable format.",
            "general": "I ask questions to understand your requirements and generate a structure that meets your needs.",
        }
        
        return explanations.get(current_topic, explanations["general"])
    
    def _format_help(self) -> str:
        """Format general help information."""
        return """I'm here to help you design a 3D vascular structure. Here's how this works:

1. Tell me what you want to create (e.g., "a branching vascular network in a 20x60x30mm box")
2. I'll ask clarifying questions about specific parameters
3. You can ask me questions at any time (e.g., "what are the defaults?", "why are you asking this?")
4. Say "change" or "actually" to correct any previous answers
5. Say "confirm" when you're happy with the specification

Current status: """ + self._get_status_summary()
    
    def _format_general_explanation(self, user_input: str) -> str:
        """Format a general explanation based on the user's question."""
        # Try to identify what they're asking about
        text_lower = user_input.lower()
        
        if "murray" in text_lower:
            return "Murray's law describes how blood vessel radii should relate at branch points for optimal flow. It states that the cube of the parent radius equals the sum of the cubes of the child radii."
        elif "topology" in text_lower:
            return "Topology refers to the overall structure type: 'path' is a simple channel, 'tree' is a branching network, 'backbone' is parallel legs, 'loop' has recirculation, and 'multi_tree' has multiple independent trees."
        elif "voxel" in text_lower:
            return "Voxel pitch is the resolution of the 3D grid used for embedding. Smaller values give finer detail but require more memory and processing time."
        else:
            return "I'm not sure what you're asking about. Could you rephrase your question, or ask about a specific parameter like 'topology', 'domain', or 'constraints'?"
    
    def _get_suggestions_for_topic(self, topic: str) -> List[str]:
        """Get suggestions for a given topic."""
        suggestions = {
            "domain": [
                "Use a 20x60x30mm box (common for testing)",
                "Match your 3D printer's build volume",
                "Use an ellipsoid for organ-like shapes",
            ],
            "ports": [
                "Single inlet on one face, outlet on opposite face",
                "Inlet at top, multiple outlets at bottom",
                "Centered inlet with 2mm radius",
            ],
            "topology": [
                "Tree topology for vascular networks",
                "Path topology for simple channels",
                "Backbone for parallel leg structures",
            ],
            "constraints": [
                "0.5mm minimum radius for FDM printing",
                "0.3mm minimum radius for SLA printing",
                "1mm clearance between channels",
            ],
            "general": [
                "Start with default values and adjust as needed",
                "Describe your use case and I'll suggest parameters",
                "Look at example configurations in the documentation",
            ],
        }
        return suggestions.get(topic, suggestions["general"])
    
    def _get_topic_prompt(self, topic: str) -> str:
        """Get the initial prompt for a topic."""
        prompts = {
            "domain": "What size and shape should the domain be? (e.g., '20x60x30mm box')",
            "ports": "Where should the inlet(s) and outlet(s) be located?",
            "topology": "What type of structure do you want? (path, tree, backbone, loop, or multi_tree)",
            "constraints": "What are your manufacturing constraints? (minimum radius, clearance, etc.)",
            "embedding": "Do you want to embed this for 3D printing? If so, what's your target resolution?",
            "general": "What would you like to know or specify?",
        }
        return prompts.get(topic, prompts["general"])
    
    def _get_question_for_field(self, field: str) -> str:
        """Get the question to ask for a specific field."""
        questions = {
            "domain_size": "What are the domain dimensions? (e.g., '20 60 30 mm')",
            "domain_type": "What shape should the domain be? (box or ellipsoid)",
            "inlet_face": "Which face should the inlet be on? (x_min, x_max, y_min, y_max, z_min, z_max)",
            "inlet_radius": "What radius should the inlet have? (in mm)",
            "outlet_face": "Which face should the outlet be on?",
            "outlet_radius": "What radius should the outlet have? (in mm)",
            "topology_kind": "What type of structure? (path, tree, backbone, loop, multi_tree)",
            "target_terminals": "How many terminal branches should the tree have?",
            "min_radius": "What's the minimum channel radius? (in mm)",
            "min_clearance": "What's the minimum clearance between channels? (in mm)",
        }
        return questions.get(field, f"Please specify {field}:")
    
    def _get_missing_fields(self) -> List[str]:
        """Get list of required fields that are still missing."""
        # Define required fields based on topology
        topology = self.context.spec_values.get("topology_kind", {}).get("value", "tree")
        
        required = ["domain_size", "inlet_face", "inlet_radius"]
        
        if topology == "path":
            required.extend(["outlet_face", "outlet_radius"])
        elif topology == "tree":
            required.append("target_terminals")
        elif topology == "backbone":
            required.extend(["outlet_face", "leg_count"])
        
        # Check which are missing
        missing = []
        for field in required:
            if field not in self.context.spec_values:
                missing.append(field)
        
        return missing
    
    def _check_readiness(self) -> bool:
        """Check if we have enough information to proceed."""
        return len(self._get_missing_fields()) == 0
    
    def _get_status_summary(self) -> str:
        """Get a summary of current status."""
        spec = self.context.get_spec_summary()
        missing = self._get_missing_fields()
        
        if not spec:
            return "No specifications captured yet."
        
        lines = [f"Captured: {', '.join(spec.keys())}"]
        if missing:
            lines.append(f"Still needed: {', '.join(missing)}")
        else:
            lines.append("Ready to proceed!")
        
        return " | ".join(lines)
    
    def run_dialogue_loop(self, initial_prompt: str = "") -> Dict[str, Any]:
        """
        Run an interactive dialogue loop until ready to proceed.
        
        Parameters
        ----------
        initial_prompt : str
            Optional initial prompt to display
            
        Returns
        -------
        Dict[str, Any]
            The final spec values collected from the dialogue
        """
        if initial_prompt:
            self._print(initial_prompt)
        else:
            self._print("Hello! I'm here to help you design a 3D vascular structure.")
            self._print("Describe what you want to create, or type 'help' for guidance.")
        
        self._print("")
        
        while True:
            user_input = self._input("You: ").strip()
            
            if not user_input:
                continue
            
            response = self.process_input(user_input)
            
            self._print(f"\nAssistant: {response.message}\n")
            
            if response.intent == DialogueIntent.CANCEL:
                return {}
            
            if response.is_ready_to_proceed:
                confirm = self._input("Proceed with generation? (y/n): ").strip().lower()
                if confirm in ["y", "yes"]:
                    return self.context.get_spec_summary()
                else:
                    self._print("Okay, let's continue refining the specification.\n")
        
        return self.context.get_spec_summary()
    
    def get_understanding_report(self) -> UnderstandingReport:
        """
        Generate an UnderstandingReport from the current dialogue context.
        
        This provides compatibility with the existing workflow system.
        """
        from .agent_dialogue import InitialRequirementsDraft
        
        spec = self.context.get_spec_summary()
        
        # Build requirements draft from spec
        draft = InitialRequirementsDraft(
            domain_type=spec.get("domain_type", "box"),
            domain_size=spec.get("domain_size"),
            num_inlets=1,  # Default
            num_outlets=1 if spec.get("outlet_face") else 0,
            inlet_radius=spec.get("inlet_radius"),
            outlet_radius=spec.get("outlet_radius"),
            target_terminals=spec.get("target_terminals"),
            min_radius=spec.get("min_radius"),
            topology_style=spec.get("topology_kind", "tree"),
        )
        
        # Build summary from conversation
        summary = self._build_summary_from_context()
        
        return UnderstandingReport(
            summary=summary,
            assumptions=self.context.assumptions,
            ambiguities=self.context.ambiguities,
            risks=self.context.risks,
            requirements_draft=draft,
            organ_type=detect_organ_type(summary),
            raw_intent=summary,
        )
    
    def _build_summary_from_context(self) -> str:
        """Build a summary string from the conversation context."""
        spec = self.context.get_spec_summary()
        
        parts = []
        
        if spec.get("topology_kind"):
            parts.append(f"a {spec['topology_kind']} structure")
        else:
            parts.append("a vascular structure")
        
        if spec.get("domain_size"):
            size = spec["domain_size"]
            if isinstance(size, tuple):
                parts.append(f"in a {size[0]*1000:.0f}x{size[1]*1000:.0f}x{size[2]*1000:.0f}mm domain")
        
        if spec.get("target_terminals"):
            parts.append(f"with {spec['target_terminals']} terminal branches")
        
        return "Generate " + " ".join(parts) + "."
