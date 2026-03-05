"""
Reactive Prompting System

Provides a conversational layer that detects meta-questions (like "what are the defaults?")
and responds appropriately before re-asking the original question. This makes the workflow
feel like a real agent rather than a form/prompt machine.

The system:
1. Classifies user input as answer vs meta-question
2. Handles meta-questions by providing information and re-asking
3. Only advances state when a valid answer is received
4. Shows understanding reflection and change summaries
"""

from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import re


class TurnIntent(Enum):
    """Classification of user input intent."""
    ANSWER = "answer"
    ASK_DEFAULTS = "ask_defaults"
    CLARIFY_DEFAULTS = "clarify_defaults"
    WHY_THIS_QUESTION = "why_this_question"
    SHOW_OPTIONS = "show_options"
    CHANGE_DEFAULT = "change_default"
    BACK = "back"
    CANCEL = "cancel"
    HELP = "help"
    FREEFORM_CONSTRAINT = "freeform_constraint"


META_QUESTION_PATTERNS = {
    TurnIntent.ASK_DEFAULTS: [
        r"\bwhat\s+(are|is)\s+(the\s+)?defaults?\b",
        r"\bshow\s+(me\s+)?(the\s+)?defaults?\b",
        r"\blist\s+(the\s+)?defaults?\b",
        r"\bdefaults?\s*\?\s*$",
        r"\bwhat\s+would\s+you\s+default\b",
    ],
    TurnIntent.CLARIFY_DEFAULTS: [
        r"\bwhat\s+does\s+default\s+mean\b",
        r"\bexplain\s+(the\s+)?defaults?\b",
        r"\bwhy\s+(these|those|this)\s+defaults?\b",
    ],
    TurnIntent.WHY_THIS_QUESTION: [
        r"\bwhy\s+(are\s+you\s+)?asking\b",
        r"\bwhy\s+do\s+(you|i)\s+need\b",
        r"\bwhat\s+is\s+this\s+(for|about)\b",
        r"\bexplain\s+(this\s+)?question\b",
        r"\bwhy\s+does\s+(this|it)\s+matter\b",
    ],
    TurnIntent.SHOW_OPTIONS: [
        r"\bwhat\s+(are\s+)?(the\s+)?options\b",
        r"\bshow\s+(me\s+)?(the\s+)?options\b",
        r"\blist\s+(the\s+)?options\b",
        r"\bwhat\s+can\s+i\s+(choose|pick|select)\b",
        r"\bwhat\s+choices\b",
    ],
    TurnIntent.HELP: [
        r"^\s*help\s*$",
        r"^\s*\?\s*$",
        r"\bhelp\s+me\b",
        r"\bi\s+(don'?t|do\s+not)\s+(understand|know)\b",
        r"\bwhat\s+should\s+i\s+(do|say|enter)\b",
        r"\bwhat\s+(does|is)\s+\w+\s+(mean|terminology)\b",
        r"\bexplain\s+(the\s+)?terminology\b",
        r"\bwhat\s+does\s+\w+\s+mean\b",
        r"\bdefine\s+\w+\b",
    ],
    TurnIntent.BACK: [
        r"^\s*back\s*$",
        r"^\s*go\s+back\s*$",
        r"^\s*previous\s*$",
        r"\bgo\s+(to\s+)?previous\b",
    ],
    TurnIntent.CANCEL: [
        r"^\s*cancel\s*$",
        r"^\s*quit\s*$",
        r"^\s*exit\s*$",
        r"^\s*abort\s*$",
        r"\bcancel\s+(this\s+)?(workflow|process|session)\b",
    ],
    TurnIntent.CHANGE_DEFAULT: [
        r"\buse\s+(a\s+)?different\s+default\b",
        r"\bchange\s+(the\s+)?default\b",
        r"\bset\s+(a\s+)?new\s+default\b",
        r"\bdon'?t\s+use\s+(the\s+)?default\b",
    ],
}

FREEFORM_CONSTRAINT_PATTERNS = [
    r"\bmake\s+sure\b",
    r"\bensure\s+that\b",
    r"\bit\s+(should|must|needs\s+to)\b",
    r"\bi\s+want\s+(it\s+to|the)\b",
    r"\bdon'?t\s+(let|allow|make)\b",
    r"\balways\b.*\bshould\b",
    r"\bnever\b.*\bshould\b",
]


def interpret_user_turn(text: str) -> TurnIntent:
    """
    Classify user input as a meta-question or answer.
    
    Uses deterministic heuristics to detect common meta-question patterns.
    If input ends with '?' or contains meta-question keywords, it's likely
    a meta-question rather than an answer.
    
    Parameters
    ----------
    text : str
        User input text
        
    Returns
    -------
    TurnIntent
        The classified intent
    """
    text_lower = text.lower().strip()
    
    if not text_lower:
        return TurnIntent.ANSWER
    
    for intent, patterns in META_QUESTION_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return intent
    
    for pattern in FREEFORM_CONSTRAINT_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return TurnIntent.FREEFORM_CONSTRAINT
    
    if text_lower.endswith("?"):
        generic_question_words = ["what", "why", "how", "which", "where", "when", "who"]
        for word in generic_question_words:
            if text_lower.startswith(word):
                return TurnIntent.HELP
    
    return TurnIntent.ANSWER


class QuestionHelp:
    """Help information for a specific question."""
    
    def __init__(
        self,
        question_id: str,
        question_text: str,
        default_value: Any = None,
        options: Optional[List[str]] = None,
        why_asking: str = "",
        impact: str = "",
        examples: Optional[List[str]] = None,
    ):
        self.question_id = question_id
        self.question_text = question_text
        self.default_value = default_value
        self.options = options or []
        self.why_asking = why_asking
        self.impact = impact
        self.examples = examples or []
    
    def format_defaults(self) -> str:
        """Format default value information."""
        if self.default_value is None:
            return "No default value set for this question."
        return f"Default: {self.default_value}"
    
    def format_options(self) -> str:
        """Format available options."""
        if not self.options:
            return "This is a free-form input field."
        return f"Options: {', '.join(self.options)}"
    
    def format_why(self) -> str:
        """Format explanation of why this question is asked."""
        if not self.why_asking:
            return "This information helps configure the generation process."
        return self.why_asking
    
    def format_help(self) -> str:
        """Format complete help text."""
        lines = [
            f"Question: {self.question_text}",
            "",
            self.format_defaults(),
            self.format_options(),
            "",
            f"Why: {self.format_why()}",
        ]
        
        if self.impact:
            lines.append(f"Impact: {self.impact}")
        
        if self.examples:
            lines.append("")
            lines.append("Examples:")
            for ex in self.examples:
                lines.append(f"  - {ex}")
        
        return "\n".join(lines)


class GlobalDefaults:
    """Global default values for the workflow."""
    
    DEFAULTS = {
        "voxel_pitch_m": 3e-4,
        "min_radius_m": 5e-5,
        "min_clearance_m": 1e-4,
        "domain_type": "box",
        "units": "meters",
        "terminal_density": "moderate",
        "frame_of_reference": "face-based",
        "output_format": "stl",
        "mesh_quality": "standard",
        "memory_policy": "balanced",
    }
    
    DESCRIPTIONS = {
        "voxel_pitch_m": "Voxel size for embedding (3e-4 m = 0.3mm)",
        "min_radius_m": "Minimum vessel radius (5e-5 m = 50 microns)",
        "min_clearance_m": "Minimum clearance between vessels (1e-4 m = 0.1mm)",
        "domain_type": "Shape of the generation domain",
        "units": "Unit system for all measurements",
        "terminal_density": "Density of terminal vessels (sparse/moderate/dense)",
        "frame_of_reference": "Coordinate system mode (face-based or explicit)",
        "output_format": "Output mesh format",
        "mesh_quality": "Mesh quality level (draft/standard/high)",
        "memory_policy": "Memory usage policy (minimal/balanced/performance)",
    }
    
    @classmethod
    def get_defaults_text(cls) -> str:
        """Get formatted text of all global defaults."""
        lines = ["Global Defaults:", ""]
        for key, value in cls.DEFAULTS.items():
            desc = cls.DESCRIPTIONS.get(key, "")
            lines.append(f"  {key}: {value}")
            if desc:
                lines.append(f"    ({desc})")
        return "\n".join(lines)
    
    @classmethod
    def get_default(cls, key: str) -> Any:
        """Get a specific default value."""
        return cls.DEFAULTS.get(key)
    
    @classmethod
    def get_defaults_dict(cls) -> Dict[str, Any]:
        """Get all defaults as a dictionary."""
        return cls.DEFAULTS.copy()


class ReactivePromptSession:
    """
    A session wrapper for reactive prompting.
    
    Handles meta-questions, validates answers, and only advances
    when a valid answer is received.
    """
    
    def __init__(
        self,
        input_func: Optional[Callable[[str], str]] = None,
        print_func: Optional[Callable[[str], None]] = None,
    ):
        self.input_func = input_func or input
        self.print_func = print_func or print
        self.context_snapshot: Dict[str, Any] = {}
        self.constraints: List[str] = []
        self.history: List[Dict[str, Any]] = []
    
    def _print(self, text: str) -> None:
        """Print text using configured print function."""
        self.print_func(text)
    
    def _input(self, prompt: str) -> str:
        """Get input using configured input function."""
        return self.input_func(prompt)
    
    def _handle_meta_intent(
        self,
        intent: TurnIntent,
        help_info: QuestionHelp,
        user_text: str,
    ) -> None:
        """Handle a meta-question intent."""
        self._print("")
        
        if intent == TurnIntent.ASK_DEFAULTS:
            self._print(GlobalDefaults.get_defaults_text())
            self._print("")
            self._print(f"For this question specifically:")
            self._print(help_info.format_defaults())
        
        elif intent == TurnIntent.CLARIFY_DEFAULTS:
            self._print(help_info.format_defaults())
            self._print("")
            if help_info.why_asking:
                self._print(f"These defaults are chosen because: {help_info.why_asking}")
        
        elif intent == TurnIntent.WHY_THIS_QUESTION:
            self._print(help_info.format_why())
            if help_info.impact:
                self._print(f"Impact: {help_info.impact}")
        
        elif intent == TurnIntent.SHOW_OPTIONS:
            self._print(help_info.format_options())
            if help_info.examples:
                self._print("Examples:")
                for ex in help_info.examples:
                    self._print(f"  - {ex}")
        
        elif intent == TurnIntent.HELP:
            self._print(help_info.format_help())
        
        elif intent == TurnIntent.CHANGE_DEFAULT:
            self._print("Please enter your preferred value instead of the default.")
            self._print(help_info.format_options())
        
        elif intent == TurnIntent.FREEFORM_CONSTRAINT:
            self.constraints.append(user_text)
            self._print(f"Noted constraint: {user_text}")
            self._print("I'll keep this in mind. Now, please answer the question:")
        
        self._print("")
    
    def ask_until_answer(
        self,
        question_id: str,
        question_text: str,
        default_value: Any = None,
        allowed_values: Optional[List[str]] = None,
        help_info: Optional[QuestionHelp] = None,
        validator: Optional[Callable[[str], Tuple[bool, str]]] = None,
        show_context: bool = True,
    ) -> Tuple[str, TurnIntent]:
        """
        Ask a question and handle meta-questions until a valid answer is received.
        
        Parameters
        ----------
        question_id : str
            Unique identifier for this question
        question_text : str
            The question to ask
        default_value : Any, optional
            Default value if user presses enter
        allowed_values : List[str], optional
            List of allowed values (for validation)
        help_info : QuestionHelp, optional
            Help information for this question
        validator : Callable, optional
            Custom validator function (returns (is_valid, error_message))
        show_context : bool
            Whether to show current context before asking
            
        Returns
        -------
        Tuple[str, TurnIntent]
            (answer, final_intent) - the validated answer and how it was classified
        """
        if help_info is None:
            help_info = QuestionHelp(
                question_id=question_id,
                question_text=question_text,
                default_value=default_value,
                options=allowed_values,
            )
        
        if show_context and self.context_snapshot:
            self._print("Current context:")
            for key, value in self.context_snapshot.items():
                self._print(f"  {key}: {value}")
            self._print("")
        
        prompt = question_text
        if default_value is not None:
            prompt = f"{question_text} [{default_value}]"
        
        while True:
            user_input = self._input(f"{prompt}: ").strip()
            
            if not user_input and default_value is not None:
                answer = str(default_value)
                intent = TurnIntent.ANSWER
            else:
                intent = interpret_user_turn(user_input)
                answer = user_input
            
            if intent == TurnIntent.BACK:
                self._print("Going back to previous question...")
                return "", TurnIntent.BACK
            
            if intent == TurnIntent.CANCEL:
                self._print("Cancelling workflow...")
                return "", TurnIntent.CANCEL
            
            if intent != TurnIntent.ANSWER and intent != TurnIntent.FREEFORM_CONSTRAINT:
                self._handle_meta_intent(intent, help_info, user_input)
                continue
            
            if intent == TurnIntent.FREEFORM_CONSTRAINT:
                self._handle_meta_intent(intent, help_info, user_input)
                continue
            
            if allowed_values and answer.lower() not in [v.lower() for v in allowed_values]:
                self._print(f"Invalid value. Please choose from: {', '.join(allowed_values)}")
                continue
            
            if validator:
                is_valid, error_msg = validator(answer)
                if not is_valid:
                    self._print(f"Invalid input: {error_msg}")
                    continue
            
            self.history.append({
                "question_id": question_id,
                "question": question_text,
                "answer": answer,
                "intent": intent.value,
            })
            
            return answer, intent
    
    def ask_yes_no(
        self,
        question_id: str,
        question_text: str,
        default: bool = True,
        help_info: Optional[QuestionHelp] = None,
    ) -> Tuple[bool, TurnIntent]:
        """
        Ask a yes/no question with meta-question handling.
        
        Parameters
        ----------
        question_id : str
            Unique identifier for this question
        question_text : str
            The question to ask
        default : bool
            Default value (True = yes, False = no)
        help_info : QuestionHelp, optional
            Help information for this question
            
        Returns
        -------
        Tuple[bool, TurnIntent]
            (answer_bool, final_intent)
        """
        default_str = "y" if default else "n"
        
        def yes_no_validator(answer: str) -> Tuple[bool, str]:
            # Strip whitespace before validation to handle "yes " or " y" inputs
            if answer.strip().lower() in ["y", "yes", "n", "no", ""]:
                return True, ""
            return False, "Please enter 'y' or 'n'"
        
        answer, intent = self.ask_until_answer(
            question_id=question_id,
            question_text=f"{question_text} (y/n)",
            default_value=default_str,
            help_info=help_info,
            validator=yes_no_validator,
            show_context=False,
        )
        
        if intent in (TurnIntent.BACK, TurnIntent.CANCEL):
            return False, intent
        
        # Strip whitespace before checking to handle "yes " or " y" inputs
        return answer.strip().lower() in ["y", "yes", ""], intent
    
    def show_understanding(self, understanding: Dict[str, Any]) -> None:
        """
        Show understanding reflection before asking next question.
        
        Parameters
        ----------
        understanding : Dict[str, Any]
            Dictionary containing what has been understood so far
        """
        self._print("")
        self._print("Here's what I've understood so far:")
        
        for key, value in understanding.items():
            if value is not None:
                self._print(f"  {key}: {value}")
        
        self._print("")
    
    def show_change_summary(self, field_name: str, old_value: Any, new_value: Any) -> None:
        """
        Show what changed after an answer.
        
        Parameters
        ----------
        field_name : str
            Name of the field that changed
        old_value : Any
            Previous value (or None if new)
        new_value : Any
            New value
        """
        if old_value is None:
            self._print(f"Set {field_name}: {new_value}")
        else:
            self._print(f"Updated {field_name}: {old_value} -> {new_value}")
    
    def update_context(self, key: str, value: Any) -> None:
        """Update the context snapshot."""
        self.context_snapshot[key] = value
    
    def get_constraints(self) -> List[str]:
        """Get all collected freeform constraints."""
        return self.constraints.copy()


def create_question_help(
    question_id: str,
    question_text: str,
    default_value: Any = None,
    options: Optional[List[str]] = None,
    why_asking: str = "",
    impact: str = "",
    examples: Optional[List[str]] = None,
) -> QuestionHelp:
    """
    Factory function to create QuestionHelp instances.
    
    Parameters
    ----------
    question_id : str
        Unique identifier for the question
    question_text : str
        The question text
    default_value : Any, optional
        Default value
    options : List[str], optional
        Available options
    why_asking : str, optional
        Explanation of why this question is asked
    impact : str, optional
        Impact of this decision on the workflow
    examples : List[str], optional
        Example answers
        
    Returns
    -------
    QuestionHelp
        Configured help instance
    """
    return QuestionHelp(
        question_id=question_id,
        question_text=question_text,
        default_value=default_value,
        options=options,
        why_asking=why_asking,
        impact=impact,
        examples=examples,
    )


WORKFLOW_QUESTION_HELP = {
    "use_defaults": create_question_help(
        question_id="use_defaults",
        question_text="Use global defaults?",
        default_value="y",
        options=["y", "n", "custom"],
        why_asking="Global defaults provide sensible starting values for voxel pitch, clearances, and other parameters. Using them speeds up configuration.",
        impact="Choosing 'n' or 'custom' will prompt you for each parameter individually.",
        examples=["y (use all defaults)", "n (configure everything)", "custom (select which defaults to use)"],
    ),
    "object_count": create_question_help(
        question_id="object_count",
        question_text="How many objects do you want to generate?",
        default_value=1,
        why_asking="Each object represents a separate vascular network to generate. Multiple objects can be generated in sequence.",
        impact="More objects means longer total generation time but allows batch processing.",
        examples=["1", "3", "5"],
    ),
    "object_description": create_question_help(
        question_id="object_description",
        question_text="Describe the vascular network you want to generate",
        why_asking="Your description helps the agent understand what kind of network to create, including topology, density, and any special requirements.",
        impact="A detailed description leads to better initial understanding and fewer follow-up questions.",
        examples=[
            "A dense hepatic vascular tree with 3 inlets and 50 terminals",
            "A simple Y-shaped bifurcation for testing",
            "A coronary-like network with tortuous vessels",
        ],
    ),
    "plan_selection": create_question_help(
        question_id="plan_selection",
        question_text="Which generation plan would you like to use?",
        options=["A", "B", "C"],
        why_asking="Different plans use different generation strategies optimized for different network types.",
        impact="The chosen plan determines which algorithms and parameters are used for generation.",
    ),
    "frame_of_reference": create_question_help(
        question_id="frame_of_reference",
        question_text="How should spatial directions be interpreted?",
        default_value="face-based",
        options=["face-based", "explicit"],
        why_asking="Frame of reference determines how terms like 'left', 'right', 'top', 'bottom' are interpreted.",
        impact="Face-based mode uses the domain faces as reference. Explicit mode requires you to define the coordinate system.",
    ),
    "domain_size": create_question_help(
        question_id="domain_size",
        question_text="What size should the generation domain be?",
        default_value="0.01 x 0.01 x 0.01 m",
        why_asking="The domain defines the 3D space where the vascular network will be generated.",
        impact="Larger domains allow for more complex networks but require more computation.",
        examples=["0.01 x 0.01 x 0.01 m", "0.02 x 0.015 x 0.01 m", "5 x 5 x 5 mm"],
    ),
}


def get_question_help(question_id: str) -> Optional[QuestionHelp]:
    """Get help information for a specific question."""
    return WORKFLOW_QUESTION_HELP.get(question_id)
