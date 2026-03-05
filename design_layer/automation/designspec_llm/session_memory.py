"""
Session Memory for DesignSpec LLM Agent

This module provides persistent memory of decisions, error resolutions,
and user preferences across conversation turns. The agent can reference
past decisions and successful error fixes.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class Decision:
    """A decision made by the agent during the session."""
    turn_number: int
    timestamp: str
    decision: str           # "used taper_factor=0.8 for gradual taper"
    reasoning: str          # Why this decision was made

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_number": self.turn_number,
            "timestamp": self.timestamp,
            "decision": self.decision,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Decision":
        return cls(
            turn_number=data.get("turn_number", 0),
            timestamp=data.get("timestamp", ""),
            decision=data.get("decision", ""),
            reasoning=data.get("reasoning", ""),
        )


@dataclass
class ErrorResolution:
    """A record of an error and how it was resolved."""
    error_pattern: str      # "PITCH_TOO_LARGE"
    error_message: str      # Full error message
    resolution: str         # "set voxel_pitch = domain_scale/100"
    success: bool           # Whether the fix worked
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_pattern": self.error_pattern,
            "error_message": self.error_message,
            "resolution": self.resolution,
            "success": self.success,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ErrorResolution":
        return cls(
            error_pattern=data.get("error_pattern", ""),
            error_message=data.get("error_message", ""),
            resolution=data.get("resolution", ""),
            success=data.get("success", False),
            timestamp=data.get("timestamp", ""),
        )


@dataclass
class SessionMemory:
    """
    Persistent memory for the session.

    Tracks decisions, error resolutions, and user preferences
    to provide context across multiple conversation turns.
    """
    session_id: str
    created_at: str
    decisions: List[Decision] = field(default_factory=list)
    error_resolutions: List[ErrorResolution] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "decisions": [d.to_dict() for d in self.decisions],
            "error_resolutions": [er.to_dict() for er in self.error_resolutions],
            "user_preferences": self.user_preferences,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionMemory":
        return cls(
            session_id=data.get("session_id", ""),
            created_at=data.get("created_at", ""),
            decisions=[Decision.from_dict(d) for d in data.get("decisions", [])],
            error_resolutions=[
                ErrorResolution.from_dict(er) for er in data.get("error_resolutions", [])
            ],
            user_preferences=data.get("user_preferences", {}),
        )

    @classmethod
    def create_new(cls, session_id: str) -> "SessionMemory":
        """Create a new session memory."""
        return cls(
            session_id=session_id,
            created_at=datetime.now().isoformat(),
        )

    def add_decision(
        self,
        turn_number: int,
        decision: str,
        reasoning: str = "",
    ) -> None:
        """
        Add a decision to the memory.

        Parameters
        ----------
        turn_number : int
            The conversation turn number
        decision : str
            Description of the decision
        reasoning : str, optional
            Why the decision was made
        """
        self.decisions.append(
            Decision(
                turn_number=turn_number,
                timestamp=datetime.now().isoformat(),
                decision=decision,
                reasoning=reasoning,
            )
        )

    def add_error_resolution(
        self,
        error_pattern: str,
        error_message: str,
        resolution: str,
        success: bool = True,
    ) -> None:
        """
        Add an error resolution to the memory.

        Parameters
        ----------
        error_pattern : str
            Type/pattern of error
        error_message : str
            Full error message
        resolution : str
            How the error was resolved
        success : bool
            Whether the resolution worked
        """
        self.error_resolutions.append(
            ErrorResolution(
                error_pattern=error_pattern,
                error_message=error_message,
                resolution=resolution,
                success=success,
                timestamp=datetime.now().isoformat(),
            )
        )

    def set_preference(self, key: str, value: Any) -> None:
        """
        Set a user preference.

        Parameters
        ----------
        key : str
            Preference key
        value : Any
            Preference value
        """
        self.user_preferences[key] = value

    def get_recent_decisions(self, limit: int = 3) -> List[Decision]:
        """Get the most recent decisions."""
        return sorted(self.decisions, key=lambda d: d.turn_number, reverse=True)[:limit]

    def get_successful_error_resolutions(self) -> List[ErrorResolution]:
        """Get all successful error resolutions."""
        return [er for er in self.error_resolutions if er.success]

    def to_summary_dict(self) -> Dict[str, Any]:
        """
        Create a summary suitable for compact context.

        Returns
        -------
        dict
            Summary with recent decisions and successful error resolutions
        """
        recent_decisions = self.get_recent_decisions(limit=3)
        successful_resolutions = self.get_successful_error_resolutions()

        return {
            "recent_decisions": [d.to_dict() for d in recent_decisions],
            "successful_error_resolutions": [
                er.to_dict() for er in successful_resolutions[-5:]  # Last 5
            ],
            "user_preferences": self.user_preferences,
        }

    def to_prompt_text(self) -> str:
        """Convert memory to text for LLM prompt."""
        lines = []

        # Recent decisions
        recent = self.get_recent_decisions(limit=3)
        if recent:
            lines.append("## Recent Decisions")
            for decision in recent:
                lines.append(f"- Turn {decision.turn_number}: {decision.decision}")
                if decision.reasoning:
                    lines.append(f"  Reasoning: {decision.reasoning}")

        # Successful error resolutions
        successful = self.get_successful_error_resolutions()
        if successful:
            lines.append("")
            lines.append("## Successful Error Resolutions")
            for resolution in successful[-5:]:  # Last 5
                lines.append(f"- {resolution.error_pattern}: {resolution.resolution}")

        # User preferences
        if self.user_preferences:
            lines.append("")
            lines.append("## User Preferences")
            for key, value in self.user_preferences.items():
                lines.append(f"- {key}: {value}")

        return "\n".join(lines) if lines else ""
