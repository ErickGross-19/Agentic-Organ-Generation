"""
V5 Single Agent Organ Generation

Goal-driven controller that feels like a confident engineer with:
- Narration and check-ins
- Always asks before generation and postprocess
- Safe fixes one-at-a-time
- Interrupts/backtracking/undo support
"""

from .controller import SingleAgentOrganGeneratorV5, ControllerConfig, ControllerStatus, RunResult
from .world_model import (
    WorldModel,
    Fact,
    FactProvenance,
    OpenQuestion,
    DecisionPoint,
    Approval,
    Artifact,
    HistoryEntry,
    Plan,
    TraceEvent,
)
from .goals import Goal, GoalStatus, GoalTracker, GOAL_DEFINITIONS
from .policies import (
    SafeFixPolicy,
    ApprovalPolicy,
    CapabilitySelectionPolicy,
    FixCandidate,
    FixSafety,
)
from .plan_synthesizer import PlanSynthesizer, PlanParameters, PlanRisk
from .io import (
    BaseIOAdapter,
    IOMessage,
    IOMessageKind,
    CLIIOAdapter,
    GUIIOAdapter,
)

__all__ = [
    "SingleAgentOrganGeneratorV5",
    "ControllerConfig",
    "ControllerStatus",
    "RunResult",
    "WorldModel",
    "Fact",
    "FactProvenance",
    "OpenQuestion",
    "DecisionPoint",
    "Approval",
    "Artifact",
    "HistoryEntry",
    "Plan",
    "TraceEvent",
    "Goal",
    "GoalStatus",
    "GoalTracker",
    "GOAL_DEFINITIONS",
    "SafeFixPolicy",
    "ApprovalPolicy",
    "CapabilitySelectionPolicy",
    "FixCandidate",
    "FixSafety",
    "PlanSynthesizer",
    "PlanParameters",
    "PlanRisk",
    "BaseIOAdapter",
    "IOMessage",
    "IOMessageKind",
    "CLIIOAdapter",
    "GUIIOAdapter",
]
