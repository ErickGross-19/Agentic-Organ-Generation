"""
Goals Module - Progress is Goal Satisfaction

Defines the goals that drive the V5 controller. Progress is measured by
goal satisfaction rather than state transitions.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from .world_model import WorldModel


class GoalStatus(Enum):
    """Status of a goal."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    SATISFIED = "satisfied"
    FAILED = "failed"


@dataclass
class Goal:
    """A goal in the V5 system."""
    goal_id: str
    name: str
    description: str
    preconditions: List[str]
    check_satisfied: Callable[["WorldModel"], bool]
    priority: int = 0
    requires_approval: bool = False
    approval_type: Optional[str] = None
    
    def get_status(self, world_model: "WorldModel", goal_statuses: Dict[str, GoalStatus]) -> GoalStatus:
        """Get the current status of this goal."""
        for precondition in self.preconditions:
            if goal_statuses.get(precondition) != GoalStatus.SATISFIED:
                return GoalStatus.BLOCKED
        
        if self.requires_approval and self.approval_type:
            if not world_model.is_approved(self.approval_type):
                return GoalStatus.BLOCKED
        
        if self.check_satisfied(world_model):
            return GoalStatus.SATISFIED
        
        return GoalStatus.IN_PROGRESS
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "name": self.name,
            "description": self.description,
            "preconditions": self.preconditions,
            "priority": self.priority,
            "requires_approval": self.requires_approval,
            "approval_type": self.approval_type,
        }


def check_spec_minimum_complete(world_model: "WorldModel") -> bool:
    """Check if the minimum viable spec is complete."""
    required_fields = [
        "domain.type",
        "domain.size",
        "topology.kind",
        "inlet.position",
        "inlet.radius",
    ]
    
    topology_kind = world_model.get_fact_value("topology.kind", "tree")
    
    if topology_kind in ("path", "backbone", "loop"):
        required_fields.extend(["outlet.position", "outlet.radius"])
    
    for field in required_fields:
        if not world_model.has_fact(field):
            return False
    
    return len(world_model.get_open_questions()) == 0


def check_spec_compiled(world_model: "WorldModel") -> bool:
    """Check if the spec has been compiled."""
    artifact = world_model.get_artifact("compiled_spec")
    return artifact is not None


def check_pregen_verified(world_model: "WorldModel") -> bool:
    """Check if pre-generation verification passed."""
    fact = world_model.get_fact("pregen_verified")
    return fact is not None and fact.value is True


def check_generation_approved(world_model: "WorldModel") -> bool:
    """Check if generation is approved."""
    return world_model.is_approved("generation")


def check_generation_done(world_model: "WorldModel") -> bool:
    """Check if generation is complete."""
    artifact = world_model.get_artifact("generated_network")
    return artifact is not None


def check_postprocess_approved(world_model: "WorldModel") -> bool:
    """Check if postprocess is approved."""
    return world_model.is_approved("postprocess")


def check_postprocess_done(world_model: "WorldModel") -> bool:
    """Check if postprocess is complete."""
    artifact = world_model.get_artifact("postprocessed_mesh")
    return artifact is not None


def check_validation_passed(world_model: "WorldModel") -> bool:
    """Check if validation passed."""
    fact = world_model.get_fact("validation_passed")
    return fact is not None and fact.value is True


def check_outputs_packaged(world_model: "WorldModel") -> bool:
    """Check if outputs are packaged."""
    artifact = world_model.get_artifact("output_package")
    return artifact is not None


GOAL_DEFINITIONS: Dict[str, Goal] = {
    "spec_minimum_complete": Goal(
        goal_id="spec_minimum_complete",
        name="Minimum Viable Spec Complete",
        description="All required fields for the chosen topology are filled",
        preconditions=[],
        check_satisfied=check_spec_minimum_complete,
        priority=100,
    ),
    "spec_compiled": Goal(
        goal_id="spec_compiled",
        name="Spec Compiled",
        description="Design spec has been compiled to executable form",
        preconditions=["spec_minimum_complete"],
        check_satisfied=check_spec_compiled,
        priority=90,
    ),
    "pregen_verified": Goal(
        goal_id="pregen_verified",
        name="Pre-Generation Verified",
        description="Feasibility and schema checks passed",
        preconditions=["spec_compiled"],
        check_satisfied=check_pregen_verified,
        priority=80,
    ),
    "generation_approved": Goal(
        goal_id="generation_approved",
        name="Generation Approved",
        description="User has approved generation",
        preconditions=["pregen_verified"],
        check_satisfied=check_generation_approved,
        priority=70,
        requires_approval=True,
        approval_type="generation",
    ),
    "generation_done": Goal(
        goal_id="generation_done",
        name="Generation Complete",
        description="Vascular network has been generated",
        preconditions=["generation_approved"],
        check_satisfied=check_generation_done,
        priority=60,
    ),
    "postprocess_approved": Goal(
        goal_id="postprocess_approved",
        name="Postprocess Approved",
        description="User has approved postprocessing",
        preconditions=["generation_done"],
        check_satisfied=check_postprocess_approved,
        priority=50,
        requires_approval=True,
        approval_type="postprocess",
    ),
    "postprocess_done": Goal(
        goal_id="postprocess_done",
        name="Postprocess Complete",
        description="Embedding/voxelization/repair/export complete",
        preconditions=["postprocess_approved"],
        check_satisfied=check_postprocess_done,
        priority=40,
    ),
    "validation_passed": Goal(
        goal_id="validation_passed",
        name="Validation Passed",
        description="All validation checks passed",
        preconditions=["postprocess_done"],
        check_satisfied=check_validation_passed,
        priority=30,
    ),
    "outputs_packaged": Goal(
        goal_id="outputs_packaged",
        name="Outputs Packaged",
        description="Final deliverables are packaged",
        preconditions=["validation_passed"],
        check_satisfied=check_outputs_packaged,
        priority=20,
    ),
}


class GoalTracker:
    """Tracks goal progress and determines next actions."""
    
    def __init__(self, world_model: "WorldModel"):
        self.world_model = world_model
        self._goal_statuses: Dict[str, GoalStatus] = {}
        self._update_statuses()
    
    def _update_statuses(self) -> None:
        """Update all goal statuses."""
        for goal_id in GOAL_DEFINITIONS:
            self._goal_statuses[goal_id] = GoalStatus.NOT_STARTED
        
        changed = True
        while changed:
            changed = False
            for goal_id, goal in GOAL_DEFINITIONS.items():
                old_status = self._goal_statuses[goal_id]
                new_status = goal.get_status(self.world_model, self._goal_statuses)
                if new_status != old_status:
                    self._goal_statuses[goal_id] = new_status
                    changed = True
    
    def get_status(self, goal_id: str) -> GoalStatus:
        """Get the status of a specific goal."""
        self._update_statuses()
        return self._goal_statuses.get(goal_id, GoalStatus.NOT_STARTED)
    
    def get_all_statuses(self) -> Dict[str, GoalStatus]:
        """Get all goal statuses."""
        self._update_statuses()
        return self._goal_statuses.copy()
    
    def get_next_goal(self) -> Optional[Goal]:
        """Get the next goal to work on (highest priority unsatisfied goal)."""
        self._update_statuses()
        
        candidates = []
        for goal_id, goal in GOAL_DEFINITIONS.items():
            status = self._goal_statuses[goal_id]
            if status == GoalStatus.IN_PROGRESS:
                candidates.append(goal)
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda g: g.priority, reverse=True)
        return candidates[0]
    
    def get_blocked_goals(self) -> List[Goal]:
        """Get all blocked goals."""
        self._update_statuses()
        return [
            GOAL_DEFINITIONS[goal_id]
            for goal_id, status in self._goal_statuses.items()
            if status == GoalStatus.BLOCKED
        ]
    
    def get_satisfied_goals(self) -> List[Goal]:
        """Get all satisfied goals."""
        self._update_statuses()
        return [
            GOAL_DEFINITIONS[goal_id]
            for goal_id, status in self._goal_statuses.items()
            if status == GoalStatus.SATISFIED
        ]
    
    def is_complete(self) -> bool:
        """Check if all goals are satisfied."""
        self._update_statuses()
        return all(
            status == GoalStatus.SATISFIED
            for status in self._goal_statuses.values()
        )
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get a summary of goal progress."""
        self._update_statuses()
        
        satisfied = []
        in_progress = []
        blocked = []
        
        for goal_id, status in self._goal_statuses.items():
            goal = GOAL_DEFINITIONS[goal_id]
            entry = {"goal_id": goal_id, "name": goal.name}
            
            if status == GoalStatus.SATISFIED:
                satisfied.append(entry)
            elif status == GoalStatus.IN_PROGRESS:
                in_progress.append(entry)
            elif status == GoalStatus.BLOCKED:
                entry["blocked_by"] = [
                    p for p in goal.preconditions
                    if self._goal_statuses.get(p) != GoalStatus.SATISFIED
                ]
                blocked.append(entry)
        
        return {
            "satisfied": satisfied,
            "in_progress": in_progress,
            "blocked": blocked,
            "is_complete": self.is_complete(),
            "next_goal": self.get_next_goal().goal_id if self.get_next_goal() else None,
        }
