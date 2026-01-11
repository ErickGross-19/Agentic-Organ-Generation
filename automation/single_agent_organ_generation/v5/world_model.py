"""
World Model - Single Source of Truth for V5

The world model stores all facts, their provenance, goals, approvals,
artifacts, and history/undo stack. This replaces the state machine approach
with a goal-driven architecture.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set
from enum import Enum
from datetime import datetime
import hashlib
import json
import copy


class FactProvenance(Enum):
    """Source of a fact."""
    USER = "user"
    INFERRED = "inferred"
    DEFAULT = "default"
    SAFE_FIX = "safe_fix"
    SYSTEM = "system"


@dataclass
class Fact:
    """A single fact in the world model."""
    field: str
    value: Any
    provenance: FactProvenance
    confidence: float = 1.0
    timestamp: str = ""
    reason: Optional[str] = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "value": self.value,
            "provenance": self.provenance.value,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "reason": self.reason,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Fact":
        return cls(
            field=d["field"],
            value=d["value"],
            provenance=FactProvenance(d["provenance"]),
            confidence=d.get("confidence", 1.0),
            timestamp=d.get("timestamp", ""),
            reason=d.get("reason"),
        )


@dataclass
class OpenQuestion:
    """An open question that needs to be resolved."""
    question_id: str
    field: str
    question_text: str
    why_it_matters: str
    options: Optional[List[str]] = None
    default_value: Optional[Any] = None
    priority: int = 0
    asked_at: Optional[str] = None
    acceptable_answers: Optional[List[str]] = None
    
    @property
    def question(self) -> str:
        """Alias for question_text for controller compatibility."""
        return self.question_text
    
    @property
    def why(self) -> str:
        """Alias for why_it_matters for controller compatibility."""
        return self.why_it_matters
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "field": self.field,
            "question_text": self.question_text,
            "why_it_matters": self.why_it_matters,
            "options": self.options,
            "default_value": self.default_value,
            "priority": self.priority,
            "asked_at": self.asked_at,
            "acceptable_answers": self.acceptable_answers,
        }


@dataclass
class DecisionPoint:
    """A decision point that can be revisited."""
    decision_id: str
    question_id: str
    what_it_resolved: str
    chosen_answer: Any
    alternatives: List[Any]
    dependencies: List[str]
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "question_id": self.question_id,
            "what_it_resolved": self.what_it_resolved,
            "chosen_answer": self.chosen_answer,
            "alternatives": self.alternatives,
            "dependencies": self.dependencies,
            "timestamp": self.timestamp,
        }


@dataclass
class Approval:
    """An approval for generation or postprocess."""
    approval_type: str
    spec_hash: str
    approved: bool
    approved_at: Optional[str] = None
    runtime_estimate: Optional[str] = None
    expected_outputs: Optional[List[str]] = None
    assumptions: Optional[List[str]] = None
    risk_flags: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "approval_type": self.approval_type,
            "spec_hash": self.spec_hash,
            "approved": self.approved,
            "approved_at": self.approved_at,
            "runtime_estimate": self.runtime_estimate,
            "expected_outputs": self.expected_outputs,
            "assumptions": self.assumptions,
            "risk_flags": self.risk_flags,
        }


@dataclass
class Artifact:
    """A generated artifact."""
    artifact_type: str
    path: str
    created_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_type": self.artifact_type,
            "path": self.path,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


@dataclass
class HistoryEntry:
    """An entry in the undo stack."""
    entry_id: str
    action: str
    description: str
    patch: Dict[str, Any]
    inverse_patch: Dict[str, Any]
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "action": self.action,
            "description": self.description,
            "patch": self.patch,
            "inverse_patch": self.inverse_patch,
            "timestamp": self.timestamp,
        }


@dataclass
class Plan:
    """A proposed generation plan."""
    plan_id: str
    name: str
    interpretation: str
    geometry_strategy: str
    parameter_draft: Dict[str, Any]
    risks: List[str]
    cost_estimate: str
    what_needed_from_user: List[str]
    patch_set: Dict[str, Any]
    recommended: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "name": self.name,
            "interpretation": self.interpretation,
            "geometry_strategy": self.geometry_strategy,
            "parameter_draft": self.parameter_draft,
            "risks": self.risks,
            "cost_estimate": self.cost_estimate,
            "what_needed_from_user": self.what_needed_from_user,
            "patch_set": self.patch_set,
            "recommended": self.recommended,
        }


@dataclass
class TraceEvent:
    """A trace event for GUI timeline."""
    event_type: str
    message: str
    timestamp: str = ""
    data: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "message": self.message,
            "timestamp": self.timestamp,
            "data": self.data,
        }


class WorldModel:
    """
    Single source of truth for the V5 controller.
    
    Stores:
    - facts (domain, topology, ports, constraints, units, etc.)
    - provenance per fact: user | inferred | default | fix
    - confidence + timestamps
    - open questions (missing/ambiguous fields + why it matters)
    - decisions (chosen plan + rationale)
    - approvals (generation, postprocess) keyed to a spec hash
    - artifacts (paths to spec, manifests, meshes, logs, reports)
    - history/undo stack (snapshots or patches)
    """
    
    def __init__(self):
        self._facts: Dict[str, Fact] = {}
        self._open_questions: Dict[str, OpenQuestion] = {}
        self._decisions: Dict[str, DecisionPoint] = {}
        self._approvals: Dict[str, Approval] = {}
        self._artifacts: Dict[str, Artifact] = {}
        self._history: List[HistoryEntry] = []
        self._plans: Dict[str, Plan] = {}
        self._selected_plan_id: Optional[str] = None
        self._trace_events: List[TraceEvent] = []
        self._event_log: List[Dict[str, Any]] = []
        self._last_user_intent: Optional[str] = None
        self._last_user_request_type: Optional[str] = None
        self._entry_counter: int = 0
        self._pending_patches: Dict[str, Any] = {}
    
    @property
    def facts(self) -> Dict[str, Fact]:
        """Read-only access to facts dictionary."""
        return self._facts
    
    @property
    def open_questions(self) -> Dict[str, "OpenQuestion"]:
        """Read-only access to open questions dictionary."""
        return self._open_questions
    
    @property
    def plans(self) -> Dict[str, Plan]:
        """Read-only access to plans dictionary."""
        return self._plans
    
    @property
    def history(self) -> List[HistoryEntry]:
        """Read-only access to history list."""
        return self._history
    
    @property
    def selected_plan(self) -> Optional[Plan]:
        """Get the currently selected plan."""
        if self._selected_plan_id and self._selected_plan_id in self._plans:
            return self._plans[self._selected_plan_id]
        return None
    
    def _generate_entry_id(self) -> str:
        """Generate a unique entry ID."""
        self._entry_counter += 1
        return f"entry_{self._entry_counter}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    def compute_spec_hash(self) -> str:
        """Compute a hash of the current spec-relevant facts."""
        spec_fields = [
            "domain.type", "domain.size", "domain.center",
            "topology.kind", "topology.terminal_mode",
            "inlet.position", "inlet.radius",
            "outlet.position", "outlet.radius",
            "colonization.influence_radius", "colonization.kill_radius",
            "colonization.step_size", "colonization.min_radius",
            "colonization.max_steps", "colonization.initial_radius",
        ]
        
        spec_data = {}
        for field in spec_fields:
            if field in self._facts:
                spec_data[field] = self._facts[field].value
        
        spec_json = json.dumps(spec_data, sort_keys=True, default=str)
        return hashlib.sha256(spec_json.encode()).hexdigest()[:16]
    
    def set_fact(
        self,
        field: str,
        value: Any,
        provenance: FactProvenance,
        confidence: float = 1.0,
        reason: Optional[str] = None,
        record_history: bool = True,
    ) -> None:
        """Set a fact in the world model."""
        old_fact = self._facts.get(field)
        
        new_fact = Fact(
            field=field,
            value=value,
            provenance=provenance,
            confidence=confidence,
            reason=reason,
        )
        
        if record_history:
            if old_fact is not None:
                patch = {"field": field, "new_value": value, "new_provenance": provenance.value}
                inverse_patch = {"field": field, "old_value": old_fact.value, "old_provenance": old_fact.provenance.value}
                
                entry = HistoryEntry(
                    entry_id=self._generate_entry_id(),
                    action="set_fact",
                    description=f"Changed {field}: {old_fact.value} -> {value}",
                    patch=patch,
                    inverse_patch=inverse_patch,
                )
                self._history.append(entry)
            else:
                patch = {"field": field, "new_value": value, "new_provenance": provenance.value}
                inverse_patch = {"field": field, "action": "delete"}
                
                entry = HistoryEntry(
                    entry_id=self._generate_entry_id(),
                    action="create_fact",
                    description=f"Created {field}: {value}",
                    patch=patch,
                    inverse_patch=inverse_patch,
                )
                self._history.append(entry)
        
        self._facts[field] = new_fact
        
        if self._is_geometry_relevant(field):
            self._invalidate_approvals()
    
    def get_fact(self, field: str) -> Optional[Fact]:
        """Get a fact from the world model."""
        return self._facts.get(field)
    
    def get_fact_value(self, field: str, default: Any = None) -> Any:
        """Get the value of a fact, or default if not set."""
        fact = self._facts.get(field)
        return fact.value if fact else default
    
    def has_fact(self, field: str) -> bool:
        """Check if a fact exists."""
        return field in self._facts
    
    def get_facts_by_provenance(self, provenance: FactProvenance) -> Dict[str, Fact]:
        """Get all facts with a specific provenance."""
        return {k: v for k, v in self._facts.items() if v.provenance == provenance}
    
    def get_confirmed_facts(self) -> Dict[str, Fact]:
        """Get all user-confirmed facts."""
        return self.get_facts_by_provenance(FactProvenance.USER)
    
    def get_inferred_facts(self) -> Dict[str, Fact]:
        """Get all inferred/default facts."""
        return {
            k: v for k, v in self._facts.items()
            if v.provenance in (FactProvenance.INFERRED, FactProvenance.DEFAULT)
        }
    
    def _is_geometry_relevant(self, field: str) -> bool:
        """Check if a field affects geometry (invalidates approvals)."""
        geometry_prefixes = [
            "domain.", "topology.", "inlet.", "outlet.",
            "colonization.", "geometry.", "port.",
        ]
        return any(field.startswith(prefix) for prefix in geometry_prefixes)
    
    def _invalidate_approvals(self) -> None:
        """Invalidate all approvals due to spec change.
        
        Clears all approvals rather than just marking them as False,
        to avoid misleading status computations that treat existing
        approvals with approved=False as "waiting for approval".
        """
        self._approvals.clear()
    
    def add_open_question(self, question: OpenQuestion) -> None:
        """Add an open question."""
        self._open_questions[question.question_id] = question
    
    def remove_open_question(self, question_id: str) -> None:
        """Remove an open question (when answered)."""
        if question_id in self._open_questions:
            del self._open_questions[question_id]
    
    def get_open_questions(self) -> List[OpenQuestion]:
        """Get all open questions sorted by priority."""
        return sorted(
            self._open_questions.values(),
            key=lambda q: q.priority,
            reverse=True,
        )
    
    def get_highest_priority_question(self) -> Optional[OpenQuestion]:
        """Get the highest priority open question."""
        questions = self.get_open_questions()
        return questions[0] if questions else None
    
    def answer_question(self, question_id: str, answer: Any) -> bool:
        """
        Answer an open question.
        
        This method:
        1. Sets the relevant fact with USER provenance
        2. Removes the open question
        3. Records a decision point
        4. Invalidates approvals if geometry-relevant
        
        Parameters
        ----------
        question_id : str
            The ID of the question to answer
        answer : Any
            The user's answer
            
        Returns
        -------
        bool
            True if the question was found and answered
        """
        question = self._open_questions.get(question_id)
        if not question:
            return False
        
        self.set_fact(
            field=question.field,
            value=answer,
            provenance=FactProvenance.USER,
            reason=f"User answered question: {question.question_text}",
        )
        
        decision = DecisionPoint(
            decision_id=self._generate_entry_id(),
            question_id=question_id,
            what_it_resolved=question.field,
            chosen_answer=answer,
            alternatives=question.options or [],
            dependencies=[question.field],
        )
        self.record_decision(decision)
        
        self.remove_open_question(question_id)
        
        return True
    
    def record_decision(self, decision: DecisionPoint) -> None:
        """Record a decision point."""
        self._decisions[decision.decision_id] = decision
    
    def get_decision(self, decision_id: str) -> Optional[DecisionPoint]:
        """Get a decision by ID."""
        return self._decisions.get(decision_id)
    
    def set_approval(
        self,
        approval_type: str,
        approved: bool,
        runtime_estimate: Optional[str] = None,
        expected_outputs: Optional[List[str]] = None,
        assumptions: Optional[List[str]] = None,
        risk_flags: Optional[List[str]] = None,
    ) -> None:
        """Set an approval (generation or postprocess)."""
        spec_hash = self.compute_spec_hash()
        
        approval = Approval(
            approval_type=approval_type,
            spec_hash=spec_hash,
            approved=approved,
            approved_at=datetime.now().isoformat() if approved else None,
            runtime_estimate=runtime_estimate,
            expected_outputs=expected_outputs,
            assumptions=assumptions,
            risk_flags=risk_flags,
        )
        
        self._approvals[approval_type] = approval
    
    def is_approved(self, approval_type: str) -> bool:
        """Check if an approval is valid for current spec."""
        approval = self._approvals.get(approval_type)
        if not approval or not approval.approved:
            return False
        
        current_hash = self.compute_spec_hash()
        return approval.spec_hash == current_hash
    
    def get_approval(self, approval_type: str) -> Optional[Approval]:
        """Get an approval by type."""
        return self._approvals.get(approval_type)
    
    def add_artifact(
        self,
        artifact_or_type: "Artifact | str",
        path_or_data: "str | Dict[str, Any] | None" = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add an artifact.
        
        Can be called in two ways:
        1. add_artifact(artifact: Artifact) - pass a pre-constructed Artifact
        2. add_artifact(artifact_type: str, path_or_data, metadata=None) - convenience form
        
        Parameters
        ----------
        artifact_or_type : Artifact or str
            Either an Artifact object or the artifact type string
        path_or_data : str or dict, optional
            Path string or data dict (for convenience form)
        metadata : dict, optional
            Additional metadata (for convenience form)
        """
        if isinstance(artifact_or_type, Artifact):
            self._artifacts[artifact_or_type.artifact_type] = artifact_or_type
        else:
            artifact_type = artifact_or_type
            if isinstance(path_or_data, dict):
                artifact = Artifact(
                    artifact_type=artifact_type,
                    path="",
                    metadata={"data": path_or_data, **(metadata or {})},
                )
            else:
                artifact = Artifact(
                    artifact_type=artifact_type,
                    path=path_or_data or "",
                    metadata=metadata or {},
                )
            self._artifacts[artifact_type] = artifact
    
    def get_artifact(self, artifact_type: str) -> Optional[Any]:
        """
        Get an artifact by type.
        
        Returns the artifact data directly if it was stored as data,
        or the path if it was stored as a path, or the Artifact object
        if neither is available.
        
        Parameters
        ----------
        artifact_type : str
            The type of artifact to retrieve
            
        Returns
        -------
        Any
            The artifact data, path, or Artifact object
        """
        artifact = self._artifacts.get(artifact_type)
        if artifact is None:
            return None
        
        if "data" in artifact.metadata:
            return artifact.metadata["data"]
        elif artifact.path:
            return artifact.path
        else:
            return artifact
    
    def get_all_artifacts(self) -> Dict[str, Artifact]:
        """Get all artifacts."""
        return self._artifacts.copy()
    
    def add_plan(self, plan: Plan) -> None:
        """Add a proposed plan."""
        self._plans[plan.plan_id] = plan
    
    def get_plan(self, plan_id: str) -> Optional[Plan]:
        """Get a plan by ID."""
        return self._plans.get(plan_id)
    
    def get_all_plans(self) -> Dict[str, Plan]:
        """Get all plans."""
        return self._plans.copy()
    
    def select_plan(self, plan_id: str) -> bool:
        """Select a plan."""
        if plan_id in self._plans:
            self._selected_plan_id = plan_id
            return True
        return False
    
    def get_selected_plan(self) -> Optional[Plan]:
        """Get the selected plan."""
        if self._selected_plan_id:
            return self._plans.get(self._selected_plan_id)
        return None
    
    def clear_plans(self) -> None:
        """Clear all plans."""
        self._plans.clear()
        self._selected_plan_id = None
    
    def add_trace_event(self, event_type: str, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Add a trace event for GUI timeline."""
        event = TraceEvent(event_type=event_type, message=message, data=data)
        self._trace_events.append(event)
    
    def get_trace_events(self) -> List[TraceEvent]:
        """Get all trace events."""
        return self._trace_events.copy()
    
    def add_event(self, event: Dict[str, Any]) -> None:
        """Add an event to the event log."""
        event["timestamp"] = datetime.now().isoformat()
        self._event_log.append(event)
    
    def get_event_log(self) -> List[Dict[str, Any]]:
        """Get the event log."""
        return self._event_log.copy()
    
    def set_last_user_intent(self, intent: str, request_type: Optional[str] = None) -> None:
        """Set the last user intent."""
        self._last_user_intent = intent
        self._last_user_request_type = request_type
    
    def get_last_user_intent(self) -> Tuple[Optional[str], Optional[str]]:
        """Get the last user intent and request type."""
        return self._last_user_intent, self._last_user_request_type
    
    def undo_last(self) -> Optional[HistoryEntry]:
        """Undo the last change.
        
        Handles both fact modifications and fact creations:
        - For set_fact: restores the previous value
        - For create_fact: deletes the fact entirely
        
        Also invalidates approvals if the undone change was geometry-relevant.
        """
        if not self._history:
            return None
        
        entry = self._history.pop()
        field = entry.inverse_patch.get("field")
        
        if entry.action == "set_fact":
            old_value = entry.inverse_patch["old_value"]
            old_provenance = FactProvenance(entry.inverse_patch["old_provenance"])
            
            self._facts[field] = Fact(
                field=field,
                value=old_value,
                provenance=old_provenance,
            )
        elif entry.action == "create_fact":
            if entry.inverse_patch.get("action") == "delete" and field in self._facts:
                del self._facts[field]
        
        if field and self._is_geometry_relevant(field):
            self._invalidate_approvals()
        
        return entry
    
    def undo_to_entry(self, entry_id: str) -> List[HistoryEntry]:
        """Undo all changes back to a specific entry."""
        undone = []
        
        while self._history:
            if self._history[-1].entry_id == entry_id:
                break
            entry = self.undo_last()
            if entry:
                undone.append(entry)
        
        return undone
    
    def get_history(self) -> List[HistoryEntry]:
        """Get the history stack."""
        return self._history.copy()
    
    def get_recent_changes(self, count: int = 5) -> List[HistoryEntry]:
        """Get recent changes."""
        return self._history[-count:] if self._history else []
    
    def apply_patch(
        self,
        patches: Dict[str, Any],
        provenance: FactProvenance,
        reason: Optional[str] = None,
    ) -> None:
        """Apply a set of patches to the world model."""
        for field, value in patches.items():
            self.set_fact(field, value, provenance, reason=reason)
    
    def get_living_spec_summary(self) -> Dict[str, Any]:
        """Get a summary of the current spec state."""
        confirmed = {}
        inferred = {}
        open_questions = []
        recent_changes = []
        
        for field, fact in self._facts.items():
            if fact.provenance == FactProvenance.USER:
                confirmed[field] = fact.value
            elif fact.provenance in (FactProvenance.INFERRED, FactProvenance.DEFAULT):
                inferred[field] = {"value": fact.value, "reason": fact.reason}
        
        for question in self.get_open_questions():
            open_questions.append({
                "field": question.field,
                "question": question.question_text,
                "why": question.why_it_matters,
            })
        
        for entry in self.get_recent_changes():
            recent_changes.append({
                "action": entry.action,
                "description": entry.description,
                "timestamp": entry.timestamp,
            })
        
        return {
            "confirmed_facts": confirmed,
            "inferred_facts": inferred,
            "open_questions": open_questions,
            "recent_changes": recent_changes,
            "spec_hash": self.compute_spec_hash(),
            "generation_approved": self.is_approved("generation"),
            "postprocess_approved": self.is_approved("postprocess"),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize the world model to a dictionary."""
        return {
            "facts": {k: v.to_dict() for k, v in self._facts.items()},
            "open_questions": {k: v.to_dict() for k, v in self._open_questions.items()},
            "decisions": {k: v.to_dict() for k, v in self._decisions.items()},
            "approvals": {k: v.to_dict() for k, v in self._approvals.items()},
            "artifacts": {k: v.to_dict() for k, v in self._artifacts.items()},
            "history": [e.to_dict() for e in self._history],
            "plans": {k: v.to_dict() for k, v in self._plans.items()},
            "selected_plan_id": self._selected_plan_id,
            "trace_events": [e.to_dict() for e in self._trace_events],
            "event_log": self._event_log,
            "last_user_intent": self._last_user_intent,
            "last_user_request_type": self._last_user_request_type,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WorldModel":
        """Deserialize a world model from a dictionary."""
        model = cls()
        
        for k, v in d.get("facts", {}).items():
            model._facts[k] = Fact.from_dict(v)
        
        for k, v in d.get("open_questions", {}).items():
            model._open_questions[k] = OpenQuestion(**v)
        
        for k, v in d.get("decisions", {}).items():
            model._decisions[k] = DecisionPoint(**v)
        
        for k, v in d.get("approvals", {}).items():
            model._approvals[k] = Approval(**v)
        
        for k, v in d.get("artifacts", {}).items():
            model._artifacts[k] = Artifact(**v)
        
        model._history = [HistoryEntry(**e) for e in d.get("history", [])]
        
        for k, v in d.get("plans", {}).items():
            model._plans[k] = Plan(**v)
        
        model._selected_plan_id = d.get("selected_plan_id")
        model._trace_events = [TraceEvent(**e) for e in d.get("trace_events", [])]
        model._event_log = d.get("event_log", [])
        model._last_user_intent = d.get("last_user_intent")
        model._last_user_request_type = d.get("last_user_request_type")
        
        return model
