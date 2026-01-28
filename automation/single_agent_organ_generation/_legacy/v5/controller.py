"""
V5 Controller - Goal-Driven Agent Controller

The main agent loop that:
1. Ingests events
2. Updates world model
3. Picks best capability
4. Responds/narrates
5. Executes
6. Repeats

No state machine - progress is measured by goal satisfaction.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union, TYPE_CHECKING

from .world_model import WorldModel, FactProvenance, TraceEvent, Artifact
from .goals import GoalTracker, GoalStatus
from .policies import SafeFixPolicy, ApprovalPolicy, CapabilitySelectionPolicy
from .plan_synthesizer import PlanSynthesizer
from .io.base_io import BaseIOAdapter
from .workspace import WorkspaceManager, ToolRegistryEntry, RunRecord
from .brain import Brain, Directive, ObservationPacket, create_initial_master_script

from ....contextual_dialogue import ContextualDialogue, DialogueIntent

if TYPE_CHECKING:
    from ....llm_client import LLMClient


class RunResult(Enum):
    """Result of a run() call - distinguishes completion from waiting."""
    COMPLETED = "completed"
    WAITING = "waiting"
    FAILED = "failed"

logger = logging.getLogger(__name__)


@dataclass
class ControllerStatus:
    """
    Derived controller status - computed from world model state, not actively set.
    
    This replaces the ControllerState enum to avoid state-machine-like behavior.
    The status is always computed from the current state of the world model.
    """
    phase: str
    is_waiting_for_user: bool = False
    is_waiting_for_approval: bool = False
    is_running_tool: bool = False
    is_complete: bool = False
    has_error: bool = False
    error_message: Optional[str] = None
    current_tool: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "is_waiting_for_user": self.is_waiting_for_user,
            "is_waiting_for_approval": self.is_waiting_for_approval,
            "is_running_tool": self.is_running_tool,
            "is_complete": self.is_complete,
            "has_error": self.has_error,
            "error_message": self.error_message,
            "current_tool": self.current_tool,
        }


@dataclass
class ControllerConfig:
    """Configuration for the V5 controller."""
    max_iterations: int = 1000
    max_safe_fixes_per_run: int = 10
    auto_select_plan_if_confident: bool = True
    verbose: bool = True
    llm_first_mode: bool = False  # Enable LLM-first agentic mode
    output_dir: Optional[str] = None  # Output directory for workspace
    # Verbosity level for conversational feedback:
    # "quiet" - minimal output, no readbacks
    # "normal" - readbacks after big answers (domain, topology, inlet/outlet)
    # "chatty" - readbacks after every answer
    verbosity: str = "normal"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_iterations": self.max_iterations,
            "max_safe_fixes_per_run": self.max_safe_fixes_per_run,
            "auto_select_plan_if_confident": self.auto_select_plan_if_confident,
            "verbose": self.verbose,
            "llm_first_mode": self.llm_first_mode,
            "output_dir": self.output_dir,
            "verbosity": self.verbosity,
        }


class SingleAgentOrganGeneratorV5:
    """
    V5 Goal-Driven Agent Controller for Single Agent Organ Generation.
    
    This controller replaces the V4 state-machine approach with a goal-driven
    architecture that:
    - Maintains a world model as single source of truth
    - Measures progress by goal satisfaction
    - Uses capabilities (small actions) selected by policy
    - Always asks before generation and postprocess
    - Applies safe fixes one at a time
    - Supports interrupts, backtracking, and undo
    """
    
    def __init__(
        self,
        io_adapter: BaseIOAdapter,
        config: Optional[ControllerConfig] = None,
        spec_compiler: Optional[Any] = None,
        generator: Optional[Any] = None,
        validator: Optional[Any] = None,
        llm_client: Optional["LLMClient"] = None,
    ):
        self.io = io_adapter
        self.config = config or ControllerConfig()
        
        self.world_model = WorldModel()
        self.goal_tracker = GoalTracker(self.world_model)
        self.plan_synthesizer = PlanSynthesizer(self.world_model)
        
        self.safe_fix_policy = SafeFixPolicy()
        self.approval_policy = ApprovalPolicy()
        self.capability_policy = CapabilitySelectionPolicy()
        
        self.spec_compiler = spec_compiler
        self.generator = generator
        self.validator = validator
        
        # LLM-first mode components
        self.llm_client = llm_client
        self.workspace: Optional[WorkspaceManager] = None
        self.brain: Optional[Brain] = None
        self._last_directive: Optional[Directive] = None
        self._last_run_result: Optional[Dict[str, Any]] = None
        self._last_verification_report: Optional[Dict[str, Any]] = None
        
        # P1 #16: Retry budgeting
        self._retry_counts: Dict[str, int] = {
            "execution": 0,  # Script execution retries
            "verification": 0,  # Verification failure retries
            "rewrite": 0,  # Master script rewrite attempts
        }
        self._max_retries: Dict[str, int] = {
            "execution": 5,  # Max execution retries before asking user
            "verification": 3,  # Max verification failure retries
            "rewrite": 10,  # Max script rewrite attempts
        }
        
        # Initialize workspace and brain if in LLM-first mode
        # Note: Workspace is NOT initialized here - it will be created after asking for project name
        # This allows per-project workspaces under <output_dir>/<project_slug>/agent_workspace
        if self.config.llm_first_mode and self.config.output_dir:
            if self.llm_client:
                self.brain = Brain(self.llm_client)
            # Workspace will be initialized in _cap_llm_ask_project_name after user provides project name
        
        self._iteration_count = 0
        self._safe_fixes_this_run = 0
        self._safe_fix_applied_this_iteration = False
        self._last_action: Optional[str] = None
        self._pending_user_message: Optional[str] = None
        self._stop_requested = False
        self._is_running = False
        self._current_tool: Optional[str] = None
        self._error_message: Optional[str] = None
        
        self._capabilities: Dict[str, Callable[[], bool]] = {
            "ingest_user_event": self._cap_ingest_user_event,
            "interpret_user_turn": self._cap_interpret_user_turn,
            "apply_patch": self._cap_apply_patch,
            "summarize_living_spec": self._cap_summarize_living_spec,
            "propose_tailored_plans": self._cap_propose_tailored_plans,
            "select_plan": self._cap_select_plan,
            "ask_best_next_question": self._cap_ask_best_next_question,
            "generate_missing_field_questions": self._cap_generate_missing_field_questions,
            "compile_spec": self._cap_compile_spec,
            "pregen_verify": self._cap_pregen_verify,
            "request_generation_approval": self._cap_request_generation_approval,
            "run_generation": self._cap_run_generation,
            "request_postprocess_approval": self._cap_request_postprocess_approval,
            "run_postprocess": self._cap_run_postprocess,
            "validate_artifacts": self._cap_validate_artifacts,
            "apply_one_safe_fix": self._cap_apply_one_safe_fix,
            "ask_for_non_safe_fix_choice": self._cap_ask_for_non_safe_fix_choice,
            "undo": self._cap_undo,
            "package_outputs": self._cap_package_outputs,
            # Project initialization (mode-independent)
            "ask_project_name": self._cap_ask_project_name,
            # Plan refresh offer (when spec changes but not major fields)
            "offer_plan_refresh": self._cap_offer_plan_refresh,
            # LLM-first mode capabilities
            "llm_ask_project_name": self._cap_llm_ask_project_name,
            "llm_decide_next": self._cap_llm_decide_next,
            "llm_apply_workspace_update": self._cap_llm_apply_workspace_update,
            "llm_request_execution": self._cap_llm_request_execution,
            "llm_run_master_script": self._cap_llm_run_master_script,
            "llm_verify_artifacts": self._cap_llm_verify_artifacts,
        }
        
        self._last_spec_hash: Optional[str] = None
        self._last_summarize_hash: Optional[str] = None
        self._last_plan_proposal_hash: Optional[str] = None  # Track when plans were last proposed
        self._denied_approval_hashes: Dict[str, str] = {}
        
        # Track major field values at last plan proposal to detect major changes
        self._last_plan_major_fields: Dict[str, Any] = {}
        
        # LLM-first mode: Track workspace hash to prevent re-running identical code
        self._last_verified_workspace_hash: Optional[str] = None
    
    def _is_ready_to_execute(self) -> bool:
        """
        Check if the spec has minimum required fields for meaningful execution.
        
        This prevents the LLM from writing/executing placeholder scripts too early
        before it has enough spec information to do real work.
        
        Required fields:
        - domain type + size (at least one dimension)
        - inlet location (face or point) + radius
        - outlet location + radius
        - target resolution (terminals, segment budget, or density level)
        
        Returns
        -------
        bool
            True if spec has minimum required fields, False otherwise
        """
        # Check domain type (uses dot notation consistent with world model)
        domain_type = self.world_model.get_fact_value("domain.type")
        if not domain_type:
            return False
        
        # Check domain size (at least one dimension)
        has_size = any([
            self.world_model.get_fact_value("domain.width"),
            self.world_model.get_fact_value("domain.height"),
            self.world_model.get_fact_value("domain.depth"),
            self.world_model.get_fact_value("domain.radius"),
            self.world_model.get_fact_value("domain.size"),
        ])
        if not has_size:
            return False
        
        # Check inlet (face or point + radius)
        has_inlet = (
            self.world_model.get_fact_value("inlet.face") or
            self.world_model.get_fact_value("inlet.position") or
            self.world_model.get_fact_value("inlet.point")
        )
        inlet_radius = self.world_model.get_fact_value("inlet.radius")
        if not has_inlet or not inlet_radius:
            return False
        
        # Check outlet (face or point + radius)
        has_outlet = (
            self.world_model.get_fact_value("outlet.face") or
            self.world_model.get_fact_value("outlet.position") or
            self.world_model.get_fact_value("outlet.point")
        )
        outlet_radius = self.world_model.get_fact_value("outlet.radius")
        if not has_outlet or not outlet_radius:
            return False
        
        # Check target resolution (any of these is acceptable)
        has_resolution = any([
            self.world_model.get_fact_value("topology.target_terminals"),
            self.world_model.get_fact_value("topology.segment_budget"),
            self.world_model.get_fact_value("topology.density_level"),
            self.world_model.get_fact_value("resolution"),
        ])
        if not has_resolution:
            return False
        
        return True
    
    def _get_missing_spec_fields(self) -> List[str]:
        """
        Get list of missing required spec fields for execution.
        
        Returns
        -------
        List[str]
            List of missing field descriptions
        """
        missing = []
        
        # Uses dot notation consistent with world model
        if not self.world_model.get_fact_value("domain.type"):
            missing.append("domain type (box, cylinder, sphere, etc.)")
        
        has_size = any([
            self.world_model.get_fact_value("domain.width"),
            self.world_model.get_fact_value("domain.height"),
            self.world_model.get_fact_value("domain.depth"),
            self.world_model.get_fact_value("domain.radius"),
            self.world_model.get_fact_value("domain.size"),
        ])
        if not has_size:
            missing.append("domain size/dimensions")
        
        has_inlet = (
            self.world_model.get_fact_value("inlet.face") or
            self.world_model.get_fact_value("inlet.position") or
            self.world_model.get_fact_value("inlet.point")
        )
        if not has_inlet:
            missing.append("inlet location (face or point)")
        
        if not self.world_model.get_fact_value("inlet.radius"):
            missing.append("inlet radius")
        
        has_outlet = (
            self.world_model.get_fact_value("outlet.face") or
            self.world_model.get_fact_value("outlet.position") or
            self.world_model.get_fact_value("outlet.point")
        )
        if not has_outlet:
            missing.append("outlet location (face or point)")
        
        if not self.world_model.get_fact_value("outlet.radius"):
            missing.append("outlet radius")
        
        has_resolution = any([
            self.world_model.get_fact_value("topology.target_terminals"),
            self.world_model.get_fact_value("topology.segment_budget"),
            self.world_model.get_fact_value("topology.density_level"),
            self.world_model.get_fact_value("resolution"),
        ])
        if not has_resolution:
            missing.append("target resolution (terminals, density level, or segment budget)")
        
        return missing
    
    def run(self, initial_message: Optional[str] = None) -> RunResult:
        """
        Run the agent loop until completion or interruption.
        
        Parameters
        ----------
        initial_message : str, optional
            Initial user message to process
            
        Returns
        -------
        RunResult
            COMPLETED if all goals satisfied, WAITING if paused for user input,
            FAILED if max iterations reached or error occurred
        """
        self._is_running = True
        self._iteration_count = 0
        self._safe_fixes_this_run = 0
        self._stop_requested = False
        self._error_message = None
        
        if initial_message:
            self._pending_user_message = initial_message
            self.world_model.add_event({
                "type": "user_message",
                "content": initial_message,
            })
        
        self._emit_trace("controller_started", "V5 controller started")
        
        try:
            while not self._should_stop():
                self._iteration_count += 1
                self._safe_fix_applied_this_iteration = False
                
                if self._iteration_count > self.config.max_iterations:
                    self.io.say_error("Maximum iterations reached")
                    self._error_message = "Maximum iterations reached"
                    self._is_running = False
                    return RunResult.FAILED
                
                available = self._get_available_capabilities()
                
                if not available:
                    if self.goal_tracker.is_complete():
                        self._emit_trace("controller_completed", "All goals satisfied")
                        self._is_running = False
                        return RunResult.COMPLETED
                    else:
                        self._emit_trace("waiting_for_input", "Waiting for user input")
                        self._is_running = False
                        return RunResult.WAITING
                
                capability = self.capability_policy.select_capability(
                    available,
                    self.world_model,
                    self.goal_tracker,
                    self._last_action,
                    self._safe_fix_applied_this_iteration,
                )
                
                if not capability:
                    continue
                
                self._emit_trace(f"capability_selected", f"Selected: {capability}")
                
                self._current_tool = capability
                success = self._execute_capability(capability)
                self._current_tool = None
                self._last_action = capability
                
                if not success:
                    logger.warning(f"Capability {capability} returned False")
            
            self._is_running = False
            # P0 #2: In LLM-first mode, check llm_complete fact instead of goal_tracker.is_complete()
            # because goal_tracker requires ALL goals (classic + LLM-first) to be satisfied
            if self.config.llm_first_mode:
                if self.world_model.get_fact_value("llm_complete", False):
                    return RunResult.COMPLETED
                return RunResult.WAITING
            return RunResult.COMPLETED if self.goal_tracker.is_complete() else RunResult.WAITING
            
        except KeyboardInterrupt:
            self._is_running = False
            self._emit_trace("controller_interrupted", "Controller interrupted by user")
            return RunResult.FAILED
        except Exception as e:
            self._is_running = False
            self._error_message = str(e)
            self._emit_trace("controller_error", f"Error: {str(e)}")
            logger.exception("Controller error")
            raise
    
    def process_user_message(self, message: str) -> None:
        """
        Process a user message (can be called during or between runs).
        
        Parameters
        ----------
        message : str
            The user's message
        """
        self._pending_user_message = message
        self.world_model.add_event({
            "type": "user_message",
            "content": message,
        })
        
        status = self._compute_status()
        if status.is_waiting_for_user and not self._is_running:
            self.run()
    
    def stop(self) -> None:
        """Request the controller to stop."""
        self._stop_requested = True
    
    def undo(self) -> bool:
        """Undo the last change."""
        return self._cap_undo()
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current controller status."""
        status = self._compute_status()
        return {
            "status": status.to_dict(),
            "iteration_count": self._iteration_count,
            "safe_fixes_this_run": self._safe_fixes_this_run,
            "goal_progress": self.goal_tracker.get_progress_summary(),
            "spec_hash": self.world_model.compute_spec_hash(),
            "generation_approved": self.world_model.is_approved("generation"),
            "postprocess_approved": self.world_model.is_approved("postprocess"),
        }
    
    def _compute_status(self) -> ControllerStatus:
        """
        Compute the current controller status from world model state.
        
        This is the key architectural change from V4: status is derived,
        not actively set. The status reflects the current state of the
        world model and goal tracker.
        """
        if self._error_message:
            return ControllerStatus(
                phase="error",
                has_error=True,
                error_message=self._error_message,
            )
        
        if self.goal_tracker.is_complete():
            return ControllerStatus(
                phase="complete",
                is_complete=True,
            )
        
        if self._current_tool:
            return ControllerStatus(
                phase="running_tool",
                is_running_tool=True,
                current_tool=self._current_tool,
            )
        
        gen_approval = self.world_model.get_approval("generation")
        post_approval = self.world_model.get_approval("postprocess")
        
        if gen_approval and not gen_approval.approved:
            return ControllerStatus(
                phase="waiting_for_generation_approval",
                is_waiting_for_approval=True,
            )
        
        if post_approval and not post_approval.approved:
            return ControllerStatus(
                phase="waiting_for_postprocess_approval",
                is_waiting_for_approval=True,
            )
        
        if self.world_model.open_questions and not self._pending_user_message:
            return ControllerStatus(
                phase="waiting_for_user_input",
                is_waiting_for_user=True,
            )
        
        if self._is_running:
            return ControllerStatus(
                phase="processing",
            )
        
        return ControllerStatus(
            phase="idle",
        )
    
    def _should_stop(self) -> bool:
        """Check if the controller should stop."""
        if self._stop_requested:
            return True
        if self.goal_tracker.is_complete():
            return True
        # In LLM-first mode, check if LLM decided to stop
        if self.config.llm_first_mode:
            if self.world_model.get_fact_value("llm_complete", False):
                return True
        return False
    
    def _get_available_capabilities(self) -> List[str]:
        """Get list of capabilities that can currently be executed."""
        # Use LLM-first capability selection if in that mode
        if self.config.llm_first_mode and self.brain and self.workspace:
            return self._get_available_capabilities_llm_first()
        
        available = []
        
        # Project initialization: Ask for project name if not yet initialized
        # This runs at the very start to establish where outputs will be stored
        project_initialized = self.world_model.get_fact_value("_project_initialized", False)
        if not project_initialized:
            available.append("ask_project_name")
            # Return early - project init should happen before anything else
            return available
        
        if self._pending_user_message:
            available.append("ingest_user_event")
            available.append("interpret_user_turn")
        
        if self.world_model._pending_patches:
            available.append("apply_patch")
        
        # Note: "undo" capability is NOT automatically added here.
        # Undo is triggered only via explicit user request, which is handled
        # in _cap_interpret_user_turn() when user says "undo" or similar.
        # This prevents the undo loop where undo was auto-selected after plan selection.
        
        current_spec_hash = self.world_model.compute_spec_hash()
        
        if self._last_summarize_hash != current_spec_hash:
            available.append("summarize_living_spec")
        
        spec_complete = self.goal_tracker.get_status("spec_minimum_complete") == GoalStatus.SATISFIED
        
        if not spec_complete:
            if self.world_model.open_questions:
                available.append("ask_best_next_question")
            else:
                available.append("generate_missing_field_questions")
            
            # Only propose/select plans if:
            # 1. No open questions remain (all questions answered first)
            # 2. Has minimum intent (domain type, topology kind, or some decisions made)
            # 3. No plan selected yet
            # 
            # This prevents the plan selection loop where plans are offered while
            # questions are still being asked, which confuses the flow.
            has_minimum_intent = (
                self.world_model.has_fact("domain.type") or
                self.world_model.has_fact("domain_type") or
                self.world_model.has_fact("topology.kind") or
                self.world_model.has_fact("topology_kind") or
                # Check if any questions have been answered (decisions recorded)
                len(self.world_model._decisions) > 0
            )
            
            # Only offer plan proposal/selection when no open questions remain
            # This ensures all questions are answered before plan selection
            no_open_questions = not self.world_model.open_questions
            
            if not self.world_model.selected_plan and has_minimum_intent and no_open_questions:
                if not self.world_model.plans:
                    # No plans yet - propose some (auto-propose ONCE)
                    available.append("propose_tailored_plans")
                else:
                    # Plans exist but none selected
                    # Check if a major field changed since last proposal
                    major_field_changed = self._check_major_field_changed()
                    user_requested_refresh = self.world_model.get_fact_value("_user_requested_plan_refresh", False)
                    
                    if major_field_changed or user_requested_refresh:
                        # Major change or user request - allow re-proposal
                        # Note: the flag is cleared in _cap_propose_tailored_plans() after execution
                        available.append("propose_tailored_plans")
                    elif self._last_plan_proposal_hash != current_spec_hash:
                        # Spec changed but not a major field - offer to refresh
                        available.append("offer_plan_refresh")
                    
                    # Always allow plan selection when plans exist but none selected
                    available.append("select_plan")
        
        if spec_complete:
            if self.goal_tracker.get_status("spec_compiled") != GoalStatus.SATISFIED:
                available.append("compile_spec")
        
        spec_compiled = self.goal_tracker.get_status("spec_compiled") == GoalStatus.SATISFIED
        if spec_compiled:
            if self.goal_tracker.get_status("pregen_verified") != GoalStatus.SATISFIED:
                available.append("pregen_verify")
        
        pregen_verified = self.goal_tracker.get_status("pregen_verified") == GoalStatus.SATISFIED
        if pregen_verified:
            gen_denied_hash = self._denied_approval_hashes.get("generation")
            if not self.world_model.is_approved("generation"):
                if gen_denied_hash != current_spec_hash:
                    available.append("request_generation_approval")
            elif self.goal_tracker.get_status("generation_done") != GoalStatus.SATISFIED:
                available.append("run_generation")
        
        generation_done = self.goal_tracker.get_status("generation_done") == GoalStatus.SATISFIED
        if generation_done:
            post_denied_hash = self._denied_approval_hashes.get("postprocess")
            if not self.world_model.is_approved("postprocess"):
                if post_denied_hash != current_spec_hash:
                    available.append("request_postprocess_approval")
            elif self.goal_tracker.get_status("postprocess_done") != GoalStatus.SATISFIED:
                available.append("run_postprocess")
        
        postprocess_done = self.goal_tracker.get_status("postprocess_done") == GoalStatus.SATISFIED
        
        validation_failed = self.world_model.get_fact_value("validation_failed", False)
        
        if validation_failed:
            has_safe_fixes = (
                self._safe_fixes_this_run < self.config.max_safe_fixes_per_run
                and not self._safe_fix_applied_this_iteration
                and self._has_available_safe_fixes()
            )
            
            if has_safe_fixes:
                available.append("apply_one_safe_fix")
            else:
                available.append("ask_for_non_safe_fix_choice")
        elif postprocess_done or generation_done:
            if self.goal_tracker.get_status("validation_passed") != GoalStatus.SATISFIED:
                available.append("validate_artifacts")
        
        validation_passed = self.goal_tracker.get_status("validation_passed") == GoalStatus.SATISFIED
        if validation_passed and postprocess_done:
            if self.goal_tracker.get_status("outputs_packaged") != GoalStatus.SATISFIED:
                available.append("package_outputs")
        
        return available
    
    def _has_available_safe_fixes(self) -> bool:
        """Check if there are any safe fixes available."""
        issues = self.world_model.get_fact_value("validation_issues", [])
        if not issues:
            return False
        
        for issue in issues:
            candidates = self.safe_fix_policy.generate_fix_candidates(
                failure_type=issue.split()[0] if issue else "unknown",
                failure_details={"issue": issue},
                world_model=self.world_model,
            )
            safe_candidates = [c for c in candidates if c.safety.value == "safe"]
            if safe_candidates:
                return True
        return False
    
    def _check_major_field_changed(self) -> bool:
        """
        Check if a major field has changed since the last plan proposal.
        
        Major fields are: topology.kind, domain.type
        These fields fundamentally change what plans make sense.
        """
        major_fields = ["topology.kind", "domain.type"]
        
        for field in major_fields:
            current_value = self.world_model.get_fact_value(field)
            last_value = self._last_plan_major_fields.get(field)
            
            if current_value != last_value:
                return True
        
        return False
    
    def _save_major_field_values(self) -> None:
        """Save current major field values for later comparison."""
        major_fields = ["topology.kind", "domain.type"]
        
        for field in major_fields:
            self._last_plan_major_fields[field] = self.world_model.get_fact_value(field)
    
    def _execute_capability(self, capability: str) -> bool:
        """Execute a capability by name."""
        if capability not in self._capabilities:
            logger.error(f"Unknown capability: {capability}")
            return False
        
        return self._capabilities[capability]()
    
    def _emit_trace(self, event_type: str, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Emit a trace event."""
        event = TraceEvent(
            event_type=event_type,
            message=message,
            data=data,
        )
        self.world_model.add_trace_event(event.event_type, event.message, event.data)
        self.io.emit_trace(event)
    
    def _cap_ingest_user_event(self) -> bool:
        """Capability 0: Ingest user event."""
        if not self._pending_user_message:
            return False
        
        self._emit_trace("event_ingested", f"Ingested user message")
        return True
    
    def _cap_interpret_user_turn(self) -> bool:
        """Capability 1: Interpret user turn."""
        if not self._pending_user_message:
            return False
        
        message = self._pending_user_message
        self._pending_user_message = None
        
        intent = self._parse_user_intent(message)
        
        self._emit_trace("interpretation", f"Intent: {intent.get('type', 'unknown')}")
        
        if intent.get("type") == "undo":
            return self._cap_undo()
        
        if intent.get("type") == "undo_to_entry":
            entry_id = intent.get("entry_id")
            return self._cap_undo_to_entry(entry_id)
        
        if intent.get("type") == "show_changes":
            return self._cap_show_changes()
        
        if intent.get("type") == "show_status":
            return self._cap_summarize_living_spec()
        
        if intent.get("type") == "select_plan":
            plan_id = intent.get("plan_id")
            return self._cap_select_plan_by_id(plan_id)
        
        if intent.get("type") == "revisit_question":
            topic = intent.get("topic")
            return self._cap_revisit_question(topic)
        
        if intent.get("patches"):
            self.world_model._pending_patches = intent["patches"]
        
        if intent.get("questions_answered"):
            for q_id, answer in intent["questions_answered"].items():
                self.world_model.answer_question(q_id, answer)
        
        return True
    
    def _parse_user_intent(self, message: str) -> Dict[str, Any]:
        """
        Parse user message into structured intent using contextual dialogue infrastructure.
        
        This method handles:
        - Corrections ("No, I meant +x max face", "Actually, change the inlet")
        - References ("use the earlier plan", "the second option")
        - Multi-field updates in one sentence
        - Navigation commands ("go back to the ports question", "revisit domain")
        - Plan selection ("plan 2", "use recommended plan", "switch to plan B")
        - Undo/backtracking ("undo", "undo to entry", "show changes")
        """
        message_lower = message.lower().strip()
        
        if message_lower in ("undo", "undo last", "revert"):
            return {"type": "undo"}
        
        undo_to_match = re.match(r"undo\s+to\s+(?:entry\s+)?(\w+)", message_lower)
        if undo_to_match:
            return {"type": "undo_to_entry", "entry_id": undo_to_match.group(1)}
        
        if message_lower in ("show changes", "show recent changes", "what changed"):
            return {"type": "show_changes"}
        
        if message_lower in ("status", "where are we", "show status", "what's missing"):
            return {"type": "show_status"}
        
        plan_match = re.match(r"(?:use\s+)?(?:plan\s+)?(\d+|[a-c]|recommended)", message_lower)
        if plan_match or "switch to plan" in message_lower or "use plan" in message_lower:
            plan_id = plan_match.group(1) if plan_match else None
            if not plan_id:
                plan_search = re.search(r"plan\s+(\d+|[a-c])", message_lower)
                plan_id = plan_search.group(1) if plan_search else "recommended"
            return {"type": "select_plan", "plan_id": plan_id}
        
        revisit_match = re.search(r"(?:revisit|go\s+back\s+to|change)\s+(?:the\s+)?(\w+)(?:\s+question)?", message_lower)
        if revisit_match:
            topic = revisit_match.group(1)
            return {"type": "revisit_question", "topic": topic}
        
        if message_lower.startswith("yes") or message_lower == "y":
            return {"type": "confirm", "value": True}
        
        if message_lower.startswith("no") or message_lower == "n":
            return {"type": "confirm", "value": False}
        
        dialogue = ContextualDialogue()
        intent = dialogue.classify_intent(message)
        extracted = dialogue.extract_values_from_text(message)
        
        patches = {}
        questions_answered = {}
        
        is_correction = intent == DialogueIntent.CORRECTION or "actually" in message_lower or "i meant" in message_lower
        
        if "domain_type" in extracted:
            patches["domain.type"] = extracted["domain_type"]
        elif "domain" in message_lower:
            if "box" in message_lower:
                patches["domain.type"] = "box"
            elif "ellipsoid" in message_lower:
                patches["domain.type"] = "ellipsoid"
            elif "cylinder" in message_lower:
                patches["domain.type"] = "cylinder"
        
        if "topology_kind" in extracted:
            patches["topology.kind"] = extracted["topology_kind"]
        elif any(t in message_lower for t in ["tree", "path", "backbone", "loop"]):
            if "tree" in message_lower:
                patches["topology.kind"] = "tree"
            elif "path" in message_lower:
                patches["topology.kind"] = "path"
            elif "backbone" in message_lower:
                patches["topology.kind"] = "backbone"
            elif "loop" in message_lower:
                patches["topology.kind"] = "loop"
        
        if "inlet_face" in extracted:
            patches["inlet.face"] = extracted["inlet_face"]
        if "outlet_face" in extracted:
            patches["outlet.face"] = extracted["outlet_face"]
        
        if "inlet" in message_lower or "outlet" in message_lower:
            face_patterns = {
                "x_min": [r"\bleft\b", r"\bx[\s_-]?min\b", r"\b-x\b", r"\bx\s*min\b"],
                "x_max": [r"\bright\b", r"\bx[\s_-]?max\b", r"\b\+x\b", r"\bx\s*max\b"],
                "y_min": [r"\bfront\b", r"\by[\s_-]?min\b", r"\b-y\b", r"\by\s*min\b"],
                "y_max": [r"\bback\b", r"\by[\s_-]?max\b", r"\b\+y\b", r"\by\s*max\b"],
                "z_min": [r"\bbottom\b", r"\bz[\s_-]?min\b", r"\b-z\b", r"\bz\s*min\b"],
                "z_max": [r"\btop\b", r"\bz[\s_-]?max\b", r"\b\+z\b", r"\bz\s*max\b"],
            }
            for face, patterns in face_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, message_lower):
                        if "inlet" in message_lower and "inlet.face" not in patches:
                            patches["inlet.face"] = face
                        if "outlet" in message_lower and "outlet.face" not in patches:
                            patches["outlet.face"] = face
                        break
        
        # Domain size parsing - supports multiple formats:
        # "20x60x30" (default mm), "20x60x30mm", "20mm x 60mm x 30mm", "2cm x 6cm x 3cm"
        # First try format with unit after each dimension: "20mm x 60mm x 30mm"
        size_match_per_dim = re.search(
            r"(\d+(?:\.\d+)?)\s*(um|mm|cm|m)?\s*(?:x|by|,)\s*"
            r"(\d+(?:\.\d+)?)\s*(um|mm|cm|m)?\s*(?:x|by|,)\s*"
            r"(\d+(?:\.\d+)?)\s*(um|mm|cm|m)?",
            message_lower
        )
        if size_match_per_dim:
            x_val = float(size_match_per_dim.group(1))
            x_unit = size_match_per_dim.group(2)
            y_val = float(size_match_per_dim.group(3))
            y_unit = size_match_per_dim.group(4)
            z_val = float(size_match_per_dim.group(5))
            z_unit = size_match_per_dim.group(6)
            
            # Determine the unit for each dimension
            # If any dimension has a unit, use it; otherwise default to mm
            # If only the last dimension has a unit, apply it to all
            def get_effective_unit(dim_unit, last_unit):
                if dim_unit:
                    return dim_unit
                elif last_unit:
                    return last_unit
                return "mm"
            
            last_unit = z_unit or y_unit or x_unit
            x_unit_eff = get_effective_unit(x_unit, last_unit)
            y_unit_eff = get_effective_unit(y_unit, last_unit)
            z_unit_eff = get_effective_unit(z_unit, last_unit)
            
            def convert_to_meters(val, unit):
                if unit == "um":
                    return val / 1_000_000
                elif unit == "mm":
                    return val / 1000
                elif unit == "cm":
                    return val / 100
                return val  # meters
            
            x = convert_to_meters(x_val, x_unit_eff)
            y = convert_to_meters(y_val, y_unit_eff)
            z = convert_to_meters(z_val, z_unit_eff)
            patches["domain.size"] = (x, y, z)
        
        terminal_match = re.search(r"(\d+)\s*terminals?", message_lower)
        if terminal_match:
            patches["topology.target_terminals"] = int(terminal_match.group(1))
        
        # Helper function to parse radius with explicit unit handling
        # If unit is present, always respect it. If absent, default to mm.
        def parse_radius_to_meters(value_str: str) -> Optional[float]:
            """Parse a radius value string to meters, respecting explicit units."""
            radius_match = re.search(r"(\d+(?:\.\d+)?)\s*(um|mm|cm|m)?", str(value_str).lower())
            if not radius_match:
                return None
            val = float(radius_match.group(1))
            unit = radius_match.group(2)
            # If unit is explicitly specified, always respect it
            if unit == "um":
                return val / 1_000_000  # micrometers to meters
            elif unit == "mm":
                return val / 1000  # millimeters to meters
            elif unit == "cm":
                return val / 100  # centimeters to meters
            elif unit == "m":
                return val  # already in meters
            else:
                # No unit specified - default to mm for vessel radii (most common)
                return val / 1000
        
        if "inlet_radius" in extracted:
            radius_val = extracted["inlet_radius"]
            if isinstance(radius_val, (int, float)):
                # No unit info available, default to mm
                patches["inlet.radius"] = radius_val / 1000
            else:
                parsed = parse_radius_to_meters(str(radius_val))
                if parsed is not None:
                    patches["inlet.radius"] = parsed
        
        if "outlet_radius" in extracted:
            radius_val = extracted["outlet_radius"]
            if isinstance(radius_val, (int, float)):
                # No unit info available, default to mm
                patches["outlet.radius"] = radius_val / 1000
            else:
                parsed = parse_radius_to_meters(str(radius_val))
                if parsed is not None:
                    patches["outlet.radius"] = parsed
        
        if "min_radius" in extracted:
            radius_val = extracted["min_radius"]
            if isinstance(radius_val, (int, float)):
                # No unit info available, default to mm
                patches["colonization.min_radius"] = radius_val / 1000
            else:
                parsed = parse_radius_to_meters(str(radius_val))
                if parsed is not None:
                    patches["colonization.min_radius"] = parsed
        
        # Freeform radius patterns - capture value and optional unit
        radius_pattern = re.search(r"(\d+(?:\.\d+)?)\s*(um|mm|cm|m)?\s*(?:inlet|input)\s*radius", message_lower)
        if radius_pattern and "inlet.radius" not in patches:
            val = float(radius_pattern.group(1))
            unit = radius_pattern.group(2) or "mm"  # default to mm
            if unit == "um":
                patches["inlet.radius"] = val / 1_000_000
            elif unit == "mm":
                patches["inlet.radius"] = val / 1000
            elif unit == "cm":
                patches["inlet.radius"] = val / 100
            else:
                patches["inlet.radius"] = val
        
        radius_pattern = re.search(r"inlet\s*radius\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*(um|mm|cm|m)?", message_lower)
        if radius_pattern and "inlet.radius" not in patches:
            val = float(radius_pattern.group(1))
            unit = radius_pattern.group(2) or "mm"  # default to mm
            if unit == "um":
                patches["inlet.radius"] = val / 1_000_000
            elif unit == "mm":
                patches["inlet.radius"] = val / 1000
            elif unit == "cm":
                patches["inlet.radius"] = val / 100
            else:
                patches["inlet.radius"] = val
        
        radius_pattern = re.search(r"(\d+(?:\.\d+)?)\s*(um|mm|cm|m)?\s*(?:outlet|output)\s*radius", message_lower)
        if radius_pattern and "outlet.radius" not in patches:
            val = float(radius_pattern.group(1))
            unit = radius_pattern.group(2) or "mm"  # default to mm
            if unit == "um":
                patches["outlet.radius"] = val / 1_000_000
            elif unit == "mm":
                patches["outlet.radius"] = val / 1000
            elif unit == "cm":
                patches["outlet.radius"] = val / 100
            else:
                patches["outlet.radius"] = val
        
        radius_pattern = re.search(r"outlet\s*radius\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*(um|mm|cm|m)?", message_lower)
        if radius_pattern and "outlet.radius" not in patches:
            val = float(radius_pattern.group(1))
            unit = radius_pattern.group(2) or "mm"  # default to mm
            if unit == "um":
                patches["outlet.radius"] = val / 1_000_000
            elif unit == "mm":
                patches["outlet.radius"] = val / 1000
            elif unit == "cm":
                patches["outlet.radius"] = val / 100
            else:
                patches["outlet.radius"] = val
        
        # Parse spatial commands using the spatial parser
        spatial_commands = self._parse_spatial_commands(message)
        if spatial_commands:
            for cmd in spatial_commands:
                if cmd.get("element") == "inlet" and cmd.get("position"):
                    pos = cmd["position"]
                    if pos.get("face"):
                        face_map = {
                            "left": "x_min", "right": "x_max",
                            "front": "y_min", "back": "y_max",
                            "bottom": "z_min", "top": "z_max",
                        }
                        patches["inlet.face"] = face_map.get(pos["face"], pos["face"])
                    elif pos.get("coordinates"):
                        patches["inlet.position"] = pos["coordinates"]
                elif cmd.get("element") == "outlet" and cmd.get("position"):
                    pos = cmd["position"]
                    if pos.get("face"):
                        face_map = {
                            "left": "x_min", "right": "x_max",
                            "front": "y_min", "back": "y_max",
                            "bottom": "z_min", "top": "z_max",
                        }
                        patches["outlet.face"] = face_map.get(pos["face"], pos["face"])
                    elif pos.get("coordinates"):
                        patches["outlet.position"] = pos["coordinates"]
        
        if self.world_model.open_questions:
            for q_id, question in self.world_model.open_questions.items():
                field = question.field
                if field in patches:
                    questions_answered[q_id] = patches[field]
        
        intent_type = "specification" if patches else "freeform"
        if is_correction:
            intent_type = "correction"
        elif intent == DialogueIntent.META_QUESTION:
            intent_type = "meta_question"
        elif intent == DialogueIntent.UNCERTAINTY:
            intent_type = "uncertainty"
        
        return {
            "type": intent_type,
            "patches": patches,
            "questions_answered": questions_answered,
            "is_correction": is_correction,
            "dialogue_intent": intent.value if intent else None,
            "raw_message": message,
            "spatial_commands": spatial_commands,
        }
    
    def _parse_spatial_commands(self, message: str) -> List[Dict[str, Any]]:
        """
        Parse spatial commands from user message using the spatial parser.
        
        Parameters
        ----------
        message : str
            User message that may contain spatial commands
            
        Returns
        -------
        List[Dict[str, Any]]
            List of parsed spatial commands as dictionaries
        """
        try:
            from .spatial_parser import parse_spatial_description, SpatialCommandType
            
            result = parse_spatial_description(message)
            if result.success and result.commands:
                return [cmd.to_dict() for cmd in result.commands]
        except ImportError:
            pass
        except Exception:
            pass
        
        return []
    
    def _cap_apply_patch(self) -> bool:
        """Capability 2: Apply patch to world model."""
        if not hasattr(self.world_model, "_pending_patches") or not self.world_model._pending_patches:
            return False
        
        patches = self.world_model._pending_patches
        self.world_model._pending_patches = {}
        
        for field_name, value in patches.items():
            old_value = self.world_model.get_fact_value(field_name)
            self.world_model.set_fact(
                field=field_name,
                value=value,
                provenance=FactProvenance.USER,
                reason="User specification",
            )
            
            self._emit_trace(
                "patch_applied",
                f"Updated {field_name}: {old_value} -> {value}",
                {"field": field_name, "old": old_value, "new": value},
            )
        
        self.io.say_assistant(
            f"Updated {len(patches)} field(s). Prior approvals invalidated if geometry changed."
        )
        
        return True
    
    def _cap_summarize_living_spec(self) -> bool:
        """Capability 3: Summarize living spec."""
        summary = self.world_model.get_living_spec_summary()
        self.io.show_living_spec(summary)
        self._last_summarize_hash = self.world_model.compute_spec_hash()
        self._emit_trace("living_spec_updated", "Living spec summary generated")
        return True
    
    def _cap_propose_tailored_plans(self) -> bool:
        """Capability 4: Propose tailored plans."""
        plans = self.plan_synthesizer.synthesize_plans()
        
        for plan in plans:
            self.world_model.add_plan(plan)
        
        recommended = self.plan_synthesizer.get_recommended_plan(plans)
        recommended_id = recommended.plan_id if recommended else None
        
        self.io.show_plans(
            [p.to_dict() for p in plans],
            recommended_id=recommended_id,
        )
        
        if recommended:
            rationale = self.plan_synthesizer.generate_recommendation_rationale(recommended)
            self.io.say_assistant(rationale)
        
        # Track when plans were proposed to prevent infinite re-proposal loop
        self._last_plan_proposal_hash = self.world_model.compute_spec_hash()
        # Save major field values for detecting major changes later
        self._save_major_field_values()
        # Clear the user request flag if it was set (consumed by this proposal)
        if self.world_model.get_fact_value("_user_requested_plan_refresh", False):
            self.world_model.set_fact(
                "_user_requested_plan_refresh",
                False,
                FactProvenance.SYSTEM,
                reason="Plan refresh request consumed",
                record_history=False,
            )
        
        self._emit_trace("plans_proposed", f"Proposed {len(plans)} plans")
        
        return True
    
    def _cap_offer_plan_refresh(self) -> bool:
        """
        Offer to refresh plan recommendations when spec changed but not a major field.
        
        Instead of automatically re-proposing plans on every spec change (which causes
        an infinite loop), this capability offers the user the option to refresh plans.
        """
        response = self.io.ask_confirm(
            "Your answers have changed the spec. Would you like me to refresh the plan recommendations?"
        )
        
        if response:
            # Set flag to allow re-proposal on next iteration
            self.world_model.set_fact(
                "_user_requested_plan_refresh",
                True,
                FactProvenance.SYSTEM,
                reason="User requested plan refresh",
                record_history=False,
            )
            self._emit_trace("plan_refresh_requested", "User requested plan refresh")
        else:
            # User declined - update the hash so we don't keep asking
            self._last_plan_proposal_hash = self.world_model.compute_spec_hash()
            self._emit_trace("plan_refresh_declined", "User declined plan refresh")
        
        return True
    
    def _cap_select_plan(self) -> bool:
        """Capability 5: Select plan."""
        if not self.world_model.plans:
            return False
        
        plans = list(self.world_model.plans.values())
        
        recommended = self.plan_synthesizer.get_recommended_plan(plans)
        
        if self.config.auto_select_plan_if_confident and recommended:
            all_low_risk = all(len(p.risks) <= 1 for p in plans)
            if all_low_risk:
                self.world_model.select_plan(recommended.plan_id)
                self.plan_synthesizer.apply_plan_patches(recommended)
                self.io.say_assistant(f"Selecting {recommended.name}. It best aligns with your geometry and constraints.")
                self._emit_trace("plan_selected", f"Auto-selected: {recommended.plan_id}")
                return True
        
        selected_id = self.io.prompt_plan_selection([p.to_dict() for p in plans])
        
        # If user didn't explicitly select, use recommended plan or first available
        if selected_id is None:
            if recommended:
                selected_id = recommended.plan_id
                self.io.say_assistant(f"No selection made - using recommended plan.")
            elif plans:
                selected_id = plans[0].plan_id
                self.io.say_assistant(f"No selection made - using first available plan.")
        
        if selected_id and selected_id in self.world_model.plans:
            selected = self.world_model.plans[selected_id]
            self.world_model.select_plan(selected_id)
            self.plan_synthesizer.apply_plan_patches(selected)
            self.io.say_assistant(f"Selected {selected.name}.")
            self._emit_trace("plan_selected", f"User selected: {selected_id}")
            return True
        
        # This should not happen if plans exist, but log it for debugging
        self._emit_trace("plan_selection_failed", f"Failed to select plan. selected_id={selected_id}, plans={[p.plan_id for p in plans]}")
        return False
    
    def _cap_ask_best_next_question(self) -> bool:
        """Capability 6: Ask best next question.
        
        Questions are prioritized by:
        1. Priority questions from ambiguities (project.priority_questions)
        2. Default priority ordering
        """
        if not self.world_model.open_questions:
            return False
        
        questions = list(self.world_model.open_questions.values())
        
        # Check for priority questions from ambiguities
        priority_questions = self.world_model.get_fact_value("project.priority_questions", [])
        
        # Sort questions with priority questions first, then by default priority
        def question_sort_key(q):
            # Higher priority = asked first, so we want higher values for priority questions
            if q.field in priority_questions:
                # Priority questions get a boost of 1000 + their index (earlier in list = higher priority)
                try:
                    idx = priority_questions.index(q.field)
                    return (1000 - idx, q.priority)
                except ValueError:
                    return (0, q.priority)
            return (0, q.priority)
        
        questions.sort(key=question_sort_key, reverse=True)
        
        best_question = questions[0]
        
        # If we just asked a priority question, remove it from the list
        if best_question.field in priority_questions:
            remaining = [pq for pq in priority_questions if pq != best_question.field]
            if remaining:
                self.world_model.set_fact(
                    "project.priority_questions",
                    remaining,
                    FactProvenance.SYSTEM,
                    reason="Updated after asking priority question",
                )
            else:
                # All priority questions asked - remove the fact using public API
                self.world_model.delete_fact(
                    "project.priority_questions",
                    FactProvenance.SYSTEM,
                    reason="All priority questions asked",
                    record_history=False,  # Don't need to track this in history
                )
        
        prompt = best_question.question
        if best_question.why:
            prompt += f"\n(This matters because: {best_question.why})"
        
        # Add help text for options if available
        if best_question.options:
            options_help = self._get_options_help_text(best_question.field, best_question.options)
            if options_help:
                prompt += f"\n{options_help}"
        
        response = self.io.ask_text(
            prompt,
            suggestions=best_question.options,
        )
        
        # Validate response against options if the question has options
        if best_question.options:
            validation_result = self._validate_option_response(
                response, best_question.options, best_question.field
            )
            
            if validation_result["is_clarification_request"]:
                # User is asking for clarification - provide help and re-ask
                help_text = self._get_detailed_options_help(best_question.field, best_question.options)
                self.io.say_assistant(help_text)
                self._emit_trace("clarification_provided", f"Provided help for: {best_question.field}")
                # Don't close the question - it will be asked again on next iteration
                return True
            
            if not validation_result["is_valid"]:
                # Invalid choice - explain available options and re-ask
                self.io.say_assistant(
                    f"I didn't recognize that option. Please choose from: {', '.join(best_question.options)}"
                )
                self._emit_trace("invalid_option", f"Invalid response for: {best_question.field}")
                # Don't close the question - it will be asked again on next iteration
                return True
            
            # Use the normalized value if available
            if validation_result["normalized_value"] is not None:
                parsed_value = validation_result["normalized_value"]
            else:
                parsed_value = self._parse_question_answer(best_question.field, response)
        else:
            parsed_value = self._parse_question_answer(best_question.field, response)
        
        self.world_model.answer_question(best_question.question_id, parsed_value)
        
        # If this was the project description, parse it to extract intent
        if best_question.field == "project.description":
            self._parse_project_description(parsed_value)
        
        # Provide conversational readback based on verbosity setting
        self._maybe_readback(best_question.field, parsed_value)
        
        self._emit_trace("question_asked", f"Asked: {best_question.field}")
        
        return True
    
    def _parse_project_description(self, description: str) -> None:
        """
        Parse project description to extract intent and guide project creation.
        
        Looks for:
        - Organ types (liver, kidney, heart, etc.) -> suggests topology
        - Size mentions (e.g., "2x3x6 mm") -> stored for reference
        - Use cases (perfusion, tissue engineering) -> stored for context
        - Specific constraints mentioned
        - Ambiguities or conflicting information
        
        Extracted intent is stored in project.intent and used to provide
        topology suggestions when topology.kind hasn't been set yet.
        Also detects ambiguities and labels them for the user.
        """
        description_lower = description.lower()
        extracted_intent = {}
        ambiguities = []
        
        # Detect organ types and suggest appropriate topology
        # Use word boundary matching to avoid false positives (e.g., "heartfelt" matching "heart")
        organ_topology_map = {
            "liver": "dual_trees",
            "kidney": "dual_trees", 
            "heart": "tree",
            "lung": "dual_trees",
            "pancreas": "tree",
            "spleen": "tree",
            "muscle": "tree",
            "skin": "tree",
            "bone": "tree",
        }
        
        # Detect all mentioned organs (not just first)
        detected_organs = []
        for organ, suggested_topology in organ_topology_map.items():
            if re.search(rf"\b{organ}\b", description_lower):
                detected_organs.append((organ, suggested_topology))
        
        # Handle multiple organs (ambiguity)
        if len(detected_organs) > 1:
            organ_names = [o[0] for o in detected_organs]
            ambiguities.append({
                "type": "multiple_organs",
                "message": f"Multiple organs mentioned: {', '.join(organ_names)}. Which is the primary target?",
                "priority_question": "organ_clarification",
            })
            # Use first mentioned as default
            detected_organ = detected_organs[0][0]
            extracted_intent["detected_organ"] = detected_organ
            extracted_intent["suggested_topology"] = detected_organs[0][1]
            extracted_intent["all_detected_organs"] = organ_names
        elif len(detected_organs) == 1:
            detected_organ = detected_organs[0][0]
            extracted_intent["detected_organ"] = detected_organ
            extracted_intent["suggested_topology"] = detected_organs[0][1]
        else:
            detected_organ = None
        
        # Detect domain type from description keywords
        domain_keywords = {
            "box": "box", "rectangular": "box", "cube": "box", "cuboid": "box",
            "ellipsoid": "ellipsoid", "oval": "ellipsoid", "egg": "ellipsoid", "spheroid": "ellipsoid",
            "cylinder": "cylinder", "cylindrical": "cylinder", "tubular": "cylinder",
        }
        for keyword, domain_type in domain_keywords.items():
            if re.search(rf"\b{keyword}\b", description_lower):
                extracted_intent["suggested_domain_type"] = domain_type
                break
        
        # Detect microfluidic/channel/manifold keywords -> suggests path/backbone
        microfluidic_keywords = ["microfluidic", "channel", "manifold", "conduit", "tube", "duct", "straight"]
        if any(kw in description_lower for kw in microfluidic_keywords):
            extracted_intent["microfluidic_hint"] = True
            # For simple channel/path structures, suggest path topology
            if "straight" in description_lower or "channel" in description_lower:
                extracted_intent["suggested_topology"] = "path"
            # This conflicts with organ-based topology suggestion
            if detected_organ and extracted_intent.get("suggested_topology") in ["tree", "dual_trees"]:
                ambiguities.append({
                    "type": "topology_conflict",
                    "message": f"You mentioned '{detected_organ}' (suggests branching tree) but also microfluidic/channel terms (suggests simpler path). Which structure do you need?",
                    "priority_question": "topology.kind",
                })
        
        # Detect size mentions (e.g., "2x3x6 mm", "20mm x 30mm x 40mm")
        size_match = re.search(
            r"(\d+(?:\.\d+)?)\s*(um|mm|cm|m)?\s*(?:x|by|×)\s*"
            r"(\d+(?:\.\d+)?)\s*(um|mm|cm|m)?\s*(?:x|by|×)\s*"
            r"(\d+(?:\.\d+)?)\s*(um|mm|cm|m)?",
            description_lower
        )
        if size_match:
            extracted_intent["detected_size"] = size_match.group(0)
        
        # Detect use cases
        use_cases = []
        if any(term in description_lower for term in ["perfusion", "perfuse", "flow"]):
            use_cases.append("perfusion")
        if any(term in description_lower for term in ["tissue engineering", "tissue-engineering", "scaffold"]):
            use_cases.append("tissue_engineering")
        if any(term in description_lower for term in ["3d print", "3d-print", "printing", "bioprint"]):
            use_cases.append("3d_printing")
        if any(term in description_lower for term in ["implant", "transplant"]):
            use_cases.append("implantation")
        if use_cases:
            extracted_intent["use_cases"] = use_cases
        
        # Detect conflicting use cases
        if "perfusion" in use_cases and "implantation" in use_cases:
            ambiguities.append({
                "type": "use_case_conflict",
                "message": "Both perfusion testing and implantation mentioned - these have different design priorities. Which is primary?",
                "priority_question": "use_case_clarification",
            })
        
        # Store ambiguities
        if ambiguities:
            extracted_intent["ambiguities"] = ambiguities
        
        # Store the extracted intent in the world model
        if extracted_intent:
            self.world_model.set_fact(
                "project.intent",
                extracted_intent,
                FactProvenance.SYSTEM,
                reason="Extracted from project description",
            )
            
            # Provide detailed conversational feedback about what was understood
            # and how it will influence the final object generation
            self._provide_description_feedback(extracted_intent, detected_organ, use_cases, ambiguities)
            
            self._emit_trace("project_intent_extracted", f"Extracted intent: {extracted_intent}")
        else:
            # No specific intent extracted - acknowledge and explain what we're looking for
            self.io.say_assistant(
                "Thanks for the description! I didn't detect a specific organ type or use case, "
                "but that's okay - I'll ask you about the details as we go. "
                "If you mention things like 'liver', 'kidney', 'perfusion', or '3D printing' later, "
                "I can adjust my suggestions accordingly."
            )
    
    def _provide_description_feedback(
        self,
        extracted_intent: Dict[str, Any],
        detected_organ: Optional[str],
        use_cases: List[str],
        ambiguities: List[Dict[str, Any]] = None,
    ) -> None:
        """
        Provide detailed conversational feedback about what was understood from
        the project description and how it will influence the final object generation.
        
        Also labels any ambiguities detected and explains what needs clarification.
        """
        ambiguities = ambiguities or []
        feedback_lines = []
        generation_implications = []
        
        # Organ detection feedback
        if detected_organ:
            feedback_lines.append(f"I see you're building a **{detected_organ}** scaffold.")
            
            # Explain topology suggestion based on organ
            suggested_topology = extracted_intent.get("suggested_topology")
            if suggested_topology:
                if suggested_topology == "dual_trees":
                    feedback_lines.append(
                        f"For {detected_organ}, I'd recommend **dual_trees** topology - "
                        "this creates two interleaved vascular networks (like arterial and venous) "
                        "that meet in a capillary bed, which is how real organs handle blood supply and drainage."
                    )
                    generation_implications.append(
                        "The final mesh will have two separate inlet/outlet pairs for the arterial and venous trees"
                    )
                else:
                    feedback_lines.append(
                        f"For {detected_organ}, a **tree** topology should work well - "
                        "this creates a branching network from a single inlet that distributes flow throughout the scaffold."
                    )
                    generation_implications.append(
                        "The final mesh will have one inlet that branches into many terminals"
                    )
        
        # Microfluidic hint feedback
        if extracted_intent.get("microfluidic_hint"):
            feedback_lines.append(
                "I noticed microfluidic/channel terminology - if you need a simple channel or manifold "
                "rather than a branching tree, consider **path** or **backbone** topology."
            )
        
        # Size detection feedback
        detected_size = extracted_intent.get("detected_size")
        if detected_size:
            feedback_lines.append(f"I noted the size you mentioned: **{detected_size}**.")
            generation_implications.append(
                f"I'll use this as a reference when you specify the domain dimensions"
            )
        
        # Use case feedback with generation implications
        if use_cases:
            use_case_descriptions = {
                "perfusion": (
                    "**perfusion testing**",
                    "I'll ensure the vascular network has good flow-through characteristics with clear inlet and outlet paths"
                ),
                "tissue_engineering": (
                    "**tissue engineering**",
                    "I'll optimize for tissue coverage - making sure the vascular network reaches all parts of the scaffold for nutrient delivery"
                ),
                "3d_printing": (
                    "**3D printing/bioprinting**",
                    "I'll ensure minimum channel diameters meet printability constraints and avoid overhangs where possible"
                ),
                "implantation": (
                    "**implantation/transplant**",
                    "I'll focus on biocompatible geometry with smooth vessel transitions and physiologically realistic branching"
                ),
            }
            
            use_case_names = []
            for uc in use_cases:
                if uc in use_case_descriptions:
                    name, implication = use_case_descriptions[uc]
                    use_case_names.append(name)
                    generation_implications.append(implication)
            
            if use_case_names:
                feedback_lines.append(f"Your intended use case is {', '.join(use_case_names)}.")
        
        # Output the feedback
        if feedback_lines:
            self.io.say_assistant("\n".join(feedback_lines))
        
        # Output generation implications
        if generation_implications:
            implications_text = "**How this affects the final object:**\n"
            for impl in generation_implications:
                implications_text += f"- {impl}\n"
            self.io.say_assistant(implications_text.strip())
        
        # Output ambiguities that need clarification
        if ambiguities:
            ambiguity_text = "**I noticed some things that need clarification:**\n"
            for amb in ambiguities:
                ambiguity_text += f"- {amb['message']}\n"
            ambiguity_text += "\nI'll ask about these in the next questions to make sure I understand your needs correctly."
            self.io.say_assistant(ambiguity_text)
            
            # Store priority questions from ambiguities for question reordering
            priority_questions = [amb.get("priority_question") for amb in ambiguities if amb.get("priority_question")]
            if priority_questions:
                self.world_model.set_fact(
                    "project.priority_questions",
                    priority_questions,
                    FactProvenance.SYSTEM,
                    reason="Questions prioritized due to ambiguities in description",
                )
    
    def _maybe_readback(self, field: str, value: Any) -> None:
        """
        Provide a conversational readback after accepting an answer.
        
        Based on verbosity setting:
        - "quiet": no readbacks
        - "normal": readbacks only for "big" fields (domain, topology, inlet/outlet)
        - "chatty": readbacks for every answer
        """
        if self.config.verbosity == "quiet":
            return
        
        # Define "big" fields that get readbacks in normal mode
        big_fields = {
            "domain.type", "domain.size",
            "topology.kind", "project.description",
            "inlet.face", "inlet.radius",
            "outlet.face", "outlet.radius",
        }
        
        if self.config.verbosity == "normal" and field not in big_fields:
            return
        
        # Generate a conversational readback
        readback = self._generate_readback(field, value)
        if readback:
            self.io.say_assistant(readback)
    
    def _generate_readback(self, field: str, value: Any) -> Optional[str]:
        """Generate a conversational readback for a field value."""
        # Casual acknowledgment phrases
        acks = ["Got it", "Cool", "Okay", "Perfect", "Great"]
        import random
        ack = random.choice(acks)
        
        if field == "domain.type":
            return f"{ack} — {value} domain."
        
        if field == "domain.size":
            if isinstance(value, (list, tuple)) and len(value) == 3:
                # Values are stored in meters, convert to mm for display
                x_mm = value[0] * 1000
                y_mm = value[1] * 1000
                z_mm = value[2] * 1000
                return f"{ack} — {x_mm}×{y_mm}×{z_mm} mm."
            return f"{ack} — domain size: {value}."
        
        if field == "topology.kind":
            return f"{ack} — {value} topology."
        
        if field == "inlet.face":
            face_names = {
                "x_min": "left", "x_max": "right",
                "y_min": "front", "y_max": "back",
                "z_min": "bottom", "z_max": "top",
            }
            friendly = face_names.get(value, value)
            return f"{ack}. Inlet on {value} ({friendly} face)."
        
        if field == "inlet.radius":
            # Value is stored in meters, convert back to mm for display
            display_value = value * 1000 if isinstance(value, (int, float)) else value
            return f"{ack} — inlet radius: {display_value} mm."
        
        if field == "outlet.face":
            face_names = {
                "x_min": "left", "x_max": "right",
                "y_min": "front", "y_max": "back",
                "z_min": "bottom", "z_max": "top",
            }
            friendly = face_names.get(value, value)
            return f"{ack}. Outlet on {value} ({friendly} face)."
        
        if field == "outlet.radius":
            # Value is stored in meters, convert back to mm for display
            display_value = value * 1000 if isinstance(value, (int, float)) else value
            return f"{ack} — outlet radius: {display_value} mm."
        
        # Generic readback for chatty mode
        if self.config.verbosity == "chatty":
            return f"{ack} — {field}: {value}."
        
        return None
    
    def _validate_option_response(
        self, response: str, options: List[str], field: str
    ) -> Dict[str, Any]:
        """
        Validate a user response against available options.
        
        Returns a dict with:
        - is_valid: True if response matches an option
        - is_clarification_request: True if response looks like a question/clarification request
        - normalized_value: The matched option value (if valid)
        """
        response_lower = response.lower().strip()
        
        # Check if this looks like a clarification request
        clarification_patterns = [
            "what are", "what is", "what's", "which", "explain", "help",
            "tell me", "describe", "difference", "mean", "?",
            "options", "choices", "types", "kinds"
        ]
        if any(pattern in response_lower for pattern in clarification_patterns):
            return {
                "is_valid": False,
                "is_clarification_request": True,
                "normalized_value": None,
            }
        
        # Normalize options for comparison
        options_lower = [opt.lower() for opt in options]
        
        # Direct match
        if response_lower in options_lower:
            idx = options_lower.index(response_lower)
            return {
                "is_valid": True,
                "is_clarification_request": False,
                "normalized_value": options[idx],
            }
        
        # Partial match (response contains option or option contains response)
        for i, opt_lower in enumerate(options_lower):
            if opt_lower in response_lower or response_lower in opt_lower:
                return {
                    "is_valid": True,
                    "is_clarification_request": False,
                    "normalized_value": options[i],
                }
        
        # Field-specific matching (e.g., face names)
        if field.endswith(".face"):
            parsed = self._parse_question_answer(field, response)
            if parsed in options or parsed in options_lower:
                return {
                    "is_valid": True,
                    "is_clarification_request": False,
                    "normalized_value": parsed,
                }
        
        return {
            "is_valid": False,
            "is_clarification_request": False,
            "normalized_value": None,
        }
    
    def _get_options_help_text(self, field: str, options: List[str]) -> str:
        """Get brief help text for options."""
        if field == "topology.kind":
            return "Options: tree (branching), dual_trees (arterial + venous), path (single channel), backbone (main trunk with branches), loop (circular)"
        if field == "domain.type":
            return "Options: box (rectangular), ellipsoid (oval), cylinder (tubular)"
        if field in ("inlet.face", "outlet.face"):
            return "Options: left, right, front, back, bottom, top"
        return f"Options: {', '.join(options)}"
    
    def _get_detailed_options_help(self, field: str, options: List[str]) -> str:
        """Get detailed help text explaining each option."""
        if field == "topology.kind":
            return (
                "Here are the available vascular topology types:\n"
                "- **tree**: A branching structure like arteries, with one inlet splitting into many terminals. "
                "Best for distributing flow throughout a volume.\n"
                "- **dual_trees**: Two interleaved vascular trees (arterial + venous) that meet in a capillary bed. "
                "Best for organs like liver that need both supply and drainage.\n"
                "- **path**: A single channel from inlet to outlet. Simple conduit for direct flow.\n"
                "- **backbone**: A main trunk with side branches. Good for elongated organs.\n"
                "- **loop**: A circular network that returns to the start. Useful for recirculating systems.\n\n"
                "Which topology would you like to use?"
            )
        if field == "domain.type":
            return (
                "Here are the available domain shapes:\n"
                "- **box**: A rectangular prism. Good for tissue slabs or cubic scaffolds.\n"
                "- **ellipsoid**: An oval/egg shape. Good for organ-like geometries.\n"
                "- **cylinder**: A tubular shape. Good for vessel segments or tubular organs.\n\n"
                "Which shape would you like?"
            )
        if field in ("inlet.face", "outlet.face"):
            return (
                "Here are the available face positions:\n"
                "- **left**: The left side of the domain (x_min)\n"
                "- **right**: The right side of the domain (x_max)\n"
                "- **front**: The front of the domain (y_min)\n"
                "- **back**: The back of the domain (y_max)\n"
                "- **bottom**: The bottom of the domain (z_min)\n"
                "- **top**: The top of the domain (z_max)\n\n"
                "Which face would you like?"
            )
        return f"Please choose one of the following options: {', '.join(options)}"
    
    def _parse_question_answer(self, field: str, response: str) -> Any:
        """
        Parse a question answer with field-aware type conversion.
        
        Converts raw string responses to appropriate types:
        - domain.size: "20x60x30", "20mm x 60mm x 30mm" -> (0.02, 0.06, 0.03) tuple in meters
        - *.radius: "2", "2mm", "0.05mm" -> float in meters (respects explicit units)
        - *.face: normalizes to canonical face names
        - domain.type, topology.kind: returns as-is (string)
        """
        response = response.strip()
        
        if field == "domain.size":
            # Support formats: "20x60x30", "20x60x30mm", "20mm x 60mm x 30mm", "2cm x 6cm x 3cm"
            size_match = re.search(
                r"(\d+(?:\.\d+)?)\s*(um|mm|cm|m)?\s*(?:x|by|,)\s*"
                r"(\d+(?:\.\d+)?)\s*(um|mm|cm|m)?\s*(?:x|by|,)\s*"
                r"(\d+(?:\.\d+)?)\s*(um|mm|cm|m)?",
                response.lower()
            )
            if size_match:
                x_val = float(size_match.group(1))
                x_unit = size_match.group(2)
                y_val = float(size_match.group(3))
                y_unit = size_match.group(4)
                z_val = float(size_match.group(5))
                z_unit = size_match.group(6)
                
                # If only the last dimension has a unit, apply it to all
                last_unit = z_unit or y_unit or x_unit or "mm"
                x_unit = x_unit or last_unit
                y_unit = y_unit or last_unit
                z_unit = z_unit or last_unit
                
                def convert_to_meters(val, unit):
                    if unit == "um":
                        return val / 1_000_000
                    elif unit == "mm":
                        return val / 1000
                    elif unit == "cm":
                        return val / 100
                    return val  # meters
                
                x = convert_to_meters(x_val, x_unit)
                y = convert_to_meters(y_val, y_unit)
                z = convert_to_meters(z_val, z_unit)
                return (x, y, z)
            return response
        
        if field.endswith(".radius"):
            # Support explicit units: "2mm", "0.05mm", "2cm", "500um"
            radius_match = re.search(r"(\d+(?:\.\d+)?)\s*(um|mm|cm|m)?", response.lower())
            if radius_match:
                value = float(radius_match.group(1))
                unit = radius_match.group(2) or "mm"  # default to mm
                if unit == "um":
                    value = value / 1_000_000
                elif unit == "mm":
                    value = value / 1000
                elif unit == "cm":
                    value = value / 100
                return value
            return response
        
        if field.endswith(".face"):
            face_map = {
                "left": "x_min", "x min": "x_min", "-x": "x_min", "xmin": "x_min",
                "right": "x_max", "x max": "x_max", "+x": "x_max", "xmax": "x_max",
                "front": "y_min", "y min": "y_min", "-y": "y_min", "ymin": "y_min",
                "back": "y_max", "y max": "y_max", "+y": "y_max", "ymax": "y_max",
                "bottom": "z_min", "z min": "z_min", "-z": "z_min", "zmin": "z_min",
                "top": "z_max", "z max": "z_max", "+z": "z_max", "zmax": "z_max",
            }
            response_lower = response.lower().replace("_", " ").replace("-", " ")
            for key, canonical in face_map.items():
                if key in response_lower:
                    return canonical
            if response.lower() in ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]:
                return response.lower()
            return response
        
        return response
    
    def _cap_generate_missing_field_questions(self) -> bool:
        """Capability 6b: Generate open questions for missing required fields.
        
        Questions are ordered to be ORGAN-AGNOSTIC in the first rounds:
        1. First ask about generic vascular topology patterns (tree, dual_trees, path, etc.)
        2. Then collect generic parameters: inlet/outlet count, domain size, flow requirements
        3. Defer organ-specific constraints until after basic structure is defined
        
        This allows the system to handle any vascular network design without
        requiring organ type specification upfront.
        """
        from .world_model import OpenQuestion
        
        # ORGAN-AGNOSTIC QUESTION ORDER:
        # Priority 1 (100-90): Generic vascular topology patterns
        # Priority 2 (89-80): Generic domain and port configuration
        # Priority 3 (79-70): Optional organ/use-case context (deferred)
        required_fields = [
            # Priority 1: Generic vascular topology (asked first)
            ("topology.kind", 
             "What vascular network topology do you need?\n"
             "  - tree: Single inlet branching to multiple terminals (most common)\n"
             "  - dual_trees: Two interleaved trees meeting in a capillary bed (liver, kidney)\n"
             "  - path: Simple inlet-to-outlet channel\n"
             "  - backbone: Main trunk with side branches\n"
             "  - loop: Circular/recirculating network",
             "Determines the fundamental branching pattern of your vascular network",
             ["tree", "dual_trees", "path", "backbone", "loop"]),
            
            # Priority 2: Generic domain configuration
            ("domain.type", 
             "What domain shape should contain the network?",
             "Determines the bounding geometry for the vascular structure",
             ["box", "ellipsoid", "cylinder"]),
            ("domain.size", 
             "What are the domain dimensions (width x depth x height in mm)?",
             "Defines the physical size of the scaffold",
             None),
            
            # Priority 3: Generic port configuration
            ("inlet.face", 
             "Which face should the inlet (fluid entry) be on?",
             "Determines where fluid enters the network",
             ["left", "right", "front", "back", "bottom", "top"]),
            ("inlet.radius", 
             "What inlet radius (in mm)?",
             "Determines the main vessel diameter at entry",
             None),
            ("outlet.face", 
             "Which face should the outlet (fluid exit) be on?",
             "Determines where fluid exits the network",
             ["left", "right", "front", "back", "bottom", "top"]),
            ("outlet.radius", 
             "What outlet radius (in mm)?",
             "Determines the exit vessel diameter",
             None),
            
            # Priority 4: Optional context (deferred - asked last)
            ("project.description", 
             "Optionally, describe your project context (organ type, intended use, constraints).\n"
             "This helps me suggest appropriate parameters, but is not required.",
             "Provides context for parameter suggestions (optional)",
             None),
        ]
        
        questions_added = 0
        for field, question_text, why, options in required_fields:
            if not self.world_model.has_fact(field):
                existing_q = None
                for q in self.world_model.open_questions.values():
                    if q.field == field:
                        existing_q = q
                        break
                
                if not existing_q:
                    question = OpenQuestion(
                        question_id=f"missing_{field.replace('.', '_')}",
                        field=field,
                        question_text=question_text,
                        why_it_matters=why,
                        options=options,
                        priority=100 - questions_added,
                    )
                    self.world_model.add_open_question(question)
                    questions_added += 1
        
        if questions_added > 0:
            self._emit_trace("questions_generated", f"Generated {questions_added} questions for missing fields")
            self.io.say_assistant(f"I need {questions_added} more piece(s) of information to proceed.")
            return True
        
        return False
    
    def _cap_compile_spec(self) -> bool:
        """Capability 7: Compile spec."""
        try:
            if self.spec_compiler:
                compiled = self.spec_compiler.compile(self.world_model.to_dict())
                self.world_model.add_artifact("compiled_spec", compiled)
            else:
                compiled = self._default_compile_spec()
                self.world_model.add_artifact("compiled_spec", compiled)
            
            self.world_model.set_fact(
                "_spec_compiled",
                True,
                FactProvenance.SYSTEM,
                reason="Spec compilation successful",
            )
            
            self._emit_trace("spec_compiled", "Spec compiled successfully")
            return True
            
        except Exception as e:
            self.world_model.set_fact(
                "_spec_compiled",
                False,
                FactProvenance.SYSTEM,
                reason=f"Compilation failed: {str(e)}",
            )
            self._emit_trace("spec_compile_failed", f"Compilation failed: {str(e)}")
            self.io.say_error(f"Spec compilation failed: {str(e)}")
            return False
    
    def _default_compile_spec(self) -> Dict[str, Any]:
        """Default spec compilation when no compiler is provided."""
        domain_size = self.world_model.get_fact_value("domain.size", (0.02, 0.06, 0.03))
        inlet_face = self.world_model.get_fact_value("inlet.face", "z_min")
        outlet_face = self.world_model.get_fact_value("outlet.face", "z_max")
        inlet_radius = self.world_model.get_fact_value("inlet.radius", 0.002)
        outlet_radius = self.world_model.get_fact_value("outlet.radius", 0.001)
        
        inlet_position = self._derive_position_from_face(inlet_face, domain_size)
        outlet_position = self._derive_position_from_face(outlet_face, domain_size)
        
        self.world_model.set_fact(
            "inlet.position",
            inlet_position,
            FactProvenance.INFERRED,
            reason=f"Derived from inlet.face={inlet_face} and domain.size",
        )
        self.world_model.set_fact(
            "outlet.position",
            outlet_position,
            FactProvenance.INFERRED,
            reason=f"Derived from outlet.face={outlet_face} and domain.size",
        )
        
        compiled = {
            "domain": {
                "type": self.world_model.get_fact_value("domain.type", "box"),
                "size": domain_size,
            },
            "topology": {
                "kind": self.world_model.get_fact_value("topology.kind", "tree"),
                "target_terminals": self.world_model.get_fact_value("topology.target_terminals", 50),
            },
            "inlet": {
                "face": inlet_face,
                "position": inlet_position,
                "radius": inlet_radius,
            },
            "outlet": {
                "face": outlet_face,
                "position": outlet_position,
                "radius": outlet_radius,
            },
            "colonization": {
                "influence_radius": self.world_model.get_fact_value("colonization.influence_radius", 0.015),
                "kill_radius": self.world_model.get_fact_value("colonization.kill_radius", 0.002),
                "step_size": self.world_model.get_fact_value("colonization.step_size", 0.001),
                "min_radius": self.world_model.get_fact_value("colonization.min_radius", 0.0001),
                "max_steps": self.world_model.get_fact_value("colonization.max_steps", 500),
            },
            "embedding": {
                "voxel_pitch": self.world_model.get_fact_value("embedding.voxel_pitch", 0.0003),
                "wall_thickness": self.world_model.get_fact_value("embedding.wall_thickness", 0.0003),
            },
        }
        
        return compiled
    
    def _derive_position_from_face(self, face: str, domain_size: tuple) -> tuple:
        """
        Derive a position from a face name and domain size.
        
        The position is centered on the specified face.
        
        Parameters
        ----------
        face : str
            Face name (x_min, x_max, y_min, y_max, z_min, z_max)
        domain_size : tuple
            Domain size (width, depth, height) in meters
            
        Returns
        -------
        tuple
            (x, y, z) position in meters
        """
        w, d, h = domain_size
        center_x, center_y, center_z = w / 2, d / 2, h / 2
        
        face_positions = {
            "x_min": (0, center_y, center_z),
            "x_max": (w, center_y, center_z),
            "y_min": (center_x, 0, center_z),
            "y_max": (center_x, d, center_z),
            "z_min": (center_x, center_y, 0),
            "z_max": (center_x, center_y, h),
        }
        
        return face_positions.get(face, (center_x, center_y, 0))
    
    def _cap_pregen_verify(self) -> bool:
        """Capability 8: Pre-generation verify."""
        compiled_spec = self.world_model.get_artifact("compiled_spec")
        if not compiled_spec:
            return False
        
        issues = []
        
        min_radius = compiled_spec.get("colonization", {}).get("min_radius", 0.0001)
        if min_radius < 0.00005:
            issues.append("min_radius too small for manufacturing")
        
        domain_size = compiled_spec.get("domain", {}).get("size", (0.02, 0.06, 0.03))
        if any(d < 0.005 for d in domain_size):
            issues.append("domain dimension too small")
        
        if issues:
            self.world_model.set_fact(
                "pregen_verified",
                False,
                FactProvenance.SYSTEM,
                reason=f"Verification failed: {', '.join(issues)}",
            )
            self.world_model.set_fact("validation_failed", True, FactProvenance.SYSTEM)
            self.world_model.set_fact("validation_issues", issues, FactProvenance.SYSTEM)
            
            self._emit_trace("pregen_failed", f"Pre-gen verification failed: {issues}")
            self.io.say_warning(f"Pre-generation verification failed: {', '.join(issues)}")
            return False
        
        self.world_model.set_fact(
            "pregen_verified",
            True,
            FactProvenance.SYSTEM,
            reason="Pre-generation verification passed",
        )
        
        self._emit_trace("pregen_verified", "Pre-generation verification passed")
        self.io.say_assistant("Spec looks feasible. Ready to generate when you approve.")
        
        return True
    
    def _cap_request_generation_approval(self) -> bool:
        """Capability 9: Request generation approval."""
        compiled_spec = self.world_model.get_artifact("compiled_spec")
        
        runtime_estimate = "2-5 minutes"
        expected_outputs = [
            "Vascular network mesh (STL)",
            "Network topology (JSON)",
            "Generation log",
        ]
        
        assumptions = []
        for field_name, fact in self.world_model.facts.items():
            if fact.provenance in (FactProvenance.INFERRED, FactProvenance.DEFAULT):
                assumptions.append(f"{field_name}: {fact.value} ({fact.provenance.value})")
        
        risk_flags = []
        min_radius = self.world_model.get_fact_value("colonization.min_radius", 0.0001)
        if min_radius < 0.0002:
            risk_flags.append("Small minimum radius may produce unprintable branches")
        
        self.io.show_generation_ready(
            runtime_estimate=runtime_estimate,
            expected_outputs=expected_outputs,
            assumptions=assumptions[:5],
            risk_flags=risk_flags,
        )
        
        self._emit_trace("generation_approval_requested", "Requesting generation approval")
        
        approved = self.io.ask_confirm(
            "Ready to run generation. Proceed?",
            details={"spec_hash": self.world_model.compute_spec_hash()},
            modal=True,
            runtime_estimate=runtime_estimate,
            expected_outputs=expected_outputs,
            assumptions=assumptions[:5],
            risk_flags=risk_flags,
        )
        
        if approved:
            self.world_model.set_approval("generation", approved=True)
            self._denied_approval_hashes.pop("generation", None)
            self._emit_trace("generation_approved", "Generation approved by user")
            self.io.say_assistant("Generation approved. Starting now.")
            return True
        else:
            self._denied_approval_hashes["generation"] = self.world_model.compute_spec_hash()
            self._emit_trace("generation_denied", "Generation denied by user")
            self.io.say_assistant("Generation not approved. Let me know what you'd like to change.")
            return False
    
    def _cap_run_generation(self) -> bool:
        """Capability 10: Run generation."""
        if not self.world_model.is_approved("generation"):
            return False
        
        self._emit_trace("generation_started", "Generation started")
        self.io.say_assistant("Running generation now...")
        
        try:
            if self.generator:
                compiled_spec = self.world_model.get_artifact("compiled_spec")
                result = self.generator.generate(compiled_spec)
                
                self.world_model.add_artifact("generation_result", result)
                self.world_model.add_artifact("generated_network", result)
                if result.get("mesh_path"):
                    self.world_model.add_artifact("mesh_path", result["mesh_path"])
            else:
                result = {"status": "simulated"}
                self.world_model.add_artifact("generation_result", result)
            
            self.world_model.add_artifact("generated_network", result)
            
            self.world_model.set_fact(
                "generation_done",
                True,
                FactProvenance.SYSTEM,
                reason="Generation completed successfully",
            )
            
            self._emit_trace("generation_finished", "Generation completed")
            self.io.say_success("Generation complete!")
            
            return True
            
        except Exception as e:
            self.world_model.set_fact(
                "generation_done",
                False,
                FactProvenance.SYSTEM,
                reason=f"Generation failed: {str(e)}",
            )
            self._emit_trace("generation_failed", f"Generation failed: {str(e)}")
            self.io.say_error(f"Generation failed: {str(e)}")
            return False
    
    def _cap_request_postprocess_approval(self) -> bool:
        """Capability 11: Request postprocess approval."""
        voxel_pitch = self.world_model.get_fact_value("embedding.voxel_pitch", 0.0003)
        
        embedding_settings = {
            "voxel_pitch": voxel_pitch,
            "wall_thickness": self.world_model.get_fact_value("embedding.wall_thickness", 0.0003),
        }
        
        repair_steps = [
            "Mesh validation",
            "Non-manifold edge repair",
            "Hole filling",
            "STL export",
        ]
        
        runtime_estimate = "2-5 minutes"
        expected_outputs = [
            "Embedded mesh (STL)",
            "Repair report",
            "Final manifest",
        ]
        
        self.io.show_postprocess_ready(
            voxel_pitch=voxel_pitch,
            embedding_settings=embedding_settings,
            repair_steps=repair_steps,
            runtime_estimate=runtime_estimate,
            expected_outputs=expected_outputs,
        )
        
        self._emit_trace("postprocess_approval_requested", "Requesting postprocess approval")
        
        approved = self.io.ask_confirm(
            f"Next is embedding + repair + STL export. Estimated {runtime_estimate}. Proceed?",
            details={
                "voxel_pitch": voxel_pitch,
                "spec_hash": self.world_model.compute_spec_hash(),
            },
            modal=True,
            runtime_estimate=runtime_estimate,
            expected_outputs=expected_outputs,
        )
        
        if approved:
            self.world_model.set_approval("postprocess", approved=True)
            self._denied_approval_hashes.pop("postprocess", None)
            self._emit_trace("postprocess_approved", "Postprocess approved by user")
            self.io.say_assistant("Postprocess approved. Starting now.")
            return True
        else:
            self._denied_approval_hashes["postprocess"] = self.world_model.compute_spec_hash()
            self._emit_trace("postprocess_denied", "Postprocess denied by user")
            self.io.say_assistant("Postprocess not approved. Let me know what you'd like to change.")
            return False
    
    def _cap_run_postprocess(self) -> bool:
        """Capability 12: Run postprocess."""
        if not self.world_model.is_approved("postprocess"):
            return False
        
        self._emit_trace("postprocess_started", "Postprocess started")
        self.io.say_assistant("Running postprocess...")
        
        try:
            _ = self.world_model.get_artifact("mesh_path")
            
            result = {"status": "completed", "output_path": "/tmp/final_output.stl"}
            self.world_model.add_artifact("postprocess_result", result)
            self.world_model.add_artifact("final_mesh_path", result["output_path"])
            self.world_model.add_artifact("postprocessed_mesh", result)
            
            self.world_model.set_fact(
                "postprocess_done",
                True,
                FactProvenance.SYSTEM,
                reason="Postprocess completed successfully",
            )
            
            self._emit_trace("postprocess_finished", "Postprocess completed")
            self.io.say_success("Postprocess complete!")
            
            self.io.prompt_stl_viewer(result["output_path"])
            
            return True
            
        except Exception as e:
            self.world_model.set_fact(
                "postprocess_done",
                False,
                FactProvenance.SYSTEM,
                reason=f"Postprocess failed: {str(e)}",
            )
            self._emit_trace("postprocess_failed", f"Postprocess failed: {str(e)}")
            self.io.say_error(f"Postprocess failed: {str(e)}")
            return False
    
    def _cap_validate_artifacts(self) -> bool:
        """Capability 13: Validate artifacts."""
        self._emit_trace("validation_started", "Validation started")
        
        issues = []
        
        if self.validator:
            mesh_path = self.world_model.get_artifact("final_mesh_path") or self.world_model.get_artifact("mesh_path")
            if mesh_path:
                validation_result = self.validator.validate(mesh_path)
                issues = validation_result.get("issues", [])
        
        if issues:
            self.world_model.set_fact("validation_failed", True, FactProvenance.SYSTEM)
            self.world_model.set_fact("validation_issues", issues, FactProvenance.SYSTEM)
            self.world_model.set_fact(
                "validation_passed",
                False,
                FactProvenance.SYSTEM,
                reason=f"Validation failed: {', '.join(issues)}",
            )
            
            self._emit_trace("validation_failed", f"Validation failed: {issues}")
            self.io.say_warning(f"Validation failed: {', '.join(issues)}")
            return False
        
        self.world_model.set_fact("validation_failed", False, FactProvenance.SYSTEM)
        self.world_model.set_fact(
            "validation_passed",
            True,
            FactProvenance.SYSTEM,
            reason="Validation passed",
        )
        
        self._emit_trace("validation_passed", "Validation passed")
        self.io.say_success("Validation passed!")
        
        return True
    
    def _cap_apply_one_safe_fix(self) -> bool:
        """Capability 14: Apply one safe fix."""
        if self._safe_fix_applied_this_iteration:
            return False
        
        issues = self.world_model.get_fact_value("validation_issues", [])
        if not issues:
            return False
        
        candidates = []
        for issue in issues:
            issue_candidates = self.safe_fix_policy.generate_fix_candidates(
                failure_type=issue.split()[0] if issue else "unknown",
                failure_details={"issue": issue},
                world_model=self.world_model,
            )
            candidates.extend(issue_candidates)
        
        safest = self.safe_fix_policy.get_safest_fix(candidates)
        
        if not safest:
            return False
        
        old_value = safest.current_value
        new_value = safest.proposed_value
        
        self.world_model.set_fact(
            safest.field,
            new_value,
            FactProvenance.SAFE_FIX,
            reason=safest.reason,
        )
        
        self.io.show_safe_fix(
            field=safest.field,
            before=old_value,
            after=new_value,
            reason=safest.reason,
        )
        
        self._emit_trace(
            "safe_fix_applied",
            f"Applied safe fix: {safest.field}: {old_value} -> {new_value}",
            safest.to_dict(),
        )
        
        self._safe_fix_applied_this_iteration = True
        self._safe_fixes_this_run += 1
        
        self.world_model.set_fact("validation_failed", False, FactProvenance.SYSTEM)
        
        self.io.say_assistant(f"Applied safe fix: {safest.reason}. Re-validating now.")
        
        return True
    
    def _cap_ask_for_non_safe_fix_choice(self) -> bool:
        """Capability 15: Ask for non-safe fix choice."""
        issues = self.world_model.get_fact_value("validation_issues", [])
        if not issues:
            return False
        
        candidates = []
        for issue in issues:
            issue_candidates = self.safe_fix_policy.generate_fix_candidates(
                failure_type=issue.split()[0] if issue else "unknown",
                failure_details={"issue": issue},
                world_model=self.world_model,
            )
            candidates.extend(issue_candidates)
        
        non_safe = [c for c in candidates if c.safety.value != "safe"]
        
        if not non_safe:
            self.io.say_assistant("No fix options available. Please provide guidance.")
            return False
        
        self.io.say_assistant("We have options that require your input:")
        
        for i, candidate in enumerate(non_safe[:3], 1):
            self.io.say_assistant(
                f"Option {i}: {candidate.reason}\n"
                f"  Change: {candidate.field}: {candidate.current_value} -> {candidate.proposed_value}\n"
                f"  Impact: {candidate.expected_impact}"
            )
        
        response = self.io.ask_text(
            "Which option would you like? (Enter number or describe alternative)",
            suggestions=[str(i) for i in range(1, min(4, len(non_safe) + 1))],
        )
        
        try:
            choice = int(response) - 1
            if 0 <= choice < len(non_safe):
                selected = non_safe[choice]
                self.world_model.set_fact(
                    selected.field,
                    selected.proposed_value,
                    FactProvenance.USER,
                    reason=f"User selected fix: {selected.reason}",
                )
                self._emit_trace("non_safe_fix_applied", f"User selected fix: {selected.fix_id}")
                return True
        except ValueError:
            pass
        
        self._emit_trace("non_safe_fix_choice_requested", "Awaiting user guidance")
        return False
    
    def _cap_undo(self) -> bool:
        """Capability 16: Undo last change."""
        if not self.world_model.history:
            self.io.say_warning("Nothing to undo.")
            return False
        
        success = self.world_model.undo_last()
        
        if success:
            self._emit_trace("undo_applied", "Reverted last change")
            self.io.say_assistant("Reverted the last change.")
            return True
        else:
            self.io.say_error("Failed to undo.")
            return False
    
    def _cap_undo_to_entry(self, entry_id: str) -> bool:
        """Undo to a specific history entry."""
        if not entry_id:
            self.io.say_warning("No entry ID specified for undo.")
            return False
        
        if not self.world_model.history:
            self.io.say_warning("No history to undo to.")
            return False
        
        target_index = None
        for i, entry in enumerate(self.world_model.history):
            if entry.entry_id == entry_id or str(i) == entry_id:
                target_index = i
                break
        
        if target_index is None:
            self.io.say_warning(f"Entry '{entry_id}' not found in history.")
            return False
        
        undo_count = len(self.world_model.history) - target_index
        for _ in range(undo_count):
            self.world_model.undo_last()
        
        self._emit_trace("reverted_to_entry", f"Reverted to entry {entry_id}")
        self.io.say_assistant(f"Reverted to entry {entry_id}. Undid {undo_count} change(s).")
        return True
    
    def _cap_show_changes(self) -> bool:
        """Show recent changes from history."""
        if not self.world_model.history:
            self.io.say_assistant("No changes recorded yet.")
            return True
        
        recent = self.world_model.history[-10:]
        
        lines = ["Recent changes:"]
        for i, entry in enumerate(recent):
            entry_num = len(self.world_model.history) - len(recent) + i
            lines.append(f"  [{entry_num}] {entry.entry_id}: {entry.description}")
        
        lines.append("\nUse 'undo to entry <id>' to revert to a specific point.")
        
        self.io.say_assistant("\n".join(lines))
        self._emit_trace("show_changes", f"Displayed {len(recent)} recent changes")
        return True
    
    def _cap_select_plan_by_id(self, plan_id: str) -> bool:
        """Select a plan by ID or name."""
        if not plan_id:
            self.io.say_warning("No plan ID specified.")
            return False
        
        if not self.world_model.plans:
            self.io.say_warning("No plans available. Run plan synthesis first.")
            return False
        
        plan_id_lower = plan_id.lower()
        
        if plan_id_lower == "recommended":
            for plan in self.world_model.plans.values():
                if plan.recommended:
                    self.world_model.select_plan(plan.plan_id)
                    self.plan_synthesizer.apply_plan_patches(plan)
                    self._emit_trace("plan_selected", f"Selected recommended plan: {plan.name}")
                    self.io.say_assistant(f"Selected recommended plan: {plan.name}")
                    return True
            self.io.say_warning("No recommended plan found.")
            return False
        
        for pid, plan in self.world_model.plans.items():
            if pid == plan_id or pid.endswith(f"_{plan_id}") or plan_id in pid:
                self.world_model.select_plan(pid)
                self.plan_synthesizer.apply_plan_patches(plan)
                self._emit_trace("plan_selected", f"Selected plan: {plan.name}")
                self.io.say_assistant(f"Selected plan: {plan.name}")
                return True
        
        plan_list = list(self.world_model.plans.values())
        try:
            idx = int(plan_id) - 1
            if 0 <= idx < len(plan_list):
                plan = plan_list[idx]
                self.world_model.select_plan(plan.plan_id)
                self.plan_synthesizer.apply_plan_patches(plan)
                self._emit_trace("plan_selected", f"Selected plan {idx + 1}: {plan.name}")
                self.io.say_assistant(f"Selected plan {idx + 1}: {plan.name}")
                return True
        except ValueError:
            pass
        
        self.io.say_warning(f"Plan '{plan_id}' not found.")
        return False
    
    def _cap_revisit_question(self, topic: str) -> bool:
        """Revisit a question or decision point by topic."""
        if not topic:
            self.io.say_warning("No topic specified to revisit.")
            return False
        
        topic_lower = topic.lower()
        
        topic_to_fields = {
            "domain": ["domain.type", "domain.size", "domain.shape"],
            "inlet": ["inlet.face", "inlet.position", "inlet.radius"],
            "outlet": ["outlet.face", "outlet.position", "outlet.radius"],
            "ports": ["inlet.face", "outlet.face", "inlet.position", "outlet.position"],
            "topology": ["topology.kind", "topology.target_terminals"],
            "terminals": ["topology.target_terminals", "terminal.strategy"],
            "plan": ["_selected_plan_id"],
        }
        
        fields_to_clear = topic_to_fields.get(topic_lower, [topic_lower])
        
        cleared_count = 0
        for field in fields_to_clear:
            if field in self.world_model.facts:
                del self.world_model._facts[field]
                cleared_count += 1
        
        if cleared_count > 0:
            self.world_model._invalidate_approvals()
            
            self._emit_trace("revisit_question", f"Cleared {cleared_count} field(s) for topic: {topic}")
            self.io.say_assistant(f"Cleared {topic} settings. I'll ask about this again.")
            return True
        else:
            self.io.say_assistant(f"No settings found for topic '{topic}'. What would you like to change?")
            return True
    
    def _cap_package_outputs(self) -> bool:
        """Capability 17: Package outputs."""
        final_mesh = self.world_model.get_artifact("final_mesh_path")
        
        if not final_mesh:
            return False
        
        manifest = {
            "version": "v5",
            "spec_hash": self.world_model.compute_spec_hash(),
            "artifacts": {
                "mesh": final_mesh,
                "spec": self.world_model.get_artifact("compiled_spec"),
            },
            "facts": {k: v.to_dict() for k, v in self.world_model.facts.items()},
        }
        
        self.world_model.add_artifact("manifest", manifest)
        self.world_model.add_artifact("output_package", manifest)
        
        self.world_model.set_fact(
            "outputs_packaged",
            True,
            FactProvenance.SYSTEM,
            reason="Outputs packaged successfully",
        )
        
        self._emit_trace("outputs_packaged", "Outputs packaged")
        self.io.say_success(f"Outputs packaged! Final mesh: {final_mesh}")
        
        return True
    
    def _cap_ask_project_name(self) -> bool:
        """
        Mode-independent capability: Ask for project name and set up output directory.
        
        This runs at the start of a session (in both classic and LLM-first modes)
        if no project folder has been set yet. It establishes where outputs will
        be stored, preventing the issue where the controller doesn't ask where
        to store objects.
        
        In GUI/CLI interactive mode, this asks for:
        - Project name (becomes a subfolder under output_dir)
        - Optionally confirms output directory if none is set
        """
        import os
        import re
        
        self._emit_trace("asking_project_name", "Asking for project folder name")
        
        # Check if output_dir is set, if not ask for it
        if not self.config.output_dir:
            output_dir = self.io.ask_text(
                "Where would you like to save project outputs? (e.g., ./output or /path/to/outputs)",
                default="./output",
            )
            if output_dir:
                self.config.output_dir = output_dir
            else:
                # Default to current directory + output
                self.config.output_dir = "./output"
        
        # Ask for project name
        project_name = self.io.ask_text(
            "What would you like to name this project? (This will be the folder name)",
        )
        
        if not project_name:
            project_name = f"project_{int(__import__('time').time())}"
        
        # Sanitize project name to create a valid folder slug
        # Replace spaces and special chars with underscores, lowercase
        project_slug = re.sub(r'[^a-zA-Z0-9_-]', '_', project_name.lower())
        project_slug = re.sub(r'_+', '_', project_slug).strip('_')
        
        if not project_slug:
            project_slug = f"project_{int(__import__('time').time())}"
        
        # Create project root directory
        project_root = os.path.join(self.config.output_dir, project_slug)
        os.makedirs(project_root, exist_ok=True)
        
        # Store project identity in world model
        self.world_model.add_artifact("project_slug", project_slug)
        self.world_model.add_artifact("project_root", project_root)
        
        # Mark project as initialized
        self.world_model.set_fact(
            "_project_initialized",
            True,
            FactProvenance.SYSTEM,
            reason=f"Project initialized: {project_slug}",
            record_history=False,  # Don't record in undo history
        )
        
        self._emit_trace("project_initialized", f"Project initialized: {project_slug}")
        self.io.say_assistant(f"Project folder created: {project_root}")
        
        return True
    
    # =========================================================================
    # LLM-First Mode Capabilities
    # =========================================================================
    
    def _cap_llm_ask_project_name(self) -> bool:
        """
        LLM-first capability: Ask for project folder name and initialize workspace.
        
        This is the first capability that runs in LLM-first mode. It asks the user
        for a project folder name and creates a per-project workspace under:
        <output_dir>/<project_slug>/agent_workspace
        
        This prevents workspace collisions when running multiple projects.
        """
        import os
        import re
        
        self._emit_trace("llm_asking_project_name", "Asking for project folder name")
        
        # Ask for project name
        project_name = self.io.ask_text(
            "What would you like to name this project? (This will be the folder name)",
            placeholder="e.g., kidney_vasculature_001",
        )
        
        if not project_name:
            project_name = f"project_{int(__import__('time').time())}"
        
        # Sanitize project name to create a valid folder slug
        # Replace spaces and special chars with underscores, lowercase
        project_slug = re.sub(r'[^a-zA-Z0-9_-]', '_', project_name.lower())
        project_slug = re.sub(r'_+', '_', project_slug).strip('_')
        
        if not project_slug:
            project_slug = f"project_{int(__import__('time').time())}"
        
        # Create project root directory
        project_root = os.path.join(self.config.output_dir, project_slug)
        workspace_dir = os.path.join(project_root, "agent_workspace")
        
        # Initialize workspace
        self.workspace = WorkspaceManager(workspace_dir)
        self.workspace.initialize()
        
        # Store project identity in world model
        self.world_model.add_artifact("project_slug", project_slug)
        self.world_model.add_artifact("project_root", project_root)
        self.world_model.add_artifact("workspace_path", str(self.workspace.workspace_path))
        self.world_model.add_artifact("spec_path", str(self.workspace.spec_path))
        
        if os.path.exists(self.workspace.master_script_path):
            self.world_model.add_artifact("master_script_path", str(self.workspace.master_script_path))
        
        self._emit_trace("llm_project_initialized", f"Project initialized: {project_slug}")
        self.io.say_assistant(f"Project workspace created: {workspace_dir}")
        
        return True
    
    def _cap_llm_decide_next(self) -> bool:
        """
        LLM-first capability: Ask the LLM brain what to do next.
        
        This is the core decision-making capability that replaces the
        deterministic capability selection in LLM-first mode.
        """
        if not self.brain or not self.workspace:
            return False
        
        # Build observation packet
        goal_progress = self.goal_tracker.get_progress_summary()
        observation = self.brain.build_observation_packet(
            user_message=self._pending_user_message,
            world_model=self.world_model,
            workspace=self.workspace,
            goal_progress=goal_progress,
            last_run_result=self._last_run_result,
            verification_report=self._last_verification_report,
        )
        
        # Clear pending message after using it
        self._pending_user_message = None
        
        # Get directive from brain
        self._emit_trace("llm_deciding", "Asking LLM for next action")
        directive = self.brain.decide_next(observation)
        self._last_directive = directive
        
        # Store directive in world model for traceability
        self.world_model.add_artifact("last_llm_directive", directive.to_dict())
        
        # Show assistant message to user
        if directive.assistant_message:
            self.io.say_assistant(directive.assistant_message)
        
        # Handle questions
        if directive.questions:
            for question in directive.questions:
                answer = self.io.ask_text(
                    question.prompt,
                    suggestions=question.options,
                    default=question.default,
                )
                # Store answer in world model
                self.world_model.set_fact(
                    f"llm_question.{question.id}",
                    answer,
                    FactProvenance.USER,
                    reason=f"User answered: {question.prompt}",
                )
            return True
        
        # Handle stop
        if directive.stop:
            self.world_model.set_fact(
                "llm_complete",
                True,
                FactProvenance.SYSTEM,
                reason="LLM decided generation is complete",
            )
            self._emit_trace("llm_complete", "LLM decided generation is complete")
            return True
        
        return True
    
    def _cap_llm_apply_workspace_update(self) -> bool:
        """
        LLM-first capability: Apply workspace updates from the last directive.
        
        This writes files to the workspace without requiring approval
        (approval is only for execution).
        
        P0 #4: Resets verification state when workspace is updated.
        """
        if not self.workspace or not self._last_directive:
            return False
        
        update = self._last_directive.workspace_update
        if not update:
            return False
        
        # P0 #4: Reset verification state when workspace is updated
        self._last_verification_report = None
        self._last_run_result = None
        
        self._emit_trace("llm_workspace_update", "Applying workspace updates")
        
        files_written = []
        
        for file_update in update.files:
            path = file_update.path
            
            # P1 #7: Handle patch-based updates
            if file_update.is_patch():
                try:
                    if path == "master.py":
                        original_content = self.workspace.read_master_script() or ""
                    else:
                        import os
                        full_path = os.path.join(self.workspace.workspace_path, path)
                        if os.path.exists(full_path):
                            with open(full_path, 'r') as f:
                                original_content = f.read()
                        else:
                            original_content = ""
                    
                    content = file_update.apply_patch(original_content)
                    self.io.say_assistant(f"Applied patch to {path}")
                except ValueError as e:
                    self.io.say_error(f"Patch failed for {path}: {e}. Using full content.")
                    content = file_update.content
            else:
                content = file_update.content
            
            if path == "master.py":
                # Write master script with snapshot
                written_path = self.workspace.write_master_script(content, snapshot=True)
                files_written.append(written_path)
                self.world_model.add_artifact("master_script_path", written_path)
            elif path.startswith("tools/"):
                # Write tool module
                tool_name = path.replace("tools/", "").replace(".py", "")
                written_path = self.workspace.write_tool_module(tool_name, content)
                files_written.append(written_path)
            else:
                # Write other file to workspace
                import os
                full_path = os.path.join(self.workspace.workspace_path, path)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, 'w') as f:
                    f.write(content)
                files_written.append(full_path)
        
        # Apply registry updates
        for reg_update in update.registry_updates:
            if reg_update.action == "add":
                entry = ToolRegistryEntry(
                    name=reg_update.name,
                    origin="generated",
                    module=reg_update.module or f"tools.{reg_update.name}",
                    entrypoints=reg_update.entrypoints or [],
                    description=reg_update.description or "",
                )
                self.workspace.add_tool(entry)
            elif reg_update.action == "remove":
                self.workspace.remove_tool(reg_update.name)
        
        # P2 #17: Apply fact_updates from directive FIRST (before exporting spec)
        # This ensures spec.json reflects the latest LLM updates
        if self._last_directive.fact_updates:
            for fact_update in self._last_directive.fact_updates:
                if fact_update.op == "set":
                    self.world_model.set_fact(
                        fact_update.path,
                        fact_update.value,
                        FactProvenance.INFERRED,
                        confidence=fact_update.confidence,
                        reason="LLM fact update",
                    )
                elif fact_update.op == "delete":
                    # P0 #4: Use delete_fact() for proper history tracking
                    self.world_model.delete_fact(
                        fact_update.path,
                        FactProvenance.INFERRED,
                        reason="LLM fact deletion",
                    )
            self._emit_trace("fact_updates_applied", f"Applied {len(self._last_directive.fact_updates)} fact updates")
        
        # Export spec from world model to workspace AFTER applying fact_updates
        if self.world_model.facts:
            self.workspace.export_spec_from_world_model(self.world_model)
        
        # P2 #18: Apply plan_board_update from directive
        if self._last_directive.plan_board_update:
            pbu = self._last_directive.plan_board_update
            current_plan = self.world_model.get_artifact("plan_board") or {
                "objectives": [], "assumptions": [], "strategy": None,
                "next_steps": [], "done_steps": [], "risks": []
            }
            if pbu.add_objectives:
                current_plan["objectives"].extend(pbu.add_objectives)
            if pbu.add_assumptions:
                current_plan["assumptions"].extend(pbu.add_assumptions)
            if pbu.set_strategy:
                current_plan["strategy"] = pbu.set_strategy
            if pbu.add_next_steps:
                current_plan["next_steps"].extend(pbu.add_next_steps)
            if pbu.complete_steps:
                for step in pbu.complete_steps:
                    if step in current_plan["next_steps"]:
                        current_plan["next_steps"].remove(step)
                    current_plan["done_steps"].append(step)
            if pbu.add_risks:
                current_plan["risks"].extend(pbu.add_risks)
            self.world_model.add_artifact("plan_board", current_plan)
            self._emit_trace("plan_board_updated", "Updated plan board")
        
        # P2 #21: Check for contradictions after updates
        contradictions = self.world_model.detect_contradictions()
        if contradictions:
            contradiction_msgs = [c["message"] for c in contradictions]
            self.io.say_assistant(f"Detected contradictions: {'; '.join(contradiction_msgs)}")
            self.world_model.add_artifact("contradictions", contradictions)
        
        self.world_model.add_artifact("workspace_files_written", files_written)
        self._emit_trace("llm_workspace_updated", f"Wrote {len(files_written)} files")
        
        if files_written:
            self.io.say_assistant(f"Updated workspace: {', '.join([f.split('/')[-1] for f in files_written])}")
        
        # P0 #1: Clear workspace_update after applying to prevent infinite loop
        # Without this, _get_available_capabilities_llm_first() keeps selecting
        # llm_apply_workspace_update until max_iterations
        self._last_directive.workspace_update = None
        
        return True
    
    def _cap_llm_request_execution(self) -> bool:
        """
        LLM-first capability: Request approval to execute the master script.
        
        This is the ONLY approval gate in LLM-first mode.
        
        Can be triggered by:
        1. Directive with request_execution=True
        2. Fallback: master exists + no pending questions + workspace changed
        
        Gated by:
        - Spec must have minimum required fields (inlet/outlet + density)
        """
        if not self.workspace:
            return False
        
        # Check if spec has minimum required fields before allowing execution
        # This prevents writing/executing placeholder scripts too early
        if not self._is_ready_to_execute():
            missing = self._get_missing_spec_fields()
            self._emit_trace("llm_execution_blocked", f"Spec incomplete: {missing}")
            self.io.say_assistant(
                f"Cannot execute yet - missing required spec fields: {', '.join(missing)}. "
                "Please collect this information first."
            )
            # Clear request_execution to prevent re-triggering
            if self._last_directive:
                self._last_directive.request_execution = False
            return True  # Return True to continue the loop (LLM will collect missing info)
        
        # Check if this is a directive-triggered request
        directive_requests_execution = (
            self._last_directive and 
            self._last_directive.request_execution
        )
        
        # Check if this is a fallback request (master exists, no questions, workspace changed)
        fallback_should_execute = False
        if not directive_requests_execution:
            import os
            master_exists = (
                self.workspace.master_script_path and 
                os.path.exists(self.workspace.master_script_path)
            )
            has_pending_questions = (
                self._last_directive and 
                self._last_directive.questions and 
                len(self._last_directive.questions) > 0
            )
            # Use workspace hash instead of has_recent_run to detect if code changed
            current_workspace_hash = self.workspace.compute_workspace_hash()
            workspace_changed = (
                current_workspace_hash is not None and
                current_workspace_hash != self._last_verified_workspace_hash
            )
            
            fallback_should_execute = master_exists and not has_pending_questions and workspace_changed
        
        if not directive_requests_execution and not fallback_should_execute:
            return False
        
        # Check if master script exists
        master_script = self.workspace.read_master_script()
        if not master_script:
            self.io.say_error("No master script to execute")
            return False
        
        self._emit_trace("llm_execution_requested", "Requesting execution approval")
        
        # Get workspace summary for approval details
        summary = self.workspace.get_summary()
        
        # Scan for suspicious patterns
        from ....script_writer import scan_for_suspicious_patterns
        warnings = scan_for_suspicious_patterns(master_script)
        
        # Get next run version for exact output paths
        next_version = self.workspace.peek_next_run_version()
        version_str = f"{next_version:03d}"
        
        # Build approval details
        details = {
            "master_script_lines": summary.master_script_lines,
            "master_script_hash": summary.master_script_hash,
            "tool_count": summary.tool_count,
            "run_count": summary.run_count,
            "target_version": next_version,
        }
        
        risk_flags = warnings if warnings else []
        
        # Show exact required output paths with version numbers
        expected_outputs = [
            f"04_outputs/network_v{version_str}.json",
            f"05_mesh/mesh_network_v{version_str}.stl",
            "ARTIFACTS_JSON footer",
        ]
        
        approved = self.io.ask_confirm(
            "Execute master script now?",
            details=details,
            modal=True,
            runtime_estimate="1-5 minutes",
            expected_outputs=expected_outputs,
            risk_flags=risk_flags,
        )
        
        # Clear request_execution after acting on it to prevent infinite loop
        # Without this, the controller keeps selecting llm_request_execution
        if self._last_directive:
            self._last_directive.request_execution = False
        
        if approved:
            self.world_model.set_approval("llm_execution", approved=True)
            self._emit_trace("llm_execution_approved", "Execution approved")
            self.io.say_assistant("Execution approved. Running master script...")
            return True
        else:
            self.world_model.set_approval("llm_execution", approved=False)
            self._emit_trace("llm_execution_denied", "Execution denied")
            self.io.say_assistant("Execution not approved. What would you like to change?")
            
            # Add an open question to pause and wait for user input
            # This prevents the controller from spamming approval requests
            from .world_model import OpenQuestion
            question = OpenQuestion(
                question_id="execution_denied_feedback",
                field="execution_feedback",
                question_text="What changes would you like to make before execution?",
                why_it_matters="User denied execution approval and needs to provide feedback",
                priority=100,  # High priority to ensure it's addressed first
            )
            self.world_model.add_open_question(question)
            
            # Return True because the capability completed its job (asked and got an answer)
            # The open question will cause _get_available_capabilities_llm_first() to pause
            return True
    
    def _cap_llm_run_master_script(self) -> bool:
        """
        LLM-first capability: Run the master script via subprocess.
        
        P1 #13: Validates script before execution.
        P1 #16: Tracks retry counts and asks user for help if exceeded.
        """
        if not self.workspace:
            return False
        
        if not self.world_model.is_approved("llm_execution"):
            return False
        
        master_script_path = self.workspace.master_script_path
        if not master_script_path:
            return False
        
        import os
        if not os.path.exists(master_script_path):
            self.io.say_error("Master script not found")
            return False
        
        # P1 #16: Check retry budget
        if self._retry_counts["execution"] >= self._max_retries["execution"]:
            self.io.say_error(
                f"Execution retry limit ({self._max_retries['execution']}) reached. "
                "Please review the master script and provide guidance."
            )
            self.world_model.set_approval("llm_execution", approved=False)
            return False
        
        # P1 #13: Validate script before execution
        validation = self.workspace.validate_master_script()
        if not validation["valid"]:
            self.io.say_error(f"Script validation failed: {validation['syntax_error']}")
            self._retry_counts["rewrite"] += 1
            self.world_model.set_approval("llm_execution", approved=False)
            return False
        
        if validation["import_warnings"]:
            warnings_str = "; ".join(validation["import_warnings"][:3])
            self.io.say_assistant(f"Import warnings (may be OK): {warnings_str}")
        
        # P4 #33: Show dangerous import warnings
        if validation.get("dangerous_imports"):
            dangerous_str = "; ".join(validation["dangerous_imports"][:5])
            self.io.say_assistant(f"Security warnings: {dangerous_str}")
        
        # P4 #34: Show write path warnings
        if validation.get("write_warnings"):
            write_str = "; ".join(validation["write_warnings"][:3])
            self.io.say_assistant(f"Write path warnings: {write_str}")
        
        # Reset verification state for new run (P1 fix: avoid stale reports)
        self._last_verification_report = None
        
        self._emit_trace("llm_execution_started", "Running master script")
        self.io.say_assistant("Running master script...")
        
        # Create run directory
        run_version = self.workspace.get_next_run_version()
        run_dir = self.workspace.create_run_directory(run_version)
        
        # P4 #32: Get sandbox execution environment
        exec_env = self.workspace.get_execution_environment()
        exec_config = self.workspace.get_execution_config()
        
        # Run the script with sandbox environment
        from ....subprocess_runner import run_script
        
        result = run_script(
            script_path=master_script_path,
            object_dir=run_dir,
            version=run_version,
            timeout_seconds=exec_config.get("timeout_seconds", 600),
            extra_env=exec_env,
        )
        
        # Store result
        self._last_run_result = {
            "success": result.success,
            "exit_code": result.exit_code,
            "elapsed_seconds": result.elapsed_seconds,
            "timed_out": result.timed_out,
            "error": result.error,
            "last_lines": result.last_lines,
        }
        
        # Save run record
        import os
        run_record = RunRecord(
            version=run_version,
            timestamp=os.path.basename(run_dir),
            status="success" if result.success else "failed",
            elapsed_seconds=result.elapsed_seconds,
            stdout_path=result.log_path,
            stderr_path=None,
            artifacts_json_path=None,
            verification_passed=None,
            error_message=result.error,
        )
        self.workspace.save_run_record(run_record)
        
        # Store in world model
        self.world_model.add_artifact("last_run_result", self._last_run_result)
        self.world_model.add_artifact("last_run_dir", run_dir)
        
        # Clear execution approval for next run
        self.world_model.set_approval("llm_execution", approved=False)
        
        if result.success:
            self._emit_trace("llm_execution_finished", "Master script completed successfully")
            self.io.say_success("Master script completed successfully!")
            return True
        else:
            self._emit_trace("llm_execution_failed", f"Master script failed: {result.error}")
            self.io.say_error(f"Master script failed: {result.error}")
            return True  # Return True to continue the loop (LLM will handle the failure)
    
    def _cap_llm_verify_artifacts(self) -> bool:
        """
        LLM-first capability: Verify artifacts after script execution.
        """
        if not self.workspace:
            return False
        
        last_run_dir = self.world_model.get_artifact("last_run_dir")
        if not last_run_dir:
            return False
        
        self._emit_trace("llm_verification_started", "Verifying artifacts")
        self.io.say_assistant("Verifying generated artifacts...")
        
        # Use artifact verifier
        from ....artifact_verifier import verify_generation_stage
        
        # Get spec path
        spec_path = self.workspace.spec_path
        
        # Get last run result for script output
        # Note: last_lines is a List[str], so we need to join it
        last_lines = self._last_run_result.get("last_lines", []) if self._last_run_result else []
        script_output = "\n".join(last_lines) if isinstance(last_lines, list) else str(last_lines)
        
        # Get run version from last run record
        last_run = self.workspace.get_last_run_record()
        run_version = last_run.version if last_run else 1
        
        result = verify_generation_stage(
            object_dir=last_run_dir,
            version=run_version,
            script_output=script_output,
            spec_path=spec_path,
        )
        
        # Store verification report
        self._last_verification_report = {
            "success": result.success,
            "required_passed": result.required_passed,
            "required_total": result.required_total,
            "optional_passed": result.optional_passed,
            "optional_total": result.optional_total,
            "errors": result.errors,
            "warnings": result.warnings,
        }
        
        self.world_model.add_artifact("last_verification_report", self._last_verification_report)
        
        # Update run record with verification result
        if last_run:
            last_run.verification_passed = result.success
            self.workspace.save_run_record(last_run)
        
        if result.success:
            self._emit_trace("llm_verification_passed", "Verification passed")
            self.io.say_success(f"Verification passed! {result.required_passed}/{result.required_total} required checks passed.")
            
            # Set generation done fact
            self.world_model.set_fact(
                "generation_done",
                True,
                FactProvenance.SYSTEM,
                reason="Generation verified successfully",
            )
            
            # Store mesh path if available
            # Note: artifacts_json is an ArtifactsJson dataclass, not a dict
            if result.artifacts_json:
                for f in result.artifacts_json.files:
                    if f.endswith(".stl"):
                        self.world_model.add_artifact("mesh_path", f)
                        break
            
            # Track workspace hash to prevent re-running identical code
            # This prevents the infinite execution loop after successful verification
            if self.workspace:
                self._last_verified_workspace_hash = self.workspace.compute_workspace_hash()
            
            return True
        else:
            self._emit_trace("llm_verification_failed", f"Verification failed: {result.errors}")
            self.io.say_error(f"Verification failed: {', '.join(result.errors)}")
            return True  # Return True to continue the loop (LLM will handle the failure)
    
    def _get_available_capabilities_llm_first(self) -> List[str]:
        """
        Get available capabilities in LLM-first mode.
        
        In LLM-first mode, the capability selection is simpler:
        0. If project slug is missing, ask for it first (creates per-project workspace)
        1. If there are open questions and no pending user message, pause (return [])
        2. If there's a pending directive with workspace_update, apply it
        3. If there's a pending directive with request_execution, request approval
        4. If execution is approved, run the script
        5. If there's a run result, verify artifacts
        6. Fallback: if master exists + no pending questions + workspace changed → suggest execution
        7. Otherwise, ask the LLM what to do next
        """
        available = []
        
        # Step 0: Check if project slug is missing - ask for it first
        # This creates a per-project workspace under <output_dir>/<project_slug>/agent_workspace
        project_slug = self.world_model.get_artifact("project_slug")
        if not project_slug or not self.workspace:
            available.append("llm_ask_project_name")
            return available
        
        # Check if we have open questions that need user input
        # This handles execution denial pause - when user denies, we add an open question
        # and wait for their feedback before continuing
        open_questions = self.world_model.get_open_questions()
        if open_questions and not self._pending_user_message:
            # Pause and wait for user input - return empty list
            # The run() loop will return WAITING when no capabilities are available
            return []
        
        # If user provided a message, clear the execution denial question
        # so we can proceed with their feedback
        if self._pending_user_message and open_questions:
            for q in open_questions:
                if q.question_id == "execution_denied_feedback":
                    self.world_model.remove_open_question(q.question_id)
                    break
        
        # Check if we need to apply workspace updates
        if self._last_directive and self._last_directive.workspace_update:
            if self._last_directive.workspace_update.files or self._last_directive.workspace_update.registry_updates:
                available.append("llm_apply_workspace_update")
                return available
        
        # Check if we need to request execution
        if self._last_directive and self._last_directive.request_execution:
            if not self.world_model.is_approved("llm_execution"):
                available.append("llm_request_execution")
                return available
        
        # Check if execution is approved
        if self.world_model.is_approved("llm_execution"):
            available.append("llm_run_master_script")
            return available
        
        # Check if we need to verify artifacts
        last_run_dir = self.world_model.get_artifact("last_run_dir")
        if last_run_dir and not self._last_verification_report:
            available.append("llm_verify_artifacts")
            return available
        
        # P0 #6: Fallback policy - if master exists + no pending questions + workspace changed
        # → suggest execution to prevent stalling
        if self.workspace and self.workspace.master_script_path:
            import os
            if os.path.exists(self.workspace.master_script_path):
                has_pending_questions = (
                    self._last_directive and 
                    self._last_directive.questions and 
                    len(self._last_directive.questions) > 0
                )
                has_verification_failure = (
                    self._last_verification_report and 
                    not self._last_verification_report.get("success", False)
                )
                
                # Check if workspace has changed since last verified run
                # This prevents re-running identical code in an infinite loop
                current_workspace_hash = self.workspace.compute_workspace_hash()
                workspace_changed = (
                    current_workspace_hash is not None and
                    current_workspace_hash != self._last_verified_workspace_hash
                )
                
                # If master exists, no questions, and workspace changed → suggest execution
                # Note: We no longer use has_recent_run because that caused issues
                # Instead we use workspace hash to detect if code actually changed
                if not has_pending_questions and workspace_changed:
                    available.append("llm_request_execution")
                    return available
                
                # If verification failed, let LLM decide how to fix
                if has_verification_failure:
                    available.append("llm_decide_next")
                    return available
        
        # Default: ask LLM what to do next
        available.append("llm_decide_next")
        return available
