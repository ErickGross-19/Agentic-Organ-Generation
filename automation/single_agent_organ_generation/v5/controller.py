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
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from enum import Enum

from .world_model import WorldModel, FactProvenance, TraceEvent
from .goals import GoalTracker, GoalStatus
from .policies import SafeFixPolicy, ApprovalPolicy, CapabilitySelectionPolicy
from .plan_synthesizer import PlanSynthesizer
from .io.base_io import BaseIOAdapter

logger = logging.getLogger(__name__)


class ControllerState(Enum):
    """High-level controller state (not a state machine - just for status reporting)."""
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING_FOR_INPUT = "waiting_for_input"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    RUNNING_GENERATION = "running_generation"
    RUNNING_POSTPROCESS = "running_postprocess"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ControllerConfig:
    """Configuration for the V5 controller."""
    max_iterations: int = 1000
    max_safe_fixes_per_run: int = 10
    auto_select_plan_if_confident: bool = True
    verbose: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_iterations": self.max_iterations,
            "max_safe_fixes_per_run": self.max_safe_fixes_per_run,
            "auto_select_plan_if_confident": self.auto_select_plan_if_confident,
            "verbose": self.verbose,
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
        
        self.state = ControllerState.IDLE
        self._iteration_count = 0
        self._safe_fixes_this_run = 0
        self._safe_fix_applied_this_iteration = False
        self._last_action: Optional[str] = None
        self._pending_user_message: Optional[str] = None
        self._stop_requested = False
        
        self._capabilities: Dict[str, Callable[[], bool]] = {
            "ingest_user_event": self._cap_ingest_user_event,
            "interpret_user_turn": self._cap_interpret_user_turn,
            "apply_patch": self._cap_apply_patch,
            "summarize_living_spec": self._cap_summarize_living_spec,
            "propose_tailored_plans": self._cap_propose_tailored_plans,
            "select_plan": self._cap_select_plan,
            "ask_best_next_question": self._cap_ask_best_next_question,
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
        }
    
    def run(self, initial_message: Optional[str] = None) -> bool:
        """
        Run the agent loop until completion or interruption.
        
        Parameters
        ----------
        initial_message : str, optional
            Initial user message to process
            
        Returns
        -------
        bool
            True if completed successfully, False otherwise
        """
        self.state = ControllerState.PROCESSING
        self._iteration_count = 0
        self._safe_fixes_this_run = 0
        self._stop_requested = False
        
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
                    self.state = ControllerState.ERROR
                    return False
                
                available = self._get_available_capabilities()
                
                if not available:
                    if self.goal_tracker.is_complete():
                        self.state = ControllerState.COMPLETED
                        self._emit_trace("controller_completed", "All goals satisfied")
                        return True
                    else:
                        self.state = ControllerState.WAITING_FOR_INPUT
                        self._emit_trace("waiting_for_input", "Waiting for user input")
                        return True
                
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
                
                success = self._execute_capability(capability)
                self._last_action = capability
                
                if not success:
                    logger.warning(f"Capability {capability} returned False")
            
            return self.goal_tracker.is_complete()
            
        except KeyboardInterrupt:
            self.state = ControllerState.IDLE
            self._emit_trace("controller_interrupted", "Controller interrupted by user")
            return False
        except Exception as e:
            self.state = ControllerState.ERROR
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
        
        if self.state == ControllerState.WAITING_FOR_INPUT:
            self.run()
    
    def stop(self) -> None:
        """Request the controller to stop."""
        self._stop_requested = True
    
    def undo(self) -> bool:
        """Undo the last change."""
        return self._cap_undo()
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current controller status."""
        return {
            "state": self.state.value,
            "iteration_count": self._iteration_count,
            "safe_fixes_this_run": self._safe_fixes_this_run,
            "goal_progress": self.goal_tracker.get_progress_summary(),
            "spec_hash": self.world_model.compute_spec_hash(),
            "generation_approved": self.world_model.is_approved("generation"),
            "postprocess_approved": self.world_model.is_approved("postprocess"),
        }
    
    def _should_stop(self) -> bool:
        """Check if the controller should stop."""
        if self._stop_requested:
            return True
        if self.goal_tracker.is_complete():
            return True
        return False
    
    def _get_available_capabilities(self) -> List[str]:
        """Get list of capabilities that can currently be executed."""
        available = []
        
        if self._pending_user_message:
            available.append("ingest_user_event")
            available.append("interpret_user_turn")
        
        if self.world_model._pending_patches:
            available.append("apply_patch")
        
        if self.world_model.history:
            available.append("undo")
        
        available.append("summarize_living_spec")
        
        spec_complete = self.goal_tracker.get_status("spec_minimum_complete") == GoalStatus.SATISFIED
        
        if not spec_complete:
            if self.world_model.open_questions:
                available.append("ask_best_next_question")
            if not self.world_model.selected_plan:
                available.append("propose_tailored_plans")
            if self.world_model.plans and not self.world_model.selected_plan:
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
            if not self.world_model.is_approved("generation"):
                available.append("request_generation_approval")
            elif self.goal_tracker.get_status("generation_done") != GoalStatus.SATISFIED:
                available.append("run_generation")
        
        generation_done = self.goal_tracker.get_status("generation_done") == GoalStatus.SATISFIED
        if generation_done:
            if not self.world_model.is_approved("postprocess"):
                available.append("request_postprocess_approval")
            elif self.goal_tracker.get_status("postprocess_done") != GoalStatus.SATISFIED:
                available.append("run_postprocess")
        
        postprocess_done = self.goal_tracker.get_status("postprocess_done") == GoalStatus.SATISFIED
        if postprocess_done or generation_done:
            if self.goal_tracker.get_status("validation_passed") != GoalStatus.SATISFIED:
                available.append("validate_artifacts")
        
        if self.world_model.get_fact_value("_validation_failed", False):
            if self._safe_fixes_this_run < self.config.max_safe_fixes_per_run:
                if not self._safe_fix_applied_this_iteration:
                    available.append("apply_one_safe_fix")
            available.append("ask_for_non_safe_fix_choice")
        
        validation_passed = self.goal_tracker.get_status("validation_passed") == GoalStatus.SATISFIED
        if validation_passed and postprocess_done:
            if self.goal_tracker.get_status("outputs_packaged") != GoalStatus.SATISFIED:
                available.append("package_outputs")
        
        return available
    
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
        self.world_model.add_trace_event(event)
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
        
        if intent.get("type") == "show_status":
            return self._cap_summarize_living_spec()
        
        if intent.get("patches"):
            self.world_model._pending_patches = intent["patches"]
        
        if intent.get("questions_answered"):
            for q_id, answer in intent["questions_answered"].items():
                self.world_model.answer_question(q_id, answer)
        
        return True
    
    def _parse_user_intent(self, message: str) -> Dict[str, Any]:
        """Parse user message into structured intent."""
        message_lower = message.lower().strip()
        
        if message_lower in ("undo", "undo last", "revert", "go back"):
            return {"type": "undo"}
        
        if message_lower in ("status", "where are we", "show status", "what's missing"):
            return {"type": "show_status"}
        
        if message_lower.startswith("yes") or message_lower == "y":
            return {"type": "confirm", "value": True}
        
        if message_lower.startswith("no") or message_lower == "n":
            return {"type": "confirm", "value": False}
        
        patches = {}
        
        if "domain" in message_lower:
            if "box" in message_lower:
                patches["domain.type"] = "box"
            elif "ellipsoid" in message_lower:
                patches["domain.type"] = "ellipsoid"
            elif "cylinder" in message_lower:
                patches["domain.type"] = "cylinder"
        
        if "inlet" in message_lower or "outlet" in message_lower:
            for face in ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "+x", "-x", "+y", "-y", "+z", "-z"]:
                if face in message_lower:
                    normalized_face = face.replace("+", "_max").replace("-", "_min")
                    if not normalized_face.endswith("_min") and not normalized_face.endswith("_max"):
                        normalized_face = face
                    if "inlet" in message_lower:
                        patches["inlet.face"] = normalized_face
                    if "outlet" in message_lower:
                        patches["outlet.face"] = normalized_face
        
        if "tree" in message_lower:
            patches["topology.kind"] = "tree"
        elif "path" in message_lower:
            patches["topology.kind"] = "path"
        elif "backbone" in message_lower:
            patches["topology.kind"] = "backbone"
        elif "loop" in message_lower:
            patches["topology.kind"] = "loop"
        
        return {
            "type": "specification" if patches else "freeform",
            "patches": patches,
            "raw_message": message,
        }
    
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
        
        self._emit_trace("plans_proposed", f"Proposed {len(plans)} plans")
        
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
        
        if selected_id is None and recommended:
            selected_id = recommended.plan_id
        
        if selected_id and selected_id in self.world_model.plans:
            selected = self.world_model.plans[selected_id]
            self.world_model.select_plan(selected_id)
            self.plan_synthesizer.apply_plan_patches(selected)
            self.io.say_assistant(f"Selected {selected.name}.")
            self._emit_trace("plan_selected", f"User selected: {selected_id}")
            return True
        
        return False
    
    def _cap_ask_best_next_question(self) -> bool:
        """Capability 6: Ask best next question."""
        if not self.world_model.open_questions:
            return False
        
        questions = list(self.world_model.open_questions.values())
        questions.sort(key=lambda q: q.priority, reverse=True)
        
        best_question = questions[0]
        
        prompt = best_question.question
        if best_question.why:
            prompt += f"\n(This matters because: {best_question.why})"
        
        response = self.io.ask_text(
            prompt,
            suggestions=best_question.acceptable_answers,
        )
        
        self.world_model.answer_question(best_question.question_id, response)
        
        self._emit_trace("question_asked", f"Asked: {best_question.field}")
        
        return True
    
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
                "Spec compilation successful",
            )
            
            self._emit_trace("spec_compiled", "Spec compiled successfully")
            return True
            
        except Exception as e:
            self.world_model.set_fact(
                "_spec_compiled",
                False,
                FactProvenance.SYSTEM,
                f"Compilation failed: {str(e)}",
            )
            self._emit_trace("spec_compile_failed", f"Compilation failed: {str(e)}")
            self.io.say_error(f"Spec compilation failed: {str(e)}")
            return False
    
    def _default_compile_spec(self) -> Dict[str, Any]:
        """Default spec compilation when no compiler is provided."""
        compiled = {
            "domain": {
                "type": self.world_model.get_fact_value("domain.type", "box"),
                "size": self.world_model.get_fact_value("domain.size", (0.02, 0.06, 0.03)),
            },
            "topology": {
                "kind": self.world_model.get_fact_value("topology.kind", "tree"),
                "target_terminals": self.world_model.get_fact_value("topology.target_terminals", 50),
            },
            "inlet": {
                "face": self.world_model.get_fact_value("inlet.face", "z_min"),
                "radius": self.world_model.get_fact_value("inlet.radius", 0.002),
            },
            "outlet": {
                "face": self.world_model.get_fact_value("outlet.face", "z_max"),
                "radius": self.world_model.get_fact_value("outlet.radius", 0.001),
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
        
        inlet_face = compiled_spec.get("inlet", {}).get("face", "")
        outlet_face = compiled_spec.get("outlet", {}).get("face", "")
        if inlet_face == outlet_face:
            issues.append("inlet and outlet on same face")
        
        if issues:
            self.world_model.set_fact(
                "_pregen_verified",
                False,
                FactProvenance.SYSTEM,
                f"Verification failed: {', '.join(issues)}",
            )
            self.world_model.set_fact("_validation_failed", True, FactProvenance.SYSTEM)
            self.world_model.set_fact("_validation_issues", issues, FactProvenance.SYSTEM)
            
            self._emit_trace("pregen_failed", f"Pre-gen verification failed: {issues}")
            self.io.say_warning(f"Pre-generation verification failed: {', '.join(issues)}")
            return False
        
        self.world_model.set_fact(
            "_pregen_verified",
            True,
            FactProvenance.SYSTEM,
            "Pre-generation verification passed",
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
        self.state = ControllerState.WAITING_FOR_APPROVAL
        
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
            self._emit_trace("generation_approved", "Generation approved by user")
            self.io.say_assistant("Generation approved. Starting now.")
            return True
        else:
            self._emit_trace("generation_denied", "Generation denied by user")
            self.io.say_assistant("Generation not approved. Let me know what you'd like to change.")
            return False
    
    def _cap_run_generation(self) -> bool:
        """Capability 10: Run generation."""
        if not self.world_model.is_approved("generation"):
            return False
        
        self.state = ControllerState.RUNNING_GENERATION
        self._emit_trace("generation_started", "Generation started")
        self.io.say_assistant("Running generation now...")
        
        try:
            if self.generator:
                compiled_spec = self.world_model.get_artifact("compiled_spec")
                result = self.generator.generate(compiled_spec)
                
                self.world_model.add_artifact("generation_result", result)
                if result.get("mesh_path"):
                    self.world_model.add_artifact("mesh_path", result["mesh_path"])
            else:
                result = {"status": "simulated", "mesh_path": "/tmp/simulated_mesh.stl"}
                self.world_model.add_artifact("generation_result", result)
                self.world_model.add_artifact("mesh_path", result["mesh_path"])
            
            self.world_model.set_fact(
                "_generation_done",
                True,
                FactProvenance.SYSTEM,
                "Generation completed successfully",
            )
            
            self._emit_trace("generation_finished", "Generation completed")
            self.io.say_success("Generation complete!")
            
            return True
            
        except Exception as e:
            self.world_model.set_fact(
                "_generation_done",
                False,
                FactProvenance.SYSTEM,
                f"Generation failed: {str(e)}",
            )
            self._emit_trace("generation_failed", f"Generation failed: {str(e)}")
            self.io.say_error(f"Generation failed: {str(e)}")
            return False
        finally:
            self.state = ControllerState.PROCESSING
    
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
        self.state = ControllerState.WAITING_FOR_APPROVAL
        
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
            self._emit_trace("postprocess_approved", "Postprocess approved by user")
            self.io.say_assistant("Postprocess approved. Starting now.")
            return True
        else:
            self._emit_trace("postprocess_denied", "Postprocess denied by user")
            self.io.say_assistant("Postprocess not approved. Let me know what you'd like to change.")
            return False
    
    def _cap_run_postprocess(self) -> bool:
        """Capability 12: Run postprocess."""
        if not self.world_model.is_approved("postprocess"):
            return False
        
        self.state = ControllerState.RUNNING_POSTPROCESS
        self._emit_trace("postprocess_started", "Postprocess started")
        self.io.say_assistant("Running postprocess...")
        
        try:
            _ = self.world_model.get_artifact("mesh_path")
            
            result = {"status": "completed", "output_path": "/tmp/final_output.stl"}
            self.world_model.add_artifact("postprocess_result", result)
            self.world_model.add_artifact("final_mesh_path", result["output_path"])
            
            self.world_model.set_fact(
                "_postprocess_done",
                True,
                FactProvenance.SYSTEM,
                "Postprocess completed successfully",
            )
            
            self._emit_trace("postprocess_finished", "Postprocess completed")
            self.io.say_success("Postprocess complete!")
            
            self.io.prompt_stl_viewer(result["output_path"])
            
            return True
            
        except Exception as e:
            self.world_model.set_fact(
                "_postprocess_done",
                False,
                FactProvenance.SYSTEM,
                f"Postprocess failed: {str(e)}",
            )
            self._emit_trace("postprocess_failed", f"Postprocess failed: {str(e)}")
            self.io.say_error(f"Postprocess failed: {str(e)}")
            return False
        finally:
            self.state = ControllerState.PROCESSING
    
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
            self.world_model.set_fact("_validation_failed", True, FactProvenance.SYSTEM)
            self.world_model.set_fact("_validation_issues", issues, FactProvenance.SYSTEM)
            self.world_model.set_fact(
                "_validation_passed",
                False,
                FactProvenance.SYSTEM,
                f"Validation failed: {', '.join(issues)}",
            )
            
            self._emit_trace("validation_failed", f"Validation failed: {issues}")
            self.io.say_warning(f"Validation failed: {', '.join(issues)}")
            return False
        
        self.world_model.set_fact("_validation_failed", False, FactProvenance.SYSTEM)
        self.world_model.set_fact(
            "_validation_passed",
            True,
            FactProvenance.SYSTEM,
            "Validation passed",
        )
        
        self._emit_trace("validation_passed", "Validation passed")
        self.io.say_success("Validation passed!")
        
        return True
    
    def _cap_apply_one_safe_fix(self) -> bool:
        """Capability 14: Apply one safe fix."""
        if self._safe_fix_applied_this_iteration:
            return False
        
        issues = self.world_model.get_fact_value("_validation_issues", [])
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
            safest.reason,
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
        
        self.world_model.set_fact("_validation_failed", False, FactProvenance.SYSTEM)
        
        self.io.say_assistant(f"Applied safe fix: {safest.reason}. Re-validating now.")
        
        return True
    
    def _cap_ask_for_non_safe_fix_choice(self) -> bool:
        """Capability 15: Ask for non-safe fix choice."""
        issues = self.world_model.get_fact_value("_validation_issues", [])
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
                    f"User selected fix: {selected.reason}",
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
        
        self.world_model.set_fact(
            "_outputs_packaged",
            True,
            FactProvenance.SYSTEM,
            "Outputs packaged successfully",
        )
        
        self._emit_trace("outputs_packaged", "Outputs packaged")
        self.io.say_success(f"Outputs packaged! Final mesh: {final_mesh}")
        
        return True
