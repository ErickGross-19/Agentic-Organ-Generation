"""
Policies Module - Safe Fix Classifier and Approval Rules

Defines policies for:
- Classifying fixes as safe or non-safe
- Determining when approval is required
- Selecting the best capability to execute
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from .world_model import WorldModel, Fact


class FixSafety(Enum):
    """Classification of a fix's safety."""
    SAFE = "safe"
    NEEDS_CONFIRMATION = "needs_confirmation"
    UNSAFE = "unsafe"


@dataclass
class FixCandidate:
    """A candidate fix for a validation failure."""
    fix_id: str
    field: str
    current_value: Any
    proposed_value: Any
    reason: str
    safety: FixSafety
    expected_impact: str
    confidence: float = 0.8
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "fix_id": self.fix_id,
            "field": self.field,
            "current_value": self.current_value,
            "proposed_value": self.proposed_value,
            "reason": self.reason,
            "safety": self.safety.value,
            "expected_impact": self.expected_impact,
            "confidence": self.confidence,
        }


class SafeFixPolicy:
    """
    Policy for classifying fixes as safe or non-safe.
    
    A fix is safe if it:
    - Doesn't change topology kind
    - Doesn't move ports across faces unless explicitly allowed
    - Doesn't change unit system
    - Stays within bounded numeric adjustments
    - Doesn't meaningfully change the user's stated intent
    """
    
    SAFE_NUMERIC_FIELDS = {
        "colonization.min_radius": {"min_factor": 0.5, "max_factor": 2.0, "max_abs_change": 0.001},
        "colonization.step_size": {"min_factor": 0.5, "max_factor": 2.0, "max_abs_change": 0.002},
        "colonization.influence_radius": {"min_factor": 0.8, "max_factor": 1.5, "max_abs_change": 0.01},
        "colonization.kill_radius": {"min_factor": 0.5, "max_factor": 2.0, "max_abs_change": 0.005},
        "colonization.max_steps": {"min_factor": 0.5, "max_factor": 2.0, "max_abs_change": 500},
        "inlet.radius": {"min_factor": 0.8, "max_factor": 1.5, "max_abs_change": 0.002},
        "outlet.radius": {"min_factor": 0.8, "max_factor": 1.5, "max_abs_change": 0.002},
        "geometry.min_clearance": {"min_factor": 0.5, "max_factor": 2.0, "max_abs_change": 0.002},
        "embedding.voxel_pitch": {"min_factor": 0.5, "max_factor": 2.0, "max_abs_change": 0.001},
    }
    
    TOPOLOGY_FIELDS = {"topology.kind", "topology.terminal_mode", "topology.style"}
    
    PORT_FACE_FIELDS = {"inlet.face", "outlet.face", "inlet.position", "outlet.position"}
    
    UNIT_FIELDS = {"units.internal", "units.export"}
    
    def classify_fix(
        self,
        field: str,
        current_value: Any,
        proposed_value: Any,
        world_model: "WorldModel",
    ) -> FixSafety:
        """
        Classify a proposed fix as safe, needs_confirmation, or unsafe.
        
        Parameters
        ----------
        field : str
            The field being changed
        current_value : Any
            The current value
        proposed_value : Any
            The proposed new value
        world_model : WorldModel
            The current world model
            
        Returns
        -------
        FixSafety
            The safety classification
        """
        if field in self.TOPOLOGY_FIELDS:
            return FixSafety.UNSAFE
        
        if field in self.UNIT_FIELDS:
            return FixSafety.UNSAFE
        
        if field in self.PORT_FACE_FIELDS:
            if self._is_cross_face_change(field, current_value, proposed_value):
                return FixSafety.NEEDS_CONFIRMATION
        
        if field in self.SAFE_NUMERIC_FIELDS:
            bounds = self.SAFE_NUMERIC_FIELDS[field]
            if self._is_within_safe_bounds(current_value, proposed_value, bounds):
                return FixSafety.SAFE
            else:
                return FixSafety.NEEDS_CONFIRMATION
        
        if self._is_small_numeric_change(current_value, proposed_value):
            return FixSafety.SAFE
        
        return FixSafety.NEEDS_CONFIRMATION
    
    def _is_cross_face_change(
        self,
        field: str,
        current_value: Any,
        proposed_value: Any,
    ) -> bool:
        """Check if a port change moves across faces."""
        if "face" in field:
            return current_value != proposed_value
        
        if "position" in field:
            if isinstance(current_value, (list, tuple)) and isinstance(proposed_value, (list, tuple)):
                if len(current_value) == 3 and len(proposed_value) == 3:
                    current_face = self._get_dominant_face(current_value)
                    proposed_face = self._get_dominant_face(proposed_value)
                    return current_face != proposed_face
        
        return False
    
    def _get_dominant_face(self, position: Tuple[float, float, float]) -> str:
        """Get the dominant face for a position."""
        x, y, z = position
        abs_coords = [abs(x), abs(y), abs(z)]
        max_idx = abs_coords.index(max(abs_coords))
        
        if max_idx == 0:
            return "x_max" if x > 0 else "x_min"
        elif max_idx == 1:
            return "y_max" if y > 0 else "y_min"
        else:
            return "z_max" if z > 0 else "z_min"
    
    def _is_within_safe_bounds(
        self,
        current_value: Any,
        proposed_value: Any,
        bounds: Dict[str, float],
    ) -> bool:
        """Check if a numeric change is within safe bounds."""
        try:
            current = float(current_value)
            proposed = float(proposed_value)
            
            if current == 0:
                return abs(proposed) <= bounds.get("max_abs_change", float("inf"))
            
            ratio = proposed / current
            min_factor = bounds.get("min_factor", 0.5)
            max_factor = bounds.get("max_factor", 2.0)
            
            if not (min_factor <= ratio <= max_factor):
                return False
            
            abs_change = abs(proposed - current)
            max_abs = bounds.get("max_abs_change", float("inf"))
            
            return abs_change <= max_abs
            
        except (TypeError, ValueError):
            return False
    
    def _is_small_numeric_change(
        self,
        current_value: Any,
        proposed_value: Any,
        threshold: float = 0.2,
    ) -> bool:
        """Check if a numeric change is small (within threshold ratio)."""
        try:
            current = float(current_value)
            proposed = float(proposed_value)
            
            if current == 0:
                return abs(proposed) < 0.001
            
            ratio = abs(proposed - current) / abs(current)
            return ratio <= threshold
            
        except (TypeError, ValueError):
            return False
    
    def generate_fix_candidates(
        self,
        failure_type: str,
        failure_details: Dict[str, Any],
        world_model: "WorldModel",
    ) -> List[FixCandidate]:
        """
        Generate fix candidates for a validation failure.
        
        Parameters
        ----------
        failure_type : str
            Type of failure (e.g., "min_radius", "collision", "coverage")
        failure_details : dict
            Details about the failure
        world_model : WorldModel
            The current world model
            
        Returns
        -------
        list
            List of FixCandidate objects
        """
        candidates = []
        
        if failure_type == "min_radius":
            candidates.extend(self._generate_min_radius_fixes(failure_details, world_model))
        elif failure_type == "collision":
            candidates.extend(self._generate_collision_fixes(failure_details, world_model))
        elif failure_type == "coverage":
            candidates.extend(self._generate_coverage_fixes(failure_details, world_model))
        elif failure_type == "mesh_quality":
            candidates.extend(self._generate_mesh_quality_fixes(failure_details, world_model))
        elif failure_type == "manufacturability":
            candidates.extend(self._generate_manufacturability_fixes(failure_details, world_model))
        
        for candidate in candidates:
            candidate.safety = self.classify_fix(
                candidate.field,
                candidate.current_value,
                candidate.proposed_value,
                world_model,
            )
        
        return candidates
    
    def _generate_min_radius_fixes(
        self,
        details: Dict[str, Any],
        world_model: "WorldModel",
    ) -> List[FixCandidate]:
        """Generate fixes for min_radius violations."""
        candidates = []
        
        current_min = world_model.get_fact_value("colonization.min_radius", 0.0001)
        required_min = details.get("required_min", 0.00025)
        
        candidates.append(FixCandidate(
            fix_id="increase_min_radius",
            field="colonization.min_radius",
            current_value=current_min,
            proposed_value=max(current_min, required_min),
            reason="Increase minimum radius to meet manufacturability constraints",
            safety=FixSafety.SAFE,
            expected_impact="Printable structure but fewer fine branches",
            confidence=0.95,
        ))
        
        return candidates
    
    def _generate_collision_fixes(
        self,
        details: Dict[str, Any],
        world_model: "WorldModel",
    ) -> List[FixCandidate]:
        """Generate fixes for collision issues."""
        candidates = []
        
        current_clearance = world_model.get_fact_value("geometry.min_clearance", 0.0005)
        
        candidates.append(FixCandidate(
            fix_id="increase_clearance",
            field="geometry.min_clearance",
            current_value=current_clearance,
            proposed_value=current_clearance * 1.5,
            reason="Increase minimum clearance to prevent collisions",
            safety=FixSafety.SAFE,
            expected_impact="Reduces collision probability but may reduce coverage",
            confidence=0.85,
        ))
        
        current_step = world_model.get_fact_value("colonization.step_size", 0.001)
        
        candidates.append(FixCandidate(
            fix_id="reduce_step_size",
            field="colonization.step_size",
            current_value=current_step,
            proposed_value=current_step * 0.8,
            reason="Smaller steps allow finer collision avoidance",
            safety=FixSafety.SAFE,
            expected_impact="Slower generation but better collision avoidance",
            confidence=0.75,
        ))
        
        return candidates
    
    def _generate_coverage_fixes(
        self,
        details: Dict[str, Any],
        world_model: "WorldModel",
    ) -> List[FixCandidate]:
        """Generate fixes for coverage issues."""
        candidates = []
        
        current_coverage = details.get("coverage", 0.5)
        current_terminals = world_model.get_fact_value("topology.target_terminals", 50)
        
        suggested_terminals = int(current_terminals * (1.0 / current_coverage) * 0.9)
        
        candidates.append(FixCandidate(
            fix_id="increase_terminals",
            field="topology.target_terminals",
            current_value=current_terminals,
            proposed_value=suggested_terminals,
            reason="More terminals improve tissue coverage",
            safety=FixSafety.NEEDS_CONFIRMATION,
            expected_impact=f"Expected coverage improvement to ~{min(current_coverage * 1.3, 0.95)*100:.0f}%",
            confidence=0.8,
        ))
        
        current_influence = world_model.get_fact_value("colonization.influence_radius", 0.015)
        
        candidates.append(FixCandidate(
            fix_id="increase_influence",
            field="colonization.influence_radius",
            current_value=current_influence,
            proposed_value=current_influence * 1.2,
            reason="Larger influence radius allows growth toward more attractors",
            safety=FixSafety.SAFE,
            expected_impact="Better coverage but potentially less uniform distribution",
            confidence=0.7,
        ))
        
        return candidates
    
    def _generate_mesh_quality_fixes(
        self,
        details: Dict[str, Any],
        world_model: "WorldModel",
    ) -> List[FixCandidate]:
        """Generate fixes for mesh quality issues."""
        candidates = []
        
        current_min = world_model.get_fact_value("colonization.min_radius", 0.0001)
        
        candidates.append(FixCandidate(
            fix_id="increase_min_radius_mesh",
            field="colonization.min_radius",
            current_value=current_min,
            proposed_value=current_min * 1.5,
            reason="Larger minimum radius produces cleaner meshes",
            safety=FixSafety.SAFE,
            expected_impact="Better mesh quality but fewer fine branches",
            confidence=0.85,
        ))
        
        candidates.append(FixCandidate(
            fix_id="enable_mesh_repair",
            field="postprocess.enable_mesh_repair",
            current_value=False,
            proposed_value=True,
            reason="Mesh repair can fix non-manifold edges and holes",
            safety=FixSafety.SAFE,
            expected_impact="Cleaner mesh but slightly longer processing",
            confidence=0.9,
        ))
        
        return candidates
    
    def _generate_manufacturability_fixes(
        self,
        details: Dict[str, Any],
        world_model: "WorldModel",
    ) -> List[FixCandidate]:
        """Generate fixes for manufacturability issues."""
        candidates = []
        
        issue_type = details.get("issue_type", "")
        
        if "diameter" in issue_type:
            current_min = world_model.get_fact_value("colonization.min_radius", 0.0001)
            manufacturing_min = 0.00025
            
            candidates.append(FixCandidate(
                fix_id="meet_min_diameter",
                field="colonization.min_radius",
                current_value=current_min,
                proposed_value=max(current_min, manufacturing_min),
                reason="Minimum diameter must meet manufacturing constraints",
                safety=FixSafety.SAFE,
                expected_impact="Printable structure but fewer fine branches",
                confidence=0.95,
            ))
        
        if "thickness" in issue_type:
            current_thickness = world_model.get_fact_value("embedding.wall_thickness", 0.0003)
            
            candidates.append(FixCandidate(
                fix_id="increase_wall_thickness",
                field="embedding.wall_thickness",
                current_value=current_thickness,
                proposed_value=0.0005,
                reason="Wall thickness must meet manufacturing constraints",
                safety=FixSafety.SAFE,
                expected_impact="Stronger walls but larger overall structure",
                confidence=0.95,
            ))
        
        return candidates
    
    def get_safest_fix(
        self,
        candidates: List[FixCandidate],
    ) -> Optional[FixCandidate]:
        """
        Get the safest fix from a list of candidates.
        
        Returns the highest-confidence safe fix, or None if no safe fixes exist.
        """
        safe_fixes = [c for c in candidates if c.safety == FixSafety.SAFE]
        
        if not safe_fixes:
            return None
        
        safe_fixes.sort(key=lambda c: c.confidence, reverse=True)
        return safe_fixes[0]


class ApprovalPolicy:
    """
    Policy for determining when approval is required.
    
    Approval is always required for:
    - Generation (any script execution producing geometry)
    - Postprocess (embedding/voxelization/repair/export)
    """
    
    APPROVAL_REQUIRED_ACTIONS = {
        "run_generation",
        "run_postprocess",
    }
    
    def requires_approval(self, action: str) -> bool:
        """Check if an action requires approval."""
        return action in self.APPROVAL_REQUIRED_ACTIONS
    
    def get_approval_type(self, action: str) -> Optional[str]:
        """Get the approval type for an action."""
        if action == "run_generation":
            return "generation"
        elif action == "run_postprocess":
            return "postprocess"
        return None


class CapabilitySelectionPolicy:
    """
    Policy for selecting the best capability to execute.
    
    Selection heuristic:
    1. Unblocks the closest gated goal
    2. Has the highest information gain if blocked
    3. Respects approval constraints
    4. Applies at most one safe fix per iteration
    """
    
    CAPABILITY_PRIORITIES = {
        "ingest_user_event": 100,
        "interpret_user_turn": 95,
        "apply_patch": 90,
        "undo": 85,
        "ask_best_next_question": 80,
        "propose_tailored_plans": 75,
        "select_plan": 70,
        "compile_spec": 65,
        "pregen_verify": 60,
        "request_generation_approval": 55,
        "run_generation": 50,
        "request_postprocess_approval": 45,
        "run_postprocess": 40,
        "validate_artifacts": 35,
        "apply_one_safe_fix": 30,
        "ask_for_non_safe_fix_choice": 25,
        "package_outputs": 20,
        "summarize_living_spec": 15,
    }
    
    def select_capability(
        self,
        available_capabilities: List[str],
        world_model: "WorldModel",
        goal_tracker: Any,
        last_action: Optional[str] = None,
        safe_fix_applied_this_iteration: bool = False,
    ) -> Optional[str]:
        """
        Select the best capability to execute.
        
        Parameters
        ----------
        available_capabilities : list
            List of available capability names
        world_model : WorldModel
            The current world model
        goal_tracker : GoalTracker
            The goal tracker
        last_action : str, optional
            The last action taken
        safe_fix_applied_this_iteration : bool
            Whether a safe fix was already applied this iteration
            
        Returns
        -------
        str or None
            The selected capability name, or None if no capability should be executed
        """
        if safe_fix_applied_this_iteration:
            available_capabilities = [
                c for c in available_capabilities
                if c != "apply_one_safe_fix"
            ]
        
        next_goal = goal_tracker.get_next_goal()
        if next_goal:
            goal_capabilities = self._get_capabilities_for_goal(next_goal.goal_id)
            for cap in goal_capabilities:
                if cap in available_capabilities:
                    return cap
        
        available_capabilities.sort(
            key=lambda c: self.CAPABILITY_PRIORITIES.get(c, 0),
            reverse=True,
        )
        
        return available_capabilities[0] if available_capabilities else None
    
    def _get_capabilities_for_goal(self, goal_id: str) -> List[str]:
        """Get capabilities that can advance a specific goal."""
        goal_to_capabilities = {
            "spec_minimum_complete": [
                "interpret_user_turn",
                "apply_patch",
                "ask_best_next_question",
                "propose_tailored_plans",
                "select_plan",
            ],
            "spec_compiled": ["compile_spec"],
            "pregen_verified": ["pregen_verify", "apply_one_safe_fix"],
            "generation_approved": ["request_generation_approval"],
            "generation_done": ["run_generation"],
            "postprocess_approved": ["request_postprocess_approval"],
            "postprocess_done": ["run_postprocess"],
            "validation_passed": ["validate_artifacts", "apply_one_safe_fix"],
            "outputs_packaged": ["package_outputs"],
        }
        
        return goal_to_capabilities.get(goal_id, [])
