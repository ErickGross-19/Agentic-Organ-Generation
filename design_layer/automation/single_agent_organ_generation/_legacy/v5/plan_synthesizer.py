"""
Plan Synthesizer Module

Generates object-specific plans based on the world model.
Plans differ on meaningful levers, not cosmetic differences.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import hashlib

if TYPE_CHECKING:
    from .world_model import WorldModel, Plan


@dataclass
class PlanRisk:
    """A risk identified in a plan."""
    category: str
    description: str
    severity: str
    mitigation: Optional[str] = None


@dataclass
class PlanParameters:
    """Parameter draft for a plan."""
    colonization_influence_radius: Optional[float] = None
    colonization_kill_radius: Optional[float] = None
    colonization_step_size: Optional[float] = None
    colonization_min_radius: Optional[float] = None
    colonization_max_steps: Optional[int] = None
    colonization_initial_radius: Optional[float] = None
    colonization_radius_decay: Optional[float] = None
    target_terminals: Optional[int] = None
    routing_strategy: Optional[str] = None
    terminal_strategy: Optional[str] = None
    embedding_strategy: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in {
            "colonization.influence_radius": self.colonization_influence_radius,
            "colonization.kill_radius": self.colonization_kill_radius,
            "colonization.step_size": self.colonization_step_size,
            "colonization.min_radius": self.colonization_min_radius,
            "colonization.max_steps": self.colonization_max_steps,
            "colonization.initial_radius": self.colonization_initial_radius,
            "colonization.radius_decay": self.colonization_radius_decay,
            "topology.target_terminals": self.target_terminals,
            "routing.strategy": self.routing_strategy,
            "terminal.strategy": self.terminal_strategy,
            "embedding.strategy": self.embedding_strategy,
        }.items() if v is not None}


class PlanSynthesizer:
    """
    Synthesizes object-specific plans based on the world model.
    
    Instead of always returning "Plan A/B/C template", it:
    - Detects what dimensions/topology/constraints imply
    - Generates plans that differ on real levers
    - Recommends one plan and explains why
    - Computes runtime estimates from world model facts
    """
    
    def __init__(self, world_model: "WorldModel"):
        self.world_model = world_model
    
    def compute_runtime_estimate(
        self,
        topology_kind: str,
        complexity_tier: str = "medium",
        target_terminals: Optional[int] = None,
    ) -> str:
        """
        Compute runtime estimate based on world model facts.
        
        Parameters
        ----------
        topology_kind : str
            The topology type (path, tree, backbone, loop)
        complexity_tier : str
            Complexity tier: "low", "medium", "high"
        target_terminals : int, optional
            Number of target terminals (for tree topology)
            
        Returns
        -------
        str
            Human-readable runtime estimate with range
        """
        domain_size = self.world_model.get_fact_value("domain.size", (0.02, 0.06, 0.03))
        if isinstance(domain_size, (list, tuple)) and len(domain_size) == 3:
            domain_volume = domain_size[0] * domain_size[1] * domain_size[2]
        else:
            domain_volume = 0.02 * 0.06 * 0.03
        
        voxel_pitch = self.world_model.get_fact_value("embedding.voxel_pitch", 3e-4)
        
        if target_terminals is None:
            target_terminals = self.world_model.get_fact_value("topology.target_terminals", 50)
        
        base_time_seconds = {
            "path": 10,
            "backbone": 25,
            "tree": 60,
            "loop": 15,
        }.get(topology_kind, 30)
        
        complexity_multiplier = {
            "low": 0.5,
            "medium": 1.0,
            "high": 2.0,
        }.get(complexity_tier, 1.0)
        
        volume_factor = (domain_volume / (0.02 * 0.06 * 0.03)) ** 0.5
        
        resolution_factor = (3e-4 / max(voxel_pitch, 1e-5)) ** 1.5
        
        terminal_factor = 1.0
        if topology_kind == "tree" and target_terminals:
            terminal_factor = (target_terminals / 50) ** 0.7
        
        estimated_seconds = (
            base_time_seconds 
            * complexity_multiplier 
            * volume_factor 
            * resolution_factor 
            * terminal_factor
        )
        
        min_seconds = estimated_seconds * 0.7
        max_seconds = estimated_seconds * 1.5
        
        def format_time(seconds: float) -> str:
            if seconds < 60:
                return f"{int(seconds)} seconds"
            elif seconds < 3600:
                minutes = seconds / 60
                if minutes < 2:
                    return f"{minutes:.1f} minute"
                return f"{minutes:.0f} minutes"
            else:
                hours = seconds / 3600
                return f"{hours:.1f} hours"
        
        return f"{format_time(min_seconds)} - {format_time(max_seconds)}"
    
    def compute_output_estimate(self, topology_kind: str) -> Dict[str, Any]:
        """
        Compute expected outputs based on world model facts.
        
        Returns
        -------
        dict
            Expected outputs including file types and sizes
        """
        domain_size = self.world_model.get_fact_value("domain.size", (0.02, 0.06, 0.03))
        if isinstance(domain_size, (list, tuple)) and len(domain_size) == 3:
            domain_volume = domain_size[0] * domain_size[1] * domain_size[2]
        else:
            domain_volume = 0.02 * 0.06 * 0.03
        
        voxel_pitch = self.world_model.get_fact_value("embedding.voxel_pitch", 3e-4)
        target_terminals = self.world_model.get_fact_value("topology.target_terminals", 50)
        
        base_vertices = {
            "path": 1000,
            "backbone": 5000,
            "tree": 20000,
            "loop": 3000,
        }.get(topology_kind, 10000)
        
        terminal_factor = (target_terminals / 50) if topology_kind == "tree" else 1.0
        estimated_vertices = int(base_vertices * terminal_factor)
        
        estimated_stl_size_mb = estimated_vertices * 0.0001
        
        voxel_count = domain_volume / (voxel_pitch ** 3)
        estimated_voxel_size_mb = voxel_count * 1e-6
        
        return {
            "files": [
                "network.json (tree structure)",
                "network.stl (mesh geometry)",
                "embedded.stl (voxelized mesh)",
                "manifest.json (metadata)",
            ],
            "estimated_vertices": estimated_vertices,
            "estimated_stl_size_mb": f"{estimated_stl_size_mb:.1f} MB",
            "estimated_voxel_count": f"{voxel_count:.0e}",
        }
    
    def synthesize_plans(self) -> List["Plan"]:
        """
        Synthesize tailored plans based on current world model state.
        
        Uses project.intent to shape plan recommendations and parameters.
        
        Returns
        -------
        list
            List of Plan objects, with one marked as recommended
        """
        from .world_model import Plan
        
        topology_kind = self.world_model.get_fact_value("topology.kind", "tree")
        domain_type = self.world_model.get_fact_value("domain.type", "box")
        domain_size = self.world_model.get_fact_value("domain.size", (0.02, 0.06, 0.03))
        
        # Get project intent to influence plan generation
        project_intent = self.world_model.get_fact_value("project.intent", {})
        
        if topology_kind == "path":
            plans = self._synthesize_path_plans(domain_type, domain_size)
        elif topology_kind == "backbone":
            plans = self._synthesize_backbone_plans(domain_type, domain_size)
        elif topology_kind == "tree":
            plans = self._synthesize_tree_plans(domain_type, domain_size)
        elif topology_kind == "loop":
            plans = self._synthesize_loop_plans(domain_type, domain_size)
        elif topology_kind == "dual_trees":
            plans = self._synthesize_dual_trees_plans(domain_type, domain_size)
        else:
            plans = self._synthesize_tree_plans(domain_type, domain_size)
        
        # Apply intent-based adjustments to plans
        plans = self._apply_intent_adjustments(plans, project_intent)
        
        return plans
    
    def _apply_intent_adjustments(
        self,
        plans: List["Plan"],
        project_intent: Dict[str, Any],
    ) -> List["Plan"]:
        """
        Adjust plans based on project intent extracted from description.
        
        This makes the intent actually drive planning, not just commentary.
        """
        if not project_intent:
            return plans
        
        use_cases = project_intent.get("use_cases", [])
        detected_organ = project_intent.get("detected_organ")
        
        for plan in plans:
            # Adjust for perfusion use case
            if "perfusion" in use_cases:
                # Prioritize plans with good flow-through characteristics
                if "balanced" in plan.plan_id or "straight" in plan.plan_id:
                    if not plan.recommended:
                        plan.risks = [r for r in plan.risks if "coverage" not in r.lower()]
                # Add flow optimization note
                plan.parameter_draft["flow.optimize_for"] = "perfusion"
            
            # Adjust for 3D printing use case
            if "3d_printing" in use_cases:
                # Add manufacturability constraints
                plan.parameter_draft["manufacturing.min_channel_diameter"] = 0.0005  # 0.5mm
                plan.parameter_draft["manufacturing.min_wall_thickness"] = 0.0003  # 0.3mm
                if "dense" in plan.plan_id:
                    plan.risks.append("Dense branching may challenge printer resolution")
            
            # Adjust for tissue engineering use case
            if "tissue_engineering" in use_cases:
                # Prioritize coverage
                if "balanced" in plan.plan_id or "dense" in plan.plan_id:
                    plan.parameter_draft["coverage.target"] = 0.9  # 90% coverage target
            
            # Adjust based on organ type
            if detected_organ:
                if detected_organ in ["liver", "kidney", "lung"]:
                    # These organs benefit from dual vascular networks
                    plan.parameter_draft["organ.type"] = detected_organ
        
        return plans
    
    def _synthesize_dual_trees_plans(
        self,
        domain_type: str,
        domain_size: tuple,
    ) -> List["Plan"]:
        """Synthesize plans for DUAL_TREES topology (e.g., liver with arterial + venous)."""
        from .world_model import Plan
        
        plans = []
        
        domain_volume = domain_size[0] * domain_size[1] * domain_size[2]
        suggested_terminals = max(20, int(domain_volume * 1e6 * 30))  # Fewer per tree
        
        inlet_radius = self.world_model.get_fact_value("inlet.radius", 0.002)
        
        interleaved_runtime = self.compute_runtime_estimate("tree", "high", suggested_terminals * 2)
        separated_runtime = self.compute_runtime_estimate("tree", "medium", suggested_terminals * 2)
        
        plans.append(Plan(
            plan_id="dual_trees_interleaved",
            name="Interleaved Dual Trees",
            interpretation="Two vascular trees (arterial + venous) that interleave and meet in a capillary bed",
            geometry_strategy="dual_space_colonization_interleaved",
            parameter_draft={
                "colonization.influence_radius": 0.012,
                "colonization.kill_radius": 0.002,
                "colonization.step_size": 0.001,
                "colonization.min_radius": 0.0001,
                "colonization.max_steps": 600,
                "colonization.initial_radius": inlet_radius,
                "colonization.radius_decay": 0.94,
                "topology.target_terminals_per_tree": suggested_terminals,
                "dual.meeting_shell_thickness": 0.002,
                "dual.anastomosis_strategy": "capillary_bed",
            },
            risks=[
                "Complex geometry may challenge mesh generation",
                "Requires careful inlet/outlet placement on opposite faces",
            ],
            cost_estimate=f"High complexity, {interleaved_runtime}",
            what_needed_from_user=["Confirm inlet/outlet face placement"],
            patch_set={
                "colonization.influence_radius": 0.012,
                "dual.anastomosis_strategy": "capillary_bed",
            },
            recommended=True,
        ))
        
        plans.append(Plan(
            plan_id="dual_trees_separated",
            name="Separated Dual Trees",
            interpretation="Two independent vascular trees in separate regions of the domain",
            geometry_strategy="dual_space_colonization_separated",
            parameter_draft={
                "colonization.influence_radius": 0.015,
                "colonization.kill_radius": 0.002,
                "colonization.step_size": 0.001,
                "colonization.min_radius": 0.00012,
                "colonization.max_steps": 400,
                "colonization.initial_radius": inlet_radius,
                "colonization.radius_decay": 0.95,
                "topology.target_terminals_per_tree": suggested_terminals,
                "dual.separation_plane": "z_mid",
            },
            risks=[
                "Trees don't connect - no flow between them",
                "May leave central region uncovered",
            ],
            cost_estimate=f"Medium complexity, {separated_runtime}",
            what_needed_from_user=["Separation plane orientation"],
            patch_set={
                "colonization.influence_radius": 0.015,
                "dual.separation_plane": "z_mid",
            },
            recommended=False,
        ))
        
        return plans
    
    def _synthesize_path_plans(
        self,
        domain_type: str,
        domain_size: tuple,
    ) -> List["Plan"]:
        """Synthesize plans for PATH topology."""
        from .world_model import Plan
        
        plans = []
        
        straight_runtime = self.compute_runtime_estimate("path", "low")
        curved_runtime = self.compute_runtime_estimate("path", "medium")
        serpentine_runtime = self.compute_runtime_estimate("path", "medium")
        
        plans.append(Plan(
            plan_id="path_straight",
            name="Straight Channel",
            interpretation="Direct path from inlet to outlet with minimal routing",
            geometry_strategy="straight_line",
            parameter_draft={
                "routing.strategy": "straight",
                "colonization.step_size": 0.001,
            },
            risks=["May not utilize full domain volume"],
            cost_estimate=f"Low complexity, {straight_runtime}",
            what_needed_from_user=[],
            patch_set={"routing.strategy": "straight"},
            recommended=True,
        ))
        
        plans.append(Plan(
            plan_id="path_curved",
            name="Curved Channel",
            interpretation="Curved path with smooth bends for better flow characteristics",
            geometry_strategy="bezier_curve",
            parameter_draft={
                "routing.strategy": "curved",
                "geometry.bend_radius": min(domain_size) * 0.3,
            },
            risks=["Requires sufficient domain depth for curves"],
            cost_estimate=f"Low complexity, {curved_runtime}",
            what_needed_from_user=["Preferred bend radius if different from default"],
            patch_set={"routing.strategy": "curved"},
            recommended=False,
        ))
        
        plans.append(Plan(
            plan_id="path_serpentine",
            name="Serpentine Channel",
            interpretation="S-curve path maximizing channel length within domain",
            geometry_strategy="serpentine",
            parameter_draft={
                "routing.strategy": "serpentine",
                "geometry.num_turns": 3,
            },
            risks=["Higher pressure drop", "More complex manufacturing"],
            cost_estimate=f"Medium complexity, {serpentine_runtime}",
            what_needed_from_user=["Number of turns if different from default"],
            patch_set={"routing.strategy": "serpentine"},
            recommended=False,
        ))
        
        return plans
    
    def _synthesize_backbone_plans(
        self,
        domain_type: str,
        domain_size: tuple,
    ) -> List["Plan"]:
        """Synthesize plans for BACKBONE topology."""
        from .world_model import Plan
        
        plans = []
        
        longest_axis = ["x", "y", "z"][domain_size.index(max(domain_size))]
        
        ladder_runtime = self.compute_runtime_estimate("backbone", "medium")
        u_shape_runtime = self.compute_runtime_estimate("backbone", "medium")
        separate_runtime = self.compute_runtime_estimate("backbone", "low")
        
        plans.append(Plan(
            plan_id="backbone_ladder",
            name="Ladder Configuration",
            interpretation="Parallel legs connected by rungs along the backbone",
            geometry_strategy="ladder",
            parameter_draft={
                "backbone.axis": longest_axis,
                "backbone.leg_connection": "ladder",
                "backbone.leg_count": 4,
            },
            risks=["Requires uniform leg spacing"],
            cost_estimate=f"Medium complexity, {ladder_runtime}",
            what_needed_from_user=["Number of legs if different from 4"],
            patch_set={
                "backbone.axis": longest_axis,
                "backbone.leg_connection": "ladder",
            },
            recommended=True,
        ))
        
        plans.append(Plan(
            plan_id="backbone_u_shape",
            name="U-Shape Configuration",
            interpretation="Legs connected at ends forming U-shapes",
            geometry_strategy="u_shape",
            parameter_draft={
                "backbone.axis": longest_axis,
                "backbone.leg_connection": "u_shape",
                "backbone.leg_count": 4,
            },
            risks=["Higher pressure drop at turns"],
            cost_estimate=f"Medium complexity, {u_shape_runtime}",
            what_needed_from_user=["Number of legs if different from 4"],
            patch_set={
                "backbone.axis": longest_axis,
                "backbone.leg_connection": "u_shape",
            },
            recommended=False,
        ))
        
        plans.append(Plan(
            plan_id="backbone_separate",
            name="Separate Legs",
            interpretation="Independent parallel legs with individual ports",
            geometry_strategy="separate",
            parameter_draft={
                "backbone.axis": longest_axis,
                "backbone.leg_connection": "separate",
                "backbone.leg_count": 4,
                "backbone.port_style": "individual",
            },
            risks=["Requires multiple inlet/outlet ports"],
            cost_estimate=f"Low complexity, {separate_runtime}",
            what_needed_from_user=["Number of legs", "Port configuration"],
            patch_set={
                "backbone.axis": longest_axis,
                "backbone.leg_connection": "separate",
            },
            recommended=False,
        ))
        
        return plans
    
    def _synthesize_tree_plans(
        self,
        domain_type: str,
        domain_size: tuple,
    ) -> List["Plan"]:
        """Synthesize plans for TREE topology."""
        from .world_model import Plan
        
        plans = []
        
        domain_volume = domain_size[0] * domain_size[1] * domain_size[2]
        suggested_terminals = max(20, int(domain_volume * 1e6 * 50))
        
        inlet_radius = self.world_model.get_fact_value("inlet.radius", 0.002)
        
        balanced_runtime = self.compute_runtime_estimate("tree", "medium", suggested_terminals)
        dense_runtime = self.compute_runtime_estimate("tree", "high", int(suggested_terminals * 1.5))
        sparse_runtime = self.compute_runtime_estimate("tree", "low", int(suggested_terminals * 0.6))
        
        plans.append(Plan(
            plan_id="tree_balanced",
            name="Balanced Tree",
            interpretation="Symmetric branching with uniform coverage",
            geometry_strategy="space_colonization_balanced",
            parameter_draft={
                "colonization.influence_radius": 0.015,
                "colonization.kill_radius": 0.002,
                "colonization.step_size": 0.001,
                "colonization.min_radius": 0.0001,
                "colonization.max_steps": 500,
                "colonization.initial_radius": inlet_radius,
                "colonization.radius_decay": 0.95,
                "topology.target_terminals": suggested_terminals,
                "terminal.strategy": "space_filling",
            },
            risks=["May not reach corners of non-convex domains"],
            cost_estimate=f"Medium complexity, {balanced_runtime}",
            what_needed_from_user=[],
            patch_set={
                "colonization.influence_radius": 0.015,
                "colonization.kill_radius": 0.002,
                "terminal.strategy": "space_filling",
            },
            recommended=True,
        ))
        
        plans.append(Plan(
            plan_id="tree_dense",
            name="Dense Tree",
            interpretation="High terminal density for maximum coverage",
            geometry_strategy="space_colonization_dense",
            parameter_draft={
                "colonization.influence_radius": 0.01,
                "colonization.kill_radius": 0.001,
                "colonization.step_size": 0.0008,
                "colonization.min_radius": 0.00008,
                "colonization.max_steps": 800,
                "colonization.initial_radius": inlet_radius,
                "colonization.radius_decay": 0.92,
                "topology.target_terminals": int(suggested_terminals * 1.5),
                "terminal.strategy": "dense",
            },
            risks=[
                "Longer generation time",
                "May have thin branches near print limit",
                "Higher mesh complexity",
            ],
            cost_estimate=f"High complexity, {dense_runtime}",
            what_needed_from_user=["Confirm acceptable generation time"],
            patch_set={
                "colonization.influence_radius": 0.01,
                "colonization.kill_radius": 0.001,
                "terminal.strategy": "dense",
            },
            recommended=False,
        ))
        
        plans.append(Plan(
            plan_id="tree_sparse",
            name="Sparse Tree",
            interpretation="Fewer, larger branches for robust manufacturing",
            geometry_strategy="space_colonization_sparse",
            parameter_draft={
                "colonization.influence_radius": 0.02,
                "colonization.kill_radius": 0.003,
                "colonization.step_size": 0.0015,
                "colonization.min_radius": 0.0002,
                "colonization.max_steps": 300,
                "colonization.initial_radius": inlet_radius,
                "colonization.radius_decay": 0.97,
                "topology.target_terminals": int(suggested_terminals * 0.6),
                "terminal.strategy": "sparse",
            },
            risks=["Lower coverage", "May leave uncovered regions"],
            cost_estimate=f"Low complexity, {sparse_runtime}",
            what_needed_from_user=[],
            patch_set={
                "colonization.influence_radius": 0.02,
                "colonization.kill_radius": 0.003,
                "terminal.strategy": "sparse",
            },
            recommended=False,
        ))
        
        return plans
    
    def _synthesize_loop_plans(
        self,
        domain_type: str,
        domain_size: tuple,
    ) -> List["Plan"]:
        """Synthesize plans for LOOP topology."""
        from .world_model import Plan
        
        plans = []
        
        single_runtime = self.compute_runtime_estimate("loop", "low")
        mesh_runtime = self.compute_runtime_estimate("loop", "high")
        spiral_runtime = self.compute_runtime_estimate("loop", "medium")
        
        plans.append(Plan(
            plan_id="loop_single",
            name="Single Loop",
            interpretation="One continuous loop from inlet back to outlet",
            geometry_strategy="single_loop",
            parameter_draft={
                "loop.style": "single_loop",
                "loop.count": 1,
            },
            risks=["Limited coverage area"],
            cost_estimate=f"Low complexity, {single_runtime}",
            what_needed_from_user=[],
            patch_set={"loop.style": "single_loop"},
            recommended=True,
        ))
        
        plans.append(Plan(
            plan_id="loop_mesh",
            name="Mesh Network",
            interpretation="Interconnected grid of loops for redundant flow paths",
            geometry_strategy="mesh",
            parameter_draft={
                "loop.style": "mesh",
                "loop.grid_size": (3, 3),
            },
            risks=["Complex manufacturing", "Many junction points"],
            cost_estimate=f"High complexity, {mesh_runtime}",
            what_needed_from_user=["Grid dimensions if different from 3x3"],
            patch_set={"loop.style": "mesh"},
            recommended=False,
        ))
        
        plans.append(Plan(
            plan_id="loop_spiral",
            name="Spiral Loop",
            interpretation="Spiral path maximizing coverage within domain",
            geometry_strategy="spiral",
            parameter_draft={
                "loop.style": "spiral",
                "loop.turns": 3,
            },
            risks=["Pressure drop increases with turns"],
            cost_estimate=f"Medium complexity, {spiral_runtime}",
            what_needed_from_user=["Number of spiral turns"],
            patch_set={"loop.style": "spiral"},
            recommended=False,
        ))
        
        return plans
    
    def get_recommended_plan(self, plans: List["Plan"]) -> Optional["Plan"]:
        """Get the recommended plan from a list."""
        for plan in plans:
            if plan.recommended:
                return plan
        return plans[0] if plans else None
    
    def generate_recommendation_rationale(self, plan: "Plan") -> str:
        """Generate a rationale for why a plan is recommended."""
        topology_kind = self.world_model.get_fact_value("topology.kind", "tree")
        domain_type = self.world_model.get_fact_value("domain.type", "box")
        
        rationale_parts = []
        
        rationale_parts.append(f"I recommend {plan.name} because:")
        
        if "balanced" in plan.plan_id:
            rationale_parts.append(
                "- It provides uniform coverage across the domain"
            )
            rationale_parts.append(
                "- The branching pattern follows physiological principles (Murray's law)"
            )
        elif "straight" in plan.plan_id:
            rationale_parts.append(
                "- It's the simplest approach for a direct channel"
            )
            rationale_parts.append(
                "- Minimizes pressure drop and manufacturing complexity"
            )
        elif "ladder" in plan.plan_id:
            rationale_parts.append(
                "- The ladder configuration provides good coverage with parallel flow"
            )
            rationale_parts.append(
                "- It's well-suited for the elongated domain shape"
            )
        elif "single_loop" in plan.plan_id:
            rationale_parts.append(
                "- A single loop is the most straightforward recirculating design"
            )
            rationale_parts.append(
                "- It ensures continuous flow with minimal dead zones"
            )
        
        if plan.risks:
            rationale_parts.append(f"\nPotential risks to be aware of:")
            for risk in plan.risks[:2]:
                rationale_parts.append(f"- {risk}")
        
        return "\n".join(rationale_parts)
    
    def apply_plan_patches(self, plan: "Plan") -> None:
        """Apply a plan's patch set to the world model."""
        from .world_model import FactProvenance
        
        for field, value in plan.patch_set.items():
            self.world_model.set_fact(
                field=field,
                value=value,
                provenance=FactProvenance.INFERRED,
                reason=f"Applied from plan: {plan.name}",
            )
