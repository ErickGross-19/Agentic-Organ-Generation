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
    """
    
    def __init__(self, world_model: "WorldModel"):
        self.world_model = world_model
    
    def synthesize_plans(self) -> List["Plan"]:
        """
        Synthesize tailored plans based on current world model state.
        
        Returns
        -------
        list
            List of Plan objects, with one marked as recommended
        """
        from .world_model import Plan
        
        topology_kind = self.world_model.get_fact_value("topology.kind", "tree")
        domain_type = self.world_model.get_fact_value("domain.type", "box")
        domain_size = self.world_model.get_fact_value("domain.size", (0.02, 0.06, 0.03))
        
        if topology_kind == "path":
            return self._synthesize_path_plans(domain_type, domain_size)
        elif topology_kind == "backbone":
            return self._synthesize_backbone_plans(domain_type, domain_size)
        elif topology_kind == "tree":
            return self._synthesize_tree_plans(domain_type, domain_size)
        elif topology_kind == "loop":
            return self._synthesize_loop_plans(domain_type, domain_size)
        else:
            return self._synthesize_tree_plans(domain_type, domain_size)
    
    def _synthesize_path_plans(
        self,
        domain_type: str,
        domain_size: tuple,
    ) -> List["Plan"]:
        """Synthesize plans for PATH topology."""
        from .world_model import Plan
        
        plans = []
        
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
            cost_estimate="Low complexity, ~10 seconds",
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
            cost_estimate="Low complexity, ~15 seconds",
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
            cost_estimate="Medium complexity, ~20 seconds",
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
            cost_estimate="Medium complexity, ~30 seconds",
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
            cost_estimate="Medium complexity, ~30 seconds",
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
            cost_estimate="Medium complexity, ~25 seconds",
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
            cost_estimate="Medium complexity, ~2-5 minutes",
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
            cost_estimate="High complexity, ~5-10 minutes",
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
            cost_estimate="Low complexity, ~1-2 minutes",
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
            cost_estimate="Low complexity, ~15 seconds",
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
            cost_estimate="Medium complexity, ~1-2 minutes",
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
            cost_estimate="Medium complexity, ~30 seconds",
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
