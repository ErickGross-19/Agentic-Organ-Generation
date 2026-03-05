"""
ExecutionPlan for DesignSpec staged execution.

This module defines the stable stage names and supports partial
execution controls for the DesignSpecRunner.

STAGES
------
The runner executes these stages in order:

1. compile_policies - Compile policy dicts to aog_policies objects
2. compile_domains - Compile domain dicts to runtime Domain objects
3. component_ports:<id> - Resolve port positions for each component
4. component_build:<id> - Generate network/mesh for each component
5. component_mesh:<id> - Convert network to void mesh for each component
6. union_voids - Union all component void meshes
7. mesh_domain - Generate domain mesh
8. embed - Embed unified void into domain
9. port_recarve - Recarve ports if enabled
10. validity - Run validity checks
11. export - Export outputs to files

PARTIAL EXECUTION
-----------------
The runner supports these controls:
- run_until: Stop after this stage
- run_only: Run only these stages
- skip: Skip these stages
- components_subset: Only process these component IDs
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set, Literal
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class Stage(str, Enum):
    """Enumeration of pipeline stages."""
    COMPILE_POLICIES = "compile_policies"
    COMPILE_DOMAINS = "compile_domains"
    COMPONENT_PORTS = "component_ports"
    COMPONENT_BUILD = "component_build"
    COMPONENT_MESH = "component_mesh"
    UNION_VOIDS = "union_voids"
    MESH_DOMAIN = "mesh_domain"
    EMBED = "embed"
    PORT_RECARVE = "port_recarve"
    VALIDITY = "validity"
    EXPORT = "export"


STAGE_ORDER: List[str] = [
    Stage.COMPILE_POLICIES.value,
    Stage.COMPILE_DOMAINS.value,
    Stage.COMPONENT_PORTS.value,
    Stage.COMPONENT_BUILD.value,
    Stage.COMPONENT_MESH.value,
    Stage.UNION_VOIDS.value,
    Stage.MESH_DOMAIN.value,
    Stage.EMBED.value,
    Stage.PORT_RECARVE.value,
    Stage.VALIDITY.value,
    Stage.EXPORT.value,
]


def _get_stage_index(stage: str) -> int:
    """Get the index of a stage in the execution order."""
    base_stage = stage.split(":")[0]
    try:
        return STAGE_ORDER.index(base_stage)
    except ValueError:
        return -1


def _expand_component_stages(
    base_stages: List[str],
    component_ids: List[str],
) -> List[str]:
    """
    Expand component stages with component IDs.
    
    For stages like component_ports, component_build, component_mesh,
    expand them to component_ports:net_1, component_ports:chan_1, etc.
    
    Parameters
    ----------
    base_stages : list of str
        Base stage names
    component_ids : list of str
        Component IDs to expand
        
    Returns
    -------
    list of str
        Expanded stage names
    """
    expanded = []
    
    for stage in base_stages:
        if stage in (
            Stage.COMPONENT_PORTS.value,
            Stage.COMPONENT_BUILD.value,
            Stage.COMPONENT_MESH.value,
        ):
            for comp_id in component_ids:
                expanded.append(f"{stage}:{comp_id}")
        else:
            expanded.append(stage)
    
    return expanded


@dataclass
class ExecutionPlan:
    """
    Execution plan for the DesignSpecRunner.
    
    Determines which stages to run based on partial execution controls.
    
    Attributes
    ----------
    run_until : str, optional
        Stop after this stage (inclusive)
    run_only : list of str, optional
        Run only these stages
    skip : list of str, optional
        Skip these stages
    components_subset : list of str, optional
        Only process these component IDs
    """
    run_until: Optional[str] = None
    run_only: Optional[List[str]] = None
    skip: Optional[List[str]] = None
    components_subset: Optional[List[str]] = None
    
    _component_ids: List[str] = field(default_factory=list, repr=False)
    _stages_to_run: List[str] = field(default_factory=list, repr=False)
    _computed: bool = field(default=False, repr=False)
    
    def set_component_ids(self, component_ids: List[str]) -> None:
        """
        Set the component IDs for stage expansion.
        
        Parameters
        ----------
        component_ids : list of str
            All component IDs from the spec
        """
        if self.components_subset:
            self._component_ids = [
                cid for cid in component_ids
                if cid in self.components_subset
            ]
        else:
            self._component_ids = list(component_ids)
        
        self._computed = False
    
    def compute_stages(self) -> List[str]:
        """
        Compute the list of stages to run.
        
        Returns
        -------
        list of str
            Ordered list of stage names to execute
        """
        if self._computed and self._stages_to_run:
            return self._stages_to_run
        
        all_stages = _expand_component_stages(STAGE_ORDER, self._component_ids)
        
        stages = list(all_stages)
        
        if self.run_until:
            run_until_base = self.run_until.split(":")[0]
            run_until_idx = _get_stage_index(run_until_base)
            if run_until_idx >= 0:
                stages = [
                    s for s in stages
                    if _get_stage_index(s.split(":")[0]) <= run_until_idx
                ]
        
        if self.run_only:
            run_only_set = set(self.run_only)
            stages = [
                s for s in stages
                if s in run_only_set or s.split(":")[0] in run_only_set
            ]
        
        if self.skip:
            skip_set = set(self.skip)
            stages = [
                s for s in stages
                if s not in skip_set and s.split(":")[0] not in skip_set
            ]
        
        self._stages_to_run = stages
        self._computed = True
        
        return stages
    
    def should_run(self, stage: str) -> bool:
        """
        Check if a stage should be run.
        
        Parameters
        ----------
        stage : str
            Stage name (may include component ID suffix)
            
        Returns
        -------
        bool
            True if stage should be run
        """
        stages = self.compute_stages()
        return stage in stages
    
    def should_run_component(self, component_id: str) -> bool:
        """
        Check if a component should be processed.
        
        Parameters
        ----------
        component_id : str
            Component ID
            
        Returns
        -------
        bool
            True if component should be processed
        """
        if self.components_subset:
            return component_id in self.components_subset
        return component_id in self._component_ids
    
    def get_next_stage(self, current_stage: str) -> Optional[str]:
        """
        Get the next stage after the current one.
        
        Parameters
        ----------
        current_stage : str
            Current stage name
            
        Returns
        -------
        str or None
            Next stage name, or None if at end
        """
        stages = self.compute_stages()
        try:
            idx = stages.index(current_stage)
            if idx + 1 < len(stages):
                return stages[idx + 1]
        except ValueError:
            pass
        return None
    
    def is_final_stage(self, stage: str) -> bool:
        """
        Check if a stage is the final stage to run.
        
        Parameters
        ----------
        stage : str
            Stage name
            
        Returns
        -------
        bool
            True if this is the final stage
        """
        stages = self.compute_stages()
        return stages and stages[-1] == stage
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
        return {
            "run_until": self.run_until,
            "run_only": self.run_only,
            "skip": self.skip,
            "components_subset": self.components_subset,
            "computed_stages": self.compute_stages(),
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExecutionPlan":
        """Create from dict."""
        return cls(
            run_until=d.get("run_until"),
            run_only=d.get("run_only"),
            skip=d.get("skip"),
            components_subset=d.get("components_subset"),
        )


def validate_plan(
    plan: ExecutionPlan,
    component_ids: List[str],
) -> List[str]:
    """
    Validate an execution plan.
    
    Parameters
    ----------
    plan : ExecutionPlan
        Plan to validate
    component_ids : list of str
        Available component IDs
        
    Returns
    -------
    list of str
        List of validation error messages (empty if valid)
    """
    errors = []
    
    if plan.run_until:
        base_stage = plan.run_until.split(":")[0]
        if base_stage not in STAGE_ORDER:
            errors.append(f"Unknown run_until stage: {plan.run_until}")
    
    if plan.run_only:
        for stage in plan.run_only:
            base_stage = stage.split(":")[0]
            if base_stage not in STAGE_ORDER:
                errors.append(f"Unknown run_only stage: {stage}")
    
    if plan.skip:
        for stage in plan.skip:
            base_stage = stage.split(":")[0]
            if base_stage not in STAGE_ORDER:
                errors.append(f"Unknown skip stage: {stage}")
    
    if plan.components_subset:
        for comp_id in plan.components_subset:
            if comp_id not in component_ids:
                errors.append(f"Unknown component in components_subset: {comp_id}")
    
    return errors


__all__ = [
    "Stage",
    "STAGE_ORDER",
    "ExecutionPlan",
    "validate_plan",
]
