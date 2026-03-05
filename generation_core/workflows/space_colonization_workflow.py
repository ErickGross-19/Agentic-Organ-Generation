"""
Condensed Space Colonization Workflow.

This module consolidates the 4 previously fragmented space colonization
entry points into a single cohesive abstraction:

Previously scattered across generation/ops/space_colonization.py:
    1. space_colonization_step()           — original multi-step (lines 121-600)
    2. space_colonization_step_v2()        — policy-driven multi-step (lines 1278-1788)
    3. space_colonization_one_step()       — single-step (lines 2131-2556)
    4. run_space_colonization_multi_step() — multi-step runner (lines 2559-2650)

Now unified into:
    SpaceColonizationWorkflow — single class with clear entry points:
        .run()           — run the full workflow (replaces #1, #2, #4)
        .step()          — execute exactly one step (replaces #3)
        .create_state()  — initialize state for step-by-step execution

The underlying implementations in generation_core/ops/space_colonization.py
are preserved and delegated to. This module provides the cohesive API layer.

Usage
-----
    from generation_core.workflows import SpaceColonizationWorkflow, WorkflowMode

    # Full run (policy-driven, recommended):
    wf = SpaceColonizationWorkflow(mode=WorkflowMode.POLICY_DRIVEN)
    result = wf.run(network, tissue_points, max_steps=500, sc_policy=policy)

    # Full run (classic mode):
    wf = SpaceColonizationWorkflow(mode=WorkflowMode.CLASSIC)
    result = wf.run(network, tissue_points, params=params)

    # Step-by-step (for multi-inlet interleaving):
    wf = SpaceColonizationWorkflow(mode=WorkflowMode.SINGLE_STEP)
    state = wf.create_state(network, tissue_points, params=params)
    for _ in range(max_steps):
        step_result = wf.step(state)
        if step_result.exhausted or step_result.stalled:
            break
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..core.network import VascularNetwork
    from ..core.result import OperationResult
    from ..ops.space_colonization import (
        SpaceColonizationParams,
        SpaceColonizationState,
        SingleStepResult,
    )
    from ..rules.constraints import BranchingConstraints
    from aog_policies.space_colonization import SpaceColonizationPolicy

logger = logging.getLogger(__name__)


class WorkflowMode(str, Enum):
    """Space colonization workflow modes.

    CLASSIC : Original algorithm (space_colonization_step).
        Simple attractor-based growth without policy-driven features.

    POLICY_DRIVEN : Policy-driven v2 algorithm (space_colonization_step_v2).
        Trunk-first growth, apical dominance, angular clustering.
        Recommended for most use cases.

    SINGLE_STEP : Step-by-step execution (space_colonization_one_step).
        For multi-inlet interleaving and fine-grained control.
        Use with create_state() + step() instead of run().
    """

    CLASSIC = "classic"
    POLICY_DRIVEN = "policy_driven"
    SINGLE_STEP = "single_step"


class SpaceColonizationWorkflow:
    """Cohesive space colonization workflow abstraction.

    Consolidates the 4 previously fragmented entry points into a single
    class with clear methods:

    - ``run()`` : Execute the full multi-step workflow.
    - ``step()`` : Execute exactly one step (for interleaved execution).
    - ``create_state()`` : Initialize persistent state for step-by-step use.

    Parameters
    ----------
    mode : WorkflowMode
        Which internal variant to use (default: POLICY_DRIVEN).
    """

    def __init__(self, mode: WorkflowMode = WorkflowMode.POLICY_DRIVEN) -> None:
        self._mode = mode

    @property
    def mode(self) -> WorkflowMode:
        return self._mode

    # ------------------------------------------------------------------
    # Full-run entry point (replaces #1, #2, #4)
    # ------------------------------------------------------------------

    def run(
        self,
        network: "VascularNetwork",
        tissue_points: np.ndarray,
        params: Optional["SpaceColonizationParams"] = None,
        constraints: Optional["BranchingConstraints"] = None,
        seed: Optional[int] = None,
        seed_nodes: Optional[List[str]] = None,
        max_steps: Optional[int] = None,
        sc_policy: Optional["SpaceColonizationPolicy"] = None,
        disable_progress: bool = False,
        progress_desc: str = "Space colonization",
        # Single-step specific kwargs
        seed_node_ids: Optional[List[int]] = None,
        inlet_id: Optional[int] = None,
        vessel_type: str = "arterial",
    ) -> "OperationResult":
        """Run the full space colonization workflow.

        Delegates to the appropriate internal variant based on ``self.mode``.

        Parameters
        ----------
        network : VascularNetwork
            Network to grow.
        tissue_points : np.ndarray
            Array of tissue points (N, 3) that need perfusion.
        params : SpaceColonizationParams, optional
            Algorithm parameters.
        constraints : BranchingConstraints, optional
            Branching constraints.
        seed : int, optional
            Random seed.
        seed_nodes : list of str, optional
            Node IDs to use as seed nodes (for CLASSIC/POLICY_DRIVEN modes).
        max_steps : int, optional
            Maximum steps (required for SINGLE_STEP mode, optional for others).
        sc_policy : SpaceColonizationPolicy, optional
            Policy (used by POLICY_DRIVEN mode).
        disable_progress : bool
            Whether to disable progress bar.
        progress_desc : str
            Progress bar description.
        seed_node_ids : list of int, optional
            Integer node IDs for SINGLE_STEP mode.
        inlet_id : int, optional
            Inlet identifier for SINGLE_STEP mode.
        vessel_type : str
            Vessel type for SINGLE_STEP mode.

        Returns
        -------
        OperationResult
            Result with metadata about growth progress.
        """
        if self._mode == WorkflowMode.CLASSIC:
            return self._run_classic(
                network=network,
                tissue_points=tissue_points,
                params=params,
                constraints=constraints,
                seed=seed,
                seed_nodes=seed_nodes,
            )

        elif self._mode == WorkflowMode.POLICY_DRIVEN:
            return self._run_policy_driven(
                network=network,
                tissue_points=tissue_points,
                params=params,
                constraints=constraints,
                seed=seed,
                seed_nodes=seed_nodes,
                sc_policy=sc_policy,
                disable_progress=disable_progress,
            )

        elif self._mode == WorkflowMode.SINGLE_STEP:
            if max_steps is None:
                max_steps = 500
                logger.warning(
                    "SINGLE_STEP mode requires max_steps; defaulting to %d",
                    max_steps,
                )
            state = self.create_state(
                network=network,
                tissue_points=tissue_points,
                params=params,
                constraints=constraints,
                seed=seed,
                seed_node_ids=seed_node_ids,
                inlet_id=inlet_id,
                vessel_type=vessel_type,
            )
            return self._run_multi_step(
                state=state,
                max_steps=max_steps,
                progress=not disable_progress,
                progress_desc=progress_desc,
            )

        else:
            raise ValueError(f"Unknown workflow mode: {self._mode}")

    # ------------------------------------------------------------------
    # Single-step entry point (replaces #3)
    # ------------------------------------------------------------------

    def step(self, state: "SpaceColonizationState") -> "SingleStepResult":
        """Execute exactly ONE iteration of space colonization.

        Designed for multi-inlet interleaving. Does not create progress
        bars or print output.

        Parameters
        ----------
        state : SpaceColonizationState
            Persistent state object (modified in-place).

        Returns
        -------
        SingleStepResult
            Result of this single step.
        """
        from ..ops.space_colonization import space_colonization_one_step

        return space_colonization_one_step(state)

    # ------------------------------------------------------------------
    # State creation (for step-by-step execution)
    # ------------------------------------------------------------------

    def create_state(
        self,
        network: "VascularNetwork",
        tissue_points: np.ndarray,
        params: Optional["SpaceColonizationParams"] = None,
        constraints: Optional["BranchingConstraints"] = None,
        seed: Optional[int] = None,
        seed_node_ids: Optional[List[int]] = None,
        inlet_id: Optional[int] = None,
        vessel_type: str = "arterial",
        kdtree_rebuild_tip_every: int = 1,
        kdtree_rebuild_all_nodes_every: int = 10,
        kdtree_rebuild_all_nodes_min_new_nodes: int = 5,
        stall_steps_threshold: int = 10,
    ) -> "SpaceColonizationState":
        """Initialize persistent state for step-by-step execution.

        Parameters
        ----------
        network : VascularNetwork
            Network to grow.
        tissue_points : np.ndarray
            Array of tissue points (N, 3).
        params : SpaceColonizationParams, optional
            Algorithm parameters.
        constraints : BranchingConstraints, optional
            Branching constraints.
        seed : int, optional
            Random seed.
        seed_node_ids : list of int, optional
            Node IDs to use as seed nodes.
        inlet_id : int, optional
            Inlet identifier for multi-inlet tracking.
        vessel_type : str
            Type of vessels.
        kdtree_rebuild_tip_every : int
            Rebuild tip KD-tree every N steps.
        kdtree_rebuild_all_nodes_every : int
            Rebuild all-nodes KD-tree every N steps.
        kdtree_rebuild_all_nodes_min_new_nodes : int
            Rebuild all-nodes KD-tree if this many nodes added.
        stall_steps_threshold : int
            Mark as stalled after this many steps with no growth.

        Returns
        -------
        SpaceColonizationState
            Initial state for step-by-step execution.
        """
        from ..ops.space_colonization import create_space_colonization_state

        return create_space_colonization_state(
            network=network,
            tissue_points=tissue_points,
            params=params,
            constraints=constraints,
            seed=seed,
            seed_node_ids=seed_node_ids,
            inlet_id=inlet_id,
            vessel_type=vessel_type,
            kdtree_rebuild_tip_every=kdtree_rebuild_tip_every,
            kdtree_rebuild_all_nodes_every=kdtree_rebuild_all_nodes_every,
            kdtree_rebuild_all_nodes_min_new_nodes=kdtree_rebuild_all_nodes_min_new_nodes,
            stall_steps_threshold=stall_steps_threshold,
        )

    # ------------------------------------------------------------------
    # Internal dispatch methods
    # ------------------------------------------------------------------

    def _run_classic(
        self,
        network: "VascularNetwork",
        tissue_points: np.ndarray,
        params: Optional["SpaceColonizationParams"] = None,
        constraints: Optional["BranchingConstraints"] = None,
        seed: Optional[int] = None,
        seed_nodes: Optional[List[str]] = None,
    ) -> "OperationResult":
        """Dispatch to the original space_colonization_step()."""
        from ..ops.space_colonization import space_colonization_step

        return space_colonization_step(
            network=network,
            tissue_points=tissue_points,
            params=params,
            constraints=constraints,
            seed=seed,
            seed_nodes=seed_nodes,
        )

    def _run_policy_driven(
        self,
        network: "VascularNetwork",
        tissue_points: np.ndarray,
        params: Optional["SpaceColonizationParams"] = None,
        constraints: Optional["BranchingConstraints"] = None,
        seed: Optional[int] = None,
        seed_nodes: Optional[List[str]] = None,
        sc_policy: Optional["SpaceColonizationPolicy"] = None,
        disable_progress: bool = False,
    ) -> "OperationResult":
        """Dispatch to the policy-driven space_colonization_step_v2()."""
        from ..ops.space_colonization import space_colonization_step_v2

        return space_colonization_step_v2(
            network=network,
            tissue_points=tissue_points,
            params=params,
            constraints=constraints,
            seed=seed,
            seed_nodes=seed_nodes,
            sc_policy=sc_policy,
            disable_progress=disable_progress,
        )

    def _run_multi_step(
        self,
        state: "SpaceColonizationState",
        max_steps: int,
        progress: bool = False,
        progress_desc: str = "Space colonization",
    ) -> "OperationResult":
        """Dispatch to run_space_colonization_multi_step()."""
        from ..ops.space_colonization import run_space_colonization_multi_step

        return run_space_colonization_multi_step(
            state=state,
            max_steps=max_steps,
            progress=progress,
            progress_desc=progress_desc,
        )
