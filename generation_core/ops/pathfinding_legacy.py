"""
Pathfinding-based growth for efficient vascular network design.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Set
import numpy as np
import heapq
from ..core.types import Point3D, Direction3D
from ..core.network import VascularNetwork, Node
from ..core.result import OperationResult, OperationStatus, Delta
from ..rules.constraints import BranchingConstraints


@dataclass
class CostWeights:
    """Cost weights for A* pathfinding."""
    
    w_length: float = 1.0
    w_clearance: float = 0.5
    w_turn: float = 0.3
    w_resistance: float = 0.1
    clearance_threshold: float = 0.002
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "w_length": self.w_length,
            "w_clearance": self.w_clearance,
            "w_turn": self.w_turn,
            "w_resistance": self.w_resistance,
            "clearance_threshold": self.clearance_threshold,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "CostWeights":
        """Create from dictionary."""
        return cls(
            w_length=d.get("w_length", 1.0),
            w_clearance=d.get("w_clearance", 0.5),
            w_turn=d.get("w_turn", 0.3),
            w_resistance=d.get("w_resistance", 0.1),
            clearance_threshold=d.get("clearance_threshold", 0.002),
        )


def grow_toward_targets(
    network: VascularNetwork,
    from_node_id: int,
    target_positions: List[Tuple[float, float, float]],
    cost_weights: Optional[CostWeights] = None,
    constraints: Optional[BranchingConstraints] = None,
    segment_length: float = 0.005,
    seed: Optional[int] = None,
) -> OperationResult:
    """
    Grow branch directly toward target positions.
    
    This is a simplified pathfinding implementation that grows sequentially
    toward each target. For more sophisticated obstacle-aware routing,
    use the full A* implementation (to be added).
    
    Parameters
    ----------
    network : VascularNetwork
        Network to modify
    from_node_id : int
        Starting node ID
    target_positions : list of tuple
        Target positions (x, y, z) to reach
    cost_weights : CostWeights, optional
        Cost weights for pathfinding
    constraints : BranchingConstraints, optional
        Branching constraints
    segment_length : float
        Length of each segment along path
    seed : int, optional
        Random seed
    
    Returns
    -------
    result : OperationResult
        Result with path information
    """
    if cost_weights is None:
        cost_weights = CostWeights()
    
    if constraints is None:
        constraints = BranchingConstraints()
    
    start_node = network.get_node(from_node_id)
    if start_node is None:
        return OperationResult.failure(
            message=f"Node {from_node_id} not found",
            errors=["Node not found"],
        )
    
    from .growth import grow_branch
    
    current_node_id = from_node_id
    new_node_ids = []
    new_segment_ids = []
    warnings = []
    
    for target in target_positions:
        target_point = Point3D.from_tuple(target)
        current_node = network.get_node(current_node_id)
        
        direction = np.array([
            target_point.x - current_node.position.x,
            target_point.y - current_node.position.y,
            target_point.z - current_node.position.z,
        ])
        
        dist = np.linalg.norm(direction)
        if dist < 1e-10:
            continue
        
        direction = direction / dist
        
        num_steps = max(1, int(dist / segment_length))
        step_length = dist / num_steps
        
        for step in range(num_steps):
            from ..core.types import Direction3D
            result = grow_branch(
                network,
                from_node_id=current_node_id,
                length=step_length,
                direction=Direction3D.from_array(direction),
                constraints=constraints,
                check_collisions=True,
                seed=seed,
            )
            
            if result.is_success():
                new_node_ids.append(result.new_ids["node"])
                new_segment_ids.append(result.new_ids["segment"])
                current_node_id = result.new_ids["node"]
            else:
                warnings.extend(result.errors)
                break
    
    if not new_node_ids:
        return OperationResult.failure(
            message="Failed to grow any branches along path",
            errors=warnings,
        )
    
    delta = Delta(
        created_node_ids=new_node_ids,
        created_segment_ids=new_segment_ids,
    )
    
    status = OperationStatus.SUCCESS if not warnings else OperationStatus.PARTIAL_SUCCESS
    
    return OperationResult(
        status=status,
        message=f"Grew {len(new_node_ids)} nodes along path to targets",
        new_ids={
            "nodes": new_node_ids,
            "segments": new_segment_ids,
        },
        warnings=warnings,
        delta=delta,
        rng_state=network.id_gen.get_state(),
        metadata={
            "nodes_grown": len(new_node_ids),
            "targets_attempted": len(target_positions),
        },
    )
