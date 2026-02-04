"""
Main tree growth algorithm for liver vascular network generation.
"""

import numpy as np
from typing import Tuple, Optional
from .config import LiverVascularConfig
from .geometry import LiverDomain
from .tree import VascularTree, Node, Segment, ActiveTip
from .rules import (
    murray_split_radii,
    compute_branching_probability,
    sample_branch_directions,
    perturb_direction,
)


class VascularNetworkGenerator:
    """Generates arterial and venous trees in liver domain."""
    
    def __init__(self, config: LiverVascularConfig):
        """
        Initialize generator.
        
        Parameters
        ----------
        config : LiverVascularConfig
            Complete configuration for generation
        """
        self.config = config
        self.domain = LiverDomain(config.geometry)
        self.rng = np.random.default_rng(config.random_seed)
        
        arterial_root_pos = self.domain.get_root_position("arterial")
        arterial_root_dir = self.domain.get_initial_direction("arterial")
        self.arterial_tree = VascularTree(
            tree_type="arterial",
            root_position=arterial_root_pos,
            root_radius=config.murray.arterial_root_radius,
            initial_direction=arterial_root_dir,
        )
        
        venous_root_pos = self.domain.get_root_position("venous")
        venous_root_dir = self.domain.get_initial_direction("venous")
        self.venous_tree = VascularTree(
            tree_type="venous",
            root_position=venous_root_pos,
            root_radius=config.murray.venous_root_radius,
            initial_direction=venous_root_dir,
        )
        
        self.next_node_id = 1  # 0 is used by roots
        self.next_segment_id = 0
    
    def grow_tree(
        self,
        tree: VascularTree,
        other_tree: Optional[VascularTree] = None,
    ) -> None:
        """
        Grow a single tree until termination criteria met.
        
        Parameters
        ----------
        tree : VascularTree
            Tree to grow
        other_tree : VascularTree, optional
            Other tree for collision avoidance
        """
        max_segments = self.config.growth.max_segments_per_tree
        max_attempts_per_tip = 10
        
        while tree.active_tips and len(tree.segments) < max_segments:
            if not tree.active_tips:
                break
            
            tip = tree.active_tips.pop(0)
            tip_node = tree.get_node(tip.node_id)
            
            if tip_node.radius < self.config.murray.min_radius:
                continue
            if tip_node.order >= self.config.branching.max_branch_order:
                continue
            if tip.attempts >= max_attempts_per_tip:
                continue
            
            root_radius = (self.config.murray.arterial_root_radius 
                          if tree.tree_type == "arterial" 
                          else self.config.murray.venous_root_radius)
            
            should_branch = compute_branching_probability(
                tip_node.radius,
                root_radius,
                self.config.branching,
                self.rng,
            )
            
            if should_branch and tip_node.radius > 1.5 * self.config.murray.min_radius:
                success = self._attempt_bifurcation(tree, tip, other_tree)
                if not success:
                    tip.attempts += 1
                    tree.active_tips.append(tip)
            else:
                success = self._attempt_continuation(tree, tip, other_tree)
                if not success:
                    tip.attempts += 1
                    tree.active_tips.append(tip)
    
    def _attempt_continuation(
        self,
        tree: VascularTree,
        tip: ActiveTip,
        other_tree: Optional[VascularTree],
    ) -> bool:
        """Attempt to extend a tip in its current direction."""
        tip_node = tree.get_node(tip.node_id)
        
        new_direction = self._compute_biased_direction(
            tip_node.position,
            tip.direction,
            tree.tree_type,
        )
        
        step_size = self.config.branching.step_size_factor * tip_node.radius
        new_position = tip_node.position + step_size * new_direction
        
        if not self._is_valid_position(new_position, tip_node.radius, tree, other_tree, 
                                       exclude_node_id=tip_node.id):
            return False
        
        new_node = Node(
            id=self.next_node_id,
            position=new_position,
            radius=tip_node.radius,  # Same radius for continuation
            parent_id=tip_node.id,
            order=tip_node.order,
        )
        self.next_node_id += 1
        
        segment = Segment(
            id=self.next_segment_id,
            parent_node_id=tip_node.id,
            child_node_id=new_node.id,
            length=step_size,
            direction=new_direction,
            radius_start=tip_node.radius,
            radius_end=new_node.radius,
        )
        self.next_segment_id += 1
        
        tip_node.children_ids.append(new_node.id)
        tree.add_node(new_node)
        tree.add_segment(segment)
        
        tree.active_tips.append(ActiveTip(
            node_id=new_node.id,
            direction=new_direction,
        ))
        
        return True
    
    def _attempt_bifurcation(
        self,
        tree: VascularTree,
        tip: ActiveTip,
        other_tree: Optional[VascularTree],
    ) -> bool:
        """Attempt to create a bifurcation at a tip."""
        tip_node = tree.get_node(tip.node_id)
        
        r1, r2 = murray_split_radii(
            tip_node.radius,
            self.config.murray,
            self.rng,
        )
        
        if r1 < self.config.murray.min_radius or r2 < self.config.murray.min_radius:
            return False
        
        dir1, dir2 = sample_branch_directions(
            tip.direction,
            self.config.branching,
            self.rng,
        )
        
        dir1 = self._compute_biased_direction(tip_node.position, dir1, tree.tree_type)
        dir2 = self._compute_biased_direction(tip_node.position, dir2, tree.tree_type)
        
        step1 = self.config.branching.step_size_factor * r1
        step2 = self.config.branching.step_size_factor * r2
        pos1 = tip_node.position + step1 * dir1
        pos2 = tip_node.position + step2 * dir2
        
        if not self._is_valid_position(pos1, r1, tree, other_tree, exclude_node_id=tip_node.id):
            return False
        if not self._is_valid_position(pos2, r2, tree, other_tree, exclude_node_id=tip_node.id):
            return False
        
        child1 = Node(
            id=self.next_node_id,
            position=pos1,
            radius=r1,
            parent_id=tip_node.id,
            order=tip_node.order + 1,
        )
        self.next_node_id += 1
        
        child2 = Node(
            id=self.next_node_id,
            position=pos2,
            radius=r2,
            parent_id=tip_node.id,
            order=tip_node.order + 1,
        )
        self.next_node_id += 1
        
        seg1 = Segment(
            id=self.next_segment_id,
            parent_node_id=tip_node.id,
            child_node_id=child1.id,
            length=step1,
            direction=dir1,
            radius_start=tip_node.radius,
            radius_end=r1,
        )
        self.next_segment_id += 1
        
        seg2 = Segment(
            id=self.next_segment_id,
            parent_node_id=tip_node.id,
            child_node_id=child2.id,
            length=step2,
            direction=dir2,
            radius_start=tip_node.radius,
            radius_end=r2,
        )
        self.next_segment_id += 1
        
        tip_node.children_ids.extend([child1.id, child2.id])
        tree.add_node(child1)
        tree.add_node(child2)
        tree.add_segment(seg1)
        tree.add_segment(seg2)
        
        tree.active_tips.append(ActiveTip(node_id=child1.id, direction=dir1))
        tree.active_tips.append(ActiveTip(node_id=child2.id, direction=dir2))
        
        return True
    
    def _compute_biased_direction(
        self,
        position: np.ndarray,
        current_direction: np.ndarray,
        tree_type: str,
    ) -> np.ndarray:
        """Compute direction with bias towards center and away from boundary."""
        weights = self.config.growth
        
        current_direction = current_direction / np.linalg.norm(current_direction)
        
        perturbed = perturb_direction(
            current_direction,
            weights.direction_noise_std,
            self.rng,
        )
        
        center_dir = self.domain.center_direction(position)
        
        combined = (
            weights.parent_direction_weight * perturbed +
            weights.center_attraction_weight * center_dir
        )
        
        if np.linalg.norm(combined) < 1e-10:
            return current_direction
        
        return combined / np.linalg.norm(combined)
    
    def _is_valid_position(
        self,
        position: np.ndarray,
        radius: float,
        tree: VascularTree,
        other_tree: Optional[VascularTree],
        exclude_node_id: Optional[int] = None,
    ) -> bool:
        """Check if a position is valid for growth."""
        if not self.domain.inside_liver(position):
            return False
        
        dist_to_boundary = self.domain.distance_to_boundary(position)
        if dist_to_boundary < self.config.growth.min_distance_to_boundary:
            return False
        
        if self.domain.beyond_meeting_shell(position, tree.tree_type):
            return False
        
        if not self._check_collision(position, radius, tree, exclude_node_id):
            return False
        
        if other_tree is not None:
            if not self._check_collision(position, radius, other_tree):
                return False
        
        return True
    
    def _check_collision(
        self,
        position: np.ndarray,
        radius: float,
        tree: VascularTree,
        exclude_node_id: Optional[int] = None,
    ) -> bool:
        """
        Check if position collides with tree segments.
        
        Parameters
        ----------
        position : np.ndarray
            Position to check
        radius : float
            Radius at position
        tree : VascularTree
            Tree to check against
        exclude_node_id : int, optional
            Node ID to exclude from collision checks (typically the parent)
        
        Returns
        -------
        valid : bool
            True if no collision detected
        """
        margin = self.config.growth.collision_margin
        search_radius = 3.0 * radius + margin
        
        nearby_segments = tree.spatial_index.query_nearby_segments(
            position,
            search_radius,
        )
        
        for segment in nearby_segments:
            if exclude_node_id is not None:
                if (segment.parent_node_id == exclude_node_id or 
                    segment.child_node_id == exclude_node_id):
                    continue
            
            parent_node = tree.get_node(segment.parent_node_id)
            child_node = tree.get_node(segment.child_node_id)
            seg_radius = max(parent_node.radius, child_node.radius)
            
            dist = tree.spatial_index._point_to_segment_distance(position, segment)
            
            min_distance = radius + seg_radius + margin
            if dist < min_distance:
                return False
        
        return True


def generate_liver_vasculature(
    config: Optional[LiverVascularConfig] = None,
) -> Tuple[VascularTree, VascularTree]:
    """
    Generate arterial and venous trees in liver domain.
    
    Parameters
    ----------
    config : LiverVascularConfig, optional
        Configuration for generation. If None, uses defaults.
    
    Returns
    -------
    arterial_tree : VascularTree
        Generated arterial tree
    venous_tree : VascularTree
        Generated venous tree
    """
    if config is None:
        config = LiverVascularConfig()
    
    generator = VascularNetworkGenerator(config)
    
    if config.growth.arterial_first:
        print("Growing arterial tree...")
        generator.grow_tree(generator.arterial_tree, other_tree=None)
        print(f"  Arterial: {len(generator.arterial_tree.nodes)} nodes, "
              f"{len(generator.arterial_tree.segments)} segments")
        
        print("Growing venous tree...")
        generator.grow_tree(generator.venous_tree, other_tree=generator.arterial_tree)
        print(f"  Venous: {len(generator.venous_tree.nodes)} nodes, "
              f"{len(generator.venous_tree.segments)} segments")
    else:
        raise NotImplementedError("Interleaved growth not yet implemented")
    
    return generator.arterial_tree, generator.venous_tree
