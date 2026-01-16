"""
K-ary tree generation backend using recursive bifurcation.

This backend generates vascular networks by recursively bifurcating
from inlet nodes to achieve a target terminal count.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import logging

from .base import GenerationBackend, BackendConfig, GenerationState
from ..core.network import VascularNetwork, Node, VesselSegment
from ..core.domain import DomainSpec
from ..core.types import Point3D, TubeGeometry

logger = logging.getLogger(__name__)


@dataclass
class KaryTreeConfig(BackendConfig):
    """
    Configuration for k-ary tree generation.
    
    Parameters
    ----------
    k : int
        Branching factor (default: 2 for binary tree)
    target_terminals : int
        Target number of terminal nodes
    terminal_tolerance : float
        Acceptable deviation from target (fraction, default: 0.1 = 10%)
    branch_length : float
        Initial branch length in meters
    branch_length_decay : float
        Factor to multiply branch length at each level
    taper_factor : float
        Radius taper factor at each bifurcation (Murray's law: 0.794)
    angle_deg : float
        Bifurcation angle in degrees
    angle_variation_deg : float
        Random variation in bifurcation angle
    min_radius : float
        Minimum vessel radius in meters
    """
    k: int = 2
    target_terminals: int = 128
    terminal_tolerance: float = 0.1
    branch_length: float = 0.001  # 1mm
    branch_length_decay: float = 0.8
    taper_factor: float = 0.794  # Murray's law: 2^(-1/3)
    angle_deg: float = 30.0
    angle_variation_deg: float = 5.0
    min_radius: float = 0.0001  # 0.1mm


class KaryTreeBackend(GenerationBackend):
    """
    Generation backend using recursive k-ary bifurcation.
    
    This backend creates tree structures by recursively splitting
    branches until a target terminal count is reached.
    """
    
    @property
    def supports_dual_tree(self) -> bool:
        return True
    
    @property
    def supports_closed_loop(self) -> bool:
        return False
    
    def generate(
        self,
        domain: DomainSpec,
        num_outlets: int,
        inlet_position: np.ndarray,
        inlet_radius: float,
        vessel_type: str = "arterial",
        config: Optional[BackendConfig] = None,
        rng_seed: Optional[int] = None,
    ) -> VascularNetwork:
        """
        Generate a k-ary tree vascular network.
        
        Parameters
        ----------
        domain : DomainSpec
            Geometric domain for the network
        num_outlets : int
            Target number of terminal outlets
        inlet_position : np.ndarray
            Position of the inlet node (x, y, z) in meters
        inlet_radius : float
            Radius of the inlet vessel in meters
        vessel_type : str
            Type of vessels ("arterial" or "venous")
        config : KaryTreeConfig, optional
            Backend configuration
        rng_seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        VascularNetwork
            Generated vascular network
        """
        if config is None:
            config = KaryTreeConfig(target_terminals=num_outlets)
        elif not isinstance(config, KaryTreeConfig):
            # Convert BackendConfig to KaryTreeConfig
            config = KaryTreeConfig(
                target_terminals=num_outlets,
                seed=config.seed,
                min_segment_length=config.min_segment_length,
                max_segment_length=config.max_segment_length,
                min_radius=config.min_radius,
                check_collisions=config.check_collisions,
                collision_clearance=config.collision_clearance,
            )
        
        # D FIX: Use local RNG instead of global np.random.seed
        effective_seed = rng_seed if rng_seed is not None else config.seed
        rng = np.random.default_rng(effective_seed)
        
        # Create network
        network = VascularNetwork(domain=domain)
        
        # D FIX: Add inlet node using network.id_gen and network.add_node
        inlet_pos = Point3D(*inlet_position)
        inlet_id = network.id_gen.next_id()
        inlet_node = Node(
            id=inlet_id,
            position=inlet_pos,
            node_type="inlet",
            vessel_type=vessel_type,
            attributes={"radius": inlet_radius},
        )
        network.add_node(inlet_node)
        
        # Calculate depth needed for target terminals
        # For k-ary tree: terminals = k^depth
        depth = self._calculate_depth(config.k, config.target_terminals)
        
        logger.info(
            f"Generating k={config.k} tree with depth={depth} "
            f"for target_terminals={config.target_terminals}"
        )
        
        # D FIX: Generate tree recursively with local RNG
        self._generate_subtree(
            network=network,
            parent_node=inlet_node,
            parent_direction=np.array([0.0, 0.0, -1.0]),  # Default: grow downward
            current_depth=0,
            max_depth=depth,
            current_radius=inlet_radius,
            current_length=config.branch_length,
            config=config,
            vessel_type=vessel_type,
            domain=domain,
            rng=rng,
        )
        
        # Count terminals and check tolerance
        terminal_count = sum(1 for n in network.nodes.values() if n.node_type == "terminal")
        target = config.target_terminals
        tolerance = config.terminal_tolerance
        
        if abs(terminal_count - target) / max(target, 1) > tolerance:
            logger.warning(
                f"Terminal count {terminal_count} outside tolerance "
                f"({tolerance*100:.0f}%) of target {target}"
            )
        
        return network
    
    def _calculate_depth(self, k: int, target_terminals: int) -> int:
        """Calculate tree depth needed for target terminal count."""
        if target_terminals <= 1:
            return 0
        
        # k^depth >= target_terminals
        # depth >= log_k(target_terminals)
        depth = int(np.ceil(np.log(target_terminals) / np.log(k)))
        return max(1, depth)
    
    def _generate_subtree(
        self,
        network: VascularNetwork,
        parent_node: Node,
        parent_direction: np.ndarray,
        current_depth: int,
        max_depth: int,
        current_radius: float,
        current_length: float,
        config: KaryTreeConfig,
        vessel_type: str,
        domain: DomainSpec,
        rng: np.random.Generator,
    ) -> None:
        """Recursively generate subtree from parent node."""
        if current_depth >= max_depth:
            # Mark as terminal
            parent_node.node_type = "terminal"
            return
        
        if current_radius < config.min_radius:
            # Too small, mark as terminal
            parent_node.node_type = "terminal"
            return
        
        # Generate k children
        k = config.k
        child_radius = current_radius * config.taper_factor
        child_length = current_length * config.branch_length_decay
        
        # Calculate bifurcation angles
        base_angle = config.angle_deg
        angle_var = config.angle_variation_deg
        
        for i in range(k):
            # D FIX: Calculate child direction with rotation using local RNG
            angle_offset = (i - (k - 1) / 2) * base_angle * 2 / max(k - 1, 1)
            angle_offset += rng.uniform(-angle_var, angle_var)
            
            # D FIX: Rotate parent direction using local RNG for azimuth
            child_direction = self._rotate_direction(
                parent_direction, 
                np.radians(angle_offset),
                np.radians(rng.uniform(0, 360))  # Random azimuth
            )
            
            # Calculate child position
            child_pos = np.array([
                parent_node.position.x,
                parent_node.position.y,
                parent_node.position.z,
            ]) + child_direction * child_length
            
            # Check if inside domain
            if hasattr(domain, 'contains'):
                if not domain.contains(Point3D(*child_pos)):
                    # Project back inside
                    child_pos = self._project_inside(child_pos, domain)
            
            # D FIX: Create child node using network.id_gen and network.add_node
            child_id = network.id_gen.next_id()
            child_node = Node(
                id=child_id,
                position=Point3D(*child_pos),
                node_type="junction",
                vessel_type=vessel_type,
                attributes={"radius": child_radius},
            )
            network.add_node(child_node)
            
            # D FIX: Create segment using network.id_gen and network.add_segment
            segment_id = network.id_gen.next_id()
            geometry = TubeGeometry(
                start=parent_node.position,
                end=child_node.position,
                radius_start=current_radius,
                radius_end=child_radius,
            )
            segment = VesselSegment(
                id=segment_id,
                start_node_id=parent_node.id,
                end_node_id=child_id,
                geometry=geometry,
                vessel_type=vessel_type,
            )
            network.add_segment(segment)
            
            # D FIX: Recurse with local RNG
            self._generate_subtree(
                network=network,
                parent_node=child_node,
                parent_direction=child_direction,
                current_depth=current_depth + 1,
                max_depth=max_depth,
                current_radius=child_radius,
                current_length=child_length,
                config=config,
                vessel_type=vessel_type,
                domain=domain,
                rng=rng,
            )
    
    def _rotate_direction(
        self,
        direction: np.ndarray,
        polar_angle: float,
        azimuth_angle: float,
    ) -> np.ndarray:
        """Rotate a direction vector by polar and azimuth angles."""
        # Normalize input
        direction = direction / np.linalg.norm(direction)
        
        # Find perpendicular vectors
        if abs(direction[2]) < 0.9:
            perp1 = np.cross(direction, np.array([0, 0, 1]))
        else:
            perp1 = np.cross(direction, np.array([1, 0, 0]))
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(direction, perp1)
        
        # Rotate by polar angle around perp1
        cos_p = np.cos(polar_angle)
        sin_p = np.sin(polar_angle)
        rotated = direction * cos_p + np.cross(perp1, direction) * sin_p
        
        # Rotate by azimuth around original direction
        cos_a = np.cos(azimuth_angle)
        sin_a = np.sin(azimuth_angle)
        final = rotated * cos_a + np.cross(direction, rotated) * sin_a
        
        return final / np.linalg.norm(final)
    
    def _project_inside(self, position: np.ndarray, domain: DomainSpec) -> np.ndarray:
        """Project a position back inside the domain."""
        if hasattr(domain, 'project_inside'):
            result = domain.project_inside(Point3D(*position))
            return np.array([result.x, result.y, result.z])
        
        # Fallback: move toward center
        if hasattr(domain, 'center'):
            center = np.array([domain.center.x, domain.center.y, domain.center.z])
        else:
            center = np.array([0, 0, 0])
        
        direction = center - position
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
            return position + direction * 0.001  # Move 1mm toward center
        
        return position


__all__ = [
    "KaryTreeBackend",
    "KaryTreeConfig",
]
