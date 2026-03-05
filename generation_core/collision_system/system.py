"""
Unified Collision System — single shared layer for ALL growth backends.

Instead of each backend reimplementing collision detection, spatial hashing,
and boundary checking, they all instantiate a CollisionSystem and use its
methods. The growth algorithm is what differs between backends; the collision
infrastructure is shared.

Usage
-----
    from generation_core.collision_system import create_collision_system

    # In any backend's generate() method:
    collision_sys = create_collision_system(cell_size=0.001, domain=domain)

    # Register segments as they are created:
    collision_sys.insert_segment(seg_id, start, end, radius)

    # Check before creating a new segment:
    if collision_sys.check_segment_collision(start, end, radius, buffer=clearance):
        # collision detected — skip or deflect

    # CCO-style insertion check:
    if collision_sys.check_insertion_collision(
        network, split_seg_id, bifurcation_pt, outlet_pt, r1, r2, clearance
    ):
        # insertion would collide

    # Post-pass detection:
    result = collision_sys.detect_all_collisions(network, domain, policy)

UNIT CONVENTIONS: All geometric values are in METERS internally.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np

from ..spatial.grid_index import (
    DynamicSpatialIndex,
    SpatialIndex,
    segment_segment_distance_exact,
)
from ..core.types import Point3D

if TYPE_CHECKING:
    from ..core.network import VascularNetwork
    from ..core.domain import DomainSpec
    from aog_policies.collision import UnifiedCollisionPolicy

logger = logging.getLogger(__name__)


@dataclass
class CollisionSystemConfig:
    """Configuration for the unified collision system.

    Parameters
    ----------
    cell_size : float
        Grid cell size for the spatial index (meters). Should be roughly
        2-3x the average segment radius.
    enabled : bool
        Master switch — if False, all checks return "no collision".
    boundary_check_enabled : bool
        Whether to check domain boundary clearance.
    ancestry_exclusion_depth : int or None
        Max hops for lazy ancestry exclusion. None means auto-compute.
    """

    cell_size: float = 0.001
    enabled: bool = True
    boundary_check_enabled: bool = True
    ancestry_exclusion_depth: Optional[int] = None


class CollisionSystem:
    """Unified collision checking service shared by ALL growth backends.

    This class wraps a DynamicSpatialIndex and provides a high-level API
    for the common collision operations that every backend needs:

    1. **Online segment collision** — check a proposed segment before creation.
    2. **Online point clearance** — check a proposed point (space colonization).
    3. **Online polyline collision** — check a curved branch (scaffold_topdown).
    4. **CCO insertion collision** — check a trifurcation insertion (cco_hybrid).
    5. **Boundary clearance** — check domain boundary distance.
    6. **Segment registration** — index a newly created segment.
    7. **Post-pass detection** — full-network collision scan after generation.
    8. **Post-pass resolution** — attempt to fix detected collisions.
    """

    def __init__(
        self,
        config: Optional[CollisionSystemConfig] = None,
        domain: Optional["DomainSpec"] = None,
    ) -> None:
        if config is None:
            config = CollisionSystemConfig()
        self._config = config
        self._domain = domain
        self._spatial_index = DynamicSpatialIndex(cell_size=config.cell_size)

        # Lookup tables for ancestry-based exclusion
        self._parent_of: Dict[int, int] = {}
        self._seg_node_map: Dict[int, Tuple[int, int]] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def spatial_index(self) -> DynamicSpatialIndex:
        """Direct access to the underlying spatial index (for advanced use)."""
        return self._spatial_index

    @property
    def config(self) -> CollisionSystemConfig:
        return self._config

    @property
    def domain(self) -> Optional["DomainSpec"]:
        return self._domain

    @domain.setter
    def domain(self, value: "DomainSpec") -> None:
        self._domain = value

    @property
    def segment_count(self) -> int:
        return self._spatial_index.segment_count

    # ------------------------------------------------------------------
    # Segment registration
    # ------------------------------------------------------------------

    def insert_segment(
        self,
        segment_id: int,
        start: np.ndarray,
        end: np.ndarray,
        radius: float,
        centerline: Optional[List[np.ndarray]] = None,
        start_node_id: Optional[int] = None,
        end_node_id: Optional[int] = None,
        parent_node_id: Optional[int] = None,
    ) -> None:
        """Register a newly created segment in the collision system.

        Parameters
        ----------
        segment_id : int
            Unique segment identifier.
        start, end : np.ndarray
            Segment endpoints (shape (3,)).
        radius : float
            Segment radius.
        centerline : list of np.ndarray, optional
            Intermediate waypoints for polyline segments.
        start_node_id, end_node_id : int, optional
            Node IDs for ancestry-based exclusion tracking.
        parent_node_id : int, optional
            Parent node ID for building the ancestry map.
        """
        self._spatial_index.insert_segment(
            segment_id, start, end, radius, centerline
        )

        # Update ancestry maps if node IDs are provided
        if start_node_id is not None and end_node_id is not None:
            self._seg_node_map[segment_id] = (start_node_id, end_node_id)
        if end_node_id is not None and parent_node_id is not None:
            self._parent_of[end_node_id] = parent_node_id

    def clear(self) -> None:
        """Clear all indexed segments and ancestry maps."""
        self._spatial_index.clear()
        self._parent_of.clear()
        self._seg_node_map.clear()

    # ------------------------------------------------------------------
    # Online collision checks (used during generation)
    # ------------------------------------------------------------------

    def check_segment_collision(
        self,
        start: np.ndarray,
        end: np.ndarray,
        radius: float,
        buffer: float = 0.0,
        exclude_adjacent_to: Optional[np.ndarray] = None,
        exclude_segment_ids: Optional[Set[int]] = None,
        from_node_id: Optional[int] = None,
        use_lazy_ancestry: bool = False,
        excl_depth: Optional[int] = None,
    ) -> bool:
        """Check if a proposed segment collides with any indexed segments.

        This is the primary online collision check used by all backends.

        Parameters
        ----------
        start, end : np.ndarray
            Proposed segment endpoints.
        radius : float
            Proposed segment radius.
        buffer : float
            Additional clearance beyond radii sum.
        exclude_adjacent_to : np.ndarray, optional
            Exclude segments sharing this endpoint (legacy).
        exclude_segment_ids : set of int, optional
            Explicit segment IDs to exclude.
        from_node_id : int, optional
            Node the new segment grows from (for lazy ancestry exclusion).
        use_lazy_ancestry : bool
            If True, use lazy per-candidate ancestry check.
        excl_depth : int, optional
            Max ancestry hops. Auto-computed if None.

        Returns
        -------
        bool
            True if collision detected.
        """
        if not self._config.enabled:
            return False

        if use_lazy_ancestry and from_node_id is not None:
            depth = excl_depth or self._config.ancestry_exclusion_depth or 10
            return self._spatial_index.check_capsule_collision_lazy(
                start=np.asarray(start, dtype=np.float64),
                end=np.asarray(end, dtype=np.float64),
                radius=radius,
                buffer=buffer,
                exclude_adjacent_to=exclude_adjacent_to,
                from_node_id=from_node_id,
                parent_of=self._parent_of if self._parent_of else None,
                seg_node_map=self._seg_node_map if self._seg_node_map else None,
                excl_depth=depth,
            )

        return self._spatial_index.check_capsule_collision(
            start=np.asarray(start, dtype=np.float64),
            end=np.asarray(end, dtype=np.float64),
            radius=radius,
            buffer=buffer,
            exclude_adjacent_to=exclude_adjacent_to,
            exclude_segment_ids=exclude_segment_ids,
        )

    def check_point_clearance(
        self,
        position: Point3D,
        network: "VascularNetwork",
        from_node_id: int,
        min_clearance: float,
        excl_depth: Optional[int] = None,
        use_lazy_ancestry: bool = True,
    ) -> bool:
        """Check if a proposed point maintains minimum clearance from segments.

        This is the space-colonization-style clearance check. Returns True
        if clearance is acceptable, False if too close to existing segments.

        Parameters
        ----------
        position : Point3D
            Proposed point position.
        network : VascularNetwork
            Current network (used for spatial index queries).
        from_node_id : int
            Node the new branch grows from.
        min_clearance : float
            Minimum required clearance from existing segments.
        excl_depth : int, optional
            Max ancestry hops for exclusion.
        use_lazy_ancestry : bool
            Whether to use lazy ancestry-based exclusion.

        Returns
        -------
        bool
            True if clearance is acceptable, False if collision.
        """
        if not self._config.enabled:
            return True

        search_radius = min_clearance * 3.0
        spatial_index = network.get_spatial_index()
        nearby_segments = spatial_index.query_nearby_segments(position, search_radius)

        if not nearby_segments:
            return True

        depth = excl_depth or self._config.ancestry_exclusion_depth or 10

        if use_lazy_ancestry and self._parent_of:
            # Build tip ancestors
            tip_ancestors: Set[int] = set()
            cur = from_node_id
            while cur is not None and cur not in tip_ancestors:
                tip_ancestors.add(cur)
                cur = self._parent_of.get(cur)

            for seg in nearby_segments:
                if seg.id in self._seg_node_map:
                    cand_start_nid, cand_end_nid = self._seg_node_map[seg.id]
                    skip = False
                    for cand_nid in (cand_start_nid, cand_end_nid):
                        cur = cand_nid
                        for _ in range(depth + 1):
                            if cur in tip_ancestors:
                                skip = True
                                break
                            nxt = self._parent_of.get(cur)
                            if nxt is None:
                                break
                            cur = nxt
                        if skip:
                            break
                    if skip:
                        continue

                p1 = network.nodes[seg.start_node_id].position
                p2 = network.nodes[seg.end_node_id].position

                v = np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z])
                w = np.array(
                    [position.x - p1.x, position.y - p1.y, position.z - p1.z]
                )

                v_len_sq = float(np.dot(v, v))
                if v_len_sq < 1e-10:
                    dist = float(np.linalg.norm(w))
                else:
                    t = float(np.clip(np.dot(w, v) / v_len_sq, 0.0, 1.0))
                    projection = p1.to_array() + t * v
                    dist = float(np.linalg.norm(position.to_array() - projection))

                seg_radius = seg.mean_radius
                required_clearance = min_clearance + seg_radius
                if dist < required_clearance:
                    return False

            return True

        # Fallback: no ancestry exclusion — check all nearby
        for seg in nearby_segments:
            p1 = network.nodes[seg.start_node_id].position
            p2 = network.nodes[seg.end_node_id].position

            v = np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z])
            w = np.array(
                [position.x - p1.x, position.y - p1.y, position.z - p1.z]
            )

            v_len_sq = float(np.dot(v, v))
            if v_len_sq < 1e-10:
                dist = float(np.linalg.norm(w))
            else:
                t = float(np.clip(np.dot(w, v) / v_len_sq, 0.0, 1.0))
                projection = p1.to_array() + t * v
                dist = float(np.linalg.norm(position.to_array() - projection))

            seg_radius = seg.mean_radius
            required_clearance = min_clearance + seg_radius
            if dist < required_clearance:
                return False

        return True

    def check_polyline_collision(
        self,
        points: List[np.ndarray],
        radius: float,
        buffer: float = 0.0,
        exclude_adjacent_to: Optional[np.ndarray] = None,
        exclude_segment_ids: Optional[Set[int]] = None,
    ) -> bool:
        """Check if a proposed polyline (curved branch) collides.

        Used by scaffold_topdown for curved path validation.

        Parameters
        ----------
        points : list of np.ndarray
            Points defining the polyline (at least 2).
        radius : float
            Polyline capsule radius.
        buffer : float
            Additional clearance.
        exclude_adjacent_to : np.ndarray, optional
            Exclude segments sharing this endpoint.
        exclude_segment_ids : set of int, optional
            Explicit IDs to exclude.

        Returns
        -------
        bool
            True if collision detected.
        """
        if not self._config.enabled:
            return False

        return self._spatial_index.check_polyline_collision(
            points=points,
            radius=radius,
            buffer=buffer,
            exclude_adjacent_to=exclude_adjacent_to,
            exclude_segment_ids=exclude_segment_ids,
        )

    def check_insertion_collision(
        self,
        network: "VascularNetwork",
        split_seg_id: int,
        bifurcation_point: Point3D,
        outlet_point: Point3D,
        r_child1: float,
        r_child2: float,
        clearance: float,
    ) -> bool:
        """Check if a CCO-style trifurcation insertion would collide.

        Tests all three new segments (A→X, X→B, X→T) for collisions
        with existing network segments using the hierarchical broad-phase
        then narrow-phase approach.

        Parameters
        ----------
        network : VascularNetwork
            Current network.
        split_seg_id : int
            ID of the segment being split.
        bifurcation_point : Point3D
            Proposed bifurcation point X.
        outlet_point : Point3D
            Target outlet point T.
        r_child1 : float
            Radius of child segment X→B.
        r_child2 : float
            Radius of new outlet segment X→T.
        clearance : float
            Minimum required clearance between vessels.

        Returns
        -------
        bool
            True if collision detected, False if safe.
        """
        if not self._config.enabled:
            return False

        X = bifurcation_point.to_array()
        T = outlet_point.to_array()

        split_seg = network.segments[split_seg_id]
        A = network.nodes[split_seg.start_node_id].position.to_array()
        B = network.nodes[split_seg.end_node_id].position.to_array()

        # Estimate radius for A→X segment
        ab_len = max(float(np.linalg.norm(B - A)), 1e-10)
        t = float(np.linalg.norm(X - A)) / ab_len
        r_at_X = (
            split_seg.geometry.radius_start
            + t * (split_seg.geometry.radius_end - split_seg.geometry.radius_start)
        )
        r_AX = (split_seg.geometry.radius_start + r_at_X) / 2

        # Use the network's spatial index for broad-phase
        spatial_index = network.get_spatial_index()
        search_radius = (
            max(
                float(np.linalg.norm(X - A)),
                float(np.linalg.norm(X - B)),
                float(np.linalg.norm(X - T)),
            )
            + clearance
            + max(r_child1, r_child2, r_AX) * 2
        )

        nearby_segments = spatial_index.query_nearby_segments(
            bifurcation_point, search_radius
        )

        for seg in nearby_segments:
            # Skip the segment being split and connected segments
            if seg.id == split_seg_id:
                continue
            if (
                seg.start_node_id == split_seg.start_node_id
                or seg.start_node_id == split_seg.end_node_id
                or seg.end_node_id == split_seg.start_node_id
                or seg.end_node_id == split_seg.end_node_id
            ):
                continue

            seg_radius = seg.geometry.mean_radius()

            if seg.geometry.centerline_points:
                seg_start_node = network.nodes[seg.start_node_id]
                seg_end_node = network.nodes[seg.end_node_id]
                seg_polyline = [seg_start_node.position.to_array()]
                seg_polyline.extend(
                    [p.to_array() for p in seg.geometry.centerline_points]
                )
                seg_polyline.append(seg_end_node.position.to_array())

                if _segment_to_polyline_distance(A, X, seg_polyline) < r_AX + seg_radius + clearance:
                    return True
                if _segment_to_polyline_distance(X, B, seg_polyline) < r_child1 + seg_radius + clearance:
                    return True
                if _segment_to_polyline_distance(X, T, seg_polyline) < r_child2 + seg_radius + clearance:
                    return True
            else:
                seg_start = network.nodes[seg.start_node_id].position.to_array()
                seg_end = network.nodes[seg.end_node_id].position.to_array()

                if segment_segment_distance_exact(A, X, seg_start, seg_end) < r_AX + seg_radius + clearance:
                    return True
                if segment_segment_distance_exact(X, B, seg_start, seg_end) < r_child1 + seg_radius + clearance:
                    return True
                if segment_segment_distance_exact(X, T, seg_start, seg_end) < r_child2 + seg_radius + clearance:
                    return True

        return False

    def check_boundary_clearance(
        self,
        point: np.ndarray,
        radius: float,
        wall_margin: float = 0.0,
        extra_margin: float = 0.0,
    ) -> bool:
        """Check if a point maintains sufficient clearance from the domain boundary.

        Parameters
        ----------
        point : np.ndarray
            Point to check (shape (3,)).
        radius : float
            Tube radius at this point.
        wall_margin : float
            Wall margin from config.
        extra_margin : float
            Additional margin for boundary checking.

        Returns
        -------
        bool
            True if boundary is violated, False if safe.
        """
        if not self._config.boundary_check_enabled or self._domain is None:
            return False

        p = Point3D.from_array(point)
        dist = self._domain.distance_to_boundary(p)
        required = wall_margin + radius + extra_margin
        return dist < required

    # ------------------------------------------------------------------
    # Post-pass full-network checks
    # ------------------------------------------------------------------

    def detect_all_collisions(
        self,
        network: "VascularNetwork",
        domain: Optional["DomainSpec"] = None,
        policy: Optional["UnifiedCollisionPolicy"] = None,
    ):
        """Run full post-pass collision detection on the entire network.

        Delegates to the existing unified collision detection module.

        Returns
        -------
        CollisionResult
        """
        from ..ops.collision.unified import detect_collisions

        return detect_collisions(network, domain=domain or self._domain, policy=policy)

    def resolve_collisions(
        self,
        network: "VascularNetwork",
        collision_result,
        domain: Optional["DomainSpec"] = None,
        policy: Optional["UnifiedCollisionPolicy"] = None,
    ):
        """Attempt to resolve detected collisions.

        Delegates to the existing unified collision resolution module.

        Returns
        -------
        ResolutionResult
        """
        from ..ops.collision.unified import resolve_collisions

        return resolve_collisions(
            network, collision_result, domain=domain or self._domain, policy=policy
        )

    # ------------------------------------------------------------------
    # Ancestry helpers
    # ------------------------------------------------------------------

    def set_parent(self, child_node_id: int, parent_node_id: int) -> None:
        """Register a parent-child node relationship for ancestry exclusion."""
        self._parent_of[child_node_id] = parent_node_id

    def register_segment_nodes(
        self, segment_id: int, start_node_id: int, end_node_id: int
    ) -> None:
        """Register a segment's node IDs for ancestry exclusion."""
        self._seg_node_map[segment_id] = (start_node_id, end_node_id)

    @property
    def parent_of(self) -> Dict[int, int]:
        """Read-only access to the parent map."""
        return self._parent_of

    @property
    def seg_node_map(self) -> Dict[int, Tuple[int, int]]:
        """Read-only access to the segment-node map."""
        return self._seg_node_map


def create_collision_system(
    cell_size: float = 0.001,
    enabled: bool = True,
    domain: Optional["DomainSpec"] = None,
    boundary_check_enabled: bool = True,
    ancestry_exclusion_depth: Optional[int] = None,
) -> CollisionSystem:
    """Factory function to create a CollisionSystem.

    Parameters
    ----------
    cell_size : float
        Grid cell size for the spatial index (meters).
    enabled : bool
        Master switch for collision checking.
    domain : DomainSpec, optional
        Domain for boundary checks.
    boundary_check_enabled : bool
        Whether to check boundary clearance.
    ancestry_exclusion_depth : int, optional
        Max hops for lazy ancestry exclusion.

    Returns
    -------
    CollisionSystem
    """
    config = CollisionSystemConfig(
        cell_size=cell_size,
        enabled=enabled,
        boundary_check_enabled=boundary_check_enabled,
        ancestry_exclusion_depth=ancestry_exclusion_depth,
    )
    return CollisionSystem(config=config, domain=domain)


# ------------------------------------------------------------------
# Module-level helper (shared by insertion collision check)
# ------------------------------------------------------------------


def _segment_to_polyline_distance(
    p1: np.ndarray,
    p2: np.ndarray,
    polyline: List[np.ndarray],
) -> float:
    """Compute minimum distance from a straight segment to a polyline."""
    min_dist = float("inf")
    for i in range(len(polyline) - 1):
        dist = segment_segment_distance_exact(p1, p2, polyline[i], polyline[i + 1])
        min_dist = min(min_dist, dist)
    return min_dist
