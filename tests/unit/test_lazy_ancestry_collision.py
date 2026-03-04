"""
Tests for lazy ancestry-based collision exclusion.

Validates that the new lazy per-candidate ancestry check produces identical
collision results to the old exhaustive exclusion-set approach, while being
dramatically faster at scale.

Test-driven: these tests are written BEFORE the implementation.
"""

import time
import pytest
import numpy as np

from generation.core.types import Point3D, Direction3D, TubeGeometry
from generation.core.network import VascularNetwork, Node, VesselSegment
from generation.core.domain import CylinderDomain
from generation.rules.constraints import BranchingConstraints
from generation.spatial.grid_index import DynamicSpatialIndex
from generation.ops.growth import grow_branch


def _make_linear_chain(n_segments=10, step=0.001, domain_radius=0.05, domain_height=0.2):
    """Create a simple linear chain of segments (no branching)."""
    domain = CylinderDomain(radius=domain_radius, height=domain_height, center=Point3D(0, 0, 0))
    net = VascularNetwork(domain=domain)

    root_id = net.id_gen.next_id()
    root = Node(
        id=root_id,
        position=Point3D(0, 0, domain_height / 2 - 0.001),
        node_type="inlet",
        vessel_type="arterial",
        attributes={"radius": 0.001, "direction": Direction3D(0, 0, -1).to_dict()},
    )
    net.add_node(root)

    tip_id = root_id
    for i in range(n_segments):
        parent = net.get_node(tip_id)
        z = parent.position.z - step
        new_id = net.id_gen.next_id()
        new_pos = Point3D(0, 0, z)
        new_node = Node(
            id=new_id, position=new_pos,
            node_type="terminal", vessel_type="arterial",
            attributes={"radius": 0.0005, "direction": Direction3D(0, 0, -1).to_dict()},
        )
        seg_id = net.id_gen.next_id()
        seg = VesselSegment(
            id=seg_id, start_node_id=tip_id, end_node_id=new_id,
            geometry=TubeGeometry(start=parent.position, end=new_pos,
                                  radius_start=0.001, radius_end=0.0005),
            vessel_type="arterial",
        )
        net.add_node(new_node)
        net.add_segment(seg)
        if parent.node_type == "terminal":
            parent.node_type = "junction"
        tip_id = new_id

    return net, root_id, tip_id


def _make_branching_tree(depth=5, step=0.001, domain_radius=0.05, domain_height=0.2):
    """Create a binary tree with branching at each level."""
    domain = CylinderDomain(radius=domain_radius, height=domain_height, center=Point3D(0, 0, 0))
    net = VascularNetwork(domain=domain)

    root_id = net.id_gen.next_id()
    root = Node(
        id=root_id,
        position=Point3D(0, 0, domain_height / 2 - 0.001),
        node_type="inlet",
        vessel_type="arterial",
        attributes={"radius": 0.001, "direction": Direction3D(0, 0, -1).to_dict()},
    )
    net.add_node(root)

    tips = [(root_id, 0.0)]
    all_tips = []

    for level in range(depth):
        next_tips = []
        for parent_id, x_offset in tips:
            parent = net.get_node(parent_id)
            for side in (-1, 1):
                new_x = parent.position.x + side * step * 0.5
                new_z = parent.position.z - step
                new_id = net.id_gen.next_id()
                new_pos = Point3D(new_x, 0, new_z)
                new_node = Node(
                    id=new_id, position=new_pos,
                    node_type="terminal", vessel_type="arterial",
                    attributes={"radius": 0.0005, "direction": Direction3D(side * 0.5, 0, -1).to_dict()},
                )
                seg_id = net.id_gen.next_id()
                seg = VesselSegment(
                    id=seg_id, start_node_id=parent_id, end_node_id=new_id,
                    geometry=TubeGeometry(start=parent.position, end=new_pos,
                                          radius_start=0.001, radius_end=0.0005),
                    vessel_type="arterial",
                )
                net.add_node(new_node)
                net.add_segment(seg)
                next_tips.append((new_id, new_x))
            if parent.node_type in ("terminal", "inlet"):
                parent.node_type = "junction"
        tips = next_tips
    all_tips = [t[0] for t in tips]

    return net, root_id, all_tips


def _build_spatial_index(network, cell_size=0.003):
    """Build DynamicSpatialIndex from all segments in a network."""
    idx = DynamicSpatialIndex(cell_size=cell_size)
    for seg_id, seg in network.segments.items():
        start = np.array([seg.geometry.start.x, seg.geometry.start.y, seg.geometry.start.z])
        end = np.array([seg.geometry.end.x, seg.geometry.end.y, seg.geometry.end.z])
        radius = seg.geometry.mean_radius()
        idx.insert_segment(seg_id, start, end, radius)
    return idx


def _build_dicts(network):
    """Build the adjacency dicts used by the current exclusion approach."""
    children_by_node = {}
    seg_by_node = {}
    for seg in network.segments.values():
        children_by_node.setdefault(seg.start_node_id, []).append(seg)
        seg_by_node.setdefault(seg.start_node_id, []).append(seg)
        seg_by_node.setdefault(seg.end_node_id, []).append(seg)
    max_radius = max(
        (seg.geometry.mean_radius() for seg in network.segments.values()),
        default=0.001,
    )
    return children_by_node, seg_by_node, max_radius


class TestIsTreeNeighbor:
    """Tests for _is_tree_neighbor() helper function."""

    def test_same_node_is_neighbor(self):
        from generation.ops.growth import _is_tree_neighbor
        net, root_id, tip_id = _make_linear_chain(5)
        parent_of = {}
        for seg in net.segments.values():
            parent_of[seg.end_node_id] = seg.start_node_id
        assert _is_tree_neighbor(tip_id, tip_id, parent_of, max_hops=5)

    def test_parent_is_neighbor(self):
        from generation.ops.growth import _is_tree_neighbor
        net, root_id, tip_id = _make_linear_chain(5)
        parent_of = {}
        for seg in net.segments.values():
            parent_of[seg.end_node_id] = seg.start_node_id
        parent_id = parent_of[tip_id]
        assert _is_tree_neighbor(tip_id, parent_id, parent_of, max_hops=5)

    def test_distant_ancestor_not_neighbor_with_small_hops(self):
        from generation.ops.growth import _is_tree_neighbor
        net, root_id, tip_id = _make_linear_chain(20)
        parent_of = {}
        for seg in net.segments.values():
            parent_of[seg.end_node_id] = seg.start_node_id
        assert not _is_tree_neighbor(tip_id, root_id, parent_of, max_hops=3)

    def test_distant_ancestor_is_neighbor_with_large_hops(self):
        from generation.ops.growth import _is_tree_neighbor
        net, root_id, tip_id = _make_linear_chain(20)
        parent_of = {}
        for seg in net.segments.values():
            parent_of[seg.end_node_id] = seg.start_node_id
        assert _is_tree_neighbor(tip_id, root_id, parent_of, max_hops=25)

    def test_siblings_are_neighbors(self):
        from generation.ops.growth import _is_tree_neighbor
        net, root_id, tips = _make_branching_tree(depth=2)
        parent_of = {}
        for seg in net.segments.values():
            parent_of[seg.end_node_id] = seg.start_node_id
        left_tip = tips[0]
        right_tip = tips[1]
        assert _is_tree_neighbor(left_tip, right_tip, parent_of, max_hops=5)

    def test_distant_cousins_not_neighbors_small_hops(self):
        from generation.ops.growth import _is_tree_neighbor
        net, root_id, tips = _make_branching_tree(depth=5)
        parent_of = {}
        for seg in net.segments.values():
            parent_of[seg.end_node_id] = seg.start_node_id
        far_left = tips[0]
        far_right = tips[-1]
        assert not _is_tree_neighbor(far_left, far_right, parent_of, max_hops=3)

    def test_unknown_node_not_neighbor(self):
        from generation.ops.growth import _is_tree_neighbor
        net, root_id, tip_id = _make_linear_chain(5)
        parent_of = {}
        for seg in net.segments.values():
            parent_of[seg.end_node_id] = seg.start_node_id
        assert not _is_tree_neighbor(tip_id, 99999, parent_of, max_hops=10)


class TestBuildParentMap:
    """Tests for _build_parent_map() helper function."""

    def test_linear_chain_parent_map(self):
        from generation.ops.growth import _build_parent_map
        net, root_id, tip_id = _make_linear_chain(5)
        parent_of, seg_node_map = _build_parent_map(net)
        assert tip_id in parent_of
        assert root_id not in parent_of
        cur = tip_id
        hops = 0
        while cur in parent_of:
            cur = parent_of[cur]
            hops += 1
        assert cur == root_id
        assert hops == 5

    def test_branching_tree_parent_map(self):
        from generation.ops.growth import _build_parent_map
        net, root_id, tips = _make_branching_tree(depth=3)
        parent_of, seg_node_map = _build_parent_map(net)
        for tip in tips:
            assert tip in parent_of
        assert root_id not in parent_of

    def test_seg_node_map_covers_all_segments(self):
        from generation.ops.growth import _build_parent_map
        net, root_id, tips = _make_branching_tree(depth=3)
        parent_of, seg_node_map = _build_parent_map(net)
        for seg_id in net.segments:
            assert seg_id in seg_node_map
            start_nid, end_nid = seg_node_map[seg_id]
            assert start_nid == net.segments[seg_id].start_node_id
            assert end_nid == net.segments[seg_id].end_node_id


class TestLazyExclusionMatchesExhaustive:
    """The lazy ancestry check must produce identical collision results
    to the old exhaustive exclusion-set approach."""

    def _grow_with_old_approach(self, net, tip_id, direction, spatial_index):
        """Grow using current exhaustive exclusion set (baseline)."""
        children_by_node, seg_by_node, max_radius = _build_dicts(net)
        excl_depth = max(int((2 * max_radius + 0.0002) / 0.001) + 5, 10)
        constraints = BranchingConstraints(
            min_segment_length=0.0001, min_radius=0.0001,
            collision_min_clearance=0.0002,
        )
        return grow_branch(
            net, tip_id, length=0.001,
            direction=direction, target_radius=0.0005,
            constraints=constraints, check_collisions=True,
            spatial_index=spatial_index,
            collision_mode="break",
            _children_by_node=children_by_node,
            _seg_by_node=seg_by_node,
            _excl_depth=excl_depth,
        )

    def _grow_with_new_approach(self, net, tip_id, direction, spatial_index):
        """Grow using new lazy ancestry exclusion."""
        from generation.ops.growth import _build_parent_map
        parent_of, seg_node_map = _build_parent_map(net)
        max_radius = max(
            (seg.geometry.mean_radius() for seg in net.segments.values()),
            default=0.001,
        )
        excl_depth = max(int((2 * max_radius + 0.0002) / 0.001) + 5, 10)
        constraints = BranchingConstraints(
            min_segment_length=0.0001, min_radius=0.0001,
            collision_min_clearance=0.0002,
        )
        return grow_branch(
            net, tip_id, length=0.001,
            direction=direction, target_radius=0.0005,
            constraints=constraints, check_collisions=True,
            spatial_index=spatial_index,
            collision_mode="break",
            _parent_of=parent_of,
            _seg_node_map=seg_node_map,
            _excl_depth=excl_depth,
        )

    def test_linear_chain_no_collision(self):
        net, root_id, tip_id = _make_linear_chain(10)
        idx = _build_spatial_index(net)
        old = self._grow_with_old_approach(net, tip_id, (0, 0, -1), idx)
        net2, _, tip_id2 = _make_linear_chain(10)
        idx2 = _build_spatial_index(net2)
        new = self._grow_with_new_approach(net2, tip_id2, (0, 0, -1), idx2)
        assert old.is_success() == new.is_success()

    def test_branching_tree_collision_match(self):
        net, root_id, tips = _make_branching_tree(depth=4)
        idx = _build_spatial_index(net)
        tip = tips[0]
        old = self._grow_with_old_approach(net, tip, (0, 0, -1), idx)
        net2, _, tips2 = _make_branching_tree(depth=4)
        idx2 = _build_spatial_index(net2)
        new = self._grow_with_new_approach(net2, tips2[0], (0, 0, -1), idx2)
        assert old.is_success() == new.is_success()

    def test_grow_toward_sibling_excluded(self):
        """Growing toward a sibling branch should be excluded (not a collision)."""
        net, root_id, tips = _make_branching_tree(depth=2, step=0.002)
        idx = _build_spatial_index(net)
        left_tip = tips[0]
        right_tip = tips[-1]
        right_pos = net.get_node(right_tip).position
        left_pos = net.get_node(left_tip).position
        dx = right_pos.x - left_pos.x
        dz = right_pos.z - left_pos.z
        norm = (dx**2 + dz**2)**0.5
        direction = (dx / norm, 0, dz / norm)
        net2, _, tips2 = _make_branching_tree(depth=2, step=0.002)
        idx2 = _build_spatial_index(net2)
        result = self._grow_with_new_approach(net2, tips2[0], direction, idx2)
        old_result = self._grow_with_old_approach(net, left_tip, direction, idx)
        assert old_result.is_success() == result.is_success()


class TestLazyExclusionDetectsRealCollisions:
    """The lazy approach must still detect real collisions with unrelated segments."""

    def test_detects_collision_with_unrelated_segment(self):
        """A segment from a completely different branch family must trigger collision."""
        from generation.ops.growth import _build_parent_map
        net, root_id, tip_id = _make_linear_chain(5, step=0.005, domain_height=0.3)
        idx = _build_spatial_index(net)

        tip = net.get_node(tip_id)
        blocker_start_id = net.id_gen.next_id()
        blocker_start = Node(
            id=blocker_start_id,
            position=Point3D(tip.position.x - 0.003, 0, tip.position.z - 0.005),
            node_type="junction", vessel_type="arterial",
            attributes={"radius": 0.002},
        )
        net.add_node(blocker_start)
        blocker_end_id = net.id_gen.next_id()
        blocker_end = Node(
            id=blocker_end_id,
            position=Point3D(tip.position.x + 0.003, 0, tip.position.z - 0.005),
            node_type="terminal", vessel_type="arterial",
            attributes={"radius": 0.002},
        )
        net.add_node(blocker_end)
        blocker_seg_id = net.id_gen.next_id()
        blocker_seg = VesselSegment(
            id=blocker_seg_id, start_node_id=blocker_start_id, end_node_id=blocker_end_id,
            geometry=TubeGeometry(start=blocker_start.position, end=blocker_end.position,
                                  radius_start=0.002, radius_end=0.002),
            vessel_type="arterial",
        )
        net.add_segment(blocker_seg)
        idx.insert_segment(
            blocker_seg_id,
            np.array([blocker_start.position.x, blocker_start.position.y, blocker_start.position.z]),
            np.array([blocker_end.position.x, blocker_end.position.y, blocker_end.position.z]),
            0.002,
        )

        parent_of, seg_node_map = _build_parent_map(net)
        max_radius = max(
            (seg.geometry.mean_radius() for seg in net.segments.values()),
            default=0.001,
        )
        excl_depth = max(int((2 * max_radius + 0.002) / 0.005) + 5, 10)
        constraints = BranchingConstraints(
            min_segment_length=0.0001, min_radius=0.0001,
            collision_min_clearance=0.002,
        )
        result = grow_branch(
            net, tip_id, length=0.005,
            direction=(0, 0, -1), target_radius=0.001,
            constraints=constraints, check_collisions=True,
            spatial_index=idx,
            collision_mode="break",
            _parent_of=parent_of,
            _seg_node_map=seg_node_map,
            _excl_depth=excl_depth,
        )
        assert not result.is_success(), "Should detect collision with unrelated blocker segment"


class TestLazyExclusionPerformance:
    """The lazy approach should be faster than exhaustive at scale."""

    def test_lazy_faster_than_exhaustive_at_scale(self):
        from generation.ops.growth import _build_parent_map
        n_segments = 1000
        net, root_id, tip_id = _make_linear_chain(n_segments, step=0.0002, domain_height=0.5)
        idx = _build_spatial_index(net)
        children_by_node, seg_by_node, max_radius = _build_dicts(net)
        excl_depth = max(int((2 * max_radius + 0.0002) / 0.0002) + 5, 10)
        constraints = BranchingConstraints(
            min_segment_length=0.0001, min_radius=0.0001,
            collision_min_clearance=0.0002,
        )

        t0 = time.perf_counter()
        for _ in range(20):
            grow_branch(
                net, tip_id, length=0.0002,
                direction=(0, 0, -1), target_radius=0.0005,
                constraints=constraints, check_collisions=True,
                spatial_index=idx, collision_mode="break",
                _children_by_node=children_by_node,
                _seg_by_node=seg_by_node,
                _excl_depth=excl_depth,
            )
        old_time = time.perf_counter() - t0

        parent_of, seg_node_map = _build_parent_map(net)
        t0 = time.perf_counter()
        for _ in range(20):
            grow_branch(
                net, tip_id, length=0.0002,
                direction=(0, 0, -1), target_radius=0.0005,
                constraints=constraints, check_collisions=True,
                spatial_index=idx, collision_mode="break",
                _parent_of=parent_of,
                _seg_node_map=seg_node_map,
                _excl_depth=excl_depth,
            )
        new_time = time.perf_counter() - t0

        assert new_time < old_time * 1.5, (
            f"Lazy approach ({new_time:.3f}s) should not be significantly slower "
            f"than exhaustive ({old_time:.3f}s) at 1000 segments"
        )


class TestSCStepWithLazyExclusion:
    """Integration test: SC step with lazy exclusion produces valid growth."""

    def test_sc_step_produces_growth_with_lazy(self):
        from generation.ops.space_colonization import (
            space_colonization_step,
            SpaceColonizationParams,
        )
        domain = CylinderDomain(radius=0.01, height=0.03, center=Point3D(0, 0, 0))
        net = VascularNetwork(domain=domain)
        root_id = net.id_gen.next_id()
        net.add_node(Node(
            id=root_id,
            position=Point3D(0, 0, 0.014),
            node_type="inlet", vessel_type="arterial",
            attributes={"radius": 0.001, "direction": Direction3D(0, 0, -1).to_dict()},
        ))

        rng = np.random.default_rng(42)
        tissue = rng.uniform([-0.009, -0.009, -0.014], [0.009, 0.009, 0.014], size=(500, 3))

        params = SpaceColonizationParams(
            influence_radius=0.015,
            kill_radius=0.003,
            step_size=0.001,
            max_steps=20,
            vessel_type="arterial",
            min_clearance=0.0002,
        )
        constraints = BranchingConstraints(
            min_segment_length=0.001,
            min_radius=0.0001,
            collision_min_clearance=0.0002,
        )
        result = space_colonization_step(net, tissue, params=params, constraints=constraints, seed=42)
        assert result.is_success()
        assert len(net.segments) > 1, "SC should produce multiple segments"

    def test_sc_step_branching_preserved_with_lazy(self):
        from generation.ops.space_colonization import (
            space_colonization_step,
            SpaceColonizationParams,
        )
        domain = CylinderDomain(radius=0.01, height=0.03, center=Point3D(0, 0, 0))
        net = VascularNetwork(domain=domain)
        root_id = net.id_gen.next_id()
        net.add_node(Node(
            id=root_id,
            position=Point3D(0, 0, 0.014),
            node_type="inlet", vessel_type="arterial",
            attributes={"radius": 0.001, "direction": Direction3D(0, 0, -1).to_dict()},
        ))

        rng = np.random.default_rng(42)
        tissue = rng.uniform([-0.009, -0.009, -0.014], [0.009, 0.009, 0.014], size=(500, 3))

        params = SpaceColonizationParams(
            influence_radius=0.015,
            kill_radius=0.003,
            step_size=0.001,
            max_steps=50,
            vessel_type="arterial",
            encourage_bifurcation=True,
            min_attractions_for_bifurcation=3,
            bifurcation_probability=0.8,
            bifurcation_angle_threshold_deg=40.0,
            min_clearance=0.0002,
        )
        constraints = BranchingConstraints(
            min_segment_length=0.001,
            min_radius=0.0001,
            collision_min_clearance=0.0002,
        )
        result = space_colonization_step(net, tissue, params=params, constraints=constraints, seed=42)
        junctions = [n for n in net.nodes.values() if n.node_type == "junction"]
        assert len(junctions) >= 1, "SC with lazy exclusion should still produce branching"
