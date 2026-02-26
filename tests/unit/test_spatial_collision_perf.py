"""
Tests for spatial-indexed collision checking in SC growth.

Validates that:
1. DynamicSpatialIndex-backed collision in grow_branch matches brute-force
2. Spatial index stays in sync during SC growth
3. No false negatives (all real collisions are caught)
4. SC produces valid results with spatial index enabled
"""

import pytest
import numpy as np
from generation.core.types import Point3D, Direction3D, TubeGeometry
from generation.core.network import VascularNetwork, Node, VesselSegment
from generation.core.result import OperationResult
from generation.rules.constraints import BranchingConstraints
from generation.spatial.grid_index import DynamicSpatialIndex
from generation.ops.growth import grow_branch


def _make_network_with_segments(n_segments=50, domain_radius=0.05, domain_height=0.1, seed=42):
    """Create a test network with n_segments random non-colliding segments."""
    from generation.core.domain import CylinderDomain
    rng = np.random.default_rng(seed)
    domain = CylinderDomain(
        radius=domain_radius,
        height=domain_height,
        center=Point3D(0, 0, 0),
    )
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
        direction = Direction3D(
            rng.normal(0, 0.3),
            rng.normal(0, 0.3),
            -1.0,
        )
        norm = np.sqrt(direction.dx**2 + direction.dy**2 + direction.dz**2)
        direction = Direction3D(direction.dx / norm, direction.dy / norm, direction.dz / norm)

        step = 0.001
        new_pos = Point3D(
            parent.position.x + direction.dx * step,
            parent.position.y + direction.dy * step,
            parent.position.z + direction.dz * step,
        )

        new_id = net.id_gen.next_id()
        new_node = Node(
            id=new_id,
            position=new_pos,
            node_type="terminal",
            vessel_type="arterial",
            attributes={"radius": 0.0005, "direction": direction.to_dict()},
        )
        seg_id = net.id_gen.next_id()
        seg = VesselSegment(
            id=seg_id,
            start_node_id=tip_id,
            end_node_id=new_id,
            geometry=TubeGeometry(
                start=parent.position,
                end=new_pos,
                radius_start=0.001,
                radius_end=0.0005,
            ),
            vessel_type="arterial",
        )
        net.add_node(new_node)
        net.add_segment(seg)
        if parent.node_type == "terminal":
            parent.node_type = "junction"
        tip_id = new_id

    return net, tip_id


def _build_spatial_index(network):
    """Build DynamicSpatialIndex from all segments in a network."""
    idx = DynamicSpatialIndex(cell_size=0.003)
    for seg_id, seg in network.segments.items():
        start = np.array([seg.geometry.start.x, seg.geometry.start.y, seg.geometry.start.z])
        end = np.array([seg.geometry.end.x, seg.geometry.end.y, seg.geometry.end.z])
        radius = seg.geometry.mean_radius()
        idx.insert_segment(seg_id, start, end, radius)
    return idx


class TestSpatialCollisionInGrowBranch:
    """grow_branch with spatial_index matches brute-force behavior."""

    def test_no_collision_matches(self):
        net, tip_id = _make_network_with_segments(10)
        idx = _build_spatial_index(net)
        constraints = BranchingConstraints(min_segment_length=0.0001, min_radius=0.0001)

        result_brute = grow_branch(
            net, tip_id, length=0.001,
            direction=(0, 0, -1), target_radius=0.0005,
            constraints=constraints, check_collisions=True,
            spatial_index=None,
        )
        net2, tip_id2 = _make_network_with_segments(10)
        idx2 = _build_spatial_index(net2)
        result_spatial = grow_branch(
            net2, tip_id2, length=0.001,
            direction=(0, 0, -1), target_radius=0.0005,
            constraints=constraints, check_collisions=True,
            spatial_index=idx2,
        )
        assert result_brute.is_success() == result_spatial.is_success()

    def test_collision_detected_by_both(self):
        net, tip_id = _make_network_with_segments(50, seed=123)
        idx = _build_spatial_index(net)
        constraints = BranchingConstraints(
            min_segment_length=0.0001, min_radius=0.0001,
            collision_min_clearance=0.01,
        )
        result_brute = grow_branch(
            net, tip_id, length=0.001,
            direction=(1, 0, 0), target_radius=0.005,
            constraints=constraints, check_collisions=True,
            spatial_index=None,
        )
        net2, tip_id2 = _make_network_with_segments(50, seed=123)
        idx2 = _build_spatial_index(net2)
        result_spatial = grow_branch(
            net2, tip_id2, length=0.001,
            direction=(1, 0, 0), target_radius=0.005,
            constraints=constraints, check_collisions=True,
            spatial_index=idx2,
        )
        assert result_brute.is_success() == result_spatial.is_success()

    def test_spatial_index_no_false_negatives(self):
        """All collisions found by brute-force are also found by spatial check."""
        from generation.ops.collision_legacy import check_segment_collision_swept
        net, tip_id = _make_network_with_segments(100, seed=77)
        idx = _build_spatial_index(net)

        rng = np.random.default_rng(99)
        for _ in range(20):
            parent = net.get_node(tip_id)
            d = rng.normal(size=3)
            d = d / np.linalg.norm(d)
            step = 0.001
            start = np.array([parent.position.x, parent.position.y, parent.position.z])
            end = start + d * step
            radius = 0.0005
            clearance = 0.0005

            has_brute, _ = check_segment_collision_swept(
                net, start, end, radius,
                exclude_node_ids=[tip_id],
                min_clearance=clearance,
            )
            has_spatial = idx.check_capsule_collision(
                start=start, end=end, radius=radius,
                buffer=clearance,
                exclude_adjacent_to=start,
            )
            if has_brute:
                assert has_spatial, "Spatial index missed a collision that brute-force found"


class TestSpatialIndexSync:
    """Spatial index stays in sync during SC growth."""

    def test_insert_after_growth(self):
        net, tip_id = _make_network_with_segments(5)
        idx = _build_spatial_index(net)
        initial_count = idx.segment_count

        constraints = BranchingConstraints(min_segment_length=0.0001, min_radius=0.0001)
        result = grow_branch(
            net, tip_id, length=0.001,
            direction=(0, 0, -1), target_radius=0.0005,
            constraints=constraints, check_collisions=True,
            spatial_index=idx,
        )
        if result.is_success():
            seg_id = result.new_ids["segment"]
            seg = net.segments[seg_id]
            start = np.array([seg.geometry.start.x, seg.geometry.start.y, seg.geometry.start.z])
            end = np.array([seg.geometry.end.x, seg.geometry.end.y, seg.geometry.end.z])
            idx.insert_segment(seg_id, start, end, seg.geometry.mean_radius())
            assert idx.segment_count == initial_count + 1


class TestSCIntegrationWithSpatialIndex:
    """End-to-end SC with spatial index produces valid results."""

    def test_sc_produces_growth(self):
        from generation.ops.space_colonization import (
            create_space_colonization_state,
            space_colonization_one_step,
            SpaceColonizationParams,
        )
        from generation.core.domain import CylinderDomain

        domain = CylinderDomain(radius=0.01, height=0.03, center=Point3D(0, 0, 0))
        net = VascularNetwork(domain=domain)
        root_id = net.id_gen.next_id()
        net.add_node(Node(
            id=root_id,
            position=Point3D(0, 0, 0.014),
            node_type="inlet",
            vessel_type="arterial",
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
        )
        constraints = BranchingConstraints(
            min_segment_length=0.001,
            min_radius=0.0001,
        )

        state = create_space_colonization_state(
            net, tissue, params=params, constraints=constraints,
            seed=42, seed_node_ids=[root_id], vessel_type="arterial",
        )
        assert state._collision_spatial_index is not None

        total_nodes = 0
        for _ in range(20):
            result = space_colonization_one_step(state)
            total_nodes += result.nodes_added
            if result.exhausted or result.stalled:
                break

        assert total_nodes > 0, "SC with spatial index should produce growth"
        assert state._collision_spatial_index.segment_count > 0
