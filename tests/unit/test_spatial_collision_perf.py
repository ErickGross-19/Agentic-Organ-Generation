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
from generation.ops._gpu_nn import (
    batch_collision_prefilter,
    batch_direction_average,
    PersistentGPUIndex,
    vectorized_direction_average,
)


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


class TestBatchCollisionPrefilter:
    """Phase 2: batch_collision_prefilter matches brute-force per-pair check."""

    def test_empty_candidates(self):
        result = batch_collision_prefilter(
            np.empty((0, 3)), np.empty((0, 3)), np.empty(0),
            np.ones((5, 3)), np.ones((5, 3)) + 0.1, np.ones(5) * 0.001,
        )
        assert len(result) == 0

    def test_empty_segments(self):
        result = batch_collision_prefilter(
            np.ones((3, 3)), np.ones((3, 3)) + 0.1, np.ones(3) * 0.001,
            np.empty((0, 3)), np.empty((0, 3)), np.empty(0),
        )
        assert len(result) == 3
        assert not np.any(result)

    def test_close_candidates_flagged(self):
        cand_starts = np.array([[0.0, 0.0, 0.0]])
        cand_ends = np.array([[0.001, 0.0, 0.0]])
        cand_radii = np.array([0.0005])
        seg_starts = np.array([[0.0005, 0.001, 0.0]])
        seg_ends = np.array([[0.0015, 0.001, 0.0]])
        seg_radii = np.array([0.0005])
        result = batch_collision_prefilter(
            cand_starts, cand_ends, cand_radii,
            seg_starts, seg_ends, seg_radii,
        )
        assert result[0], "Close candidate should be flagged as might-collide"

    def test_far_candidates_clear(self):
        cand_starts = np.array([[0.0, 0.0, 0.0]])
        cand_ends = np.array([[0.001, 0.0, 0.0]])
        cand_radii = np.array([0.0005])
        seg_starts = np.array([[10.0, 10.0, 10.0]])
        seg_ends = np.array([[10.001, 10.0, 10.0]])
        seg_radii = np.array([0.0005])
        result = batch_collision_prefilter(
            cand_starts, cand_ends, cand_radii,
            seg_starts, seg_ends, seg_radii,
        )
        assert not result[0], "Far candidate should be clear"

    def test_batch_consistency(self):
        rng = np.random.default_rng(42)
        n_cand, n_seg = 20, 50
        cand_starts = rng.uniform(-0.01, 0.01, (n_cand, 3))
        cand_ends = cand_starts + rng.uniform(0.0005, 0.002, (n_cand, 3))
        cand_radii = rng.uniform(0.0001, 0.001, n_cand)
        seg_starts = rng.uniform(-0.01, 0.01, (n_seg, 3))
        seg_ends = seg_starts + rng.uniform(0.0005, 0.002, (n_seg, 3))
        seg_radii = rng.uniform(0.0001, 0.001, n_seg)

        batch_result = batch_collision_prefilter(
            cand_starts, cand_ends, cand_radii,
            seg_starts, seg_ends, seg_radii,
        )

        for i in range(n_cand):
            single = batch_collision_prefilter(
                cand_starts[i:i+1], cand_ends[i:i+1], cand_radii[i:i+1],
                seg_starts, seg_ends, seg_radii,
            )
            assert batch_result[i] == single[0], f"Mismatch at candidate {i}"


class TestBatchDirectionAverage:
    """Phase 3: batch_direction_average matches per-tip vectorized_direction_average."""

    def test_single_tip_matches(self):
        rng = np.random.default_rng(42)
        tip_pos = rng.uniform(-0.01, 0.01, (1, 3))
        attracted = [rng.uniform(-0.01, 0.01, (10, 3))]
        batch = batch_direction_average(tip_pos, attracted)
        single = vectorized_direction_average(attracted[0], tip_pos[0])
        np.testing.assert_allclose(batch[0], single, atol=1e-10)

    def test_multiple_tips_match(self):
        rng = np.random.default_rng(99)
        n_tips = 10
        tip_pos = rng.uniform(-0.01, 0.01, (n_tips, 3))
        attracted_list = [rng.uniform(-0.01, 0.01, (rng.integers(3, 20), 3)) for _ in range(n_tips)]
        batch = batch_direction_average(tip_pos, attracted_list)
        for i in range(n_tips):
            single = vectorized_direction_average(attracted_list[i], tip_pos[i])
            np.testing.assert_allclose(batch[i], single, atol=1e-10)

    def test_empty_attracted(self):
        tip_pos = np.array([[0.0, 0.0, 0.0]])
        attracted_list = [np.empty((0, 3))]
        batch = batch_direction_average(tip_pos, attracted_list)
        assert np.allclose(batch[0], 0.0)


class TestPersistentGPUIndex:
    """Phase 4: PersistentGPUIndex NN and kill queries match CPU fallback."""

    def test_nn_query_matches_cpu(self):
        rng = np.random.default_rng(42)
        tissue = rng.uniform(-0.01, 0.01, (200, 3))
        database = rng.uniform(-0.01, 0.01, (10, 3))
        active = np.arange(200, dtype=np.intp)

        idx = PersistentGPUIndex(tissue)
        gpu_dist, gpu_idx_arr = idx.nn_query(active, database)

        from generation.ops._gpu_nn import nearest_neighbor
        cpu_dist, cpu_idx_arr = nearest_neighbor(tissue, database, k=1)

        np.testing.assert_allclose(gpu_dist, cpu_dist, atol=1e-5)
        np.testing.assert_array_equal(gpu_idx_arr, cpu_idx_arr)

    def test_kill_within_radius_matches_cpu(self):
        rng = np.random.default_rng(42)
        tissue = rng.uniform(-0.01, 0.01, (200, 3))
        nodes = rng.uniform(-0.005, 0.005, (5, 3))
        active = np.arange(200, dtype=np.intp)
        radius = 0.008

        idx = PersistentGPUIndex(tissue)
        gpu_mask = idx.kill_within_radius(active, nodes, radius)

        from generation.ops._gpu_nn import range_search
        cpu_mask = range_search(tissue, nodes, radius)

        np.testing.assert_array_equal(gpu_mask, cpu_mask)

    def test_empty_active(self):
        tissue = np.ones((10, 3))
        idx = PersistentGPUIndex(tissue)
        dist, indices = idx.nn_query(np.empty(0, dtype=np.intp), np.ones((5, 3)))
        assert len(dist) == 0
        assert len(indices) == 0

    def test_subset_active_indices(self):
        rng = np.random.default_rng(42)
        tissue = rng.uniform(-0.01, 0.01, (200, 3))
        database = rng.uniform(-0.01, 0.01, (10, 3))
        active = np.array([0, 10, 50, 100, 199], dtype=np.intp)

        idx = PersistentGPUIndex(tissue)
        dist, indices = idx.nn_query(active, database)

        from generation.ops._gpu_nn import nearest_neighbor
        cpu_dist, cpu_idx = nearest_neighbor(tissue[active], database, k=1)

        np.testing.assert_allclose(dist, cpu_dist, atol=1e-5)
        np.testing.assert_array_equal(indices, cpu_idx)


class TestSCIntegrationWithPhases234:
    """End-to-end SC with all phases produces valid growth."""

    def test_sc_growth_with_all_phases(self):
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
        assert state._persistent_gpu_index is not None
        assert state._collision_spatial_index is not None

        total_nodes = 0
        for _ in range(20):
            result = space_colonization_one_step(state)
            total_nodes += result.nodes_added
            if result.exhausted or result.stalled:
                break

        assert total_nodes > 0, "SC with all phases should produce growth"
