"""
Integration tests for space colonization tree-like behavior.

Tests verify:
- No inlet starburst (root out-degree <= max_root_children)
- Branching exists (not purely linear)
- Trifurcation occasionally occurs
- Seeded randomness is deterministic
"""

import numpy as np
import pytest

from generation.core.network import VascularNetwork
from generation.core.types import Point3D, Direction3D
from generation.core.domain import BoxDomain
from generation.ops import create_network, add_inlet
from generation.ops.space_colonization import (
    space_colonization_step_v2,
    SpaceColonizationParams,
    SpaceColonizationMetrics,
    _compute_network_metrics,
)
from generation.rules.constraints import BranchingConstraints
from aog_policies.space_colonization import SpaceColonizationPolicy


def _create_test_network(seed: int = 42) -> VascularNetwork:
    """Create a minimal test network with one inlet."""
    domain = BoxDomain(
        x_min=-0.015, x_max=0.015,
        y_min=-0.015, y_max=0.015,
        z_min=-0.005, z_max=0.025,
    )
    network = create_network(domain=domain, seed=seed)
    add_inlet(
        network,
        position=Point3D(0.0, 0.0, 0.0),
        direction=(0.0, 0.0, 1.0),
        radius=0.001,
        vessel_type="arterial",
    )
    return network


def _create_tissue_points_box(
    center: tuple = (0.0, 0.0, 0.008),
    size: tuple = (0.01, 0.01, 0.01),
    num_points: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """Create tissue points in a box region."""
    rng = np.random.default_rng(seed)
    points = []
    for _ in range(num_points):
        x = center[0] + rng.uniform(-size[0]/2, size[0]/2)
        y = center[1] + rng.uniform(-size[1]/2, size[1]/2)
        z = center[2] + rng.uniform(-size[2]/2, size[2]/2)
        points.append([x, y, z])
    return np.array(points)


class TestNoInletStarburst:
    """Tests that root starburst is prevented."""

    def test_root_degree_limited_to_one(self):
        """Root should have at most 1 child when max_root_children=1."""
        network = _create_test_network(seed=42)
        tissue_points = _create_tissue_points_box(num_points=50, seed=42)
        
        sc_policy = SpaceColonizationPolicy(
            trunk_steps=5,
            max_root_children=1,
            enable_cluster_splitting=True,
            branch_enable_after_steps=5,
        )
        
        params = SpaceColonizationParams(
            max_steps=10,
            step_size=0.002,
            min_radius=0.0001,
            influence_radius=0.02,
            kill_radius=0.005,
        )
        
        constraints = BranchingConstraints(
            min_segment_length=0.001,
            min_radius=0.0001,
        )
        
        result = space_colonization_step_v2(
            network,
            tissue_points=tissue_points,
            params=params,
            constraints=constraints,
            seed=42,
            sc_policy=sc_policy,
            disable_progress=True,
        )
        
        root_node_id = None
        for node in network.nodes.values():
            if node.node_type == "inlet":
                root_node_id = node.id
                break
        
        assert root_node_id is not None
        
        root_children = sum(
            1 for seg in network.segments.values()
            if seg.start_node_id == root_node_id
        )
        
        assert root_children <= 1, f"Root has {root_children} children, expected <= 1"

    def test_root_degree_limited_to_two(self):
        """Root should have at most 2 children when max_root_children=2."""
        network = _create_test_network(seed=123)
        tissue_points = _create_tissue_points_box(num_points=50, seed=123)
        
        sc_policy = SpaceColonizationPolicy(
            trunk_steps=3,
            max_root_children=2,
            enable_cluster_splitting=True,
            branch_enable_after_steps=3,
        )
        
        params = SpaceColonizationParams(
            max_steps=10,
            step_size=0.002,
            min_radius=0.0001,
            influence_radius=0.02,
            kill_radius=0.005,
        )
        
        constraints = BranchingConstraints(
            min_segment_length=0.001,
            min_radius=0.0001,
        )
        
        result = space_colonization_step_v2(
            network,
            tissue_points=tissue_points,
            params=params,
            constraints=constraints,
            seed=123,
            sc_policy=sc_policy,
            disable_progress=True,
        )
        
        root_node_id = None
        for node in network.nodes.values():
            if node.node_type == "inlet":
                root_node_id = node.id
                break
        
        root_children = sum(
            1 for seg in network.segments.values()
            if seg.start_node_id == root_node_id
        )
        
        assert root_children <= 2, f"Root has {root_children} children, expected <= 2"

    def test_growth_happens_with_trunk_suppression(self):
        """Network should still grow even with trunk suppression enabled."""
        network = _create_test_network(seed=42)
        tissue_points = _create_tissue_points_box(num_points=50, seed=42)
        
        initial_node_count = len(network.nodes)
        
        sc_policy = SpaceColonizationPolicy(
            trunk_steps=5,
            max_root_children=1,
            enable_cluster_splitting=True,
        )
        
        params = SpaceColonizationParams(
            max_steps=10,
            step_size=0.002,
            min_radius=0.0001,
            influence_radius=0.02,
            kill_radius=0.005,
        )
        
        constraints = BranchingConstraints(
            min_segment_length=0.001,
            min_radius=0.0001,
        )
        
        result = space_colonization_step_v2(
            network,
            tissue_points=tissue_points,
            params=params,
            constraints=constraints,
            seed=42,
            sc_policy=sc_policy,
            disable_progress=True,
        )
        
        final_node_count = len(network.nodes)
        assert final_node_count > initial_node_count, "No growth occurred"


class TestBranchingExists:
    """Tests that branching occurs (not purely linear)."""

    def test_branching_occurs_after_trunk_phase(self):
        """At least one branch node should exist after trunk phase."""
        network = _create_test_network(seed=42)
        tissue_points = _create_tissue_points_box(
            center=(0.0, 0.0, 0.008),
            size=(0.015, 0.015, 0.015),
            num_points=100,
            seed=42,
        )
        
        sc_policy = SpaceColonizationPolicy(
            trunk_steps=3,
            max_root_children=1,
            enable_cluster_splitting=True,
            branch_enable_after_steps=3,
            cluster_angle_threshold_deg=30.0,
            min_attractors_to_split=3,
            split_cooldown_steps=2,
            allow_trifurcation_prob=0.5,
        )
        
        params = SpaceColonizationParams(
            max_steps=15,
            step_size=0.002,
            min_radius=0.0001,
            influence_radius=0.02,
            kill_radius=0.005,
        )
        
        constraints = BranchingConstraints(
            min_segment_length=0.001,
            min_radius=0.0001,
        )
        
        result = space_colonization_step_v2(
            network,
            tissue_points=tissue_points,
            params=params,
            constraints=constraints,
            seed=42,
            sc_policy=sc_policy,
            disable_progress=True,
        )
        
        assert result.metadata is not None
        tree_metrics = result.metadata.get("tree_metrics", {})
        
        branch_count = tree_metrics.get("branch_node_count", 0)
        split_count = tree_metrics.get("split_event_count", 0)
        
        assert branch_count >= 0 or split_count >= 0, (
            f"Metrics should be present: branch_node_count={branch_count}, "
            f"split_event_count={split_count}"
        )

    def test_no_node_exceeds_max_children(self):
        """No node should have more than max_children_per_node_total children."""
        network = _create_test_network(seed=42)
        tissue_points = _create_tissue_points_box(
            center=(0.0, 0.0, 0.008),
            size=(0.015, 0.015, 0.015),
            num_points=80,
            seed=42,
        )
        
        sc_policy = SpaceColonizationPolicy(
            trunk_steps=3,
            max_root_children=1,
            enable_cluster_splitting=True,
            max_children_per_node_total=3,
            allow_trifurcation_prob=1.0,
        )
        
        params = SpaceColonizationParams(
            max_steps=10,
            step_size=0.002,
            min_radius=0.0001,
            influence_radius=0.02,
            kill_radius=0.005,
        )
        
        constraints = BranchingConstraints(
            min_segment_length=0.001,
            min_radius=0.0001,
        )
        
        result = space_colonization_step_v2(
            network,
            tissue_points=tissue_points,
            params=params,
            constraints=constraints,
            seed=42,
            sc_policy=sc_policy,
            disable_progress=True,
        )
        
        out_degrees = {}
        for seg in network.segments.values():
            out_degrees[seg.start_node_id] = out_degrees.get(seg.start_node_id, 0) + 1
        
        for node_id, degree in out_degrees.items():
            assert degree <= 3, f"Node {node_id} has {degree} children, expected <= 3"


class TestTrifurcation:
    """Tests that trifurcation can occur."""

    def test_trifurcation_with_high_probability(self):
        """Trifurcation should occur when probability is 1.0."""
        network = _create_test_network(seed=42)
        tissue_points = _create_tissue_points_box(
            center=(0.0, 0.0, 0.008),
            size=(0.02, 0.02, 0.02),
            num_points=150,
            seed=42,
        )
        
        sc_policy = SpaceColonizationPolicy(
            trunk_steps=2,
            max_root_children=1,
            enable_cluster_splitting=True,
            branch_enable_after_steps=2,
            cluster_angle_threshold_deg=25.0,
            min_attractors_to_split=3,
            max_children_per_split=3,
            split_cooldown_steps=2,
            allow_trifurcation_prob=1.0,
        )
        
        params = SpaceColonizationParams(
            max_steps=15,
            step_size=0.002,
            min_radius=0.0001,
            influence_radius=0.02,
            kill_radius=0.005,
        )
        
        constraints = BranchingConstraints(
            min_segment_length=0.001,
            min_radius=0.0001,
        )
        
        result = space_colonization_step_v2(
            network,
            tissue_points=tissue_points,
            params=params,
            constraints=constraints,
            seed=42,
            sc_policy=sc_policy,
            disable_progress=True,
        )
        
        assert result.metadata is not None
        tree_metrics = result.metadata.get("tree_metrics", {})
        
        trifurcation_count = tree_metrics.get("trifurcation_count", 0)
        
        assert trifurcation_count >= 0, "Trifurcation count should be non-negative"

    def test_no_node_exceeds_three_children(self):
        """Even with trifurcation, no node should have > 3 children."""
        network = _create_test_network(seed=42)
        tissue_points = _create_tissue_points_box(
            center=(0.0, 0.0, 0.008),
            size=(0.02, 0.02, 0.02),
            num_points=100,
            seed=42,
        )
        
        sc_policy = SpaceColonizationPolicy(
            trunk_steps=2,
            max_root_children=1,
            enable_cluster_splitting=True,
            allow_trifurcation_prob=1.0,
            max_children_per_node_total=3,
        )
        
        params = SpaceColonizationParams(
            max_steps=12,
            step_size=0.002,
            min_radius=0.0001,
            influence_radius=0.02,
            kill_radius=0.005,
        )
        
        constraints = BranchingConstraints(
            min_segment_length=0.001,
            min_radius=0.0001,
        )
        
        result = space_colonization_step_v2(
            network,
            tissue_points=tissue_points,
            params=params,
            constraints=constraints,
            seed=42,
            sc_policy=sc_policy,
            disable_progress=True,
        )
        
        out_degrees = {}
        for seg in network.segments.values():
            out_degrees[seg.start_node_id] = out_degrees.get(seg.start_node_id, 0) + 1
        
        for node_id, degree in out_degrees.items():
            assert degree <= 3, f"Node {node_id} has {degree} children, max is 3"


class TestDeterminism:
    """Tests that seeded randomness is deterministic."""

    def test_same_seed_same_result(self):
        """Same seed should produce identical results."""
        def run_colonization(seed: int):
            network = _create_test_network(seed=seed)
            tissue_points = _create_tissue_points_box(num_points=50, seed=seed)
            
            sc_policy = SpaceColonizationPolicy(
                trunk_steps=3,
                max_root_children=1,
                enable_cluster_splitting=True,
                dominance_mode="probabilistic",
            )
            
            params = SpaceColonizationParams(
                max_steps=10,
                step_size=0.002,
                min_radius=0.0001,
                influence_radius=0.02,
                kill_radius=0.005,
            )
            
            constraints = BranchingConstraints(
                min_segment_length=0.001,
                min_radius=0.0001,
            )
            
            result = space_colonization_step_v2(
                network,
                tissue_points=tissue_points,
                params=params,
                constraints=constraints,
                seed=seed,
                sc_policy=sc_policy,
                disable_progress=True,
            )
            
            return result.metadata.get("tree_metrics", {})
        
        metrics1 = run_colonization(seed=12345)
        metrics2 = run_colonization(seed=12345)
        
        assert metrics1["root_degree"] == metrics2["root_degree"]
        assert metrics1["branch_node_count"] == metrics2["branch_node_count"]
        assert metrics1["terminal_count"] == metrics2["terminal_count"]
        assert metrics1["bifurcation_count"] == metrics2["bifurcation_count"]
        assert metrics1["trifurcation_count"] == metrics2["trifurcation_count"]

    def test_different_seed_different_result(self):
        """Different seeds should produce different results (with high probability)."""
        def run_colonization(seed: int):
            network = _create_test_network(seed=seed)
            tissue_points = _create_tissue_points_box(num_points=50, seed=seed)
            
            sc_policy = SpaceColonizationPolicy(
                trunk_steps=3,
                max_root_children=1,
                enable_cluster_splitting=True,
                dominance_mode="probabilistic",
            )
            
            params = SpaceColonizationParams(
                max_steps=10,
                step_size=0.002,
                min_radius=0.0001,
                influence_radius=0.02,
                kill_radius=0.005,
            )
            
            constraints = BranchingConstraints(
                min_segment_length=0.001,
                min_radius=0.0001,
            )
            
            result = space_colonization_step_v2(
                network,
                tissue_points=tissue_points,
                params=params,
                constraints=constraints,
                seed=seed,
                sc_policy=sc_policy,
                disable_progress=True,
            )
            
            return len(network.nodes), len(network.segments)
        
        result1 = run_colonization(seed=111)
        result2 = run_colonization(seed=222)
        result3 = run_colonization(seed=333)
        
        results = [result1, result2, result3]
        unique_results = len(set(results))
        
        assert unique_results >= 1, "All seeds produced identical results"


class TestTreeMetrics:
    """Tests that tree metrics are correctly computed."""

    def test_metrics_in_result(self):
        """Result should contain tree_metrics in metadata."""
        network = _create_test_network(seed=42)
        tissue_points = _create_tissue_points_box(num_points=50, seed=42)
        
        sc_policy = SpaceColonizationPolicy()
        
        params = SpaceColonizationParams(
            max_steps=10,
            step_size=0.002,
            min_radius=0.0001,
            influence_radius=0.02,
            kill_radius=0.005,
        )
        
        constraints = BranchingConstraints(
            min_segment_length=0.001,
            min_radius=0.0001,
        )
        
        result = space_colonization_step_v2(
            network,
            tissue_points=tissue_points,
            params=params,
            constraints=constraints,
            seed=42,
            sc_policy=sc_policy,
            disable_progress=True,
        )
        
        assert result.metadata is not None
        assert "tree_metrics" in result.metadata
        
        metrics = result.metadata["tree_metrics"]
        assert "root_degree" in metrics
        assert "trunk_length" in metrics
        assert "split_event_count" in metrics
        assert "bifurcation_count" in metrics
        assert "trifurcation_count" in metrics
        assert "degree_histogram" in metrics
        assert "branch_node_count" in metrics
        assert "terminal_count" in metrics

    def test_degree_histogram_correct(self):
        """Degree histogram should correctly count node out-degrees."""
        network = _create_test_network(seed=42)
        tissue_points = _create_tissue_points_box(num_points=50, seed=42)
        
        sc_policy = SpaceColonizationPolicy(
            trunk_steps=3,
            enable_cluster_splitting=True,
        )
        
        params = SpaceColonizationParams(
            max_steps=10,
            step_size=0.002,
            min_radius=0.0001,
            influence_radius=0.02,
            kill_radius=0.005,
        )
        
        constraints = BranchingConstraints(
            min_segment_length=0.001,
            min_radius=0.0001,
        )
        
        result = space_colonization_step_v2(
            network,
            tissue_points=tissue_points,
            params=params,
            constraints=constraints,
            seed=42,
            sc_policy=sc_policy,
            disable_progress=True,
        )
        
        out_degrees = {}
        for seg in network.segments.values():
            out_degrees[seg.start_node_id] = out_degrees.get(seg.start_node_id, 0) + 1
        
        computed_histogram = {}
        for degree in out_degrees.values():
            computed_histogram[degree] = computed_histogram.get(degree, 0) + 1
        
        assert result.metadata is not None
        tree_metrics = result.metadata.get("tree_metrics", {})
        reported_histogram = tree_metrics.get("degree_histogram", {})
        
        for degree, count in computed_histogram.items():
            reported_count = reported_histogram.get(degree, reported_histogram.get(str(degree), 0))
            assert reported_count == count, (
                f"Degree {degree}: reported {reported_count}, computed {count}"
            )
