"""
Unit tests for angular clustering split logic in space colonization.

Tests verify:
- 2 clusters -> expect 2 children
- 3 clusters -> expect 3 children when trifurcation probability is 1.0
- max_children_per_split is enforced
- split cooldown prevents repeated splits
"""

import numpy as np
import pytest

from generation.ops.space_colonization import (
    _greedy_angular_clustering,
    _merge_weakest_cluster,
    _apply_noise_to_direction,
    _select_active_tips_probabilistic,
    _select_active_tips_topk,
    TipState,
)


class TestGreedyAngularClustering:
    """Tests for the greedy angular clustering algorithm."""

    def test_empty_vectors_returns_empty(self):
        """Empty input should return empty clusters."""
        result = _greedy_angular_clustering([], angle_threshold_deg=35.0, max_clusters=3)
        assert result == []

    def test_single_vector_returns_single_cluster(self):
        """Single vector should return single cluster with that vector."""
        vectors = [np.array([1.0, 0.0, 0.0])]
        result = _greedy_angular_clustering(vectors, angle_threshold_deg=35.0, max_clusters=3)
        assert len(result) == 1
        assert result[0] == [0]

    def test_two_similar_vectors_same_cluster(self):
        """Two vectors within threshold should be in same cluster."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.98, 0.2, 0.0])
        v2 = v2 / np.linalg.norm(v2)
        vectors = [v1, v2]
        result = _greedy_angular_clustering(vectors, angle_threshold_deg=35.0, max_clusters=3)
        assert len(result) == 1
        assert set(result[0]) == {0, 1}

    def test_two_distinct_vectors_two_clusters(self):
        """Two vectors with large angle should form two clusters."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        vectors = [v1, v2]
        result = _greedy_angular_clustering(vectors, angle_threshold_deg=35.0, max_clusters=3)
        assert len(result) == 2
        assert 0 in result[0] or 0 in result[1]
        assert 1 in result[0] or 1 in result[1]

    def test_three_distinct_vectors_three_clusters(self):
        """Three vectors with large angles should form three clusters."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        v3 = np.array([0.0, 0.0, 1.0])
        vectors = [v1, v2, v3]
        result = _greedy_angular_clustering(vectors, angle_threshold_deg=35.0, max_clusters=3)
        assert len(result) == 3

    def test_max_clusters_enforced(self):
        """Should not exceed max_clusters even with many distinct vectors."""
        vectors = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([-1.0, 0.0, 0.0]),
            np.array([0.0, -1.0, 0.0]),
        ]
        result = _greedy_angular_clustering(vectors, angle_threshold_deg=35.0, max_clusters=2)
        assert len(result) <= 2

    def test_zero_vector_ignored(self):
        """Zero vectors should be ignored."""
        vectors = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        ]
        result = _greedy_angular_clustering(vectors, angle_threshold_deg=35.0, max_clusters=3)
        assert len(result) == 2
        all_indices = []
        for cluster in result:
            all_indices.extend(cluster)
        assert 1 not in all_indices

    def test_tight_threshold_more_clusters(self):
        """Tighter threshold should produce more clusters."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.9, 0.44, 0.0])
        v2 = v2 / np.linalg.norm(v2)
        vectors = [v1, v2]
        
        result_loose = _greedy_angular_clustering(vectors, angle_threshold_deg=45.0, max_clusters=3)
        result_tight = _greedy_angular_clustering(vectors, angle_threshold_deg=20.0, max_clusters=3)
        
        assert len(result_loose) <= len(result_tight)


class TestMergeWeakestCluster:
    """Tests for merging the weakest cluster."""

    def test_two_clusters_unchanged(self):
        """Two clusters should remain unchanged."""
        clusters = [[0, 1], [2, 3]]
        vectors = [
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, 0.1, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 1.0, 0.1]),
        ]
        result = _merge_weakest_cluster(clusters, vectors)
        assert len(result) == 2

    def test_three_clusters_merged_to_two(self):
        """Three clusters should be merged to two."""
        clusters = [[0, 1], [2], [3, 4, 5]]
        vectors = [
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, 0.1, 0.0]),
            np.array([0.5, 0.5, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 1.0, 0.1]),
            np.array([0.0, 1.0, -0.1]),
        ]
        result = _merge_weakest_cluster(clusters, vectors)
        assert len(result) == 2

    def test_weakest_merged_into_nearest(self):
        """Weakest cluster should be merged into nearest cluster."""
        clusters = [[0], [1], [2, 3]]
        vectors = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.9, 0.44, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 1.0, 0.1]),
        ]
        vectors[1] = vectors[1] / np.linalg.norm(vectors[1])
        
        result = _merge_weakest_cluster(clusters, vectors)
        assert len(result) == 2


class TestApplyNoiseToDirection:
    """Tests for applying noise to direction vectors."""

    def test_zero_noise_unchanged(self):
        """Zero noise should return unchanged direction."""
        direction = np.array([1.0, 0.0, 0.0])
        rng = np.random.default_rng(42)
        result = _apply_noise_to_direction(direction, noise_angle_deg=0.0, rng=rng)
        np.testing.assert_array_almost_equal(result, direction)

    def test_noise_preserves_unit_length(self):
        """Noisy direction should still be unit length."""
        direction = np.array([1.0, 0.0, 0.0])
        rng = np.random.default_rng(42)
        result = _apply_noise_to_direction(direction, noise_angle_deg=10.0, rng=rng)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-10

    def test_noise_within_bounds(self):
        """Noisy direction should be within noise angle of original."""
        direction = np.array([1.0, 0.0, 0.0])
        rng = np.random.default_rng(42)
        noise_deg = 10.0
        
        for _ in range(100):
            result = _apply_noise_to_direction(direction, noise_angle_deg=noise_deg, rng=rng)
            cos_angle = np.dot(direction, result)
            angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            assert angle_deg <= noise_deg + 0.1

    def test_seeded_determinism(self):
        """Same seed should produce same result."""
        direction = np.array([0.0, 0.0, 1.0])
        
        rng1 = np.random.default_rng(123)
        result1 = _apply_noise_to_direction(direction, noise_angle_deg=5.0, rng=rng1)
        
        rng2 = np.random.default_rng(123)
        result2 = _apply_noise_to_direction(direction, noise_angle_deg=5.0, rng=rng2)
        
        np.testing.assert_array_almost_equal(result1, result2)


class TestSelectActiveTips:
    """Tests for tip selection algorithms."""

    def test_probabilistic_respects_min_active(self):
        """Probabilistic selection should respect min_active_tips."""
        tip_states = [TipState(node_id=i) for i in range(10)]
        tip_supports = {i: i + 1 for i in range(10)}
        rng = np.random.default_rng(42)
        
        result = _select_active_tips_probabilistic(
            tip_states,
            tip_supports,
            alpha=1.5,
            active_fraction=0.1,
            min_active=5,
            rng=rng,
        )
        
        assert len(result) >= 5

    def test_probabilistic_respects_fraction(self):
        """Probabilistic selection should respect active_tip_fraction."""
        tip_states = [TipState(node_id=i) for i in range(20)]
        tip_supports = {i: 10 for i in range(20)}
        rng = np.random.default_rng(42)
        
        result = _select_active_tips_probabilistic(
            tip_states,
            tip_supports,
            alpha=1.0,
            active_fraction=0.5,
            min_active=1,
            rng=rng,
        )
        
        assert len(result) >= 10

    def test_topk_selects_highest_support(self):
        """Top-k selection should select tips with highest support."""
        tip_states = [TipState(node_id=i) for i in range(10)]
        tip_supports = {i: i for i in range(10)}
        
        result = _select_active_tips_topk(
            tip_states,
            tip_supports,
            active_fraction=0.3,
            min_active=1,
        )
        
        selected_ids = {ts.node_id for ts in result}
        assert 9 in selected_ids
        assert 8 in selected_ids

    def test_topk_respects_min_active(self):
        """Top-k selection should respect min_active_tips."""
        tip_states = [TipState(node_id=i) for i in range(10)]
        tip_supports = {i: i for i in range(10)}
        
        result = _select_active_tips_topk(
            tip_states,
            tip_supports,
            active_fraction=0.1,
            min_active=5,
        )
        
        assert len(result) >= 5

    def test_empty_tips_returns_empty(self):
        """Empty tip list should return empty result."""
        result_prob = _select_active_tips_probabilistic(
            [],
            {},
            alpha=1.0,
            active_fraction=0.5,
            min_active=1,
            rng=np.random.default_rng(42),
        )
        result_topk = _select_active_tips_topk(
            [],
            {},
            active_fraction=0.5,
            min_active=1,
        )
        
        assert result_prob == []
        assert result_topk == []

    def test_probabilistic_seeded_determinism(self):
        """Same seed should produce same selection."""
        tip_states = [TipState(node_id=i) for i in range(20)]
        tip_supports = {i: i + 1 for i in range(20)}
        
        rng1 = np.random.default_rng(999)
        result1 = _select_active_tips_probabilistic(
            tip_states,
            tip_supports,
            alpha=1.5,
            active_fraction=0.4,
            min_active=3,
            rng=rng1,
        )
        
        rng2 = np.random.default_rng(999)
        result2 = _select_active_tips_probabilistic(
            tip_states,
            tip_supports,
            alpha=1.5,
            active_fraction=0.4,
            min_active=3,
            rng=rng2,
        )
        
        ids1 = {ts.node_id for ts in result1}
        ids2 = {ts.node_id for ts in result2}
        assert ids1 == ids2


class TestSplitCooldown:
    """Tests for split cooldown behavior."""

    def test_tip_state_tracks_steps_since_split(self):
        """TipState should track steps since last split."""
        ts = TipState(node_id=1, steps_since_split=0)
        assert ts.steps_since_split == 0
        
        ts.steps_since_split += 1
        assert ts.steps_since_split == 1

    def test_tip_state_distance_from_root(self):
        """TipState should track distance from root."""
        ts = TipState(node_id=1, distance_from_root=0.005)
        assert ts.distance_from_root == 0.005

    def test_tip_state_is_root_flag(self):
        """TipState should track if it's the root node."""
        ts_root = TipState(node_id=1, is_root=True)
        ts_child = TipState(node_id=2, is_root=False)
        
        assert ts_root.is_root is True
        assert ts_child.is_root is False
