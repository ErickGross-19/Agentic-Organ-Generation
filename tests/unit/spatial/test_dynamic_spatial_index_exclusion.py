"""
Unit tests for DynamicSpatialIndex segment ID-based adjacency exclusion.

These tests verify that exclude_segment_ids works correctly for deterministic
adjacency exclusion, replacing the fragile coordinate-based exclusion approach.
"""

import pytest
import numpy as np
from generation.spatial.grid_index import DynamicSpatialIndex


class TestDynamicSpatialIndexSegmentIDExclusion:
    """Tests for segment ID-based adjacency exclusion in DynamicSpatialIndex."""
    
    def test_collision_detected_without_exclusion(self):
        """
        Without exclude_segment_ids, collision should be detected when
        segments share a node and their radii + buffer overlap.
        """
        index = DynamicSpatialIndex(cell_size=0.1)
        
        # Insert segment A from (0,0,0) to (1,0,0) with radius 0.1
        seg_a_id = 1
        seg_a_start = np.array([0.0, 0.0, 0.0])
        seg_a_end = np.array([1.0, 0.0, 0.0])
        seg_a_radius = 0.1
        index.insert_segment(seg_a_id, seg_a_start, seg_a_end, seg_a_radius)
        
        # Propose segment B sharing node with A at (1,0,0)
        # Goes from (1,0,0) to (2,0,0) with radius 0.1
        seg_b_start = np.array([1.0, 0.0, 0.0])
        seg_b_end = np.array([2.0, 0.0, 0.0])
        seg_b_radius = 0.1
        buffer = 0.05
        
        # Without exclusion, collision should be detected at the shared node
        # because the capsules overlap at (1,0,0)
        collision = index.check_capsule_collision(
            start=seg_b_start,
            end=seg_b_end,
            radius=seg_b_radius,
            buffer=buffer,
            exclude_segment_ids=None,
        )
        
        assert collision is True, "Collision should be detected without exclusion"
    
    def test_no_collision_with_segment_id_exclusion(self):
        """
        With exclude_segment_ids={A}, collision at shared node should not
        be counted because segment A is excluded.
        """
        index = DynamicSpatialIndex(cell_size=0.1)
        
        # Insert segment A from (0,0,0) to (1,0,0) with radius 0.1
        seg_a_id = 1
        seg_a_start = np.array([0.0, 0.0, 0.0])
        seg_a_end = np.array([1.0, 0.0, 0.0])
        seg_a_radius = 0.1
        index.insert_segment(seg_a_id, seg_a_start, seg_a_end, seg_a_radius)
        
        # Propose segment B sharing node with A at (1,0,0)
        seg_b_start = np.array([1.0, 0.0, 0.0])
        seg_b_end = np.array([2.0, 0.0, 0.0])
        seg_b_radius = 0.1
        buffer = 0.05
        
        # With exclusion of segment A, no collision should be detected
        collision = index.check_capsule_collision(
            start=seg_b_start,
            end=seg_b_end,
            radius=seg_b_radius,
            buffer=buffer,
            exclude_segment_ids={seg_a_id},
        )
        
        assert collision is False, "No collision should be detected with segment A excluded"
    
    def test_collision_with_non_adjacent_segment(self):
        """
        Collision with a non-adjacent segment should still be detected
        even when excluding the parent segment.
        """
        index = DynamicSpatialIndex(cell_size=0.1)
        
        # Insert segment A (parent) from (0,0,0) to (1,0,0)
        seg_a_id = 1
        index.insert_segment(seg_a_id, np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 0.1)
        
        # Insert segment C (non-adjacent) from (1.5,0,0) to (2.5,0,0)
        seg_c_id = 3
        index.insert_segment(seg_c_id, np.array([1.5, 0.0, 0.0]), np.array([2.5, 0.0, 0.0]), 0.1)
        
        # Propose segment B from (1,0,0) to (2,0,0) - shares node with A, collides with C
        seg_b_start = np.array([1.0, 0.0, 0.0])
        seg_b_end = np.array([2.0, 0.0, 0.0])
        seg_b_radius = 0.1
        buffer = 0.05
        
        # With exclusion of segment A only, collision with C should still be detected
        collision = index.check_capsule_collision(
            start=seg_b_start,
            end=seg_b_end,
            radius=seg_b_radius,
            buffer=buffer,
            exclude_segment_ids={seg_a_id},
        )
        
        assert collision is True, "Collision with non-adjacent segment C should be detected"
    
    def test_multiple_segment_exclusion(self):
        """
        Multiple segments can be excluded from collision checks.
        """
        index = DynamicSpatialIndex(cell_size=0.1)
        
        # Insert segment A from (0,0,0) to (1,0,0)
        seg_a_id = 1
        index.insert_segment(seg_a_id, np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 0.1)
        
        # Insert segment B from (1,0,0) to (1,1,0) (shares node with A)
        seg_b_id = 2
        index.insert_segment(seg_b_id, np.array([1.0, 0.0, 0.0]), np.array([1.0, 1.0, 0.0]), 0.1)
        
        # Propose segment C from (1,0,0) to (2,0,0) - shares node with both A and B
        seg_c_start = np.array([1.0, 0.0, 0.0])
        seg_c_end = np.array([2.0, 0.0, 0.0])
        seg_c_radius = 0.1
        buffer = 0.05
        
        # Without exclusion, collision should be detected
        collision_no_exclude = index.check_capsule_collision(
            start=seg_c_start,
            end=seg_c_end,
            radius=seg_c_radius,
            buffer=buffer,
            exclude_segment_ids=None,
        )
        assert collision_no_exclude is True
        
        # With exclusion of only A, collision with B should still be detected
        collision_exclude_a = index.check_capsule_collision(
            start=seg_c_start,
            end=seg_c_end,
            radius=seg_c_radius,
            buffer=buffer,
            exclude_segment_ids={seg_a_id},
        )
        assert collision_exclude_a is True, "Collision with B should be detected when only A excluded"
        
        # With exclusion of both A and B, no collision should be detected
        collision_exclude_both = index.check_capsule_collision(
            start=seg_c_start,
            end=seg_c_end,
            radius=seg_c_radius,
            buffer=buffer,
            exclude_segment_ids={seg_a_id, seg_b_id},
        )
        assert collision_exclude_both is False, "No collision when both A and B excluded"


class TestPolylineCollisionSegmentIDExclusion:
    """Tests for segment ID-based exclusion in check_polyline_collision."""
    
    def test_polyline_collision_with_exclusion(self):
        """
        Polyline collision check should respect exclude_segment_ids.
        """
        index = DynamicSpatialIndex(cell_size=0.1)
        
        # Insert parent segment
        parent_id = 1
        index.insert_segment(parent_id, np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 0.1)
        
        # Polyline starting from parent endpoint
        polyline_points = [
            np.array([1.0, 0.0, 0.0]),
            np.array([1.5, 0.1, 0.0]),
            np.array([2.0, 0.0, 0.0]),
        ]
        
        # Without exclusion, collision at shared node
        collision_no_exclude = index.check_polyline_collision(
            points=polyline_points,
            radius=0.1,
            buffer=0.05,
            exclude_segment_ids=None,
        )
        assert collision_no_exclude is True
        
        # With parent excluded, no collision
        collision_with_exclude = index.check_polyline_collision(
            points=polyline_points,
            radius=0.1,
            buffer=0.05,
            exclude_segment_ids={parent_id},
        )
        assert collision_with_exclude is False
    
    def test_polyline_collision_detects_non_excluded_segments(self):
        """
        Polyline collision should still detect collisions with non-excluded segments.
        """
        index = DynamicSpatialIndex(cell_size=0.1)
        
        # Insert parent segment
        parent_id = 1
        index.insert_segment(parent_id, np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 0.1)
        
        # Insert obstacle segment in the path
        obstacle_id = 2
        index.insert_segment(obstacle_id, np.array([1.5, -0.5, 0.0]), np.array([1.5, 0.5, 0.0]), 0.1)
        
        # Polyline that passes through obstacle
        polyline_points = [
            np.array([1.0, 0.0, 0.0]),
            np.array([1.5, 0.0, 0.0]),
            np.array([2.0, 0.0, 0.0]),
        ]
        
        # With parent excluded, collision with obstacle should still be detected
        collision = index.check_polyline_collision(
            points=polyline_points,
            radius=0.1,
            buffer=0.05,
            exclude_segment_ids={parent_id},
        )
        assert collision is True, "Collision with obstacle should be detected"


class TestSegmentIDExclusionDeterminism:
    """Tests to verify segment ID-based exclusion is deterministic."""
    
    def test_exclusion_is_deterministic(self):
        """
        Segment ID-based exclusion should produce consistent results
        regardless of insertion order or floating-point variations.
        """
        # Run the same test multiple times to verify determinism
        for _ in range(10):
            index = DynamicSpatialIndex(cell_size=0.1)
            
            # Insert segment with exact coordinates
            seg_id = 42
            index.insert_segment(
                seg_id,
                np.array([0.0, 0.0, 0.0]),
                np.array([1.0, 0.0, 0.0]),
                0.1
            )
            
            # Check collision with exclusion
            collision = index.check_capsule_collision(
                start=np.array([1.0, 0.0, 0.0]),
                end=np.array([2.0, 0.0, 0.0]),
                radius=0.1,
                buffer=0.05,
                exclude_segment_ids={seg_id},
            )
            
            # Should consistently return False
            assert collision is False, "Exclusion should be deterministic"
    
    def test_exclusion_with_small_coordinate_variations(self):
        """
        Segment ID-based exclusion should work even with small coordinate variations
        that would break coordinate-based exclusion.
        """
        index = DynamicSpatialIndex(cell_size=0.1)
        
        # Insert segment
        seg_id = 1
        index.insert_segment(
            seg_id,
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            0.1
        )
        
        # Check collision with slightly different coordinates (simulating floating-point issues)
        # This would fail with coordinate-based exclusion using tight tolerance
        collision = index.check_capsule_collision(
            start=np.array([1.0 + 1e-10, 0.0, 0.0]),  # Tiny variation
            end=np.array([2.0, 0.0, 0.0]),
            radius=0.1,
            buffer=0.05,
            exclude_segment_ids={seg_id},
        )
        
        # Should still exclude correctly based on ID
        assert collision is False, "ID-based exclusion should work despite coordinate variations"
