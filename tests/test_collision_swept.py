"""
Tests for swept-volume collision detection functions.

These tests validate:
- capsule_collision_check() for capsule-to-capsule collision detection
- check_segment_collision_swept() for network segment collision detection
- segment_segment_distance() for correct distance computation
"""

import pytest
import numpy as np


class TestSegmentSegmentDistance:
    """
    Regression tests for segment-segment distance computation.
    
    A1 FIX: These tests ensure that the canonical segment_segment_distance
    function correctly returns 0.0 for intersecting/crossing segments.
    """
    
    def test_perpendicular_crossing_segments_distance_zero(self):
        """
        REGRESSION TEST (A1): Perpendicular crossing segments must have distance == 0.
        
        This was a critical bug where the old implementation returned nonzero
        distance for segments that literally intersect, breaking collision detection.
        """
        from generation.utils.geometry import segment_segment_distance
        
        # Segment 1: along X axis from (0,0,0) to (1,0,0)
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        
        # Segment 2: along Y axis crossing at (0.5, 0, 0)
        p3 = np.array([0.5, -0.5, 0.0])
        p4 = np.array([0.5, 0.5, 0.0])
        
        dist = segment_segment_distance(p1, p2, p3, p4)
        
        assert dist == pytest.approx(0.0, abs=1e-10), \
            f"Crossing segments should have distance 0, got {dist}"
    
    def test_crossing_segments_in_3d_distance_zero(self):
        """
        REGRESSION TEST (A1): 3D crossing segments must have distance == 0.
        """
        from generation.utils.geometry import segment_segment_distance
        
        # Segment 1: diagonal in XY plane
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 1.0, 0.0])
        
        # Segment 2: crosses segment 1 at (0.5, 0.5, 0)
        p3 = np.array([0.0, 1.0, 0.0])
        p4 = np.array([1.0, 0.0, 0.0])
        
        dist = segment_segment_distance(p1, p2, p3, p4)
        
        assert dist == pytest.approx(0.0, abs=1e-10), \
            f"Crossing segments should have distance 0, got {dist}"
    
    def test_parallel_segments_nonzero_distance(self):
        """Test that parallel non-intersecting segments have correct distance."""
        from generation.utils.geometry import segment_segment_distance
        
        # Segment 1: along X axis
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        
        # Segment 2: parallel, offset by 1 in Y
        p3 = np.array([0.0, 1.0, 0.0])
        p4 = np.array([1.0, 1.0, 0.0])
        
        dist = segment_segment_distance(p1, p2, p3, p4)
        
        assert dist == pytest.approx(1.0, rel=1e-6), \
            f"Parallel segments should have distance 1.0, got {dist}"
    
    def test_skew_segments_correct_distance(self):
        """Test that skew (non-intersecting, non-parallel) segments have correct distance."""
        from generation.utils.geometry import segment_segment_distance
        
        # Segment 1: along X axis at z=0
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        
        # Segment 2: along Y axis at z=1 (skew, closest distance is 1.0)
        p3 = np.array([0.5, -0.5, 1.0])
        p4 = np.array([0.5, 0.5, 1.0])
        
        dist = segment_segment_distance(p1, p2, p3, p4)
        
        assert dist == pytest.approx(1.0, rel=1e-6), \
            f"Skew segments should have distance 1.0, got {dist}"
    
    def test_degenerate_point_segments(self):
        """Test distance between degenerate (point) segments."""
        from generation.utils.geometry import segment_segment_distance
        
        # Two point segments
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])  # Same as p1 (degenerate)
        
        p3 = np.array([1.0, 0.0, 0.0])
        p4 = np.array([1.0, 0.0, 0.0])  # Same as p3 (degenerate)
        
        dist = segment_segment_distance(p1, p2, p3, p4)
        
        assert dist == pytest.approx(1.0, rel=1e-6), \
            f"Point-to-point distance should be 1.0, got {dist}"
    
    def test_t_intersection_distance_zero(self):
        """Test T-intersection where one segment endpoint touches another segment."""
        from generation.utils.geometry import segment_segment_distance
        
        # Segment 1: along X axis
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        
        # Segment 2: perpendicular, starting at midpoint of segment 1
        p3 = np.array([0.5, 0.0, 0.0])
        p4 = np.array([0.5, 1.0, 0.0])
        
        dist = segment_segment_distance(p1, p2, p3, p4)
        
        assert dist == pytest.approx(0.0, abs=1e-10), \
            f"T-intersection should have distance 0, got {dist}"


class TestCapsuleCollisionCheck:
    """Tests for capsule_collision_check function."""
    
    def test_parallel_capsules_no_collision(self):
        """Test two parallel capsules that don't collide."""
        from generation.ops.collision import capsule_collision_check
        
        seg1_start = np.array([0.0, 0.0, 0.0])
        seg1_end = np.array([0.01, 0.0, 0.0])
        seg1_radius = 0.001
        
        seg2_start = np.array([0.0, 0.01, 0.0])
        seg2_end = np.array([0.01, 0.01, 0.0])
        seg2_radius = 0.001
        
        collides, clearance = capsule_collision_check(
            seg1_start, seg1_end, seg1_radius,
            seg2_start, seg2_end, seg2_radius,
        )
        
        assert collides == False
        assert clearance > 0
        assert clearance == pytest.approx(0.01 - 0.002, rel=0.01)
    
    def test_parallel_capsules_collision(self):
        """Test two parallel capsules that collide."""
        from generation.ops.collision import capsule_collision_check
        
        seg1_start = np.array([0.0, 0.0, 0.0])
        seg1_end = np.array([0.01, 0.0, 0.0])
        seg1_radius = 0.003
        
        seg2_start = np.array([0.0, 0.004, 0.0])
        seg2_end = np.array([0.01, 0.004, 0.0])
        seg2_radius = 0.003
        
        collides, clearance = capsule_collision_check(
            seg1_start, seg1_end, seg1_radius,
            seg2_start, seg2_end, seg2_radius,
        )
        
        assert collides == True
        assert clearance < 0
    
    def test_capsules_with_clearance_requirement(self):
        """Test capsules that pass without clearance but fail with clearance."""
        from generation.ops.collision import capsule_collision_check
        
        seg1_start = np.array([0.0, 0.0, 0.0])
        seg1_end = np.array([0.01, 0.0, 0.0])
        seg1_radius = 0.001
        
        seg2_start = np.array([0.0, 0.003, 0.0])
        seg2_end = np.array([0.01, 0.003, 0.0])
        seg2_radius = 0.001
        
        collides_no_clearance, clearance = capsule_collision_check(
            seg1_start, seg1_end, seg1_radius,
            seg2_start, seg2_end, seg2_radius,
            min_clearance=0.0,
        )
        
        collides_with_clearance, _ = capsule_collision_check(
            seg1_start, seg1_end, seg1_radius,
            seg2_start, seg2_end, seg2_radius,
            min_clearance=0.002,
        )
        
        assert collides_no_clearance == False
        assert collides_with_clearance == True
        assert clearance == pytest.approx(0.001, rel=0.01)
    
    def test_same_segment_collision(self):
        """Test that identical segments collide."""
        from generation.ops.collision import capsule_collision_check
        
        seg_start = np.array([0.0, 0.0, 0.0])
        seg_end = np.array([0.01, 0.0, 0.0])
        seg_radius = 0.001
        
        collides, clearance = capsule_collision_check(
            seg_start, seg_end, seg_radius,
            seg_start, seg_end, seg_radius,
        )
        
        assert collides == True
        assert clearance < 0
    
    def test_touching_capsules(self):
        """Test capsules that are exactly touching (clearance = 0)."""
        from generation.ops.collision import capsule_collision_check
        
        seg1_start = np.array([0.0, 0.0, 0.0])
        seg1_end = np.array([0.01, 0.0, 0.0])
        seg1_radius = 0.001
        
        seg2_start = np.array([0.0, 0.002, 0.0])
        seg2_end = np.array([0.01, 0.002, 0.0])
        seg2_radius = 0.001
        
        collides, clearance = capsule_collision_check(
            seg1_start, seg1_end, seg1_radius,
            seg2_start, seg2_end, seg2_radius,
        )
        
        assert clearance == pytest.approx(0.0, abs=1e-6)
    
    def test_clearance_calculation(self):
        """Test that clearance is calculated correctly."""
        from generation.ops.collision import capsule_collision_check
        
        seg1_start = np.array([0.0, 0.0, 0.0])
        seg1_end = np.array([0.01, 0.0, 0.0])
        seg1_radius = 0.001
        
        seg2_start = np.array([0.0, 0.005, 0.0])
        seg2_end = np.array([0.01, 0.005, 0.0])
        seg2_radius = 0.001
        
        collides, clearance = capsule_collision_check(
            seg1_start, seg1_end, seg1_radius,
            seg2_start, seg2_end, seg2_radius,
        )
        
        expected_clearance = 0.005 - 0.001 - 0.001
        assert clearance == pytest.approx(expected_clearance, rel=0.01)
        assert collides == False
    
    def test_negative_clearance_means_collision(self):
        """Test that negative clearance indicates collision."""
        from generation.ops.collision import capsule_collision_check
        
        seg1_start = np.array([0.0, 0.0, 0.0])
        seg1_end = np.array([0.01, 0.0, 0.0])
        seg1_radius = 0.003
        
        seg2_start = np.array([0.0, 0.002, 0.0])
        seg2_end = np.array([0.01, 0.002, 0.0])
        seg2_radius = 0.003
        
        collides, clearance = capsule_collision_check(
            seg1_start, seg1_end, seg1_radius,
            seg2_start, seg2_end, seg2_radius,
        )
        
        assert clearance < 0
        assert collides == True


class TestCheckSegmentCollisionSwept:
    """Tests for check_segment_collision_swept function."""
    
    def test_no_collision_in_empty_network(self):
        """Test that no collision is detected in empty network."""
        from generation.ops.collision import check_segment_collision_swept
        from generation.core.network import VascularNetwork
        from generation.core.domain import EllipsoidDomain
        
        domain = EllipsoidDomain(
            semi_axis_a=0.05,
            semi_axis_b=0.045,
            semi_axis_c=0.035,
        )
        network = VascularNetwork(domain=domain)
        
        new_seg_start = np.array([0.0, 0.0, 0.0])
        new_seg_end = np.array([0.01, 0.0, 0.0])
        new_seg_radius = 0.001
        
        has_collision, details = check_segment_collision_swept(
            network,
            new_seg_start,
            new_seg_end,
            new_seg_radius,
        )
        
        assert has_collision == False
        assert len(details) == 0
    
    def test_collision_with_parallel_segment(self):
        """Test collision detection with parallel segment that is too close."""
        from generation.ops.collision import check_segment_collision_swept
        from generation.core.network import VascularNetwork, Node, VesselSegment
        from generation.core.domain import EllipsoidDomain
        from generation.core.types import Point3D, TubeGeometry
        
        domain = EllipsoidDomain(
            semi_axis_a=0.05,
            semi_axis_b=0.045,
            semi_axis_c=0.035,
        )
        network = VascularNetwork(domain=domain)
        
        node0 = Node(id=0, position=Point3D(0.0, 0.0, 0.0), node_type="inlet", vessel_type="arterial")
        node1 = Node(id=1, position=Point3D(0.01, 0.0, 0.0), node_type="terminal", vessel_type="arterial")
        network.add_node(node0)
        network.add_node(node1)
        
        seg0 = VesselSegment(
            id=0,
            start_node_id=0,
            end_node_id=1,
            geometry=TubeGeometry(
                start=node0.position,
                end=node1.position,
                radius_start=0.003,
                radius_end=0.003,
            ),
            vessel_type="arterial",
        )
        network.add_segment(seg0)
        
        new_seg_start = np.array([0.0, 0.004, 0.0])
        new_seg_end = np.array([0.01, 0.004, 0.0])
        new_seg_radius = 0.003
        
        has_collision, details = check_segment_collision_swept(
            network,
            new_seg_start,
            new_seg_end,
            new_seg_radius,
            min_clearance=0.0,
        )
        
        assert has_collision == True
        assert len(details) > 0
        assert details[0]["segment_id"] == 0
    
    def test_no_collision_with_distant_segment(self):
        """Test no collision with segment that is far away."""
        from generation.ops.collision import check_segment_collision_swept
        from generation.core.network import VascularNetwork, Node, VesselSegment
        from generation.core.domain import EllipsoidDomain
        from generation.core.types import Point3D, TubeGeometry
        
        domain = EllipsoidDomain(
            semi_axis_a=0.05,
            semi_axis_b=0.045,
            semi_axis_c=0.035,
        )
        network = VascularNetwork(domain=domain)
        
        node0 = Node(id=0, position=Point3D(0.0, 0.0, 0.0), node_type="inlet", vessel_type="arterial")
        node1 = Node(id=1, position=Point3D(0.01, 0.0, 0.0), node_type="terminal", vessel_type="arterial")
        network.add_node(node0)
        network.add_node(node1)
        
        seg0 = VesselSegment(
            id=0,
            start_node_id=0,
            end_node_id=1,
            geometry=TubeGeometry(
                start=node0.position,
                end=node1.position,
                radius_start=0.001,
                radius_end=0.001,
            ),
            vessel_type="arterial",
        )
        network.add_segment(seg0)
        
        new_seg_start = np.array([0.0, 0.02, 0.0])
        new_seg_end = np.array([0.01, 0.02, 0.0])
        new_seg_radius = 0.001
        
        has_collision, details = check_segment_collision_swept(
            network,
            new_seg_start,
            new_seg_end,
            new_seg_radius,
            min_clearance=0.001,
        )
        
        assert has_collision == False
        assert len(details) == 0
    
    def test_exclude_node_ids(self):
        """Test that excluded node IDs are not checked for collision."""
        from generation.ops.collision import check_segment_collision_swept
        from generation.core.network import VascularNetwork, Node, VesselSegment
        from generation.core.domain import EllipsoidDomain
        from generation.core.types import Point3D, TubeGeometry
        
        domain = EllipsoidDomain(
            semi_axis_a=0.05,
            semi_axis_b=0.045,
            semi_axis_c=0.035,
        )
        network = VascularNetwork(domain=domain)
        
        node0 = Node(id=0, position=Point3D(0.0, 0.0, 0.0), node_type="inlet", vessel_type="arterial")
        node1 = Node(id=1, position=Point3D(0.01, 0.0, 0.0), node_type="terminal", vessel_type="arterial")
        network.add_node(node0)
        network.add_node(node1)
        
        seg0 = VesselSegment(
            id=0,
            start_node_id=0,
            end_node_id=1,
            geometry=TubeGeometry(
                start=node0.position,
                end=node1.position,
                radius_start=0.003,
                radius_end=0.003,
            ),
            vessel_type="arterial",
        )
        network.add_segment(seg0)
        
        new_seg_start = np.array([0.0, 0.004, 0.0])
        new_seg_end = np.array([0.01, 0.004, 0.0])
        new_seg_radius = 0.003
        
        has_collision_without_exclude, _ = check_segment_collision_swept(
            network,
            new_seg_start,
            new_seg_end,
            new_seg_radius,
            exclude_node_ids=[],
            min_clearance=0.0,
        )
        
        has_collision_with_exclude, _ = check_segment_collision_swept(
            network,
            new_seg_start,
            new_seg_end,
            new_seg_radius,
            exclude_node_ids=[0, 1],
            min_clearance=0.0,
        )
        
        assert has_collision_without_exclude == True
        assert has_collision_with_exclude == False
