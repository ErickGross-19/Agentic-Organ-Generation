"""
Smoke tests for network topology operations.

These tests validate that the core network topology helpers work correctly
after the refactoring to use network.get_connected_segment_ids() instead of
node.connected_segment_ids.
"""

import pytest


class TestNetworkTopology:
    """Tests for VascularNetwork topology helpers."""
    
    def test_get_connected_segment_ids_empty_network(self):
        """Test get_connected_segment_ids on empty network."""
        from generation.core.network import VascularNetwork, Node
        from generation.core.domain import EllipsoidDomain
        from generation.core.types import Point3D
        
        domain = EllipsoidDomain(
            semi_axis_a=0.05,
            semi_axis_b=0.045,
            semi_axis_c=0.035,
        )
        network = VascularNetwork(domain=domain)
        
        node = Node(
            id=0,
            position=Point3D(0.0, 0.0, 0.0),
            node_type="inlet",
            vessel_type="arterial",
        )
        network.add_node(node)
        
        connected = network.get_connected_segment_ids(0)
        assert connected == []
    
    def test_get_connected_segment_ids_with_segments(self):
        """Test get_connected_segment_ids with connected segments."""
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
        node1 = Node(id=1, position=Point3D(0.01, 0.0, 0.0), node_type="junction", vessel_type="arterial")
        node2 = Node(id=2, position=Point3D(0.02, 0.01, 0.0), node_type="terminal", vessel_type="arterial")
        node3 = Node(id=3, position=Point3D(0.02, -0.01, 0.0), node_type="terminal", vessel_type="arterial")
        
        network.add_node(node0)
        network.add_node(node1)
        network.add_node(node2)
        network.add_node(node3)
        
        seg0 = VesselSegment(
            id=0,
            start_node_id=0,
            end_node_id=1,
            geometry=TubeGeometry(
                start=node0.position,
                end=node1.position,
                radius_start=0.002,
                radius_end=0.0015,
            ),
            vessel_type="arterial",
        )
        seg1 = VesselSegment(
            id=1,
            start_node_id=1,
            end_node_id=2,
            geometry=TubeGeometry(
                start=node1.position,
                end=node2.position,
                radius_start=0.001,
                radius_end=0.0008,
            ),
            vessel_type="arterial",
        )
        seg2 = VesselSegment(
            id=2,
            start_node_id=1,
            end_node_id=3,
            geometry=TubeGeometry(
                start=node1.position,
                end=node3.position,
                radius_start=0.001,
                radius_end=0.0008,
            ),
            vessel_type="arterial",
        )
        
        network.add_segment(seg0)
        network.add_segment(seg1)
        network.add_segment(seg2)
        
        connected_to_0 = network.get_connected_segment_ids(0)
        assert len(connected_to_0) == 1
        assert 0 in connected_to_0
        
        connected_to_1 = network.get_connected_segment_ids(1)
        assert len(connected_to_1) == 3
        assert 0 in connected_to_1
        assert 1 in connected_to_1
        assert 2 in connected_to_1
        
        connected_to_2 = network.get_connected_segment_ids(2)
        assert len(connected_to_2) == 1
        assert 1 in connected_to_2
    
    def test_vessel_segment_properties(self):
        """Test VesselSegment length, direction, and mean_radius properties."""
        from generation.core.network import VesselSegment
        from generation.core.types import Point3D, TubeGeometry
        
        geometry = TubeGeometry(
            start=Point3D(0.0, 0.0, 0.0),
            end=Point3D(0.01, 0.0, 0.0),
            radius_start=0.002,
            radius_end=0.001,
        )
        
        segment = VesselSegment(
            id=0,
            start_node_id=0,
            end_node_id=1,
            geometry=geometry,
            vessel_type="arterial",
        )
        
        assert abs(segment.length - 0.01) < 1e-10
        
        direction = segment.direction
        assert abs(direction.dx - 1.0) < 1e-10
        assert abs(direction.dy) < 1e-10
        assert abs(direction.dz) < 1e-10
        
        assert abs(segment.mean_radius - 0.0015) < 1e-10


class TestBranchStats:
    """Tests for compute_branch_stats function."""
    
    def test_compute_branch_stats_simple_network(self):
        """Test compute_branch_stats on a simple bifurcating network."""
        from generation.core.network import VascularNetwork, Node, VesselSegment
        from generation.core.domain import EllipsoidDomain
        from generation.core.types import Point3D, TubeGeometry
        from generation.analysis.structure import compute_branch_stats
        
        domain = EllipsoidDomain(
            semi_axis_a=0.05,
            semi_axis_b=0.045,
            semi_axis_c=0.035,
        )
        network = VascularNetwork(domain=domain)
        
        node0 = Node(id=0, position=Point3D(0.0, 0.0, 0.0), node_type="inlet", vessel_type="arterial")
        node1 = Node(id=1, position=Point3D(0.01, 0.0, 0.0), node_type="junction", vessel_type="arterial")
        node2 = Node(id=2, position=Point3D(0.02, 0.01, 0.0), node_type="terminal", vessel_type="arterial")
        node3 = Node(id=3, position=Point3D(0.02, -0.01, 0.0), node_type="terminal", vessel_type="arterial")
        
        network.add_node(node0)
        network.add_node(node1)
        network.add_node(node2)
        network.add_node(node3)
        
        seg0 = VesselSegment(
            id=0,
            start_node_id=0,
            end_node_id=1,
            geometry=TubeGeometry(
                start=node0.position,
                end=node1.position,
                radius_start=0.002,
                radius_end=0.0015,
            ),
            vessel_type="arterial",
        )
        seg1 = VesselSegment(
            id=1,
            start_node_id=1,
            end_node_id=2,
            geometry=TubeGeometry(
                start=node1.position,
                end=node2.position,
                radius_start=0.001,
                radius_end=0.0008,
            ),
            vessel_type="arterial",
        )
        seg2 = VesselSegment(
            id=2,
            start_node_id=1,
            end_node_id=3,
            geometry=TubeGeometry(
                start=node1.position,
                end=node3.position,
                radius_start=0.001,
                radius_end=0.0008,
            ),
            vessel_type="arterial",
        )
        
        network.add_segment(seg0)
        network.add_segment(seg1)
        network.add_segment(seg2)
        
        stats = compute_branch_stats(network)
        
        assert 'degree_histogram' in stats
        assert 'num_bifurcations' in stats
        assert 'branching_angle_distribution' in stats
        assert 'mean_branching_angle' in stats
        
        assert stats['degree_histogram'].get(1, 0) == 3
        assert stats['degree_histogram'].get(3, 0) == 1
    
    def test_compute_branch_stats_empty_network(self):
        """Test compute_branch_stats on empty network."""
        from generation.core.network import VascularNetwork
        from generation.core.domain import EllipsoidDomain
        from generation.analysis.structure import compute_branch_stats
        
        domain = EllipsoidDomain(
            semi_axis_a=0.05,
            semi_axis_b=0.045,
            semi_axis_c=0.035,
        )
        network = VascularNetwork(domain=domain)
        
        stats = compute_branch_stats(network)
        
        assert stats['degree_histogram'] == {}
        assert stats['num_bifurcations'] == 0
        assert stats['branching_angle_distribution'] == []
        assert stats['mean_branching_angle'] == 0.0
