"""
Smoke tests for validity checks.

These tests validate that the validity module works correctly with the
canonical VascularNetwork API after the refactoring.
"""

import pytest


class TestGraphChecks:
    """Tests for pre-embedding graph checks."""
    
    def test_check_murrays_law_simple_network(self):
        """Test Murray's law check on a simple bifurcating network."""
        from generation.core.network import VascularNetwork, Node, VesselSegment
        from generation.core.domain import EllipsoidDomain
        from generation.core.types import Point3D, TubeGeometry
        from validity.pre_embedding.graph_checks import check_murrays_law
        
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
        
        parent_radius = 0.002
        child_radius = parent_radius * (0.5 ** (1/3))
        
        seg0 = VesselSegment(
            id=0,
            start_node_id=0,
            end_node_id=1,
            geometry=TubeGeometry(
                start=node0.position,
                end=node1.position,
                radius_start=parent_radius,
                radius_end=parent_radius,
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
                radius_start=child_radius,
                radius_end=child_radius,
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
                radius_start=child_radius,
                radius_end=child_radius,
            ),
            vessel_type="arterial",
        )
        
        network.add_segment(seg0)
        network.add_segment(seg1)
        network.add_segment(seg2)
        
        result = check_murrays_law(network, gamma=3.0, tolerance=0.15)
        
        assert result.check_name == "murrays_law"
        assert 'bifurcation_count' in result.details
        assert 'mean_deviation' in result.details
    
    def test_check_collisions_no_collisions(self):
        """Test collision check on network with no collisions."""
        from generation.core.network import VascularNetwork, Node, VesselSegment
        from generation.core.domain import EllipsoidDomain
        from generation.core.types import Point3D, TubeGeometry
        from validity.pre_embedding.graph_checks import check_collisions
        
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
        
        result = check_collisions(network, min_clearance=0.0, max_collisions=0)
        
        assert result.check_name == "collisions"
        assert result.passed is True
        assert result.details['collision_count'] == 0
    
    def test_check_self_intersections_no_intersections(self):
        """Test self-intersection check on valid network."""
        from generation.core.network import VascularNetwork, Node, VesselSegment
        from generation.core.domain import EllipsoidDomain
        from generation.core.types import Point3D, TubeGeometry
        from validity.pre_embedding.graph_checks import check_self_intersections
        
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
        
        result = check_self_intersections(network)
        
        assert result.check_name == "self_intersections"
        assert result.passed is True
        assert result.details['self_intersection_count'] == 0
    
    def test_run_all_graph_checks(self):
        """Test running all graph checks on a simple network."""
        from generation.core.network import VascularNetwork, Node, VesselSegment
        from generation.core.domain import EllipsoidDomain
        from generation.core.types import Point3D, TubeGeometry
        from validity.pre_embedding.graph_checks import run_all_graph_checks
        
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
        
        report = run_all_graph_checks(network)
        
        assert hasattr(report, 'passed')
        assert hasattr(report, 'status')
        assert hasattr(report, 'checks')
        assert hasattr(report, 'summary')
        
        assert len(report.checks) == 4
        assert report.summary['total_checks'] == 4


class TestRadiusRetrieval:
    """Tests for radius retrieval preferring geometry over attributes."""
    
    def test_collision_check_uses_geometry_radius(self):
        """Test that collision check uses geometry radius when available."""
        from generation.core.network import VascularNetwork, Node, VesselSegment
        from generation.core.domain import EllipsoidDomain
        from generation.core.types import Point3D, TubeGeometry
        from validity.pre_embedding.graph_checks import check_collisions
        
        domain = EllipsoidDomain(
            semi_axis_a=0.05,
            semi_axis_b=0.045,
            semi_axis_c=0.035,
        )
        network = VascularNetwork(domain=domain)
        
        node0 = Node(id=0, position=Point3D(0.0, 0.0, 0.0), node_type="inlet", vessel_type="arterial")
        node1 = Node(id=1, position=Point3D(0.01, 0.0, 0.0), node_type="junction", vessel_type="arterial")
        node2 = Node(id=2, position=Point3D(0.0, 0.01, 0.0), node_type="inlet", vessel_type="arterial")
        node3 = Node(id=3, position=Point3D(0.01, 0.01, 0.0), node_type="terminal", vessel_type="arterial")
        
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
                radius_start=0.001,
                radius_end=0.001,
            ),
            vessel_type="arterial",
        )
        seg1 = VesselSegment(
            id=1,
            start_node_id=2,
            end_node_id=3,
            geometry=TubeGeometry(
                start=node2.position,
                end=node3.position,
                radius_start=0.001,
                radius_end=0.001,
            ),
            vessel_type="arterial",
        )
        
        network.add_segment(seg0)
        network.add_segment(seg1)
        
        result = check_collisions(network, min_clearance=0.0, max_collisions=10)
        
        assert result.check_name == "collisions"
        assert 'collision_count' in result.details
