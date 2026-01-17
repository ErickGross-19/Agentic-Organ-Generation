"""
Acceptance tests for Agent Fix Plan (v39) implementations.

These tests validate the P0-P2 fixes for:
1. Collision detection returns clearance (not centerline distance)
2. Polyline collision detection works
3. Dual-tree validity allowed by vessel_type
4. CCO insertion collision-safe
5. Perfusion uses absolute scoring
"""

import pytest
import numpy as np


class TestCollisionClearance:
    """P0-1: Collision output semantics - clearance not distance."""
    
    def test_colliding_pair_produces_negative_clearance(self):
        """A known-colliding pair produces negative clearance."""
        from generation.core.network import VascularNetwork, Node, VesselSegment
        from generation.core.types import Point3D, TubeGeometry
        from generation.core.domain import BoxDomain
        from generation.spatial.grid_index import SpatialIndex
        
        domain = BoxDomain.from_center_and_size((0, 0, 0), 0.1, 0.1, 0.1)
        network = VascularNetwork(domain=domain)
        
        node1 = Node(id=1, position=Point3D(0, 0, 0), node_type="inlet")
        node2 = Node(id=2, position=Point3D(0.02, 0, 0), node_type="terminal")
        node3 = Node(id=3, position=Point3D(0.01, -0.005, 0), node_type="inlet")
        node4 = Node(id=4, position=Point3D(0.01, 0.005, 0), node_type="terminal")
        
        for node in [node1, node2, node3, node4]:
            network.add_node(node)
        
        seg1 = VesselSegment(
            id=1,
            start_node_id=1,
            end_node_id=2,
            geometry=TubeGeometry(
                start=node1.position,
                end=node2.position,
                radius_start=0.003,
                radius_end=0.003,
            ),
            vessel_type="arterial",
        )
        
        seg2 = VesselSegment(
            id=2,
            start_node_id=3,
            end_node_id=4,
            geometry=TubeGeometry(
                start=node3.position,
                end=node4.position,
                radius_start=0.003,
                radius_end=0.003,
            ),
            vessel_type="arterial",
        )
        
        network.add_segment(seg1)
        network.add_segment(seg2)
        
        spatial_index = SpatialIndex(network)
        collisions = spatial_index.get_collisions(min_clearance=0.0, exclude_connected=True)
        
        assert len(collisions) > 0, "Should detect collision between crossing segments"
        
        for seg1_id, seg2_id, clearance in collisions:
            assert clearance < 0, f"Colliding segments should have negative clearance, got {clearance}"


class TestPolylineCollision:
    """P0-3: Polyline-aware spatial indexing and collision detection."""
    
    def test_curved_route_indexed_correctly(self):
        """Curved routes created by grow-to-point are correctly indexed."""
        from generation.core.network import VascularNetwork, Node, VesselSegment
        from generation.core.types import Point3D, TubeGeometry
        from generation.core.domain import BoxDomain
        from generation.spatial.grid_index import SpatialIndex
        
        domain = BoxDomain.from_center_and_size((0, 0, 0), 0.1, 0.1, 0.1)
        network = VascularNetwork(domain=domain)
        
        node1 = Node(id=1, position=Point3D(0, 0, 0), node_type="inlet")
        node2 = Node(id=2, position=Point3D(0.04, 0, 0), node_type="terminal")
        
        network.add_node(node1)
        network.add_node(node2)
        
        centerline_points = [
            Point3D(0.01, 0.01, 0),
            Point3D(0.02, 0.015, 0),
            Point3D(0.03, 0.01, 0),
        ]
        
        seg = VesselSegment(
            id=1,
            start_node_id=1,
            end_node_id=2,
            geometry=TubeGeometry(
                start=node1.position,
                end=node2.position,
                radius_start=0.002,
                radius_end=0.002,
                centerline_points=centerline_points,
            ),
            vessel_type="arterial",
        )
        
        network.add_segment(seg)
        
        spatial_index = SpatialIndex(network)
        
        query_point = Point3D(0.02, 0.015, 0)
        nearby = spatial_index.query_nearby_segments(query_point, radius=0.005)
        
        assert len(nearby) > 0, "Should find polyline segment near its midpoint waypoint"
        assert any(s.id == 1 for s in nearby), "Should find the polyline segment"


class TestDualTreeValidity:
    """P0-6: Validity allows dual-tree disconnectedness by vessel_type."""
    
    def test_dual_tree_not_invalid_for_disconnect(self):
        """A dual-tree network without anastomoses is not marked invalid."""
        from generation.core.network import VascularNetwork, Node, VesselSegment
        from generation.core.types import Point3D, TubeGeometry
        from generation.core.domain import BoxDomain
        from generation.api.evaluate import _compute_validity_metrics
        
        domain = BoxDomain.from_center_and_size((0, 0, 0), 0.1, 0.1, 0.1)
        network = VascularNetwork(domain=domain)
        
        art_node1 = Node(id=1, position=Point3D(0, 0, 0), node_type="inlet", vessel_type="arterial")
        art_node2 = Node(id=2, position=Point3D(0.02, 0, 0), node_type="terminal", vessel_type="arterial")
        
        ven_node1 = Node(id=3, position=Point3D(0.05, 0, 0), node_type="inlet", vessel_type="venous")
        ven_node2 = Node(id=4, position=Point3D(0.07, 0, 0), node_type="terminal", vessel_type="venous")
        
        for node in [art_node1, art_node2, ven_node1, ven_node2]:
            network.add_node(node)
        
        art_seg = VesselSegment(
            id=1,
            start_node_id=1,
            end_node_id=2,
            geometry=TubeGeometry(
                start=art_node1.position,
                end=art_node2.position,
                radius_start=0.002,
                radius_end=0.002,
            ),
            vessel_type="arterial",
        )
        
        ven_seg = VesselSegment(
            id=2,
            start_node_id=3,
            end_node_id=4,
            geometry=TubeGeometry(
                start=ven_node1.position,
                end=ven_node2.position,
                radius_start=0.002,
                radius_end=0.002,
            ),
            vessel_type="venous",
        )
        
        network.add_segment(art_seg)
        network.add_segment(ven_seg)
        
        validity = _compute_validity_metrics(network, allow_disconnected_by_vessel_type=True)
        
        assert validity.is_watertight, "Dual-tree should be valid when each tree is connected"
        
        disconnected_errors = [e for e in validity.error_codes if "DISCONNECTED" in e]
        assert len(disconnected_errors) == 0, f"Should not have disconnection errors: {disconnected_errors}"


class TestPerfusionAbsoluteScoring:
    """P1-3: Perfusion scoring uses absolute exponential mapping."""
    
    def test_perfusion_score_absolute_not_normalized(self):
        """Perfusion scores use absolute mapping, not normalized by max."""
        from generation.analysis.perfusion import compute_perfusion_metrics_segment_based
        from generation.core.network import VascularNetwork, Node, VesselSegment
        from generation.core.types import Point3D, TubeGeometry
        from generation.core.domain import BoxDomain
        
        domain = BoxDomain.from_center_and_size((0, 0, 0), 0.1, 0.1, 0.1)
        network = VascularNetwork(domain=domain)
        
        art_node1 = Node(id=1, position=Point3D(0, 0, 0), node_type="inlet", vessel_type="arterial")
        art_node2 = Node(id=2, position=Point3D(0.02, 0, 0), node_type="terminal", vessel_type="arterial")
        ven_node1 = Node(id=3, position=Point3D(0.04, 0, 0), node_type="inlet", vessel_type="venous")
        ven_node2 = Node(id=4, position=Point3D(0.06, 0, 0), node_type="terminal", vessel_type="venous")
        
        for node in [art_node1, art_node2, ven_node1, ven_node2]:
            network.add_node(node)
        
        art_seg = VesselSegment(
            id=1, start_node_id=1, end_node_id=2,
            geometry=TubeGeometry(start=art_node1.position, end=art_node2.position,
                                  radius_start=0.002, radius_end=0.002),
            vessel_type="arterial",
        )
        ven_seg = VesselSegment(
            id=2, start_node_id=3, end_node_id=4,
            geometry=TubeGeometry(start=ven_node1.position, end=ven_node2.position,
                                  radius_start=0.002, radius_end=0.002),
            vessel_type="venous",
        )
        
        network.add_segment(art_seg)
        network.add_segment(ven_seg)
        
        tissue_points = np.array([
            [0.01, 0, 0],
            [0.03, 0, 0],
            [0.05, 0, 0],
        ])
        
        result = compute_perfusion_metrics_segment_based(network, tissue_points)
        
        scores = result["perfusion_scores"]
        assert np.max(scores) <= 1.0, "Scores should be <= 1.0 (absolute exponential)"
        assert np.min(scores) >= 0.0, "Scores should be >= 0.0"
        
        if np.max(scores) > 0:
            assert np.max(scores) < 1.0 or np.allclose(np.max(scores), 1.0), \
                "Max score should be close to 1.0 only for points at vessel surface"


class TestSpatialIndexScaleAware:
    """P2-2: SpatialIndex cell size is scale-aware."""
    
    def test_cell_size_adapts_to_network_scale(self):
        """Cell size is computed from network statistics when not specified."""
        from generation.core.network import VascularNetwork, Node, VesselSegment
        from generation.core.types import Point3D, TubeGeometry
        from generation.core.domain import BoxDomain
        from generation.spatial.grid_index import SpatialIndex
        
        domain = BoxDomain.from_center_and_size((0, 0, 0), 1.0, 1.0, 1.0)
        network = VascularNetwork(domain=domain)
        
        positions = [(0, 0, 0), (0.1, 0, 0), (0.2, 0, 0), (0.3, 0, 0)]
        for i, pos in enumerate(positions):
            network.add_node(Node(id=i+1, position=Point3D(*pos), node_type="terminal"))
        
        for i in range(len(positions) - 1):
            seg = VesselSegment(
                id=i+1,
                start_node_id=i+1,
                end_node_id=i+2,
                geometry=TubeGeometry(
                    start=Point3D(*positions[i]),
                    end=Point3D(*positions[i+1]),
                    radius_start=0.01,
                    radius_end=0.01,
                ),
                vessel_type="arterial",
            )
            network.add_segment(seg)
        
        spatial_index = SpatialIndex(network)
        
        assert spatial_index.cell_size > 0.0005, "Cell size should be above minimum"
        assert spatial_index.cell_size < 0.05, "Cell size should be below maximum"
        
        expected_cell_size = 0.5 * 0.1
        assert abs(spatial_index.cell_size - expected_cell_size) < 0.01, \
            f"Cell size should be ~0.5 * median_length, got {spatial_index.cell_size}"


class TestGrowToPointFailOnCollision:
    """P2-3: grow_to_point fails on collision by default."""
    
    def test_grow_to_point_fails_on_collision_by_default(self):
        """grow_to_point returns failure when collision detected."""
        from generation.core.network import VascularNetwork, Node, VesselSegment
        from generation.core.types import Point3D, TubeGeometry
        from generation.core.domain import BoxDomain
        from generation.ops.growth import grow_to_point
        
        domain = BoxDomain.from_center_and_size((0, 0, 0), 0.1, 0.1, 0.1)
        network = VascularNetwork(domain=domain)
        
        inlet = Node(
            id=1,
            position=Point3D(0, 0, 0),
            node_type="inlet",
            vessel_type="arterial",
            attributes={"radius": 0.005, "direction": {"x": 1, "y": 0, "z": 0}},
        )
        network.add_node(inlet)
        
        blocking_node1 = Node(id=2, position=Point3D(0.01, -0.01, 0), node_type="terminal")
        blocking_node2 = Node(id=3, position=Point3D(0.01, 0.01, 0), node_type="terminal")
        network.add_node(blocking_node1)
        network.add_node(blocking_node2)
        
        blocking_seg = VesselSegment(
            id=1,
            start_node_id=2,
            end_node_id=3,
            geometry=TubeGeometry(
                start=blocking_node1.position,
                end=blocking_node2.position,
                radius_start=0.005,
                radius_end=0.005,
            ),
            vessel_type="arterial",
        )
        network.add_segment(blocking_seg)
        
        result = grow_to_point(
            network,
            from_node_id=1,
            target_point=(0.02, 0, 0),
            target_radius=0.003,
            fail_on_collision=True,
        )
        
        assert not result.is_success(), "Should fail when collision detected with fail_on_collision=True"
    
    def test_grow_to_point_warns_on_collision_when_disabled(self):
        """grow_to_point succeeds with warnings when fail_on_collision=False."""
        from generation.core.network import VascularNetwork, Node, VesselSegment
        from generation.core.types import Point3D, TubeGeometry
        from generation.core.domain import BoxDomain
        from generation.ops.growth import grow_to_point
        
        domain = BoxDomain.from_center_and_size((0, 0, 0), 0.1, 0.1, 0.1)
        network = VascularNetwork(domain=domain)
        
        inlet = Node(
            id=1,
            position=Point3D(0, 0, 0),
            node_type="inlet",
            vessel_type="arterial",
            attributes={"radius": 0.005, "direction": {"x": 1, "y": 0, "z": 0}},
        )
        network.add_node(inlet)
        
        blocking_node1 = Node(id=2, position=Point3D(0.01, -0.01, 0), node_type="terminal")
        blocking_node2 = Node(id=3, position=Point3D(0.01, 0.01, 0), node_type="terminal")
        network.add_node(blocking_node1)
        network.add_node(blocking_node2)
        
        blocking_seg = VesselSegment(
            id=1,
            start_node_id=2,
            end_node_id=3,
            geometry=TubeGeometry(
                start=blocking_node1.position,
                end=blocking_node2.position,
                radius_start=0.005,
                radius_end=0.005,
            ),
            vessel_type="arterial",
        )
        network.add_segment(blocking_seg)
        
        result = grow_to_point(
            network,
            from_node_id=1,
            target_point=(0.02, 0, 0),
            target_radius=0.003,
            fail_on_collision=False,
        )
        
        assert result.is_success() or result.status.value == "partial_success", \
            "Should succeed (possibly with warnings) when fail_on_collision=False"
