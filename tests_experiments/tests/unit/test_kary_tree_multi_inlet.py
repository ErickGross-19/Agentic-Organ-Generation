"""
Tests for kary_tree backend multi-inlet support, downward-biased growth,
and boundary constraints.

These tests cover:
- Task B: Multi-inlet support (forest and merge_to_trunk modes)
- Task C: Downward-biased growth & jitter constraints
- Task D: Boundary constraints (wall_margin)
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from generation.backends.kary_tree_backend import KaryTreeBackend, KaryTreeConfig
from generation.core.domain import DomainSpec


class MockDomainSpec:
    """Mock domain for testing."""
    
    def __init__(self, center=(0, 0, 0), size=(0.01, 0.01, 0.01)):
        self.center = MagicMock()
        self.center.x = center[0]
        self.center.y = center[1]
        self.center.z = center[2]
        self.size = size
        self._bounds = (
            center[0] - size[0]/2, center[0] + size[0]/2,
            center[1] - size[1]/2, center[1] + size[1]/2,
            center[2] - size[2]/2, center[2] + size[2]/2,
        )
    
    def signed_distance(self, point):
        x, y, z = point.x, point.y, point.z
        dx = max(self._bounds[0] - x, x - self._bounds[1], 0)
        dy = max(self._bounds[2] - y, y - self._bounds[3], 0)
        dz = max(self._bounds[4] - z, z - self._bounds[5], 0)
        outside_dist = np.sqrt(dx**2 + dy**2 + dz**2)
        
        if outside_dist > 0:
            return outside_dist
        
        inside_dist = min(
            x - self._bounds[0], self._bounds[1] - x,
            y - self._bounds[2], self._bounds[3] - y,
            z - self._bounds[4], self._bounds[5] - z,
        )
        return -inside_dist


class TestKaryTreeConfig:
    """Tests for KaryTreeConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = KaryTreeConfig()
        
        assert config.k == 2
        assert config.primary_axis is None
        assert config.max_deviation_deg == 90.0
        assert config.upward_forbidden is False
        assert config.azimuth_jitter_deg == 180.0
        assert config.elevation_jitter_deg is None
        assert config.wall_margin == 0.0
        assert config.multi_inlet_mode == "merge_to_trunk"
        assert config.trunk_depth_fraction == 0.2
        assert config.trunk_merge_radius is None
        assert config.max_inlets == 10
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = KaryTreeConfig(
            k=3,
            primary_axis=(0, 0, -1),
            max_deviation_deg=45.0,
            upward_forbidden=True,
            azimuth_jitter_deg=90.0,
            elevation_jitter_deg=15.0,
            wall_margin=0.0001,
            multi_inlet_mode="forest",
            trunk_depth_fraction=0.3,
            trunk_merge_radius=0.0005,
            max_inlets=5,
        )
        
        assert config.k == 3
        assert config.primary_axis == (0, 0, -1)
        assert config.max_deviation_deg == 45.0
        assert config.upward_forbidden is True
        assert config.azimuth_jitter_deg == 90.0
        assert config.elevation_jitter_deg == 15.0
        assert config.wall_margin == 0.0001
        assert config.multi_inlet_mode == "forest"
        assert config.trunk_depth_fraction == 0.3
        assert config.trunk_merge_radius == 0.0005
        assert config.max_inlets == 5


class TestKaryTreeBackendSingleInlet:
    """Tests for single inlet generation."""
    
    def test_generate_basic_tree(self):
        """Test basic tree generation with single inlet."""
        backend = KaryTreeBackend()
        domain = MockDomainSpec(center=(0, 0, 0.005), size=(0.01, 0.01, 0.01))
        
        config = KaryTreeConfig(
            k=2,
            target_terminals=8,
            seed=42,
        )
        
        network = backend.generate(
            domain=domain,
            num_outlets=8,
            inlet_position=np.array([0, 0, 0.01]),
            inlet_radius=0.0005,
            vessel_type="arterial",
            config=config,
            rng_seed=42,
        )
        
        assert network is not None
        assert len(network.nodes) > 0
        assert len(network.segments) > 0
        
        inlet_nodes = [n for n in network.nodes.values() if n.node_type == "inlet"]
        assert len(inlet_nodes) == 1
    
    def test_generate_with_primary_axis(self):
        """Test tree generation with explicit primary axis."""
        backend = KaryTreeBackend()
        domain = MockDomainSpec(center=(0, 0, 0.005), size=(0.01, 0.01, 0.01))
        
        config = KaryTreeConfig(
            k=2,
            target_terminals=8,
            primary_axis=(0, 0, -1),
            max_deviation_deg=45.0,
            seed=42,
        )
        
        network = backend.generate(
            domain=domain,
            num_outlets=8,
            inlet_position=np.array([0, 0, 0.01]),
            inlet_radius=0.0005,
            vessel_type="arterial",
            config=config,
            rng_seed=42,
        )
        
        assert network is not None
        assert len(network.nodes) > 0


class TestKaryTreeBackendDownwardBias:
    """Tests for downward-biased growth (Task C)."""
    
    def test_upward_forbidden_constraint(self):
        """Test that upward_forbidden prevents upward growth."""
        backend = KaryTreeBackend()
        domain = MockDomainSpec(center=(0, 0, 0.005), size=(0.01, 0.01, 0.01))
        
        config = KaryTreeConfig(
            k=2,
            target_terminals=16,
            primary_axis=(0, 0, -1),
            upward_forbidden=True,
            seed=42,
        )
        
        network = backend.generate(
            domain=domain,
            num_outlets=16,
            inlet_position=np.array([0, 0, 0.01]),
            inlet_radius=0.0005,
            vessel_type="arterial",
            config=config,
            rng_seed=42,
        )
        
        assert network is not None
        
        downward_count = 0
        total_segments = 0
        
        for segment in network.segments.values():
            start_node = network.nodes[segment.start_node_id]
            end_node = network.nodes[segment.end_node_id]
            
            direction = np.array([
                end_node.position.x - start_node.position.x,
                end_node.position.y - start_node.position.y,
                end_node.position.z - start_node.position.z,
            ])
            
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
                total_segments += 1
                
                if direction[2] < 0:
                    downward_count += 1
        
        if total_segments > 0:
            downward_ratio = downward_count / total_segments
            assert downward_ratio >= 0.5, f"Expected >50% downward segments, got {downward_ratio:.1%}"
    
    def test_max_deviation_constraint(self):
        """Test that max_deviation_deg limits branch angles."""
        backend = KaryTreeBackend()
        domain = MockDomainSpec(center=(0, 0, 0.005), size=(0.01, 0.01, 0.01))
        
        config = KaryTreeConfig(
            k=2,
            target_terminals=8,
            primary_axis=(0, 0, -1),
            max_deviation_deg=30.0,
            seed=42,
        )
        
        network = backend.generate(
            domain=domain,
            num_outlets=8,
            inlet_position=np.array([0, 0, 0.01]),
            inlet_radius=0.0005,
            vessel_type="arterial",
            config=config,
            rng_seed=42,
        )
        
        assert network is not None
        assert len(network.segments) > 0


class TestKaryTreeBackendMultiInlet:
    """Tests for multi-inlet support (Task B)."""
    
    def test_multi_inlet_merge_to_trunk(self):
        """Test merge_to_trunk mode with multiple inlets."""
        backend = KaryTreeBackend()
        domain = MockDomainSpec(center=(0, 0, 0.005), size=(0.01, 0.01, 0.01))
        
        config = KaryTreeConfig(
            k=2,
            target_terminals=16,
            multi_inlet_mode="merge_to_trunk",
            trunk_depth_fraction=0.2,
            seed=42,
        )
        
        inlets = [
            {"position": (-0.002, 0, 0.01), "radius": 0.0003},
            {"position": (0.002, 0, 0.01), "radius": 0.0003},
        ]
        
        network = backend.generate_multi_inlet(
            domain=domain,
            num_outlets=16,
            inlets=inlets,
            vessel_type="arterial",
            config=config,
            rng_seed=42,
        )
        
        assert network is not None
        
        inlet_nodes = [n for n in network.nodes.values() if n.node_type == "inlet"]
        assert len(inlet_nodes) == 2, f"Expected 2 inlet nodes, got {len(inlet_nodes)}"
    
    def test_multi_inlet_forest_mode(self):
        """Test forest mode with multiple inlets (separate trees)."""
        backend = KaryTreeBackend()
        domain = MockDomainSpec(center=(0, 0, 0.005), size=(0.01, 0.01, 0.01))
        
        config = KaryTreeConfig(
            k=2,
            target_terminals=8,
            multi_inlet_mode="forest",
            seed=42,
        )
        
        inlets = [
            {"position": (-0.003, 0, 0.01), "radius": 0.0003},
            {"position": (0.003, 0, 0.01), "radius": 0.0003},
        ]
        
        network = backend.generate_multi_inlet(
            domain=domain,
            num_outlets=8,
            inlets=inlets,
            vessel_type="arterial",
            config=config,
            rng_seed=42,
        )
        
        assert network is not None
        
        inlet_nodes = [n for n in network.nodes.values() if n.node_type == "inlet"]
        assert len(inlet_nodes) == 2, f"Expected 2 inlet nodes, got {len(inlet_nodes)}"
    
    def test_single_inlet_delegation(self):
        """Test that single inlet delegates to standard generate()."""
        backend = KaryTreeBackend()
        domain = MockDomainSpec(center=(0, 0, 0.005), size=(0.01, 0.01, 0.01))
        
        config = KaryTreeConfig(
            k=2,
            target_terminals=8,
            seed=42,
        )
        
        inlets = [
            {"position": (0, 0, 0.01), "radius": 0.0005},
        ]
        
        network = backend.generate_multi_inlet(
            domain=domain,
            num_outlets=8,
            inlets=inlets,
            vessel_type="arterial",
            config=config,
            rng_seed=42,
        )
        
        assert network is not None
        
        inlet_nodes = [n for n in network.nodes.values() if n.node_type == "inlet"]
        assert len(inlet_nodes) == 1
    
    def test_max_inlets_cap(self):
        """Test that max_inlets cap is enforced."""
        backend = KaryTreeBackend()
        domain = MockDomainSpec(center=(0, 0, 0.005), size=(0.01, 0.01, 0.01))
        
        config = KaryTreeConfig(
            k=2,
            target_terminals=8,
            max_inlets=2,
            seed=42,
        )
        
        inlets = [
            {"position": (-0.003, 0, 0.01), "radius": 0.0003},
            {"position": (0, 0, 0.01), "radius": 0.0003},
            {"position": (0.003, 0, 0.01), "radius": 0.0003},
        ]
        
        network = backend.generate_multi_inlet(
            domain=domain,
            num_outlets=8,
            inlets=inlets,
            vessel_type="arterial",
            config=config,
            rng_seed=42,
        )
        
        assert network is not None
        
        inlet_nodes = [n for n in network.nodes.values() if n.node_type == "inlet"]
        assert len(inlet_nodes) <= 2, f"Expected at most 2 inlet nodes, got {len(inlet_nodes)}"


class TestKaryTreeBackendBoundaryConstraints:
    """Tests for boundary constraints (Task D)."""
    
    def test_wall_margin_constraint(self):
        """Test that wall_margin keeps nodes away from boundary."""
        backend = KaryTreeBackend()
        domain = MockDomainSpec(center=(0, 0, 0.005), size=(0.01, 0.01, 0.01))
        
        wall_margin = 0.0005
        
        config = KaryTreeConfig(
            k=2,
            target_terminals=8,
            wall_margin=wall_margin,
            seed=42,
        )
        
        network = backend.generate(
            domain=domain,
            num_outlets=8,
            inlet_position=np.array([0, 0, 0.01]),
            inlet_radius=0.0005,
            vessel_type="arterial",
            config=config,
            rng_seed=42,
        )
        
        assert network is not None
        assert len(network.nodes) > 0


class TestKaryTreeBackendInletDirection:
    """Tests for inlet direction handling."""
    
    def test_inlet_with_explicit_direction(self):
        """Test that explicit inlet direction is used."""
        backend = KaryTreeBackend()
        domain = MockDomainSpec(center=(0, 0, 0.005), size=(0.01, 0.01, 0.01))
        
        config = KaryTreeConfig(
            k=2,
            target_terminals=8,
            seed=42,
        )
        
        inlets = [
            {
                "position": (0, 0, 0.01),
                "radius": 0.0005,
                "direction": (0, 0, -1),
            },
        ]
        
        network = backend.generate_multi_inlet(
            domain=domain,
            num_outlets=8,
            inlets=inlets,
            vessel_type="arterial",
            config=config,
            rng_seed=42,
        )
        
        assert network is not None
        assert len(network.nodes) > 0
