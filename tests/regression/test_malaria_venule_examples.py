"""
Regression tests for malaria venule DesignSpec examples.

This module validates that the malaria venule examples grow properly and don't get stuck:
1. malaria_venule_bifurcating_tree.json - should grow downward into the cylinder
2. malaria_venule_space_colonization.json - should produce organic blended growth, not cross patterns

Acceptance criteria:
- Network has > N nodes (threshold: > 50 for basic growth)
- Max depth in -Z direction exceeds fraction of domain height (> 0.3 * height)
- For space colonization: growth not confined to single cross pattern
  - XY variance must exceed threshold (organic spread)
  - Number of degree>=3 nodes > K (branching occurred)
"""

import json
import pytest
import numpy as np
from pathlib import Path

from generation.api.generate import generate_network
from generation.specs.design_spec import DesignSpec


EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples" / "designspec"
BIFURCATING_TREE_JSON = EXAMPLES_DIR / "malaria_venule_bifurcating_tree.json"
SPACE_COLONIZATION_JSON = EXAMPLES_DIR / "malaria_venule_space_colonization.json"

MIN_NODE_COUNT = 50
MIN_DEPTH_FRACTION = 0.3
MIN_XY_VARIANCE = 0.1
MIN_BRANCHING_NODES = 3


def load_design_spec(json_path: Path) -> DesignSpec:
    """Load a DesignSpec from a JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return DesignSpec.from_dict(data)


def get_node_positions(network) -> np.ndarray:
    """Extract node positions as numpy array."""
    positions = []
    for node in network.nodes.values():
        positions.append([node.position.x, node.position.y, node.position.z])
    return np.array(positions) if positions else np.zeros((0, 3))


def get_node_degrees(network) -> dict:
    """Compute degree (number of connections) for each node."""
    degrees = {node_id: 0 for node_id in network.nodes}
    for segment in network.segments.values():
        degrees[segment.source_id] = degrees.get(segment.source_id, 0) + 1
        degrees[segment.target_id] = degrees.get(segment.target_id, 0) + 1
    return degrees


def get_domain_height(network) -> float:
    """Get the domain height from the network's domain."""
    bounds = network.domain.get_bounds()
    z_min, z_max = bounds[4], bounds[5]
    return z_max - z_min


def get_z_depth_range(positions: np.ndarray) -> tuple:
    """Get the Z depth range of node positions."""
    if len(positions) == 0:
        return 0.0, 0.0
    z_values = positions[:, 2]
    return z_values.min(), z_values.max()


def get_xy_variance(positions: np.ndarray) -> tuple:
    """Get the variance of X and Y coordinates."""
    if len(positions) < 2:
        return 0.0, 0.0
    x_var = np.var(positions[:, 0])
    y_var = np.var(positions[:, 1])
    return x_var, y_var


class TestMalariaVenuleBifurcatingTree:
    """Tests for malaria_venule_bifurcating_tree.json example."""
    
    @pytest.fixture
    def network(self):
        """Generate network from bifurcating tree example."""
        spec = load_design_spec(BIFURCATING_TREE_JSON)
        network, stats = generate_network(
            domain=spec.domains,
            ports=spec.ports,
            growth_policy=spec.growth,
            collision_policy=spec.collision,
            seed=42,
        )
        return network, stats
    
    def test_network_has_sufficient_nodes(self, network):
        """Test that network grows beyond initial inlet stub."""
        net, stats = network
        node_count = len(net.nodes)
        assert node_count > MIN_NODE_COUNT, (
            f"Network has only {node_count} nodes, expected > {MIN_NODE_COUNT}. "
            f"Tree may be stuck at inlet."
        )
    
    def test_network_grows_downward(self, network):
        """Test that network grows downward into the domain."""
        net, stats = network
        positions = get_node_positions(net)
        
        if len(positions) == 0:
            pytest.fail("Network has no nodes")
        
        domain_height = get_domain_height(net)
        z_min, z_max = get_z_depth_range(positions)
        z_span = z_max - z_min
        
        min_required_depth = MIN_DEPTH_FRACTION * domain_height
        
        assert z_span > min_required_depth, (
            f"Network Z span ({z_span:.6f}) is less than {MIN_DEPTH_FRACTION*100}% "
            f"of domain height ({domain_height:.6f}). Tree may not be growing downward."
        )
    
    def test_network_has_branching(self, network):
        """Test that network has branching (degree >= 3 nodes)."""
        net, stats = network
        degrees = get_node_degrees(net)
        
        branching_nodes = sum(1 for d in degrees.values() if d >= 3)
        
        assert branching_nodes >= MIN_BRANCHING_NODES, (
            f"Network has only {branching_nodes} branching nodes (degree >= 3), "
            f"expected >= {MIN_BRANCHING_NODES}. Tree may not be bifurcating properly."
        )


class TestMalariaVenuleSpaceColonization:
    """Tests for malaria_venule_space_colonization.json example."""
    
    @pytest.fixture
    def network(self):
        """Generate network from space colonization example."""
        spec = load_design_spec(SPACE_COLONIZATION_JSON)
        network, stats = generate_network(
            domain=spec.domains,
            ports=spec.ports,
            growth_policy=spec.growth,
            collision_policy=spec.collision,
            seed=42,
        )
        return network, stats
    
    def test_network_has_sufficient_nodes(self, network):
        """Test that network grows beyond initial inlet stubs."""
        net, stats = network
        node_count = len(net.nodes)
        assert node_count > MIN_NODE_COUNT, (
            f"Network has only {node_count} nodes, expected > {MIN_NODE_COUNT}. "
            f"Space colonization may have stopped early."
        )
    
    def test_network_grows_downward(self, network):
        """Test that network grows downward into the domain."""
        net, stats = network
        positions = get_node_positions(net)
        
        if len(positions) == 0:
            pytest.fail("Network has no nodes")
        
        domain_height = get_domain_height(net)
        z_min, z_max = get_z_depth_range(positions)
        z_span = z_max - z_min
        
        min_required_depth = MIN_DEPTH_FRACTION * domain_height
        
        assert z_span > min_required_depth, (
            f"Network Z span ({z_span:.6f}) is less than {MIN_DEPTH_FRACTION*100}% "
            f"of domain height ({domain_height:.6f}). Growth may have stopped early."
        )
    
    def test_network_not_cross_pattern(self, network):
        """Test that network is not confined to a rigid cross pattern.
        
        A cross pattern would have low variance in both X and Y because
        nodes would be clustered along the axes connecting inlets.
        Organic blended growth should have higher variance in both dimensions.
        """
        net, stats = network
        positions = get_node_positions(net)
        
        if len(positions) < 10:
            pytest.fail(f"Network has only {len(positions)} nodes, not enough for variance test")
        
        x_var, y_var = get_xy_variance(positions)
        
        assert x_var > MIN_XY_VARIANCE or y_var > MIN_XY_VARIANCE, (
            f"Network XY variance (x={x_var:.6f}, y={y_var:.6f}) is too low. "
            f"Growth may be confined to a cross pattern. Expected at least one > {MIN_XY_VARIANCE}."
        )
    
    def test_network_not_axis_concentrated(self, network):
        """Test that network nodes are not concentrated along X or Y axes.
        
        A cross pattern would have many nodes with |x| small OR |y| small
        (clustered along the axes). Organic growth should have nodes spread
        throughout the domain, not just along axis lines.
        
        This test fails if > 50% of nodes are within 0.5 units of either axis.
        """
        net, stats = network
        positions = get_node_positions(net)
        
        if len(positions) < 10:
            pytest.fail(f"Network has only {len(positions)} nodes, not enough for axis test")
        
        axis_threshold = 0.5
        max_axis_fraction = 0.5
        
        x_near_axis = np.abs(positions[:, 0]) < axis_threshold
        y_near_axis = np.abs(positions[:, 1]) < axis_threshold
        
        near_x_axis_count = np.sum(x_near_axis)
        near_y_axis_count = np.sum(y_near_axis)
        total_nodes = len(positions)
        
        near_x_axis_fraction = near_x_axis_count / total_nodes
        near_y_axis_fraction = near_y_axis_count / total_nodes
        
        assert near_x_axis_fraction < max_axis_fraction, (
            f"{near_x_axis_fraction*100:.1f}% of nodes are near X axis (|y| < {axis_threshold}), "
            f"expected < {max_axis_fraction*100:.0f}%. Growth may be cross-dominated."
        )
        
        assert near_y_axis_fraction < max_axis_fraction, (
            f"{near_y_axis_fraction*100:.1f}% of nodes are near Y axis (|x| < {axis_threshold}), "
            f"expected < {max_axis_fraction*100:.0f}%. Growth may be cross-dominated."
        )
    
    def test_network_has_branching(self, network):
        """Test that network has branching beyond immediate inlet stubs."""
        net, stats = network
        degrees = get_node_degrees(net)
        
        branching_nodes = sum(1 for d in degrees.values() if d >= 3)
        
        assert branching_nodes >= MIN_BRANCHING_NODES, (
            f"Network has only {branching_nodes} branching nodes (degree >= 3), "
            f"expected >= {MIN_BRANCHING_NODES}. Growth may have stopped at inlet stubs."
        )
    
    def test_blended_mode_used(self, network):
        """Test that blended multi-inlet mode was used (from stats)."""
        net, stats = network
        multi_inlet_mode = stats.get("multi_inlet_mode", "unknown")
        assert multi_inlet_mode == "blended", (
            f"Expected multi_inlet_mode='blended', got '{multi_inlet_mode}'. "
            f"Example may not be using the new blended mode."
        )


class TestInletDirectionValidation:
    """Tests for inlet direction validation guard."""
    
    def test_bifurcating_tree_inlet_directions_correct(self):
        """Test that bifurcating tree example has correct inlet directions."""
        with open(BIFURCATING_TREE_JSON, "r") as f:
            data = json.load(f)
        
        ports = data.get("ports", {})
        inlets = ports.get("inlets", [])
        
        for inlet in inlets:
            direction = inlet.get("direction", [0, 0, 0])
            assert direction[2] < 0, (
                f"Inlet '{inlet.get('name', 'unnamed')}' has direction[2]={direction[2]}, "
                f"expected < 0 (pointing downward/inward). Direction: {direction}"
            )
    
    def test_space_colonization_inlet_directions_correct(self):
        """Test that space colonization example has correct inlet directions."""
        with open(SPACE_COLONIZATION_JSON, "r") as f:
            data = json.load(f)
        
        ports = data.get("ports", {})
        inlets = ports.get("inlets", [])
        
        for inlet in inlets:
            direction = inlet.get("direction", [0, 0, 0])
            assert direction[2] < 0, (
                f"Inlet '{inlet.get('name', 'unnamed')}' has direction[2]={direction[2]}, "
                f"expected < 0 (pointing downward/inward). Direction: {direction}"
            )
    
    def test_space_colonization_uses_blended_mode(self):
        """Test that space colonization example is configured for blended mode."""
        with open(SPACE_COLONIZATION_JSON, "r") as f:
            data = json.load(f)
        
        growth = data.get("growth", {})
        backend_params = growth.get("backend_params", {})
        multi_inlet_mode = backend_params.get("multi_inlet_mode", "")
        
        assert multi_inlet_mode == "blended", (
            f"Expected multi_inlet_mode='blended' in space colonization example, "
            f"got '{multi_inlet_mode}'"
        )
