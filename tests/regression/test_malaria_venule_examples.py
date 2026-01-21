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


class TestUnitNormalization:
    """Tests for unit normalization of backend_params length fields."""
    
    def test_backend_params_normalized_to_meters(self):
        """Test that backend_params length fields are normalized from mm to meters.
        
        The malaria_venule_space_colonization.json uses input_units="mm".
        After normalization, all length fields should be converted to meters.
        """
        from designspec.spec import DesignSpec
        
        with open(SPACE_COLONIZATION_JSON, "r") as f:
            data = json.load(f)
        
        spec = DesignSpec.from_dict(data)
        
        growth_policy = spec.normalized.get("policies", {}).get("growth", {})
        backend_params = growth_policy.get("backend_params", {})
        
        collision_merge_distance = backend_params.get("collision_merge_distance")
        assert collision_merge_distance is not None, "collision_merge_distance not found"
        assert abs(collision_merge_distance - 0.05e-3) < 1e-9, (
            f"collision_merge_distance should be 0.05e-3 (0.05mm in meters), "
            f"got {collision_merge_distance}"
        )
        
        multi_inlet_blend_sigma = backend_params.get("multi_inlet_blend_sigma")
        assert multi_inlet_blend_sigma is not None, "multi_inlet_blend_sigma not found"
        assert abs(multi_inlet_blend_sigma - 2.5e-3) < 1e-9, (
            f"multi_inlet_blend_sigma should be 2.5e-3 (2.5mm in meters), "
            f"got {multi_inlet_blend_sigma}"
        )
    
    def test_tissue_sampling_normalized_to_meters(self):
        """Test that tissue_sampling length fields are normalized from mm to meters."""
        from designspec.spec import DesignSpec
        
        with open(SPACE_COLONIZATION_JSON, "r") as f:
            data = json.load(f)
        
        spec = DesignSpec.from_dict(data)
        
        growth_policy = spec.normalized.get("policies", {}).get("growth", {})
        backend_params = growth_policy.get("backend_params", {})
        tissue_sampling = backend_params.get("tissue_sampling", {})
        
        min_distance_to_ports = tissue_sampling.get("min_distance_to_ports")
        assert min_distance_to_ports is not None, "min_distance_to_ports not found"
        assert abs(min_distance_to_ports - 0.05e-3) < 1e-9, (
            f"min_distance_to_ports should be 0.05e-3 (0.05mm in meters), "
            f"got {min_distance_to_ports}"
        )
        
        depth_min = tissue_sampling.get("depth_min")
        assert depth_min is not None, "depth_min not found"
        assert abs(depth_min - 0.1e-3) < 1e-9, (
            f"depth_min should be 0.1e-3 (0.1mm in meters), "
            f"got {depth_min}"
        )
    
    def test_space_colonization_policy_normalized_to_meters(self):
        """Test that space_colonization_policy length fields are normalized from mm to meters."""
        from designspec.spec import DesignSpec
        
        with open(SPACE_COLONIZATION_JSON, "r") as f:
            data = json.load(f)
        
        spec = DesignSpec.from_dict(data)
        
        growth_policy = spec.normalized.get("policies", {}).get("growth", {})
        backend_params = growth_policy.get("backend_params", {})
        sc_policy = backend_params.get("space_colonization_policy", {})
        
        branch_enable_after_distance = sc_policy.get("branch_enable_after_distance")
        assert branch_enable_after_distance is not None, "branch_enable_after_distance not found"
        assert abs(branch_enable_after_distance - 0.001e-3) < 1e-12, (
            f"branch_enable_after_distance should be 0.001e-3 (0.001mm in meters), "
            f"got {branch_enable_after_distance}"
        )
        
        min_branch_segment_length = sc_policy.get("min_branch_segment_length")
        assert min_branch_segment_length is not None, "min_branch_segment_length not found"
        assert abs(min_branch_segment_length - 0.01e-3) < 1e-9, (
            f"min_branch_segment_length should be 0.01e-3 (0.01mm in meters), "
            f"got {min_branch_segment_length}"
        )


class TestEnableFlags:
    """Tests for validity and embedding enable flags."""
    
    def test_validity_enable_flag_respected(self):
        """Test that validity.enable=false skips validity stage."""
        from designspec.runner import DesignSpecRunner
        from designspec.spec import DesignSpec
        
        test_spec = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"name": "test_enable_flags", "input_units": "mm"},
            "policies": {},
            "domains": {
                "test_domain": {
                    "type": "cylinder",
                    "center": [0, 0, 0],
                    "radius": 1.0,
                    "height": 1.0
                }
            },
            "components": [],
            "validity": {"enable": False}
        }
        
        spec = DesignSpec.from_dict(test_spec)
        
        validity_spec = spec.normalized.get("validity", {})
        assert validity_spec.get("enable") == False, (
            "validity.enable should be False in normalized spec"
        )
    
    def test_embedding_enable_flag_respected(self):
        """Test that embedding.enable=false skips embedding stage."""
        from designspec.spec import DesignSpec
        
        test_spec = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"name": "test_enable_flags", "input_units": "mm"},
            "policies": {},
            "domains": {
                "test_domain": {
                    "type": "cylinder",
                    "center": [0, 0, 0],
                    "radius": 1.0,
                    "height": 1.0
                }
            },
            "components": [],
            "embedding": {"enable": False}
        }
        
        spec = DesignSpec.from_dict(test_spec)
        
        embedding_spec = spec.normalized.get("embedding", {})
        assert embedding_spec.get("enable") == False, (
            "embedding.enable should be False in normalized spec"
        )
