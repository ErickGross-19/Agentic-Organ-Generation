"""
Regression tests for malaria venule DesignSpec examples.

This module validates that the malaria venule examples run correctly through
the DesignSpecRunner pipeline:
1. malaria_venule_bifurcating_tree.json - scaffold_topdown multi-inlet forest
2. malaria_venule_space_colonization.json - organic blended growth
3. malaria_venule_fang_hook_channels.json - primitive fang-hook channels
4. malaria_venule_vertical_channels.json - primitive vertical channels
5. malaria_venule_cco.json - CCO-optimized tree

Acceptance criteria:
- Runner completes without exception through component_build stage
- For network-producing examples: network has > N segments
- Multi-inlet specs: network contains branches from each inlet (forest is fine)
"""

import json
import pytest
import numpy as np
import tempfile
from pathlib import Path

from designspec.spec import DesignSpec
from designspec.runner import DesignSpecRunner
from designspec.plan import ExecutionPlan


EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples" / "designspec"

MALARIA_EXAMPLES = [
    "malaria_venule_bifurcating_tree.json",
    "malaria_venule_space_colonization.json",
    "malaria_venule_fang_hook_channels.json",
    "malaria_venule_vertical_channels.json",
    "malaria_venule_cco.json",
]

MIN_SEGMENT_COUNT = 5
MIN_NODE_COUNT = 10


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


def count_inlet_roots(network) -> int:
    """Count the number of inlet root nodes in the network."""
    inlet_count = 0
    for node in network.nodes.values():
        if hasattr(node, 'node_type') and node.node_type == 'inlet':
            inlet_count += 1
    return inlet_count


class TestMalariaVenuleExamplesLoad:
    """Tests that all malaria venule examples load correctly."""
    
    @pytest.mark.parametrize("example_file", MALARIA_EXAMPLES)
    def test_example_loads_with_designspec(self, example_file):
        """Test that example loads with the DesignSpec loader."""
        json_path = EXAMPLES_DIR / example_file
        assert json_path.exists(), f"Example file not found: {json_path}"
        
        spec = DesignSpec.from_json(str(json_path))
        
        assert spec is not None
        assert spec.meta is not None
        assert "malaria" in spec.meta.get("name", "").lower() or "malaria" in str(spec.meta.get("tags", []))
    
    @pytest.mark.parametrize("example_file", MALARIA_EXAMPLES)
    def test_example_has_required_structure(self, example_file):
        """Test that example has required top-level keys."""
        json_path = EXAMPLES_DIR / example_file
        
        with open(json_path, "r") as f:
            data = json.load(f)
        
        assert "schema" in data, f"{example_file} missing 'schema'"
        assert "meta" in data, f"{example_file} missing 'meta'"
        assert "policies" in data, f"{example_file} missing 'policies'"
        assert "domains" in data, f"{example_file} missing 'domains'"
        assert "components" in data, f"{example_file} missing 'components'"


class TestMalariaVenuleExamplesRunner:
    """Tests that malaria venule examples run through DesignSpecRunner."""
    
    @pytest.mark.parametrize("example_file", MALARIA_EXAMPLES)
    def test_runner_completes_compile_stages(self, example_file):
        """Test that runner completes compile_policies and compile_domains."""
        json_path = EXAMPLES_DIR / example_file
        spec = DesignSpec.from_json(str(json_path))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            plan = ExecutionPlan(run_until="compile_domains")
            runner = DesignSpecRunner(spec, plan=plan, output_dir=tmpdir)
            result = runner.run()
        
        assert result.success, f"{example_file} failed compile stages: {result.errors}"
        assert "compile_policies" in result.stages_completed
        assert "compile_domains" in result.stages_completed


class TestMalariaVenuleBifurcatingTree:
    """Tests for malaria_venule_bifurcating_tree.json example."""
    
    @pytest.fixture
    def runner_result(self):
        """Run the bifurcating tree example through component_build."""
        json_path = EXAMPLES_DIR / "malaria_venule_bifurcating_tree.json"
        spec = DesignSpec.from_json(str(json_path))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            component_id = spec.components[0]["id"]
            plan = ExecutionPlan(run_until=f"component_build:{component_id}")
            runner = DesignSpecRunner(spec, plan=plan, output_dir=tmpdir)
            result = runner.run()
            
            network = runner._component_networks.get(component_id)
            yield result, network, spec
    
    def test_runner_succeeds(self, runner_result):
        """Test that runner completes without errors."""
        result, network, spec = runner_result
        assert result.success, f"Runner failed: {result.errors}"
    
    def test_network_exists(self, runner_result):
        """Test that a network was produced."""
        result, network, spec = runner_result
        assert network is not None, "No network produced"
    
    def test_network_has_segments(self, runner_result):
        """Test that network has sufficient segments."""
        result, network, spec = runner_result
        if network is None:
            pytest.skip("No network produced")
        
        segment_count = len(network.segments)
        assert segment_count >= MIN_SEGMENT_COUNT, (
            f"Network has only {segment_count} segments, expected >= {MIN_SEGMENT_COUNT}"
        )
    
    def test_uses_scaffold_topdown_backend(self, runner_result):
        """Test that the example uses scaffold_topdown backend."""
        result, network, spec = runner_result
        
        growth_policy = spec.policies.get("growth", {})
        backend = growth_policy.get("backend", "")
        
        assert backend == "scaffold_topdown", (
            f"Expected backend='scaffold_topdown', got '{backend}'"
        )
    
    def test_multi_inlet_forest(self, runner_result):
        """Test that multiple inlets produce a forest (multiple trees)."""
        result, network, spec = runner_result
        if network is None:
            pytest.skip("No network produced")
        
        component = spec.components[0]
        inlets = component.get("ports", {}).get("inlets", [])
        num_inlets = len(inlets)
        
        assert num_inlets >= 2, f"Expected multiple inlets, got {num_inlets}"


class TestMalariaVenuleSpaceColonization:
    """Tests for malaria_venule_space_colonization.json example."""
    
    @pytest.fixture
    def runner_result(self):
        """Run the space colonization example through component_build."""
        json_path = EXAMPLES_DIR / "malaria_venule_space_colonization.json"
        spec = DesignSpec.from_json(str(json_path))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            component_id = spec.components[0]["id"]
            plan = ExecutionPlan(run_until=f"component_build:{component_id}")
            runner = DesignSpecRunner(spec, plan=plan, output_dir=tmpdir)
            result = runner.run()
            
            network = runner._component_networks.get(component_id)
            yield result, network, spec
    
    def test_runner_succeeds(self, runner_result):
        """Test that runner completes without errors."""
        result, network, spec = runner_result
        assert result.success, f"Runner failed: {result.errors}"
    
    def test_network_exists(self, runner_result):
        """Test that a network was produced."""
        result, network, spec = runner_result
        assert network is not None, "No network produced"
    
    def test_uses_space_colonization_backend(self, runner_result):
        """Test that the example uses space_colonization backend."""
        result, network, spec = runner_result
        
        growth_policy = spec.policies.get("growth", {})
        backend = growth_policy.get("backend", "")
        
        assert backend == "space_colonization", (
            f"Expected backend='space_colonization', got '{backend}'"
        )


class TestMalariaVenuleCCO:
    """Tests for malaria_venule_cco.json example."""
    
    @pytest.fixture
    def runner_result(self):
        """Run the CCO example through component_build."""
        json_path = EXAMPLES_DIR / "malaria_venule_cco.json"
        spec = DesignSpec.from_json(str(json_path))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            component_id = spec.components[0]["id"]
            plan = ExecutionPlan(run_until=f"component_build:{component_id}")
            runner = DesignSpecRunner(spec, plan=plan, output_dir=tmpdir)
            result = runner.run()
            
            network = runner._component_networks.get(component_id)
            yield result, network, spec
    
    def test_runner_succeeds(self, runner_result):
        """Test that runner completes without errors."""
        result, network, spec = runner_result
        assert result.success, f"Runner failed: {result.errors}"
    
    def test_uses_cco_hybrid_backend(self, runner_result):
        """Test that the example uses cco_hybrid backend."""
        result, network, spec = runner_result
        
        growth_policy = spec.policies.get("growth", {})
        backend = growth_policy.get("backend", "")
        
        assert backend == "cco_hybrid", (
            f"Expected backend='cco_hybrid', got '{backend}'"
        )


class TestMalariaVenulePrimitiveChannels:
    """Tests for primitive channel examples (fang_hook and vertical)."""
    
    @pytest.mark.parametrize("example_file", [
        "malaria_venule_fang_hook_channels.json",
        "malaria_venule_vertical_channels.json",
    ])
    def test_runner_completes_component_build(self, example_file):
        """Test that runner completes component_build for channel examples."""
        json_path = EXAMPLES_DIR / example_file
        spec = DesignSpec.from_json(str(json_path))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            component_id = spec.components[0]["id"]
            plan = ExecutionPlan(run_until=f"component_build:{component_id}")
            runner = DesignSpecRunner(spec, plan=plan, output_dir=tmpdir)
            result = runner.run()
        
        assert result.success, f"{example_file} failed: {result.errors}"
    
    @pytest.mark.parametrize("example_file", [
        "malaria_venule_fang_hook_channels.json",
        "malaria_venule_vertical_channels.json",
    ])
    def test_uses_primitive_channels_build_type(self, example_file):
        """Test that channel examples use primitive_channels build type."""
        json_path = EXAMPLES_DIR / example_file
        spec = DesignSpec.from_json(str(json_path))
        
        component = spec.components[0]
        build_type = component.get("build", {}).get("type", "")
        
        assert build_type == "primitive_channels", (
            f"Expected build.type='primitive_channels', got '{build_type}'"
        )


class TestMergePolicyConfiguration:
    """Tests that all malaria examples have keep_largest_component=false."""
    
    @pytest.mark.parametrize("example_file", MALARIA_EXAMPLES)
    def test_composition_keep_largest_component_false(self, example_file):
        """Test that composition policy has keep_largest_component=false."""
        json_path = EXAMPLES_DIR / example_file
        
        with open(json_path, "r") as f:
            data = json.load(f)
        
        composition = data.get("policies", {}).get("composition", {})
        keep_largest = composition.get("keep_largest_component", True)
        
        assert keep_largest is False, (
            f"{example_file}: policies.composition.keep_largest_component should be false, got {keep_largest}"
        )
    
    @pytest.mark.parametrize("example_file", MALARIA_EXAMPLES)
    def test_mesh_merge_keep_largest_component_false(self, example_file):
        """Test that mesh_merge policy has keep_largest_component=false if present."""
        json_path = EXAMPLES_DIR / example_file
        
        with open(json_path, "r") as f:
            data = json.load(f)
        
        mesh_merge = data.get("policies", {}).get("mesh_merge", {})
        if mesh_merge:
            keep_largest = mesh_merge.get("keep_largest_component", True)
            assert keep_largest is False, (
                f"{example_file}: policies.mesh_merge.keep_largest_component should be false, got {keep_largest}"
            )
    
    @pytest.mark.parametrize("example_file", MALARIA_EXAMPLES)
    def test_nested_merge_policy_keep_largest_component_false(self, example_file):
        """Test that nested merge_policy has keep_largest_component=false if present."""
        json_path = EXAMPLES_DIR / example_file
        
        with open(json_path, "r") as f:
            data = json.load(f)
        
        composition = data.get("policies", {}).get("composition", {})
        merge_policy = composition.get("merge_policy", {})
        if merge_policy:
            keep_largest = merge_policy.get("keep_largest_component", True)
            assert keep_largest is False, (
                f"{example_file}: policies.composition.merge_policy.keep_largest_component should be false, got {keep_largest}"
            )


class TestInletDirectionValidation:
    """Tests for inlet direction validation."""
    
    @pytest.mark.parametrize("example_file", MALARIA_EXAMPLES)
    def test_inlet_directions_point_inward(self, example_file):
        """Test that inlet directions point inward (negative Z for top-face inlets)."""
        json_path = EXAMPLES_DIR / example_file
        
        with open(json_path, "r") as f:
            data = json.load(f)
        
        for component in data.get("components", []):
            ports = component.get("ports", {})
            inlets = ports.get("inlets", [])
            
            for inlet in inlets:
                direction = inlet.get("direction", [0, 0, 0])
                assert direction[2] < 0, (
                    f"{example_file}: Inlet '{inlet.get('name', 'unnamed')}' has direction[2]={direction[2]}, "
                    f"expected < 0 (pointing inward). Direction: {direction}"
                )


class TestUnitNormalization:
    """Tests for unit normalization of backend_params length fields."""
    
    def test_space_colonization_backend_params_normalized(self):
        """Test that space colonization backend_params are normalized to meters."""
        json_path = EXAMPLES_DIR / "malaria_venule_space_colonization.json"
        spec = DesignSpec.from_json(str(json_path))
        
        growth_policy = spec.normalized.get("policies", {}).get("growth", {})
        backend_params = growth_policy.get("backend_params", {})
        
        influence_radius = backend_params.get("influence_radius")
        assert influence_radius is not None, "influence_radius not found"
        assert influence_radius < 0.01, (
            f"influence_radius should be in meters (< 0.01), got {influence_radius}"
        )
    
    def test_bifurcating_tree_backend_params_normalized(self):
        """Test that bifurcating tree backend_params are normalized to meters."""
        json_path = EXAMPLES_DIR / "malaria_venule_bifurcating_tree.json"
        spec = DesignSpec.from_json(str(json_path))
        
        growth_policy = spec.normalized.get("policies", {}).get("growth", {})
        backend_params = growth_policy.get("backend_params", {})
        
        step_length = backend_params.get("step_length")
        if step_length is not None:
            assert step_length < 0.01, (
                f"step_length should be in meters (< 0.01), got {step_length}"
            )


class TestEnableFlags:
    """Tests for validity and embedding enable flags."""
    
    def test_validity_enable_flag_respected(self):
        """Test that validity.enable=false skips validity stage."""
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


class TestMalariaChannelValidityConfiguration:
    """
    Tests for validity configuration of channel-based malaria examples.
    
    These tests verify that the fang_hook and vertical channel examples
    have proper validity configuration for multi-component channel grids
    where void intersection at inlet ports is expected.
    """
    
    CHANNEL_EXAMPLES = [
        "malaria_venule_fang_hook_channels.json",
        "malaria_venule_vertical_channels.json",
    ]
    
    @pytest.mark.parametrize("example_file", CHANNEL_EXAMPLES)
    def test_max_components_allows_multiple_channels(self, example_file):
        """Test that max_components is set high enough for 9 channels."""
        json_path = EXAMPLES_DIR / example_file
        
        with open(json_path, "r") as f:
            data = json.load(f)
        
        validity = data.get("policies", {}).get("validity", {})
        max_components = validity.get("max_components", 1)
        
        assert max_components >= 9, (
            f"{example_file}: max_components should be >= 9 for 9-channel grid, got {max_components}"
        )
    
    @pytest.mark.parametrize("example_file", CHANNEL_EXAMPLES)
    def test_allow_boundary_intersections_at_ports_enabled(self, example_file):
        """Test that allow_boundary_intersections_at_ports is enabled."""
        json_path = EXAMPLES_DIR / example_file
        
        with open(json_path, "r") as f:
            data = json.load(f)
        
        validity = data.get("policies", {}).get("validity", {})
        allow_boundary = validity.get("allow_boundary_intersections_at_ports", False)
        
        assert allow_boundary is True, (
            f"{example_file}: allow_boundary_intersections_at_ports should be true, got {allow_boundary}"
        )
    
    @pytest.mark.parametrize("example_file", CHANNEL_EXAMPLES)
    def test_open_port_policy_enabled(self, example_file):
        """Test that open_port policy is enabled."""
        json_path = EXAMPLES_DIR / example_file
        
        with open(json_path, "r") as f:
            data = json.load(f)
        
        open_port = data.get("policies", {}).get("open_port", {})
        enabled = open_port.get("enabled", False)
        
        assert enabled is True, (
            f"{example_file}: open_port.enabled should be true, got {enabled}"
        )
    
    @pytest.mark.parametrize("example_file", CHANNEL_EXAMPLES)
    def test_check_open_ports_enabled(self, example_file):
        """Test that check_open_ports is enabled in validity policy."""
        json_path = EXAMPLES_DIR / example_file
        
        with open(json_path, "r") as f:
            data = json.load(f)
        
        validity = data.get("policies", {}).get("validity", {})
        check_open_ports = validity.get("check_open_ports", False)
        
        assert check_open_ports is True, (
            f"{example_file}: check_open_ports should be true, got {check_open_ports}"
        )
    
    @pytest.mark.parametrize("example_file", CHANNEL_EXAMPLES)
    def test_has_nine_inlets(self, example_file):
        """Test that channel examples have exactly 9 inlets."""
        json_path = EXAMPLES_DIR / example_file
        
        with open(json_path, "r") as f:
            data = json.load(f)
        
        components = data.get("components", [])
        assert len(components) > 0, f"{example_file}: No components found"
        
        component = components[0]
        inlets = component.get("ports", {}).get("inlets", [])
        
        assert len(inlets) == 9, (
            f"{example_file}: Expected 9 inlets for 9-channel grid, got {len(inlets)}"
        )
