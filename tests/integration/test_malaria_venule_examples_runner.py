"""
Integration tests for malaria venule DesignSpec examples.

This module tests all malaria venule examples through the DesignSpecRunner
to verify they run successfully and produce expected outputs.

All malaria venule examples:
- malaria_venule_bifurcating_tree.json (scaffold_topdown backend)
- malaria_venule_space_colonization.json (space_colonization backend)
- malaria_venule_cco.json (cco_hybrid backend)
- malaria_venule_fang_hook_channels.json (primitive_channels)
- malaria_venule_vertical_channels.json (primitive_channels)

Usage:
    pytest -q tests/integration/test_malaria_venule_examples_runner.py
    pytest -q tests/integration/test_malaria_venule_examples_runner.py -k bifurcating
"""

import json
import pytest
import tempfile
from pathlib import Path

from designspec.spec import DesignSpec
from designspec.runner import DesignSpecRunner
from designspec.plan import ExecutionPlan


EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples" / "designspec"

MALARIA_EXAMPLES = [
    "malaria_venule_bifurcating_tree.json",
    "malaria_venule_space_colonization.json",
    "malaria_venule_cco.json",
    "malaria_venule_fang_hook_channels.json",
    "malaria_venule_vertical_channels.json",
]


def load_malaria_example(example_name: str) -> DesignSpec:
    """Load a malaria venule DesignSpec example by name."""
    example_path = EXAMPLES_DIR / example_name
    if not example_path.exists():
        pytest.skip(f"Example not found: {example_path}")
    return DesignSpec.from_json(str(example_path))


class TestMalariaVenuleExamplesSmoke:
    """Smoke tests that all malaria examples load and run through component_build."""

    @pytest.mark.parametrize("example_file", MALARIA_EXAMPLES)
    def test_example_loads_successfully(self, example_file):
        """Test that malaria examples can be loaded."""
        spec = load_malaria_example(example_file)
        assert spec is not None
        assert spec.meta is not None
        assert "seed" in spec.meta
        assert "input_units" in spec.meta

    @pytest.mark.parametrize("example_file", MALARIA_EXAMPLES)
    def test_example_runs_through_component_build(self, example_file, tmp_path):
        """Test that malaria examples run through component_build stage."""
        spec = load_malaria_example(example_file)
        
        component_id = spec.components[0]["id"]
        plan = ExecutionPlan(run_until=f"component_build:{component_id}")
        runner = DesignSpecRunner(spec, plan=plan, output_dir=str(tmp_path))
        result = runner.run()
        
        assert result.success, f"Runner failed for {example_file}: {result.errors}"


class TestMalariaVenueMergePolicyConfiguration:
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
    def test_merge_policy_keep_largest_component_false(self, example_file):
        """Test that merge_policy has keep_largest_component=false."""
        json_path = EXAMPLES_DIR / example_file
        
        with open(json_path, "r") as f:
            data = json.load(f)
        
        composition = data.get("policies", {}).get("composition", {})
        merge_policy = composition.get("merge_policy", {})
        keep_largest = merge_policy.get("keep_largest_component", True)
        
        assert keep_largest is False, (
            f"{example_file}: policies.composition.merge_policy.keep_largest_component should be false, got {keep_largest}"
        )


class TestMalariaBifurcatingTreeScaffoldTopdown:
    """Tests specific to the bifurcating tree example with scaffold_topdown backend."""

    @pytest.fixture
    def spec(self):
        """Load the bifurcating tree spec."""
        return load_malaria_example("malaria_venule_bifurcating_tree.json")

    def test_uses_scaffold_topdown_backend(self, spec):
        """Test that the example uses scaffold_topdown backend."""
        growth_policy = spec.policies.get("growth", {})
        backend = growth_policy.get("backend", "")
        
        assert backend == "scaffold_topdown", (
            f"Expected backend='scaffold_topdown', got '{backend}'"
        )

    def test_has_multiple_inlets(self, spec):
        """Test that the example has multiple inlets for forest generation."""
        component = spec.components[0]
        inlets = component.get("ports", {}).get("inlets", [])
        
        assert len(inlets) >= 2, (
            f"Expected multiple inlets for multi-inlet forest, got {len(inlets)}"
        )

    def test_has_collision_settings(self, spec):
        """Test that collision settings are configured."""
        growth_policy = spec.policies.get("growth", {})
        backend_params = growth_policy.get("backend_params", {})
        
        collision_online = backend_params.get("collision_online", {})
        collision_postpass = backend_params.get("collision_postpass", {})
        
        assert collision_online.get("enabled", False), (
            "collision_online should be enabled"
        )
        assert collision_postpass.get("enabled", False), (
            "collision_postpass should be enabled"
        )

    def test_runner_produces_network(self, spec, tmp_path):
        """Test that runner produces a network artifact."""
        component_id = spec.components[0]["id"]
        plan = ExecutionPlan(run_until=f"component_build:{component_id}")
        runner = DesignSpecRunner(spec, plan=plan, output_dir=str(tmp_path))
        result = runner.run()
        
        assert result.success, f"Runner failed: {result.errors}"
        
        network = runner._component_networks.get(component_id)
        assert network is not None, "Expected network artifact to be produced"
        
        segment_count = len(network.segments) if hasattr(network, 'segments') else 0
        assert segment_count > 0, f"Expected non-trivial segment count, got {segment_count}"


class TestMalariaSpaceColonization:
    """Tests specific to the space colonization example."""

    @pytest.fixture
    def spec(self):
        """Load the space colonization spec."""
        return load_malaria_example("malaria_venule_space_colonization.json")

    def test_uses_space_colonization_backend(self, spec):
        """Test that the example uses space_colonization backend."""
        growth_policy = spec.policies.get("growth", {})
        backend = growth_policy.get("backend", "")
        
        assert backend == "space_colonization", (
            f"Expected backend='space_colonization', got '{backend}'"
        )

    def test_has_multiple_inlets(self, spec):
        """Test that the example has multiple inlets."""
        component = spec.components[0]
        inlets = component.get("ports", {}).get("inlets", [])
        
        assert len(inlets) >= 2, (
            f"Expected multiple inlets, got {len(inlets)}"
        )


class TestMalariaCCO:
    """Tests specific to the CCO hybrid example."""

    @pytest.fixture
    def spec(self):
        """Load the CCO spec."""
        return load_malaria_example("malaria_venule_cco.json")

    def test_uses_cco_hybrid_backend(self, spec):
        """Test that the example uses cco_hybrid backend."""
        growth_policy = spec.policies.get("growth", {})
        backend = growth_policy.get("backend", "")
        
        assert backend == "cco_hybrid", (
            f"Expected backend='cco_hybrid', got '{backend}'"
        )


class TestMalariaChannelExamples:
    """Tests for channel-based malaria examples (fang_hook and vertical)."""

    @pytest.mark.parametrize("example_file", [
        "malaria_venule_fang_hook_channels.json",
        "malaria_venule_vertical_channels.json",
    ])
    def test_uses_primitive_channels_build(self, example_file):
        """Test that channel examples use primitive_channels build type."""
        spec = load_malaria_example(example_file)
        component = spec.components[0]
        build_type = component.get("build", {}).get("type", "")
        
        assert build_type == "primitive_channels", (
            f"Expected build.type='primitive_channels', got '{build_type}'"
        )

    @pytest.mark.parametrize("example_file", [
        "malaria_venule_fang_hook_channels.json",
        "malaria_venule_vertical_channels.json",
    ])
    def test_has_channels_policy(self, example_file):
        """Test that channel examples have channels policy enabled."""
        spec = load_malaria_example(example_file)
        channels_policy = spec.policies.get("channels", {})
        
        assert channels_policy.get("enabled", False), (
            f"{example_file}: channels policy should be enabled"
        )


class TestMalariaUnitNormalization:
    """Tests that malaria examples have proper unit configuration."""

    @pytest.mark.parametrize("example_file", MALARIA_EXAMPLES)
    def test_has_valid_input_units(self, example_file):
        """Test that examples have valid input_units."""
        json_path = EXAMPLES_DIR / example_file
        
        with open(json_path, "r") as f:
            data = json.load(f)
        
        input_units = data.get("meta", {}).get("input_units", None)
        valid_units = ["m", "mm", "cm", "um"]
        
        assert input_units in valid_units, (
            f"{example_file}: input_units should be one of {valid_units}, got '{input_units}'"
        )

    @pytest.mark.parametrize("example_file", MALARIA_EXAMPLES)
    def test_resolution_policy_matches_input_units(self, example_file):
        """Test that resolution policy input_units matches meta.input_units."""
        json_path = EXAMPLES_DIR / example_file
        
        with open(json_path, "r") as f:
            data = json.load(f)
        
        meta_units = data.get("meta", {}).get("input_units", None)
        resolution_units = data.get("policies", {}).get("resolution", {}).get("input_units", None)
        
        if resolution_units is not None:
            assert resolution_units == meta_units, (
                f"{example_file}: resolution.input_units ({resolution_units}) should match meta.input_units ({meta_units})"
            )
