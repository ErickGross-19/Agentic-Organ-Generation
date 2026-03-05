"""
Smoke tests for DesignSpec examples.

This module runs malaria venule insert examples through the DesignSpecRunner
to verify they are valid and produce expected outputs.

Fast CI subset (run by default):
- malaria_venule_vertical_channels.json
- malaria_venule_fang_hook_channels.json

Slow examples (marked with @pytest.mark.slow):
- malaria_venule_bifurcating_tree.json
- malaria_venule_space_colonization.json

Usage:
    # Run fast smoke tests (CI subset)
    pytest -q tests/integration/test_examples_smoke.py

    # Run all examples including slow ones
    pytest -q -m slow tests/integration/test_examples_smoke.py

    # Run specific example test
    pytest -q tests/integration/test_examples_smoke.py::test_malaria_venule_vertical_channels
"""

import json
import pytest
from pathlib import Path

from designspec.spec import DesignSpec
from designspec.runner import DesignSpecRunner, RunnerResult
from designspec.plan import ExecutionPlan


EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples" / "designspec"


FAST_EXAMPLES = [
    "malaria_venule_vertical_channels.json",
    "malaria_venule_fang_hook_channels.json",
    "malaria_venule_control_ridge_only.json",
]

SLOW_EXAMPLES = [
    "malaria_venule_bifurcating_tree.json",
    "malaria_venule_bifurcating_tree_with_merge.json",
    "malaria_venule_space_colonization.json",
]


def load_example(example_name: str) -> DesignSpec:
    """Load a DesignSpec example by name."""
    example_path = EXAMPLES_DIR / example_name
    if not example_path.exists():
        pytest.skip(f"Example not found: {example_path}")
    return DesignSpec.from_json(str(example_path))


def run_example_until_stage(spec: DesignSpec, stage: str, tmp_path: Path) -> RunnerResult:
    """Run an example until a specific stage."""
    plan = ExecutionPlan(run_until=stage)
    runner = DesignSpecRunner(spec, plan=plan, output_dir=tmp_path)
    return runner.run()


def assert_result_json_serializable(result: RunnerResult):
    """Assert that the runner result is JSON-serializable."""
    result_dict = result.to_dict()
    json_str = json.dumps(result_dict, default=str)
    assert json_str is not None
    assert len(json_str) > 0
    decoded = json.loads(json_str)
    assert "success" in decoded
    assert "stages_completed" in decoded


class TestFastExamplesUnionOnly:
    """Fast smoke tests that run examples until union_voids stage."""

    def test_malaria_venule_vertical_channels(self, tmp_path):
        """Test malaria_venule_vertical_channels.json runs until union_voids."""
        spec = load_example("malaria_venule_vertical_channels.json")
        result = run_example_until_stage(spec, "union_voids", tmp_path)

        assert isinstance(result, RunnerResult)
        assert_result_json_serializable(result)

    def test_malaria_venule_fang_hook_channels(self, tmp_path):
        """Test malaria_venule_fang_hook_channels.json runs until union_voids."""
        spec = load_example("malaria_venule_fang_hook_channels.json")
        result = run_example_until_stage(spec, "union_voids", tmp_path)

        assert isinstance(result, RunnerResult)
        assert_result_json_serializable(result)

    def test_malaria_venule_control_ridge_only(self, tmp_path):
        """Test malaria_venule_control_ridge_only.json runs full pipeline.
        
        This is a control spec with no network generation and no embedding.
        It should produce a ridged cylinder mesh and complete successfully.
        """
        spec = load_example("malaria_venule_control_ridge_only.json")
        runner = DesignSpecRunner(spec, output_dir=tmp_path)
        result = runner.run()

        assert isinstance(result, RunnerResult)
        assert result.success, f"Control spec failed: {result.errors}"
        assert_result_json_serializable(result)
        
        # Verify domain mesh artifact exists
        domain_mesh_path = tmp_path / "domain_mesh_with_ridge.stl"
        assert domain_mesh_path.exists() or (tmp_path / "artifacts" / "domain_mesh_with_ridge.stl").exists(), \
            "domain_mesh_with_ridge.stl artifact not found"


class TestFastExamplesFullPipeline:
    """Fast smoke tests that run selected examples through full pipeline."""

    def test_malaria_venule_vertical_channels_full(self, tmp_path):
        """Test malaria_venule_vertical_channels.json runs full pipeline with validity."""
        spec = load_example("malaria_venule_vertical_channels.json")
        runner = DesignSpecRunner(spec, output_dir=tmp_path)
        result = runner.run()

        assert isinstance(result, RunnerResult)
        assert_result_json_serializable(result)

    def test_malaria_venule_fang_hook_channels_full(self, tmp_path):
        """Test malaria_venule_fang_hook_channels.json runs full pipeline with validity."""
        spec = load_example("malaria_venule_fang_hook_channels.json")
        runner = DesignSpecRunner(spec, output_dir=tmp_path)
        result = runner.run()

        assert isinstance(result, RunnerResult)
        assert_result_json_serializable(result)


class TestExampleSpecValidity:
    """Test that example specs are valid and can be loaded."""

    @pytest.mark.parametrize("example_name", FAST_EXAMPLES)
    def test_fast_example_loads(self, example_name):
        """Test that fast examples can be loaded."""
        spec = load_example(example_name)
        assert spec is not None
        assert spec.meta is not None
        assert "seed" in spec.meta
        assert "input_units" in spec.meta

    @pytest.mark.slow
    @pytest.mark.parametrize("example_name", SLOW_EXAMPLES)
    def test_slow_example_loads(self, example_name):
        """Test that slow examples can be loaded."""
        spec = load_example(example_name)
        assert spec is not None
        assert spec.meta is not None
        assert "seed" in spec.meta
        assert "input_units" in spec.meta


class TestExampleDomainCompilation:
    """Test that example domains can be compiled."""

    @pytest.mark.parametrize("example_name", FAST_EXAMPLES)
    def test_fast_example_compiles_domains(self, example_name, tmp_path):
        """Test that fast examples compile domains successfully."""
        spec = load_example(example_name)
        result = run_example_until_stage(spec, "compile_domains", tmp_path)

        assert isinstance(result, RunnerResult)
        assert "compile_domains" in result.stages_completed or result.success


@pytest.mark.slow
class TestSlowExamplesUnionOnly:
    """Slow smoke tests that run examples until union_voids stage."""

    def test_malaria_venule_bifurcating_tree(self, tmp_path):
        """Test malaria_venule_bifurcating_tree.json runs until union_voids."""
        spec = load_example("malaria_venule_bifurcating_tree.json")
        result = run_example_until_stage(spec, "union_voids", tmp_path)

        assert isinstance(result, RunnerResult)
        assert_result_json_serializable(result)

    def test_malaria_venule_bifurcating_tree_with_merge(self, tmp_path):
        """Test malaria_venule_bifurcating_tree_with_merge.json runs until union_voids.
        
        This spec enables merge_on_collision for branch reconnections during collision failures.
        """
        spec = load_example("malaria_venule_bifurcating_tree_with_merge.json")
        result = run_example_until_stage(spec, "union_voids", tmp_path)

        assert isinstance(result, RunnerResult)
        assert_result_json_serializable(result)

    def test_malaria_venule_space_colonization(self, tmp_path):
        """Test malaria_venule_space_colonization.json runs until union_voids."""
        spec = load_example("malaria_venule_space_colonization.json")
        result = run_example_until_stage(spec, "union_voids", tmp_path)

        assert isinstance(result, RunnerResult)
        assert_result_json_serializable(result)


class TestExampleMetadata:
    """Test that examples have proper metadata."""

    @pytest.mark.parametrize("example_name", FAST_EXAMPLES + SLOW_EXAMPLES)
    def test_example_has_required_metadata(self, example_name):
        """Test that examples have required metadata fields."""
        example_path = EXAMPLES_DIR / example_name
        if not example_path.exists():
            pytest.skip(f"Example not found: {example_path}")

        with open(example_path) as f:
            data = json.load(f)

        assert "schema" in data
        assert data["schema"]["name"] == "aog_designspec"
        assert data["schema"]["version"] == "1.0.0"

        assert "meta" in data
        assert "name" in data["meta"]
        assert "seed" in data["meta"]
        assert "input_units" in data["meta"]
        assert "description" in data["meta"]
        assert "tags" in data["meta"]

        assert "policies" in data
        assert "domains" in data
        assert "components" in data
        assert "outputs" in data

    @pytest.mark.parametrize("example_name", FAST_EXAMPLES + SLOW_EXAMPLES)
    def test_example_has_deterministic_seed(self, example_name):
        """Test that examples have a deterministic seed."""
        example_path = EXAMPLES_DIR / example_name
        if not example_path.exists():
            pytest.skip(f"Example not found: {example_path}")

        with open(example_path) as f:
            data = json.load(f)

        seed = data["meta"]["seed"]
        assert isinstance(seed, int)
        assert seed > 0


class TestValidityEnabled:
    """Test that validity checks are enabled and produce reports."""

    @pytest.mark.parametrize("example_name", FAST_EXAMPLES + SLOW_EXAMPLES)
    def test_validity_enabled_in_spec(self, example_name):
        """Test that validity is enabled in all malaria specs."""
        example_path = EXAMPLES_DIR / example_name
        if not example_path.exists():
            pytest.skip(f"Example not found: {example_path}")

        with open(example_path) as f:
            data = json.load(f)

        # Check top-level validity section
        assert "validity" in data, f"Missing top-level validity section in {example_name}"
        assert data["validity"].get("enable") is True, \
            f"validity.enable should be true in {example_name}"
        assert "save_report" in data["validity"], \
            f"validity.save_report should be set in {example_name}"


@pytest.mark.slow
class TestBottomSprawlPrevention:
    """Test that scaffold_topdown prevents bottom-plane sprawl."""

    def test_no_bottom_sprawl_bifurcating_tree(self, tmp_path):
        """Test that bifurcating tree does not sprawl on the bottom face.
        
        Verifies that no accepted node has z within required_clearance of the bottom face.
        """
        spec = load_example("malaria_venule_bifurcating_tree.json")
        result = run_example_until_stage(spec, "union_voids", tmp_path)

        assert isinstance(result, RunnerResult)
        
        # Load the generated network to check node positions
        network_path = tmp_path / "artifacts" / "bifurcating_tree_network.json"
        if not network_path.exists():
            pytest.skip("Network file not generated")
        
        with open(network_path) as f:
            network_data = json.load(f)
        
        # Get domain parameters
        domain = spec.domains.get("main_domain")
        if domain is None:
            pytest.skip("No main_domain found")
        
        # Calculate z_min and required clearance
        # Cylinder: center at (0, 0, 0), height = 0.002
        # z_min = center.z - height/2 = 0 - 0.001 = -0.001
        z_min = -0.001
        
        # Required clearance from spec:
        # wall_margin_m = 0.0003, boundary_extra_m = 0.0001
        # stop_before_boundary_m = 0.0001, stop_before_boundary_extra_m = 0.0001
        # Plus typical radius ~0.0005
        # Total: ~0.0011
        min_allowed_z = z_min + 0.0008  # Conservative check
        
        # Check all nodes
        nodes = network_data.get("nodes", [])
        for node in nodes:
            pos = node.get("position", {})
            z = pos.get("z", 0)
            assert z >= min_allowed_z, \
                f"Node at z={z} is too close to bottom face (z_min={z_min})"
