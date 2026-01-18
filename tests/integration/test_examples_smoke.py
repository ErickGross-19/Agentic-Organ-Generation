"""
Smoke tests for DesignSpec examples.

This module runs a fast subset of examples through the DesignSpecRunner
to verify they are valid and produce expected outputs.

Fast CI subset (run by default):
- 01_minimal_box_network.json
- 02_multicomponent_union_embed.json
- 03_transform_domain.json
- 05_implicit_ast_domain.json

Slow examples (marked with @pytest.mark.slow):
- All other examples that take longer to run

Usage:
    # Run fast smoke tests (CI subset)
    pytest -q tests/integration/test_examples_smoke.py

    # Run all examples including slow ones
    pytest -q -m slow tests/integration/test_examples_smoke.py

    # Run specific example test
    pytest -q tests/integration/test_examples_smoke.py::test_01_minimal_box_network
"""

import json
import pytest
from pathlib import Path

from designspec.spec import DesignSpec
from designspec.runner import DesignSpecRunner, RunnerResult
from designspec.plan import ExecutionPlan


EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples" / "designspec"


FAST_EXAMPLES = [
    "01_minimal_box_network.json",
    "02_multicomponent_union_embed.json",
    "03_transform_domain.json",
    "05_implicit_ast_domain.json",
]

SLOW_EXAMPLES = [
    "04_composite_domain_boolean.json",
    "06_mesh_domain_user_faces.json",
    "07_fang_hook_channels.json",
    "08_path_channel_tube_sweep.json",
    "09_hierarchical_pathfinding_waypoints.json",
    "10_programmatic_backend_dsl.json",
    "11_kary_backend.json",
    "12_cco_hybrid_backend.json",
    "13_validity_open_ports_focus.json",
    "14_budget_relaxation_showcase.json",
]


def load_example(example_name: str) -> DesignSpec:
    """Load a DesignSpec example by name."""
    example_path = EXAMPLES_DIR / example_name
    if not example_path.exists():
        pytest.skip(f"Example not found: {example_path}")
    return DesignSpec.from_file(str(example_path))


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

    def test_01_minimal_box_network(self, tmp_path):
        """Test 01_minimal_box_network.json runs until union_voids."""
        spec = load_example("01_minimal_box_network.json")
        result = run_example_until_stage(spec, "union_voids", tmp_path)

        assert isinstance(result, RunnerResult)
        assert_result_json_serializable(result)

    def test_02_multicomponent_union_embed(self, tmp_path):
        """Test 02_multicomponent_union_embed.json runs until union_voids."""
        spec = load_example("02_multicomponent_union_embed.json")
        result = run_example_until_stage(spec, "union_voids", tmp_path)

        assert isinstance(result, RunnerResult)
        assert_result_json_serializable(result)

    def test_03_transform_domain(self, tmp_path):
        """Test 03_transform_domain.json runs until union_voids."""
        spec = load_example("03_transform_domain.json")
        result = run_example_until_stage(spec, "union_voids", tmp_path)

        assert isinstance(result, RunnerResult)
        assert_result_json_serializable(result)

    def test_05_implicit_ast_domain(self, tmp_path):
        """Test 05_implicit_ast_domain.json runs until union_voids."""
        spec = load_example("05_implicit_ast_domain.json")
        result = run_example_until_stage(spec, "union_voids", tmp_path)

        assert isinstance(result, RunnerResult)
        assert_result_json_serializable(result)


class TestFastExamplesFullPipeline:
    """Fast smoke tests that run selected examples through full pipeline."""

    def test_01_minimal_box_network_full(self, tmp_path):
        """Test 01_minimal_box_network.json runs full pipeline with validity."""
        spec = load_example("01_minimal_box_network.json")
        runner = DesignSpecRunner(spec, output_dir=tmp_path)
        result = runner.run()

        assert isinstance(result, RunnerResult)
        assert_result_json_serializable(result)

    def test_03_transform_domain_full(self, tmp_path):
        """Test 03_transform_domain.json runs full pipeline with validity."""
        spec = load_example("03_transform_domain.json")
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

    def test_04_composite_domain_boolean(self, tmp_path):
        """Test 04_composite_domain_boolean.json runs until union_voids."""
        spec = load_example("04_composite_domain_boolean.json")
        result = run_example_until_stage(spec, "union_voids", tmp_path)

        assert isinstance(result, RunnerResult)
        assert_result_json_serializable(result)

    def test_06_mesh_domain_user_faces(self, tmp_path):
        """Test 06_mesh_domain_user_faces.json runs until union_voids."""
        spec = load_example("06_mesh_domain_user_faces.json")
        result = run_example_until_stage(spec, "union_voids", tmp_path)

        assert isinstance(result, RunnerResult)
        assert_result_json_serializable(result)

    def test_07_fang_hook_channels(self, tmp_path):
        """Test 07_fang_hook_channels.json runs until union_voids."""
        spec = load_example("07_fang_hook_channels.json")
        result = run_example_until_stage(spec, "union_voids", tmp_path)

        assert isinstance(result, RunnerResult)
        assert_result_json_serializable(result)

    def test_08_path_channel_tube_sweep(self, tmp_path):
        """Test 08_path_channel_tube_sweep.json runs until union_voids."""
        spec = load_example("08_path_channel_tube_sweep.json")
        result = run_example_until_stage(spec, "union_voids", tmp_path)

        assert isinstance(result, RunnerResult)
        assert_result_json_serializable(result)

    def test_09_hierarchical_pathfinding_waypoints(self, tmp_path):
        """Test 09_hierarchical_pathfinding_waypoints.json runs until union_voids."""
        spec = load_example("09_hierarchical_pathfinding_waypoints.json")
        result = run_example_until_stage(spec, "union_voids", tmp_path)

        assert isinstance(result, RunnerResult)
        assert_result_json_serializable(result)

    def test_10_programmatic_backend_dsl(self, tmp_path):
        """Test 10_programmatic_backend_dsl.json runs until union_voids."""
        spec = load_example("10_programmatic_backend_dsl.json")
        result = run_example_until_stage(spec, "union_voids", tmp_path)

        assert isinstance(result, RunnerResult)
        assert_result_json_serializable(result)

    def test_11_kary_backend(self, tmp_path):
        """Test 11_kary_backend.json runs until union_voids."""
        spec = load_example("11_kary_backend.json")
        result = run_example_until_stage(spec, "union_voids", tmp_path)

        assert isinstance(result, RunnerResult)
        assert_result_json_serializable(result)

    def test_12_cco_hybrid_backend(self, tmp_path):
        """Test 12_cco_hybrid_backend.json runs until union_voids."""
        spec = load_example("12_cco_hybrid_backend.json")
        result = run_example_until_stage(spec, "union_voids", tmp_path)

        assert isinstance(result, RunnerResult)
        assert_result_json_serializable(result)

    def test_13_validity_open_ports_focus(self, tmp_path):
        """Test 13_validity_open_ports_focus.json runs until union_voids."""
        spec = load_example("13_validity_open_ports_focus.json")
        result = run_example_until_stage(spec, "union_voids", tmp_path)

        assert isinstance(result, RunnerResult)
        assert_result_json_serializable(result)

    def test_14_budget_relaxation_showcase(self, tmp_path):
        """Test 14_budget_relaxation_showcase.json runs until union_voids."""
        spec = load_example("14_budget_relaxation_showcase.json")
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
