"""Tests for ExecutionPlan stage filtering."""

import pytest
from designspec.plan import (
    ExecutionPlan,
    Stage,
    STAGE_ORDER,
    validate_plan,
)


class TestExecutionPlanBasics:
    """Tests for basic ExecutionPlan functionality."""
    
    def test_create_default_plan(self):
        plan = ExecutionPlan()
        assert plan.run_until is None
        assert plan.run_only is None
        assert plan.skip is None
        assert plan.components_subset is None
    
    def test_create_plan_with_run_until(self):
        plan = ExecutionPlan(run_until=Stage.UNION_VOIDS.value)
        assert plan.run_until == Stage.UNION_VOIDS.value
    
    def test_create_plan_with_run_only(self):
        plan = ExecutionPlan(run_only=[Stage.COMPILE_POLICIES.value, Stage.COMPILE_DOMAINS.value])
        assert len(plan.run_only) == 2
    
    def test_create_plan_with_skip(self):
        plan = ExecutionPlan(skip=[Stage.VALIDITY.value])
        assert Stage.VALIDITY.value in plan.skip
    
    def test_create_plan_with_components_subset(self):
        plan = ExecutionPlan(components_subset=["net_1"])
        assert "net_1" in plan.components_subset


class TestExecutionPlanRunUntil:
    """Tests for run_until functionality."""
    
    def test_run_until_union_voids_stops_before_embed(self):
        plan = ExecutionPlan(run_until=Stage.UNION_VOIDS.value)
        plan.set_component_ids(["net_1"])
        
        stages = plan.compute_stages()
        
        assert Stage.UNION_VOIDS.value in stages
        assert Stage.EMBED.value not in stages
    
    def test_run_until_compile_domains_stops_early(self):
        plan = ExecutionPlan(run_until=Stage.COMPILE_DOMAINS.value)
        plan.set_component_ids(["net_1"])
        
        stages = plan.compute_stages()
        
        assert Stage.COMPILE_POLICIES.value in stages
        assert Stage.COMPILE_DOMAINS.value in stages
        assert f"{Stage.COMPONENT_PORTS.value}:net_1" not in stages
    
    def test_run_until_export_includes_all(self):
        plan = ExecutionPlan(run_until=Stage.EXPORT.value)
        plan.set_component_ids(["net_1"])
        
        stages = plan.compute_stages()
        
        assert Stage.EXPORT.value in stages
        assert Stage.VALIDITY.value in stages


class TestExecutionPlanRunOnly:
    """Tests for run_only functionality."""
    
    def test_run_only_single_stage(self):
        plan = ExecutionPlan(run_only=[Stage.COMPILE_POLICIES.value])
        plan.set_component_ids(["net_1"])
        
        stages = plan.compute_stages()
        
        assert Stage.COMPILE_POLICIES.value in stages
        assert len(stages) == 1
    
    def test_run_only_multiple_stages(self):
        plan = ExecutionPlan(run_only=[Stage.COMPILE_POLICIES.value, Stage.COMPILE_DOMAINS.value])
        plan.set_component_ids(["net_1"])
        
        stages = plan.compute_stages()
        
        assert Stage.COMPILE_POLICIES.value in stages
        assert Stage.COMPILE_DOMAINS.value in stages
        assert len(stages) == 2


class TestExecutionPlanSkip:
    """Tests for skip functionality."""
    
    def test_skip_validity(self):
        plan = ExecutionPlan(skip=[Stage.VALIDITY.value])
        plan.set_component_ids(["net_1"])
        
        stages = plan.compute_stages()
        
        assert Stage.VALIDITY.value not in stages
        assert Stage.EXPORT.value in stages
    
    def test_skip_multiple_stages(self):
        plan = ExecutionPlan(skip=[Stage.VALIDITY.value, Stage.PORT_RECARVE.value])
        plan.set_component_ids(["net_1"])
        
        stages = plan.compute_stages()
        
        assert Stage.VALIDITY.value not in stages
        assert Stage.PORT_RECARVE.value not in stages


class TestExecutionPlanComponentsSubset:
    """Tests for components_subset functionality."""
    
    def test_components_subset_filters_component_stages(self):
        plan = ExecutionPlan(components_subset=["net_1"])
        plan.set_component_ids(["net_1", "chan_1"])
        
        stages = plan.compute_stages()
        
        assert f"{Stage.COMPONENT_BUILD.value}:net_1" in stages
        assert f"{Stage.COMPONENT_BUILD.value}:chan_1" not in stages
    
    def test_components_subset_includes_non_component_stages(self):
        plan = ExecutionPlan(components_subset=["net_1"])
        plan.set_component_ids(["net_1", "chan_1"])
        
        stages = plan.compute_stages()
        
        assert Stage.COMPILE_POLICIES.value in stages
        assert Stage.UNION_VOIDS.value in stages
    
    def test_should_run_component(self):
        plan = ExecutionPlan(components_subset=["net_1"])
        plan.set_component_ids(["net_1", "chan_1"])
        
        assert plan.should_run_component("net_1") is True
        assert plan.should_run_component("chan_1") is False
    
    def test_should_run_component_no_subset(self):
        plan = ExecutionPlan()
        plan.set_component_ids(["net_1", "chan_1"])
        
        assert plan.should_run_component("net_1") is True
        assert plan.should_run_component("chan_1") is True


class TestExecutionPlanShouldRun:
    """Tests for should_run method."""
    
    def test_should_run_default_plan(self):
        plan = ExecutionPlan()
        plan.set_component_ids(["net_1"])
        
        assert plan.should_run(Stage.COMPILE_POLICIES.value) is True
        assert plan.should_run(Stage.EXPORT.value) is True
    
    def test_should_run_with_run_until(self):
        plan = ExecutionPlan(run_until=Stage.UNION_VOIDS.value)
        plan.set_component_ids(["net_1"])
        
        assert plan.should_run(Stage.COMPILE_POLICIES.value) is True
        assert plan.should_run(Stage.UNION_VOIDS.value) is True
        assert plan.should_run(Stage.EMBED.value) is False
    
    def test_should_run_with_skip(self):
        plan = ExecutionPlan(skip=[Stage.VALIDITY.value])
        plan.set_component_ids(["net_1"])
        
        assert plan.should_run(Stage.EMBED.value) is True
        assert plan.should_run(Stage.VALIDITY.value) is False
        assert plan.should_run(Stage.EXPORT.value) is True


class TestExecutionPlanStageHelpers:
    """Tests for stage helper methods."""
    
    def test_get_next_stage(self):
        plan = ExecutionPlan()
        plan.set_component_ids(["net_1"])
        
        next_stage = plan.get_next_stage(Stage.COMPILE_POLICIES.value)
        assert next_stage == Stage.COMPILE_DOMAINS.value
    
    def test_get_next_stage_at_end(self):
        plan = ExecutionPlan()
        plan.set_component_ids(["net_1"])
        
        next_stage = plan.get_next_stage(Stage.EXPORT.value)
        assert next_stage is None
    
    def test_is_final_stage(self):
        plan = ExecutionPlan()
        plan.set_component_ids(["net_1"])
        
        assert plan.is_final_stage(Stage.EXPORT.value) is True
        assert plan.is_final_stage(Stage.VALIDITY.value) is False


class TestStageConstants:
    """Tests for stage constants."""
    
    def test_stage_order_starts_with_compile_policies(self):
        assert STAGE_ORDER[0] == Stage.COMPILE_POLICIES
    
    def test_stage_order_ends_with_export(self):
        assert STAGE_ORDER[-1] == Stage.EXPORT
    
    def test_stage_order_has_all_stages(self):
        assert len(STAGE_ORDER) == len(Stage)
    
    def test_union_voids_before_embed(self):
        union_idx = STAGE_ORDER.index(Stage.UNION_VOIDS)
        embed_idx = STAGE_ORDER.index(Stage.EMBED)
        assert union_idx < embed_idx


class TestValidatePlan:
    """Tests for plan validation."""
    
    def test_validate_valid_plan(self):
        plan = ExecutionPlan()
        errors = validate_plan(plan, component_ids=["net_1"])
        assert len(errors) == 0
    
    def test_validate_plan_with_invalid_run_until(self):
        plan = ExecutionPlan(run_until="invalid_stage")
        errors = validate_plan(plan, component_ids=["net_1"])
        assert len(errors) > 0
    
    def test_validate_plan_with_invalid_skip(self):
        plan = ExecutionPlan(skip=["invalid_stage"])
        errors = validate_plan(plan, component_ids=["net_1"])
        assert len(errors) > 0
