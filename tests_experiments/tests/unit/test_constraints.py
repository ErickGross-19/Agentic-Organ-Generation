"""
Tests for the constraints module with meter-scale defaults.

These tests validate that:
- BranchingConstraints uses meter-scale defaults (not mm-scale)
- InteractionRuleSpec uses meter-scale defaults
- DegradationRuleSpec uses meter-scale defaults
- All spatial parameters are in METERS (SI units)
"""

import pytest


class TestBranchingConstraintsMeterScale:
    """Tests for BranchingConstraints meter-scale defaults."""
    
    def test_branching_constraints_default_min_radius_is_meters(self):
        """Test that default min_radius is in meters (3e-4 = 0.3mm)."""
        from generation.rules.constraints import BranchingConstraints
        
        constraints = BranchingConstraints()
        
        assert constraints.min_radius == pytest.approx(3e-4)
        assert constraints.min_radius < 0.001
    
    def test_branching_constraints_default_max_radius_is_meters(self):
        """Test that default max_radius is in meters (1e-2 = 10mm)."""
        from generation.rules.constraints import BranchingConstraints
        
        constraints = BranchingConstraints()
        
        assert constraints.max_radius == pytest.approx(1e-2)
        assert constraints.max_radius < 0.1
    
    def test_branching_constraints_default_min_segment_length_is_meters(self):
        """Test that default min_segment_length is in meters (1e-3 = 1mm)."""
        from generation.rules.constraints import BranchingConstraints
        
        constraints = BranchingConstraints()
        
        assert constraints.min_segment_length == pytest.approx(1e-3)
        assert constraints.min_segment_length < 0.01
    
    def test_branching_constraints_default_max_segment_length_is_meters(self):
        """Test that default max_segment_length is in meters (5e-2 = 50mm)."""
        from generation.rules.constraints import BranchingConstraints
        
        constraints = BranchingConstraints()
        
        assert constraints.max_segment_length == pytest.approx(5e-2)
        assert constraints.max_segment_length < 0.1
    
    def test_branching_constraints_from_dict_uses_meter_defaults(self):
        """Test that from_dict uses meter-scale defaults for missing values."""
        from generation.rules.constraints import BranchingConstraints
        
        constraints = BranchingConstraints.from_dict({})
        
        assert constraints.min_radius == pytest.approx(3e-4)
        assert constraints.max_radius == pytest.approx(1e-2)
        assert constraints.min_segment_length == pytest.approx(1e-3)
        assert constraints.max_segment_length == pytest.approx(5e-2)
    
    def test_branching_constraints_to_dict_roundtrip(self):
        """Test that to_dict/from_dict roundtrip preserves values."""
        from generation.rules.constraints import BranchingConstraints
        
        original = BranchingConstraints(
            min_radius=5e-4,
            max_radius=2e-2,
            min_segment_length=2e-3,
            max_segment_length=1e-1,
        )
        
        d = original.to_dict()
        restored = BranchingConstraints.from_dict(d)
        
        assert restored.min_radius == pytest.approx(original.min_radius)
        assert restored.max_radius == pytest.approx(original.max_radius)
        assert restored.min_segment_length == pytest.approx(original.min_segment_length)
        assert restored.max_segment_length == pytest.approx(original.max_segment_length)
    
    def test_branching_constraints_physically_plausible(self):
        """Test that default constraints are physically plausible for bioprinting.
        
        Bioprinting scale:
        - Capillaries: 0.1-0.3mm diameter (0.05-0.15mm radius)
        - Small vessels: 0.5-2mm diameter
        - Large vessels: up to 10mm diameter
        
        So min_radius=3e-4 (0.3mm) and max_radius=1e-2 (10mm) are reasonable.
        """
        from generation.rules.constraints import BranchingConstraints
        
        constraints = BranchingConstraints()
        
        min_radius_mm = constraints.min_radius * 1000
        max_radius_mm = constraints.max_radius * 1000
        
        assert 0.1 <= min_radius_mm <= 1.0
        assert 5.0 <= max_radius_mm <= 20.0


class TestInteractionRuleSpecMeterScale:
    """Tests for InteractionRuleSpec meter-scale defaults."""
    
    def test_interaction_rule_spec_default_distances_are_meters(self):
        """Test that default min_distance_between_types is in meters."""
        from generation.rules.constraints import InteractionRuleSpec
        
        spec = InteractionRuleSpec()
        
        arterial_venous_dist = spec.get_min_distance("arterial", "venous")
        arterial_arterial_dist = spec.get_min_distance("arterial", "arterial")
        
        assert arterial_venous_dist == pytest.approx(1e-3)
        assert arterial_arterial_dist == pytest.approx(5e-4)
    
    def test_interaction_rule_spec_default_distance_fallback_is_meters(self):
        """Test that default fallback distance is in meters."""
        from generation.rules.constraints import InteractionRuleSpec
        
        spec = InteractionRuleSpec()
        
        unknown_dist = spec.get_min_distance("unknown1", "unknown2")
        
        assert unknown_dist == pytest.approx(1e-3)
    
    def test_interaction_rule_spec_distances_physically_plausible(self):
        """Test that default distances are physically plausible.
        
        1mm clearance between arterial/venous is reasonable for bioprinting.
        0.5mm clearance within same type is also reasonable.
        """
        from generation.rules.constraints import InteractionRuleSpec
        
        spec = InteractionRuleSpec()
        
        arterial_venous_mm = spec.get_min_distance("arterial", "venous") * 1000
        arterial_arterial_mm = spec.get_min_distance("arterial", "arterial") * 1000
        
        assert 0.5 <= arterial_venous_mm <= 2.0
        assert 0.25 <= arterial_arterial_mm <= 1.0


class TestDegradationRuleSpecMeterScale:
    """Tests for DegradationRuleSpec meter-scale defaults."""
    
    def test_degradation_rule_spec_default_min_terminal_radius_is_meters(self):
        """Test that default min_terminal_radius is in meters (1e-4 = 0.1mm)."""
        from generation.rules.constraints import DegradationRuleSpec
        
        spec = DegradationRuleSpec()
        
        assert spec.min_terminal_radius == pytest.approx(1e-4)
        assert spec.min_terminal_radius < 0.001
    
    def test_degradation_rule_spec_from_dict_uses_meter_defaults(self):
        """Test that from_dict uses meter-scale defaults."""
        from generation.rules.constraints import DegradationRuleSpec
        
        spec = DegradationRuleSpec.from_dict({})
        
        assert spec.min_terminal_radius == pytest.approx(1e-4)
    
    def test_degradation_rule_spec_exponential_factory_uses_meters(self):
        """Test that exponential() factory uses meter-scale default."""
        from generation.rules.constraints import DegradationRuleSpec
        
        spec = DegradationRuleSpec.exponential()
        
        assert spec.min_terminal_radius == pytest.approx(1e-4)
        assert spec.model == "exponential"
    
    def test_degradation_rule_spec_linear_factory_uses_meters(self):
        """Test that linear() factory uses meter-scale default."""
        from generation.rules.constraints import DegradationRuleSpec
        
        spec = DegradationRuleSpec.linear()
        
        assert spec.min_terminal_radius == pytest.approx(1e-4)
        assert spec.model == "linear"
    
    def test_degradation_rule_spec_apply_degradation(self):
        """Test that apply_degradation works correctly."""
        from generation.rules.constraints import DegradationRuleSpec
        
        spec = DegradationRuleSpec(
            model="exponential",
            degradation_factor=0.85,
            min_terminal_radius=1e-4,
        )
        
        parent_radius = 1e-3
        child_radius = spec.apply_degradation(parent_radius, generation=1)
        
        assert child_radius < parent_radius
        assert child_radius >= spec.min_terminal_radius
    
    def test_degradation_rule_spec_should_terminate_at_min_radius(self):
        """Test that should_terminate returns True at min_terminal_radius."""
        from generation.rules.constraints import DegradationRuleSpec
        
        spec = DegradationRuleSpec(min_terminal_radius=1e-4)
        
        should_term, reason = spec.should_terminate(radius=1e-4, generation=0)
        
        assert should_term is True
        assert reason is not None
        assert "m" in reason
    
    def test_degradation_rule_spec_min_terminal_radius_physically_plausible(self):
        """Test that default min_terminal_radius is physically plausible.
        
        0.1mm (1e-4 m) is capillary scale, which is reasonable for
        the smallest vessels in a vascular network.
        """
        from generation.rules.constraints import DegradationRuleSpec
        
        spec = DegradationRuleSpec()
        
        min_radius_mm = spec.min_terminal_radius * 1000
        
        assert 0.05 <= min_radius_mm <= 0.5


class TestConstraintsUnitAudit:
    """Unit audit tests to ensure defaults are physically plausible.
    
    These tests fail if defaults are not physically plausible under
    the stated internal unit convention (meters).
    """
    
    def test_all_spatial_defaults_are_sub_meter(self):
        """Test that all spatial defaults are less than 1 meter.
        
        This catches the common mistake of using mm-scale values
        when the internal unit is meters.
        """
        from generation.rules.constraints import (
            BranchingConstraints,
            InteractionRuleSpec,
            DegradationRuleSpec,
        )
        
        bc = BranchingConstraints()
        assert bc.min_radius < 1.0
        assert bc.max_radius < 1.0
        assert bc.min_segment_length < 1.0
        assert bc.max_segment_length < 1.0
        
        ir = InteractionRuleSpec()
        for key, value in ir.min_distance_between_types.items():
            assert value < 1.0, f"min_distance for {key} is >= 1m"
        
        dr = DegradationRuleSpec()
        assert dr.min_terminal_radius < 1.0
    
    def test_all_spatial_defaults_are_positive(self):
        """Test that all spatial defaults are positive."""
        from generation.rules.constraints import (
            BranchingConstraints,
            InteractionRuleSpec,
            DegradationRuleSpec,
        )
        
        bc = BranchingConstraints()
        assert bc.min_radius > 0
        assert bc.max_radius > 0
        assert bc.min_segment_length > 0
        assert bc.max_segment_length > 0
        
        ir = InteractionRuleSpec()
        for key, value in ir.min_distance_between_types.items():
            assert value > 0, f"min_distance for {key} is <= 0"
        
        dr = DegradationRuleSpec()
        assert dr.min_terminal_radius > 0
    
    def test_min_max_ordering(self):
        """Test that min values are less than max values."""
        from generation.rules.constraints import BranchingConstraints
        
        bc = BranchingConstraints()
        
        assert bc.min_radius < bc.max_radius
        assert bc.min_segment_length < bc.max_segment_length
