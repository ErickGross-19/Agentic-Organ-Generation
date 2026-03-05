"""
Tests for the generation library functionality.

These tests validate:
- Design specifications (DesignSpec, TreeSpec, etc.)
- Network creation and manipulation
- Space colonization algorithm
- Serialization/deserialization

NOTE: This file is LEGACY and tests deprecated APIs (generation/specs/design_spec.py).
These tests are not run by default. Use `pytest -m legacy` to run them.
"""

import pytest
import json
import tempfile
import os


pytestmark = pytest.mark.legacy


class TestDesignSpec:
    """Tests for DesignSpec and related classes."""
    
    def test_ellipsoid_spec_creation(self):
        """Test creating an EllipsoidSpec."""
        from generation.specs.design_spec import EllipsoidSpec
        
        spec = EllipsoidSpec(
            center=(0.0, 0.0, 0.0),
            semi_axes=(50.0, 45.0, 35.0),
        )
        
        assert spec.type == "ellipsoid"
        assert spec.center == (0.0, 0.0, 0.0)
        assert spec.semi_axes == (50.0, 45.0, 35.0)
    
    def test_ellipsoid_spec_to_dict(self):
        """Test EllipsoidSpec serialization."""
        from generation.specs.design_spec import EllipsoidSpec
        
        spec = EllipsoidSpec(
            center=(1.0, 2.0, 3.0),
            semi_axes=(10.0, 20.0, 30.0),
        )
        
        d = spec.to_dict()
        assert d["type"] == "ellipsoid"
        assert d["center"] == (1.0, 2.0, 3.0)
        assert d["semi_axes"] == (10.0, 20.0, 30.0)
    
    def test_ellipsoid_spec_from_dict(self):
        """Test EllipsoidSpec deserialization."""
        from generation.specs.design_spec import EllipsoidSpec
        
        d = {
            "type": "ellipsoid",
            "center": [5.0, 6.0, 7.0],
            "semi_axes": [15.0, 25.0, 35.0],
        }
        
        spec = EllipsoidSpec.from_dict(d)
        assert spec.center == (5.0, 6.0, 7.0)
        assert spec.semi_axes == (15.0, 25.0, 35.0)
    
    def test_box_spec_creation(self):
        """Test creating a BoxSpec."""
        from generation.specs.design_spec import BoxSpec
        
        spec = BoxSpec(
            center=(0.0, 0.0, 0.0),
            size=(100.0, 90.0, 70.0),
        )
        
        assert spec.type == "box"
        assert spec.center == (0.0, 0.0, 0.0)
        assert spec.size == (100.0, 90.0, 70.0)
    
    def test_colonization_spec_defaults(self):
        """Test ColonizationSpec default values."""
        from generation.specs.design_spec import ColonizationSpec
        
        spec = ColonizationSpec()
        
        assert spec.influence_radius == 0.015
        assert spec.kill_radius == 0.002
        assert spec.step_size == 0.001
        assert spec.max_steps == 500
        assert spec.initial_radius == 0.0005
        assert spec.min_radius == 0.0001
        assert spec.radius_decay == 0.95
    
    def test_colonization_spec_to_dict(self):
        """Test ColonizationSpec serialization."""
        from generation.specs.design_spec import ColonizationSpec
        
        spec = ColonizationSpec(
            influence_radius=0.02,
            kill_radius=0.003,
        )
        
        d = spec.to_dict()
        assert d["influence_radius"] == 0.02
        assert d["kill_radius"] == 0.003
    
    def test_inlet_spec_creation(self):
        """Test creating an InletSpec."""
        from generation.specs.design_spec import InletSpec
        
        spec = InletSpec(
            position=(0.0, 0.0, 0.0),
            radius=5.0,
            vessel_type="arterial",
        )
        
        assert spec.position == (0.0, 0.0, 0.0)
        assert spec.radius == 5.0
        assert spec.vessel_type == "arterial"
    
    def test_outlet_spec_creation(self):
        """Test creating an OutletSpec."""
        from generation.specs.design_spec import OutletSpec
        
        spec = OutletSpec(
            position=(10.0, 10.0, 10.0),
            radius=3.0,
            vessel_type="venous",
        )
        
        assert spec.position == (10.0, 10.0, 10.0)
        assert spec.radius == 3.0
        assert spec.vessel_type == "venous"
    
    def test_tree_spec_single_inlet(self):
        """Test TreeSpec.single_inlet convenience constructor."""
        from generation.specs.design_spec import TreeSpec, ColonizationSpec
        
        colonization = ColonizationSpec()
        spec = TreeSpec.single_inlet(
            inlet_position=(0.0, 0.0, 0.0),
            inlet_radius=5.0,
            colonization=colonization,
        )
        
        assert len(spec.inlets) == 1
        assert len(spec.outlets) == 0
        assert spec.inlets[0].position == (0.0, 0.0, 0.0)
        assert spec.inlets[0].radius == 5.0
    
    def test_design_spec_with_tree(self):
        """Test creating a DesignSpec with a single tree."""
        from generation.specs.design_spec import (
            DesignSpec, EllipsoidSpec, TreeSpec, InletSpec, ColonizationSpec
        )
        
        domain = EllipsoidSpec(
            center=(0.0, 0.0, 0.0),
            semi_axes=(50.0, 45.0, 35.0),
        )
        
        tree = TreeSpec(
            inlets=[InletSpec(position=(-50.0, 0.0, 0.0), radius=5.0)],
            outlets=[],
            colonization=ColonizationSpec(),
        )
        
        spec = DesignSpec(
            domain=domain,
            tree=tree,
            seed=42,
        )
        
        assert spec.domain == domain
        assert spec.tree == tree
        assert spec.dual_tree is None
        assert spec.seed == 42
        assert spec.output_units == "mm"
    
    def test_design_spec_output_units(self):
        """Test DesignSpec with custom output_units."""
        from generation.specs.design_spec import (
            DesignSpec, EllipsoidSpec, TreeSpec, InletSpec, ColonizationSpec
        )
        
        domain = EllipsoidSpec()
        tree = TreeSpec(
            inlets=[InletSpec(position=(0.0, 0.0, 0.0), radius=1.0)],
            outlets=[],
            colonization=ColonizationSpec(),
        )
        
        spec = DesignSpec(
            domain=domain,
            tree=tree,
            output_units="cm",
        )
        
        assert spec.output_units == "cm"
    
    def test_design_spec_invalid_output_units(self):
        """Test that invalid output_units raises ValueError."""
        from generation.specs.design_spec import (
            DesignSpec, EllipsoidSpec, TreeSpec, InletSpec, ColonizationSpec
        )
        
        domain = EllipsoidSpec()
        tree = TreeSpec(
            inlets=[InletSpec(position=(0.0, 0.0, 0.0), radius=1.0)],
            outlets=[],
            colonization=ColonizationSpec(),
        )
        
        with pytest.raises(ValueError, match="Unknown output_units"):
            DesignSpec(
                domain=domain,
                tree=tree,
                output_units="invalid",
            )
    
    def test_design_spec_requires_tree_or_dual_tree(self):
        """Test that DesignSpec requires either tree or dual_tree."""
        from generation.specs.design_spec import DesignSpec, EllipsoidSpec
        
        domain = EllipsoidSpec()
        
        with pytest.raises(ValueError, match="Must specify either"):
            DesignSpec(domain=domain)
    
    def test_design_spec_to_dict_includes_output_units(self):
        """Test that DesignSpec.to_dict includes output_units."""
        from generation.specs.design_spec import (
            DesignSpec, EllipsoidSpec, TreeSpec, InletSpec, ColonizationSpec
        )
        
        domain = EllipsoidSpec()
        tree = TreeSpec(
            inlets=[InletSpec(position=(0.0, 0.0, 0.0), radius=1.0)],
            outlets=[],
            colonization=ColonizationSpec(),
        )
        
        spec = DesignSpec(
            domain=domain,
            tree=tree,
            output_units="um",
        )
        
        d = spec.to_dict()
        assert d["output_units"] == "um"
    
    def test_design_spec_from_dict_reads_output_units(self):
        """Test that DesignSpec.from_dict reads output_units."""
        from generation.specs.design_spec import (
            DesignSpec, EllipsoidSpec, TreeSpec, InletSpec, ColonizationSpec
        )
        
        domain = EllipsoidSpec()
        tree = TreeSpec(
            inlets=[InletSpec(position=(0.0, 0.0, 0.0), radius=1.0)],
            outlets=[],
            colonization=ColonizationSpec(),
        )
        
        original = DesignSpec(
            domain=domain,
            tree=tree,
            output_units="cm",
        )
        
        d = original.to_dict()
        restored = DesignSpec.from_dict(d)
        
        assert restored.output_units == "cm"
    
    def test_design_spec_from_dict_defaults_output_units(self):
        """Test that DesignSpec.from_dict defaults output_units to mm."""
        from generation.specs.design_spec import DesignSpec
        
        d = {
            "domain": {
                "type": "ellipsoid",
                "center": [0.0, 0.0, 0.0],
                "semi_axes": [50.0, 45.0, 35.0],
            },
            "tree": {
                "inlets": [{"position": [0.0, 0.0, 0.0], "radius": 1.0}],
                "outlets": [],
                "colonization": {},
            },
        }
        
        spec = DesignSpec.from_dict(d)
        assert spec.output_units == "mm"
    
    def test_design_spec_json_roundtrip(self):
        """Test DesignSpec JSON serialization roundtrip."""
        from generation.specs.design_spec import (
            DesignSpec, EllipsoidSpec, TreeSpec, InletSpec, ColonizationSpec
        )
        
        domain = EllipsoidSpec(
            center=(1.0, 2.0, 3.0),
            semi_axes=(50.0, 45.0, 35.0),
        )
        tree = TreeSpec(
            inlets=[InletSpec(position=(-50.0, 0.0, 0.0), radius=5.0)],
            outlets=[],
            colonization=ColonizationSpec(influence_radius=0.02),
        )
        
        original = DesignSpec(
            domain=domain,
            tree=tree,
            seed=42,
            output_units="cm",
            metadata={"test": "value"},
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            original.to_json(f.name)
            restored = DesignSpec.from_json(f.name)
        
        os.unlink(f.name)
        
        assert restored.seed == original.seed
        assert restored.output_units == original.output_units
        assert restored.metadata == original.metadata


class TestDualTreeSpec:
    """Tests for DualTreeSpec."""
    
    def test_dual_tree_spec_creation(self):
        """Test creating a DualTreeSpec."""
        from generation.specs.design_spec import (
            DualTreeSpec, InletSpec, OutletSpec, ColonizationSpec
        )
        
        spec = DualTreeSpec(
            arterial_inlets=[InletSpec(position=(0.0, 0.0, 0.0), radius=5.0)],
            venous_outlets=[OutletSpec(position=(10.0, 0.0, 0.0), radius=3.0)],
            arterial_colonization=ColonizationSpec(),
            venous_colonization=ColonizationSpec(),
        )
        
        assert len(spec.arterial_inlets) == 1
        assert len(spec.venous_outlets) == 1
    
    def test_dual_tree_spec_single_inlet_outlet(self):
        """Test DualTreeSpec.single_inlet_outlet convenience constructor."""
        from generation.specs.design_spec import DualTreeSpec, ColonizationSpec
        
        spec = DualTreeSpec.single_inlet_outlet(
            arterial_inlet_position=(0.0, 0.0, 0.0),
            arterial_inlet_radius=5.0,
            venous_outlet_position=(10.0, 0.0, 0.0),
            venous_outlet_radius=3.0,
            arterial_colonization=ColonizationSpec(),
            venous_colonization=ColonizationSpec(),
        )
        
        assert len(spec.arterial_inlets) == 1
        assert len(spec.venous_outlets) == 1
        assert spec.arterial_inlets[0].vessel_type == "arterial"
        assert spec.venous_outlets[0].vessel_type == "venous"
