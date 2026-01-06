"""
Tests for the unit conversion system.

These tests validate the dimensionless internal units system where:
- All internal calculations use dimensionless values (1 is 1)
- Scaling to user-specified units happens only at export/output time
- When output_units="mm", 1 internal unit = 1 mm in output files
"""

import pytest
import numpy as np


class TestUnitConversion:
    """Tests for basic unit conversion functions."""
    
    def test_to_si_length_mm(self):
        """Test conversion from mm to SI (meters)."""
        from generation.utils.units import to_si_length
        
        result = to_si_length(5.0, "mm")
        assert result == pytest.approx(0.005)
    
    def test_to_si_length_cm(self):
        """Test conversion from cm to SI (meters)."""
        from generation.utils.units import to_si_length
        
        result = to_si_length(10.0, "cm")
        assert result == pytest.approx(0.1)
    
    def test_to_si_length_m(self):
        """Test conversion from m to SI (meters) - identity."""
        from generation.utils.units import to_si_length
        
        result = to_si_length(1.0, "m")
        assert result == pytest.approx(1.0)
    
    def test_to_si_length_um(self):
        """Test conversion from um to SI (meters)."""
        from generation.utils.units import to_si_length
        
        result = to_si_length(1000.0, "um")
        assert result == pytest.approx(0.001)
    
    def test_from_si_length_mm(self):
        """Test conversion from SI (meters) to mm."""
        from generation.utils.units import from_si_length
        
        result = from_si_length(0.005, "mm")
        assert result == pytest.approx(5.0)
    
    def test_from_si_length_cm(self):
        """Test conversion from SI (meters) to cm."""
        from generation.utils.units import from_si_length
        
        result = from_si_length(0.1, "cm")
        assert result == pytest.approx(10.0)
    
    def test_convert_length_mm_to_cm(self):
        """Test conversion between mm and cm."""
        from generation.utils.units import convert_length
        
        result = convert_length(100.0, "mm", "cm")
        assert result == pytest.approx(10.0)
    
    def test_convert_length_cm_to_mm(self):
        """Test conversion between cm and mm."""
        from generation.utils.units import convert_length
        
        result = convert_length(5.0, "cm", "mm")
        assert result == pytest.approx(50.0)
    
    def test_convert_length_same_unit(self):
        """Test conversion when source and target are the same."""
        from generation.utils.units import convert_length
        
        result = convert_length(42.0, "mm", "mm")
        assert result == pytest.approx(42.0)
    
    def test_convert_length_roundtrip(self):
        """Test that conversion is reversible."""
        from generation.utils.units import convert_length
        
        original = 123.456
        converted = convert_length(original, "mm", "m")
        back = convert_length(converted, "m", "mm")
        assert back == pytest.approx(original)
    
    def test_convert_length_array(self):
        """Test conversion with numpy arrays."""
        from generation.utils.units import convert_length
        
        values = np.array([1.0, 2.0, 3.0])
        result = convert_length(values, "mm", "cm")
        expected = np.array([0.1, 0.2, 0.3])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_invalid_unit_raises(self):
        """Test that invalid units raise ValueError."""
        from generation.utils.units import to_si_length
        
        with pytest.raises(ValueError, match="Unknown unit"):
            to_si_length(1.0, "invalid_unit")


class TestUnitContext:
    """Tests for the UnitContext class."""
    
    def test_unit_context_default(self):
        """Test default UnitContext (mm output)."""
        from generation.utils.units import UnitContext
        
        ctx = UnitContext()
        assert ctx.output_units == "mm"
        assert ctx.scale_factor == pytest.approx(1.0)
    
    def test_unit_context_mm(self):
        """Test UnitContext with mm output."""
        from generation.utils.units import UnitContext
        
        ctx = UnitContext(output_units="mm")
        assert ctx.scale_factor == pytest.approx(1.0)
    
    def test_unit_context_m(self):
        """Test UnitContext with m output."""
        from generation.utils.units import UnitContext
        
        ctx = UnitContext(output_units="m")
        assert ctx.scale_factor == pytest.approx(0.001)
    
    def test_unit_context_cm(self):
        """Test UnitContext with cm output."""
        from generation.utils.units import UnitContext
        
        ctx = UnitContext(output_units="cm")
        assert ctx.scale_factor == pytest.approx(0.1)
    
    def test_unit_context_um(self):
        """Test UnitContext with um output."""
        from generation.utils.units import UnitContext
        
        ctx = UnitContext(output_units="um")
        assert ctx.scale_factor == pytest.approx(1000.0)
    
    def test_unit_context_to_output(self):
        """Test converting internal value to output units."""
        from generation.utils.units import UnitContext
        
        ctx = UnitContext(output_units="m")
        result = ctx.to_output(50.0)
        assert result == pytest.approx(0.05)
    
    def test_unit_context_from_output(self):
        """Test converting output value to internal units."""
        from generation.utils.units import UnitContext
        
        ctx = UnitContext(output_units="m")
        result = ctx.from_output(0.05)
        assert result == pytest.approx(50.0)
    
    def test_unit_context_roundtrip(self):
        """Test that to_output and from_output are inverses."""
        from generation.utils.units import UnitContext
        
        ctx = UnitContext(output_units="cm")
        original = 123.456
        converted = ctx.to_output(original)
        back = ctx.from_output(converted)
        assert back == pytest.approx(original)
    
    def test_unit_context_to_dict(self):
        """Test serialization to dict."""
        from generation.utils.units import UnitContext
        
        ctx = UnitContext(output_units="cm")
        d = ctx.to_dict()
        assert d["output_units"] == "cm"
        assert d["scale_factor"] == pytest.approx(0.1)
    
    def test_unit_context_from_dict(self):
        """Test deserialization from dict."""
        from generation.utils.units import UnitContext
        
        d = {"output_units": "um"}
        ctx = UnitContext.from_dict(d)
        assert ctx.output_units == "um"
        assert ctx.scale_factor == pytest.approx(1000.0)
    
    def test_unit_context_get_metadata(self):
        """Test getting metadata dict."""
        from generation.utils.units import UnitContext
        
        ctx = UnitContext(output_units="mm")
        metadata = ctx.get_metadata()
        assert metadata["units"] == "mm"
        assert "scale_factor_applied" in metadata
        assert "internal_units" in metadata
    
    def test_unit_context_invalid_unit_raises(self):
        """Test that invalid output_units raises ValueError."""
        from generation.utils.units import UnitContext
        
        with pytest.raises(ValueError, match="Unknown output_units"):
            UnitContext(output_units="invalid")
    
    def test_unit_context_array_conversion(self):
        """Test conversion with numpy arrays."""
        from generation.utils.units import UnitContext
        
        ctx = UnitContext(output_units="cm")
        values = np.array([10.0, 20.0, 30.0])
        result = ctx.to_output(values)
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(result, expected)


class TestDetectUnit:
    """Tests for unit detection heuristics."""
    
    def test_detect_unit_small_value_meters(self):
        """Test that small values are detected as meters."""
        from generation.utils.units import detect_unit
        
        result = detect_unit(0.005)
        assert result == "m"
    
    def test_detect_unit_large_value_mm(self):
        """Test that larger values are detected as mm."""
        from generation.utils.units import detect_unit
        
        result = detect_unit(5.0)
        assert result == "mm"
    
    def test_detect_unit_domain_size_small(self):
        """Test domain size detection for small values."""
        from generation.utils.units import detect_unit
        
        result = detect_unit(0.12, context="domain_size")
        assert result == "m"
    
    def test_detect_unit_domain_size_large(self):
        """Test domain size detection for large values."""
        from generation.utils.units import detect_unit
        
        result = detect_unit(120.0, context="domain_size")
        assert result == "mm"


class TestSameGeometryDifferentUnits:
    """Tests to verify same physical design produces same internal geometry."""
    
    def test_same_internal_geometry_mm_vs_m(self):
        """Test that same physical design in mm vs m produces same internal geometry."""
        from generation.utils.units import UnitContext
        
        ctx_mm = UnitContext(output_units="mm")
        ctx_m = UnitContext(output_units="m")
        
        internal_value = 50.0
        
        output_mm = ctx_mm.to_output(internal_value)
        output_m = ctx_m.to_output(internal_value)
        
        assert output_mm == pytest.approx(50.0)
        assert output_m == pytest.approx(0.05)
        
        back_from_mm = ctx_mm.from_output(output_mm)
        back_from_m = ctx_m.from_output(output_m)
        
        assert back_from_mm == pytest.approx(internal_value)
        assert back_from_m == pytest.approx(internal_value)
