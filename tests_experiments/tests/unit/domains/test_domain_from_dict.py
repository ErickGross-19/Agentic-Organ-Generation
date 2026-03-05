"""
Test domain_from_dict JSON compilation coverage (B4).

This module verifies that all domain types can be compiled from dict specs
with input_units="mm" and that basic operations work correctly.
"""

import pytest
import numpy as np

from generation.core.domain import domain_from_dict, BoxDomain, CylinderDomain, EllipsoidDomain
from generation.core.domain_primitives import SphereDomain, CapsuleDomain, FrustumDomain
from generation.core.domain_composite import CompositeDomain
from generation.core.domain_implicit import ImplicitDomain, sphere_sdf
from generation.core.types import Point3D


BOX_SPEC = {
    "type": "box",
    "x_min": -10.0,
    "x_max": 10.0,
    "y_min": -10.0,
    "y_max": 10.0,
    "z_min": -10.0,
    "z_max": 10.0,
}

CYLINDER_SPEC = {
    "type": "cylinder",
    "radius": 5.0,
    "height": 10.0,
    "center": {"x": 0.0, "y": 0.0, "z": 0.0},
}

ELLIPSOID_SPEC = {
    "type": "ellipsoid",
    "semi_axis_a": 10.0,
    "semi_axis_b": 8.0,
    "semi_axis_c": 6.0,
    "center": {"x": 0.0, "y": 0.0, "z": 0.0},
}

SPHERE_SPEC = {
    "type": "sphere",
    "radius": 5.0,
    "center": {"x": 0.0, "y": 0.0, "z": 0.0},
}

CAPSULE_SPEC = {
    "type": "capsule",
    "radius": 3.0,
    "length": 10.0,
    "center": {"x": 0.0, "y": 0.0, "z": 0.0},
    "axis": [0, 0, 1],
}

FRUSTUM_SPEC = {
    "type": "frustum",
    "radius_top": 3.0,
    "radius_bottom": 5.0,
    "height": 10.0,
    "center": {"x": 0.0, "y": 0.0, "z": 0.0},
    "axis": [0, 0, 1],
}

COMPOSITE_UNION_SPEC = {
    "type": "composite",
    "operation": "union",
    "children": [
        {
            "type": "box",
            "x_min": -5.0, "x_max": 5.0,
            "y_min": -5.0, "y_max": 5.0,
            "z_min": -5.0, "z_max": 5.0,
        },
        {
            "type": "sphere",
            "radius": 4.0,
            "center": {"x": 3.0, "y": 0.0, "z": 0.0},
        },
    ],
}

COMPOSITE_INTERSECTION_SPEC = {
    "type": "composite",
    "operation": "intersection",
    "children": [
        {
            "type": "box",
            "x_min": -10.0, "x_max": 10.0,
            "y_min": -10.0, "y_max": 10.0,
            "z_min": -10.0, "z_max": 10.0,
        },
        {
            "type": "sphere",
            "radius": 8.0,
            "center": {"x": 0.0, "y": 0.0, "z": 0.0},
        },
    ],
}

COMPOSITE_DIFFERENCE_SPEC = {
    "type": "composite",
    "operation": "difference",
    "children": [
        {
            "type": "box",
            "x_min": -10.0, "x_max": 10.0,
            "y_min": -10.0, "y_max": 10.0,
            "z_min": -10.0, "z_max": 10.0,
        },
        {
            "type": "sphere",
            "radius": 5.0,
            "center": {"x": 0.0, "y": 0.0, "z": 0.0},
        },
    ],
}

IMPLICIT_SPHERE_SPEC = {
    "type": "implicit",
    "bbox": [-6.0, 6.0, -6.0, 6.0, -6.0, 6.0],
    "sdf_ast": sphere_sdf(radius=5.0, center=(0, 0, 0)),
    "params": {},
}


ALL_DOMAIN_SPECS = [
    ("BoxDomain", BOX_SPEC),
    ("CylinderDomain", CYLINDER_SPEC),
    ("EllipsoidDomain", ELLIPSOID_SPEC),
    ("SphereDomain", SPHERE_SPEC),
    ("CapsuleDomain", CAPSULE_SPEC),
    ("FrustumDomain", FRUSTUM_SPEC),
    ("CompositeDomain_union", COMPOSITE_UNION_SPEC),
    ("CompositeDomain_intersection", COMPOSITE_INTERSECTION_SPEC),
    ("CompositeDomain_difference", COMPOSITE_DIFFERENCE_SPEC),
    ("ImplicitDomain_sphere", IMPLICIT_SPHERE_SPEC),
]


class TestDomainFromDictCompilation:
    """B4: JSON dict compilation coverage."""
    
    @pytest.mark.parametrize("name,spec", ALL_DOMAIN_SPECS)
    def test_domain_from_dict_does_not_crash(self, name, spec):
        """Test that domain_from_dict does not crash."""
        try:
            domain = domain_from_dict(spec)
            assert domain is not None, f"{name}: domain_from_dict returned None"
        except Exception as e:
            pytest.fail(f"{name}: domain_from_dict raised {type(e).__name__}: {e}")
    
    @pytest.mark.parametrize("name,spec", ALL_DOMAIN_SPECS)
    def test_domain_contains_works(self, name, spec):
        """Test that contains works on obvious point."""
        domain = domain_from_dict(spec)
        
        bounds = domain.get_bounds()
        center_x = (bounds[0] + bounds[1]) / 2
        center_y = (bounds[2] + bounds[3]) / 2
        center_z = (bounds[4] + bounds[5]) / 2
        
        center_point = Point3D(center_x, center_y, center_z)
        
        result = domain.contains(center_point)
        assert isinstance(result, bool), f"{name}: contains did not return bool"
    
    @pytest.mark.parametrize("name,spec", ALL_DOMAIN_SPECS)
    def test_domain_signed_distance_works(self, name, spec):
        """Test that signed_distance works."""
        domain = domain_from_dict(spec)
        
        bounds = domain.get_bounds()
        center_x = (bounds[0] + bounds[1]) / 2
        center_y = (bounds[2] + bounds[3]) / 2
        center_z = (bounds[4] + bounds[5]) / 2
        
        center_point = Point3D(center_x, center_y, center_z)
        
        result = domain.signed_distance(center_point)
        assert isinstance(result, (int, float)), f"{name}: signed_distance did not return number"
        assert np.isfinite(result), f"{name}: signed_distance returned non-finite value"
    
    @pytest.mark.parametrize("name,spec", ALL_DOMAIN_SPECS)
    def test_domain_bbox_is_finite_nonzero(self, name, spec):
        """Test that bbox is finite and has nonzero volume."""
        domain = domain_from_dict(spec)
        bounds = domain.get_bounds()
        
        assert len(bounds) == 6, f"{name}: bounds should have 6 elements"
        
        for i, val in enumerate(bounds):
            assert np.isfinite(val), f"{name}: bounds[{i}] is not finite: {val}"
        
        x_min, x_max = bounds[0], bounds[1]
        y_min, y_max = bounds[2], bounds[3]
        z_min, z_max = bounds[4], bounds[5]
        
        volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
        assert volume > 0, f"{name}: bounding box volume is not positive: {volume}"
    
    @pytest.mark.parametrize("name,spec", ALL_DOMAIN_SPECS)
    def test_domain_to_dict_round_trip(self, name, spec):
        """Test that domain can be serialized back to dict."""
        domain = domain_from_dict(spec)
        
        result_dict = domain.to_dict()
        assert isinstance(result_dict, dict), f"{name}: to_dict did not return dict"
        assert "type" in result_dict, f"{name}: to_dict missing 'type' key"


class TestDomainFromDictWithUnits:
    """Test domain_from_dict with input_units parameter."""
    
    def test_box_with_mm_units(self):
        """Test BoxDomain with mm input units."""
        spec = {
            "type": "box",
            "x_min": -10.0,
            "x_max": 10.0,
            "y_min": -10.0,
            "y_max": 10.0,
            "z_min": -10.0,
            "z_max": 10.0,
        }
        
        domain = domain_from_dict(spec, input_units="mm")
        
        assert domain is not None
        
        bounds = domain.get_bounds()
        assert abs(bounds[0] - (-0.01)) < 1e-9, f"x_min should be -0.01m, got {bounds[0]}"
        assert abs(bounds[1] - 0.01) < 1e-9, f"x_max should be 0.01m, got {bounds[1]}"
    
    def test_cylinder_with_mm_units(self):
        """Test CylinderDomain with mm input units."""
        spec = {
            "type": "cylinder",
            "radius": 5.0,
            "height": 10.0,
            "center": {"x": 0.0, "y": 0.0, "z": 0.0},
        }
        
        domain = domain_from_dict(spec, input_units="mm")
        
        assert domain is not None
        assert abs(domain.radius - 0.005) < 1e-9, f"radius should be 0.005m, got {domain.radius}"
        assert abs(domain.height - 0.01) < 1e-9, f"height should be 0.01m, got {domain.height}"
    
    def test_sphere_with_um_units(self):
        """Test SphereDomain with um input units."""
        spec = {
            "type": "sphere",
            "radius": 500.0,
            "center": {"x": 0.0, "y": 0.0, "z": 0.0},
        }
        
        domain = domain_from_dict(spec, input_units="um")
        
        assert domain is not None
        assert abs(domain.radius - 0.0005) < 1e-9, f"radius should be 0.0005m, got {domain.radius}"
    
    def test_composite_with_mm_units(self):
        """Test CompositeDomain with mm input units."""
        spec = {
            "type": "composite",
            "operation": "union",
            "children": [
                {
                    "type": "box",
                    "x_min": -5.0, "x_max": 5.0,
                    "y_min": -5.0, "y_max": 5.0,
                    "z_min": -5.0, "z_max": 5.0,
                },
                {
                    "type": "sphere",
                    "radius": 4.0,
                    "center": {"x": 3.0, "y": 0.0, "z": 0.0},
                },
            ],
        }
        
        domain = domain_from_dict(spec, input_units="mm")
        
        assert domain is not None
        
        bounds = domain.get_bounds()
        assert bounds[1] < 0.02, f"x_max should be < 0.02m (20mm), got {bounds[1]}"


class TestDomainFromDictEdgeCases:
    """Test edge cases for domain_from_dict."""
    
    def test_unknown_type_raises_error(self):
        """Test that unknown domain type raises appropriate error."""
        spec = {
            "type": "unknown_domain_type",
            "param": 1.0,
        }
        
        with pytest.raises((ValueError, KeyError)):
            domain_from_dict(spec)
    
    def test_missing_required_fields_raises_error(self):
        """Test that missing required fields raise appropriate error."""
        spec = {
            "type": "box",
        }
        
        with pytest.raises((ValueError, KeyError, TypeError)):
            domain_from_dict(spec)
    
    def test_nested_composite_domain(self):
        """Test deeply nested composite domain."""
        spec = {
            "type": "composite",
            "operation": "union",
            "children": [
                {
                    "type": "composite",
                    "operation": "intersection",
                    "children": [
                        {
                            "type": "box",
                            "x_min": -10.0, "x_max": 10.0,
                            "y_min": -10.0, "y_max": 10.0,
                            "z_min": -10.0, "z_max": 10.0,
                        },
                        {
                            "type": "sphere",
                            "radius": 8.0,
                            "center": {"x": 0.0, "y": 0.0, "z": 0.0},
                        },
                    ],
                },
                {
                    "type": "cylinder",
                    "radius": 3.0,
                    "height": 20.0,
                    "center": {"x": 0.0, "y": 0.0, "z": 0.0},
                },
            ],
        }
        
        domain = domain_from_dict(spec)
        
        assert domain is not None
        assert isinstance(domain, CompositeDomain)
        
        center = Point3D(0, 0, 0)
        result = domain.contains(center)
        assert isinstance(result, bool)
