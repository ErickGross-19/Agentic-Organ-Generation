"""
Unit tests for project_inside() margin enforcement in domain classes.

These tests verify that project_inside(point, margin=...) enforces the margin
even when the point is already inside the domain but too close to the boundary.

This is a regression test for the bug where domain classes returned early
when contains(point) was True, ignoring the margin parameter.
"""

import pytest
import numpy as np
from generation.core.domain import BoxDomain, CylinderDomain, EllipsoidDomain
from generation.core.types import Point3D


class TestBoxDomainProjectInsideMargin:
    """Tests for BoxDomain.project_inside() margin enforcement."""
    
    def test_inside_with_sufficient_margin_unchanged(self):
        """Point inside with sufficient margin should be unchanged."""
        domain = BoxDomain(x_min=0, x_max=10, y_min=0, y_max=10, z_min=0, z_max=10)
        margin = 1.0
        
        # Point at center has plenty of margin
        point = Point3D(5.0, 5.0, 5.0)
        result = domain.project_inside(point, margin=margin)
        
        assert result.x == point.x
        assert result.y == point.y
        assert result.z == point.z
    
    def test_inside_near_boundary_moved_inward(self):
        """Point inside but too close to boundary should be moved inward."""
        domain = BoxDomain(x_min=0, x_max=10, y_min=0, y_max=10, z_min=0, z_max=10)
        margin = 1.0
        
        # Point is inside but only 0.25*margin from +X wall
        point = Point3D(9.75, 5.0, 5.0)
        result = domain.project_inside(point, margin=margin)
        
        # Should be moved to x = 9.0 (x_max - margin)
        assert domain.contains(result)
        dist = domain.distance_to_boundary(result)
        assert dist >= margin - 1e-9, f"Distance {dist} should be >= margin {margin}"
    
    def test_inside_near_multiple_boundaries(self):
        """Point inside but near corner should be moved inward on all axes."""
        domain = BoxDomain(x_min=0, x_max=10, y_min=0, y_max=10, z_min=0, z_max=10)
        margin = 1.0
        
        # Point is inside but too close to corner
        point = Point3D(9.5, 9.5, 9.5)
        result = domain.project_inside(point, margin=margin)
        
        assert domain.contains(result)
        dist = domain.distance_to_boundary(result)
        assert dist >= margin - 1e-9
    
    def test_outside_projected_with_margin(self):
        """Point outside should be projected inside with margin."""
        domain = BoxDomain(x_min=0, x_max=10, y_min=0, y_max=10, z_min=0, z_max=10)
        margin = 1.0
        
        # Point is outside
        point = Point3D(15.0, 5.0, 5.0)
        result = domain.project_inside(point, margin=margin)
        
        assert domain.contains(result)
        dist = domain.distance_to_boundary(result)
        assert dist >= margin - 1e-9


class TestCylinderDomainProjectInsideMargin:
    """Tests for CylinderDomain.project_inside() margin enforcement."""
    
    def test_inside_with_sufficient_margin_unchanged(self):
        """Point inside with sufficient margin should be unchanged."""
        domain = CylinderDomain(radius=5.0, height=10.0, center=Point3D(0, 0, 0))
        margin = 0.5
        
        # Point at center has plenty of margin
        point = Point3D(0.0, 0.0, 0.0)
        result = domain.project_inside(point, margin=margin)
        
        assert result.x == point.x
        assert result.y == point.y
        assert result.z == point.z
    
    def test_inside_near_radial_boundary_moved_inward(self):
        """Point inside but too close to radial boundary should be moved inward."""
        domain = CylinderDomain(radius=5.0, height=10.0, center=Point3D(0, 0, 0))
        margin = 1.0
        
        # Point with radial distance = radius - 0.5*margin (too close)
        r_xy = 5.0 - 0.5 * margin  # 4.5
        point = Point3D(r_xy, 0.0, 0.0)
        result = domain.project_inside(point, margin=margin)
        
        # Check result is inside with sufficient margin
        assert domain.contains(result)
        
        # Check radial distance is <= radius - margin
        result_r_xy = np.sqrt(result.x**2 + result.y**2)
        assert result_r_xy <= domain.radius - margin + 1e-9
        
        dist = domain.distance_to_boundary(result)
        assert dist >= margin - 1e-9, f"Distance {dist} should be >= margin {margin}"
    
    def test_inside_near_axial_boundary_moved_inward(self):
        """Point inside but too close to top/bottom cap should be moved inward."""
        domain = CylinderDomain(radius=5.0, height=10.0, center=Point3D(0, 0, 0))
        margin = 1.0
        half_height = domain.height / 2
        
        # Point near top cap
        point = Point3D(0.0, 0.0, half_height - 0.3)  # 0.3 from top, less than margin
        result = domain.project_inside(point, margin=margin)
        
        assert domain.contains(result)
        dist = domain.distance_to_boundary(result)
        assert dist >= margin - 1e-9
    
    def test_on_axis_stays_on_axis(self):
        """Point on cylinder axis should stay on axis after projection."""
        domain = CylinderDomain(radius=5.0, height=10.0, center=Point3D(0, 0, 0))
        margin = 1.0
        
        # Point on axis but near top
        point = Point3D(0.0, 0.0, 4.5)
        result = domain.project_inside(point, margin=margin)
        
        # Should stay on axis (x=0, y=0)
        assert abs(result.x) < 1e-9
        assert abs(result.y) < 1e-9
        assert domain.contains(result)


class TestEllipsoidDomainProjectInsideMargin:
    """Tests for EllipsoidDomain.project_inside() margin enforcement."""
    
    def test_inside_with_sufficient_margin_unchanged(self):
        """Point inside with sufficient margin should be unchanged."""
        domain = EllipsoidDomain(
            semi_axis_a=5.0, semi_axis_b=4.0, semi_axis_c=3.0,
            center=Point3D(0, 0, 0)
        )
        margin = 0.5
        
        # Point at center has plenty of margin
        point = Point3D(0.0, 0.0, 0.0)
        result = domain.project_inside(point, margin=margin)
        
        assert result.x == point.x
        assert result.y == point.y
        assert result.z == point.z
    
    def test_inside_near_boundary_on_major_axis_moved_inward(self):
        """Point inside but near boundary on major axis should be moved inward."""
        domain = EllipsoidDomain(
            semi_axis_a=5.0, semi_axis_b=4.0, semi_axis_c=3.0,
            center=Point3D(0, 0, 0)
        )
        margin = 0.5
        
        # Point near boundary on x-axis (major axis)
        point = Point3D(4.8, 0.0, 0.0)  # Very close to boundary at x=5
        result = domain.project_inside(point, margin=margin)
        
        assert domain.contains(result)
        dist = domain.distance_to_boundary(result)
        assert dist >= margin - 1e-9, f"Distance {dist} should be >= margin {margin}"
    
    def test_inside_near_boundary_on_minor_axis_moved_inward(self):
        """Point inside but near boundary on minor axis should be moved inward."""
        domain = EllipsoidDomain(
            semi_axis_a=5.0, semi_axis_b=4.0, semi_axis_c=3.0,
            center=Point3D(0, 0, 0)
        )
        margin = 0.5
        
        # Point near boundary on z-axis (minor axis)
        point = Point3D(0.0, 0.0, 2.8)  # Very close to boundary at z=3
        result = domain.project_inside(point, margin=margin)
        
        assert domain.contains(result)
        dist = domain.distance_to_boundary(result)
        assert dist >= margin - 1e-9


class TestDomainProjectInsideMarginRegression:
    """Regression tests to ensure the early-return bug is fixed."""
    
    def test_box_early_return_bug_fixed(self):
        """
        Regression test: BoxDomain.project_inside() used to return early
        when contains(point) was True, ignoring the margin parameter.
        """
        domain = BoxDomain(x_min=0, x_max=10, y_min=0, y_max=10, z_min=0, z_max=10)
        margin = 2.0
        
        # Point is inside but distance to boundary is only 0.5 (less than margin)
        point = Point3D(9.5, 5.0, 5.0)
        
        # Before fix: would return point unchanged
        # After fix: should move point inward
        result = domain.project_inside(point, margin=margin)
        
        # The result should NOT be the same as input
        # (unless input already satisfies margin, which it doesn't)
        dist_before = domain.distance_to_boundary(point)
        dist_after = domain.distance_to_boundary(result)
        
        assert dist_before < margin, "Test setup: point should be too close to boundary"
        assert dist_after >= margin - 1e-9, "After projection, distance should satisfy margin"
    
    def test_cylinder_early_return_bug_fixed(self):
        """
        Regression test: CylinderDomain.project_inside() used to return early
        when contains(point) was True, ignoring the margin parameter.
        """
        domain = CylinderDomain(radius=5.0, height=10.0, center=Point3D(0, 0, 0))
        margin = 1.0
        
        # Point is inside but radial distance is radius - 0.3 (too close)
        point = Point3D(4.7, 0.0, 0.0)
        
        result = domain.project_inside(point, margin=margin)
        
        dist_before = domain.distance_to_boundary(point)
        dist_after = domain.distance_to_boundary(result)
        
        assert dist_before < margin, "Test setup: point should be too close to boundary"
        assert dist_after >= margin - 1e-9, "After projection, distance should satisfy margin"
    
    def test_ellipsoid_early_return_bug_fixed(self):
        """
        Regression test: EllipsoidDomain.project_inside() used to return early
        when contains(point) was True, ignoring the margin parameter.
        """
        domain = EllipsoidDomain(
            semi_axis_a=5.0, semi_axis_b=4.0, semi_axis_c=3.0,
            center=Point3D(0, 0, 0)
        )
        margin = 0.5
        
        # Point is inside but very close to boundary
        point = Point3D(4.9, 0.0, 0.0)
        
        result = domain.project_inside(point, margin=margin)
        
        dist_before = domain.distance_to_boundary(point)
        dist_after = domain.distance_to_boundary(result)
        
        assert dist_before < margin, "Test setup: point should be too close to boundary"
        assert dist_after >= margin - 1e-9, "After projection, distance should satisfy margin"
