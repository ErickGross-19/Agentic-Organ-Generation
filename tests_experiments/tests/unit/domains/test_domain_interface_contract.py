"""
Test domain interface contract (B1, B2, B3).

This module verifies that all domain types implement the DomainSpec interface
correctly, including:
- B1: contains and signed_distance sign agreement
- B2: project_inside correctness
- B3: faces contract (for domains that support faces)
"""

import pytest
import numpy as np
from typing import List, Tuple

from generation.core.domain import (
    DomainSpec,
    BoxDomain,
    CylinderDomain,
    EllipsoidDomain,
    MeshDomain,
)
from generation.core.domain_primitives import (
    SphereDomain,
    CapsuleDomain,
    FrustumDomain,
)
from generation.core.domain_composite import CompositeDomain
from generation.core.domain_implicit import ImplicitDomain, sphere_sdf
from generation.core.types import Point3D


def create_box_domain():
    """Create a test box domain."""
    return BoxDomain(
        x_min=-0.01, x_max=0.01,
        y_min=-0.01, y_max=0.01,
        z_min=-0.01, z_max=0.01,
    )


def create_cylinder_domain():
    """Create a test cylinder domain."""
    return CylinderDomain(
        radius=0.005,
        height=0.01,
        center=Point3D(0, 0, 0),
    )


def create_ellipsoid_domain():
    """Create a test ellipsoid domain."""
    return EllipsoidDomain(
        semi_axis_a=0.01,
        semi_axis_b=0.008,
        semi_axis_c=0.006,
        center=Point3D(0, 0, 0),
    )


def create_sphere_domain():
    """Create a test sphere domain."""
    return SphereDomain(
        radius=0.005,
        center=Point3D(0, 0, 0),
    )


def create_capsule_domain():
    """Create a test capsule domain."""
    return CapsuleDomain(
        radius=0.003,
        length=0.01,
        center=Point3D(0, 0, 0),
        axis=(0, 0, 1),
    )


def create_frustum_domain():
    """Create a test frustum domain."""
    return FrustumDomain(
        radius_top=0.003,
        radius_bottom=0.005,
        height=0.01,
        center=Point3D(0, 0, 0),
        axis=(0, 0, 1),
    )


def create_composite_union_domain():
    """Create a test composite union domain."""
    box = BoxDomain(
        x_min=-0.005, x_max=0.005,
        y_min=-0.005, y_max=0.005,
        z_min=-0.005, z_max=0.005,
    )
    sphere = SphereDomain(radius=0.004, center=Point3D(0.003, 0, 0))
    return CompositeDomain.union(box, sphere)


def create_composite_intersection_domain():
    """Create a test composite intersection domain."""
    box = BoxDomain(
        x_min=-0.01, x_max=0.01,
        y_min=-0.01, y_max=0.01,
        z_min=-0.01, z_max=0.01,
    )
    sphere = SphereDomain(radius=0.008, center=Point3D(0, 0, 0))
    return CompositeDomain.intersection(box, sphere)


def create_composite_difference_domain():
    """Create a test composite difference domain."""
    box = BoxDomain(
        x_min=-0.01, x_max=0.01,
        y_min=-0.01, y_max=0.01,
        z_min=-0.01, z_max=0.01,
    )
    sphere = SphereDomain(radius=0.005, center=Point3D(0, 0, 0))
    return CompositeDomain.difference(box, sphere)


def create_implicit_sphere_domain():
    """Create a test implicit sphere domain."""
    return ImplicitDomain.sphere(radius=0.005, center=(0, 0, 0))


ALL_DOMAIN_FACTORIES = [
    ("BoxDomain", create_box_domain),
    ("CylinderDomain", create_cylinder_domain),
    ("EllipsoidDomain", create_ellipsoid_domain),
    ("SphereDomain", create_sphere_domain),
    ("CapsuleDomain", create_capsule_domain),
    ("FrustumDomain", create_frustum_domain),
    ("CompositeDomain_union", create_composite_union_domain),
    ("CompositeDomain_intersection", create_composite_intersection_domain),
    ("CompositeDomain_difference", create_composite_difference_domain),
    ("ImplicitDomain_sphere", create_implicit_sphere_domain),
]

DOMAINS_WITH_FACES = [
    ("BoxDomain", create_box_domain),
    ("CylinderDomain", create_cylinder_domain),
    ("SphereDomain", create_sphere_domain),
    ("CapsuleDomain", create_capsule_domain),
    ("FrustumDomain", create_frustum_domain),
]


def sample_inside_points(domain: DomainSpec, n_points: int = 10, seed: int = 42) -> List[Point3D]:
    """Sample points that are inside the domain."""
    samples = domain.sample_points(n_points * 2, seed=seed)
    points = []
    for sample in samples:
        p = Point3D(sample[0], sample[1], sample[2])
        if domain.contains(p):
            points.append(p)
            if len(points) >= n_points:
                break
    return points


def sample_outside_points(domain: DomainSpec, n_points: int = 10, seed: int = 42) -> List[Point3D]:
    """Sample points that are outside the domain."""
    bounds = domain.get_bounds()
    x_min, x_max = bounds[0], bounds[1]
    y_min, y_max = bounds[2], bounds[3]
    z_min, z_max = bounds[4], bounds[5]
    
    margin = max(x_max - x_min, y_max - y_min, z_max - z_min) * 0.5
    
    rng = np.random.default_rng(seed)
    points = []
    
    for _ in range(n_points * 10):
        x = rng.uniform(x_min - margin, x_max + margin)
        y = rng.uniform(y_min - margin, y_max + margin)
        z = rng.uniform(z_min - margin, z_max + margin)
        
        p = Point3D(x, y, z)
        if not domain.contains(p):
            points.append(p)
            if len(points) >= n_points:
                break
    
    return points


class TestContainsSignedDistanceAgreement:
    """B1: contains and signed_distance sign agreement."""
    
    @pytest.mark.parametrize("name,factory", ALL_DOMAIN_FACTORIES)
    def test_inside_points_have_negative_signed_distance(self, name, factory):
        """Test that points inside the domain have signed_distance <= 0."""
        domain = factory()
        inside_points = sample_inside_points(domain, n_points=20)
        
        if len(inside_points) == 0:
            pytest.skip(f"Could not sample inside points for {name}")
        
        tolerance = 1e-9
        
        for p in inside_points:
            sd = domain.signed_distance(p)
            assert sd <= tolerance, (
                f"{name}: Point {p} is inside (contains=True) but signed_distance={sd} > 0"
            )
    
    @pytest.mark.parametrize("name,factory", ALL_DOMAIN_FACTORIES)
    def test_outside_points_have_positive_signed_distance(self, name, factory):
        """Test that points outside the domain have signed_distance > 0."""
        domain = factory()
        outside_points = sample_outside_points(domain, n_points=20)
        
        if len(outside_points) == 0:
            pytest.skip(f"Could not sample outside points for {name}")
        
        tolerance = 1e-9
        
        for p in outside_points:
            sd = domain.signed_distance(p)
            assert sd >= -tolerance, (
                f"{name}: Point {p} is outside (contains=False) but signed_distance={sd} < 0"
            )
    
    @pytest.mark.parametrize("name,factory", ALL_DOMAIN_FACTORIES)
    def test_contains_signed_distance_consistency(self, name, factory):
        """Test that contains and signed_distance are consistent."""
        domain = factory()
        
        bounds = domain.get_bounds()
        x_min, x_max = bounds[0], bounds[1]
        y_min, y_max = bounds[2], bounds[3]
        z_min, z_max = bounds[4], bounds[5]
        
        margin = max(x_max - x_min, y_max - y_min, z_max - z_min) * 0.3
        
        rng = np.random.default_rng(42)
        
        tolerance = 1e-8
        
        for _ in range(50):
            x = rng.uniform(x_min - margin, x_max + margin)
            y = rng.uniform(y_min - margin, y_max + margin)
            z = rng.uniform(z_min - margin, z_max + margin)
            
            p = Point3D(x, y, z)
            is_inside = domain.contains(p)
            sd = domain.signed_distance(p)
            
            if is_inside:
                assert sd <= tolerance, (
                    f"{name}: contains=True but signed_distance={sd} > 0 at {p}"
                )
            else:
                assert sd >= -tolerance, (
                    f"{name}: contains=False but signed_distance={sd} < 0 at {p}"
                )


class TestProjectInsideCorrectness:
    """B2: project_inside correctness."""
    
    @pytest.mark.parametrize("name,factory", ALL_DOMAIN_FACTORIES)
    def test_projected_point_is_inside(self, name, factory):
        """Test that projected point is inside or on boundary."""
        domain = factory()
        outside_points = sample_outside_points(domain, n_points=20)
        
        if len(outside_points) == 0:
            pytest.skip(f"Could not sample outside points for {name}")
        
        for p in outside_points:
            projected = domain.project_inside(p)
            
            sd = domain.signed_distance(projected)
            tolerance = 1e-6
            
            assert sd <= tolerance, (
                f"{name}: Projected point {projected} has signed_distance={sd} > 0 "
                f"(original point: {p})"
            )
    
    @pytest.mark.parametrize("name,factory", ALL_DOMAIN_FACTORIES)
    def test_inside_points_unchanged(self, name, factory):
        """Test that points already inside are returned unchanged or very close."""
        domain = factory()
        inside_points = sample_inside_points(domain, n_points=20)
        
        if len(inside_points) == 0:
            pytest.skip(f"Could not sample inside points for {name}")
        
        for p in inside_points:
            projected = domain.project_inside(p)
            
            dist = np.sqrt(
                (p.x - projected.x)**2 +
                (p.y - projected.y)**2 +
                (p.z - projected.z)**2
            )
            
            bounds = domain.get_bounds()
            domain_scale = max(
                bounds[1] - bounds[0],
                bounds[3] - bounds[2],
                bounds[5] - bounds[4],
            )
            tolerance = domain_scale * 0.01
            
            assert dist <= tolerance, (
                f"{name}: Inside point {p} moved to {projected} (dist={dist})"
            )
    
    @pytest.mark.parametrize("name,factory", ALL_DOMAIN_FACTORIES)
    def test_projected_point_near_boundary(self, name, factory):
        """Test that projected point is near the boundary."""
        domain = factory()
        outside_points = sample_outside_points(domain, n_points=10)
        
        if len(outside_points) == 0:
            pytest.skip(f"Could not sample outside points for {name}")
        
        for p in outside_points:
            projected = domain.project_inside(p)
            
            sd = abs(domain.signed_distance(projected))
            
            bounds = domain.get_bounds()
            domain_scale = max(
                bounds[1] - bounds[0],
                bounds[3] - bounds[2],
                bounds[5] - bounds[4],
            )
            tolerance = domain_scale * 0.05
            
            assert sd <= tolerance, (
                f"{name}: Projected point {projected} is far from boundary "
                f"(signed_distance={sd}, tolerance={tolerance})"
            )


class TestFacesContract:
    """B3: faces contract for domains that support faces."""
    
    CANONICAL_FACES = ["top", "bottom", "+x", "-x", "+y", "-y", "+z", "-z"]
    
    @pytest.mark.parametrize("name,factory", DOMAINS_WITH_FACES)
    def test_face_frame_returns_valid_structure(self, name, factory):
        """Test that get_face_frame returns valid structure."""
        domain = factory()
        
        if not hasattr(domain, 'get_face_frame'):
            pytest.skip(f"{name} does not have get_face_frame method")
        
        for face in ["top", "bottom"]:
            try:
                frame = domain.get_face_frame(face)
            except (ValueError, KeyError):
                continue
            
            assert "origin" in frame, f"{name}: face_frame missing 'origin'"
            assert "normal" in frame, f"{name}: face_frame missing 'normal'"
            assert "u" in frame, f"{name}: face_frame missing 'u'"
            assert "v" in frame, f"{name}: face_frame missing 'v'"
    
    @pytest.mark.parametrize("name,factory", DOMAINS_WITH_FACES)
    def test_face_normal_is_unit_vector(self, name, factory):
        """Test that face normals are unit vectors."""
        domain = factory()
        
        if not hasattr(domain, 'get_face_frame'):
            pytest.skip(f"{name} does not have get_face_frame method")
        
        for face in ["top", "bottom"]:
            try:
                frame = domain.get_face_frame(face)
            except (ValueError, KeyError):
                continue
            
            normal = np.array(frame["normal"])
            norm = np.linalg.norm(normal)
            
            assert abs(norm - 1.0) < 1e-6, (
                f"{name}: face '{face}' normal is not unit vector (norm={norm})"
            )
    
    @pytest.mark.parametrize("name,factory", DOMAINS_WITH_FACES)
    def test_face_uv_orthonormal(self, name, factory):
        """Test that u,v axes are orthonormal and perpendicular to normal."""
        domain = factory()
        
        if not hasattr(domain, 'get_face_frame'):
            pytest.skip(f"{name} does not have get_face_frame method")
        
        for face in ["top", "bottom"]:
            try:
                frame = domain.get_face_frame(face)
            except (ValueError, KeyError):
                continue
            
            normal = np.array(frame["normal"])
            u = np.array(frame["u"])
            v = np.array(frame["v"])
            
            u_norm = np.linalg.norm(u)
            v_norm = np.linalg.norm(v)
            assert abs(u_norm - 1.0) < 1e-6, (
                f"{name}: face '{face}' u is not unit vector (norm={u_norm})"
            )
            assert abs(v_norm - 1.0) < 1e-6, (
                f"{name}: face '{face}' v is not unit vector (norm={v_norm})"
            )
            
            u_dot_v = np.dot(u, v)
            assert abs(u_dot_v) < 1e-6, (
                f"{name}: face '{face}' u and v are not orthogonal (dot={u_dot_v})"
            )
            
            u_dot_n = np.dot(u, normal)
            v_dot_n = np.dot(v, normal)
            assert abs(u_dot_n) < 1e-6, (
                f"{name}: face '{face}' u is not perpendicular to normal (dot={u_dot_n})"
            )
            assert abs(v_dot_n) < 1e-6, (
                f"{name}: face '{face}' v is not perpendicular to normal (dot={v_dot_n})"
            )


class TestDomainBoundsContract:
    """Test that get_bounds returns valid bounding boxes."""
    
    @pytest.mark.parametrize("name,factory", ALL_DOMAIN_FACTORIES)
    def test_bounds_are_finite(self, name, factory):
        """Test that bounding box values are finite."""
        domain = factory()
        bounds = domain.get_bounds()
        
        assert len(bounds) == 6, f"{name}: bounds should have 6 elements"
        
        for i, val in enumerate(bounds):
            assert np.isfinite(val), f"{name}: bounds[{i}] is not finite: {val}"
    
    @pytest.mark.parametrize("name,factory", ALL_DOMAIN_FACTORIES)
    def test_bounds_have_positive_volume(self, name, factory):
        """Test that bounding box has positive volume."""
        domain = factory()
        bounds = domain.get_bounds()
        
        x_min, x_max = bounds[0], bounds[1]
        y_min, y_max = bounds[2], bounds[3]
        z_min, z_max = bounds[4], bounds[5]
        
        assert x_max > x_min, f"{name}: x_max ({x_max}) <= x_min ({x_min})"
        assert y_max > y_min, f"{name}: y_max ({y_max}) <= y_min ({y_min})"
        assert z_max > z_min, f"{name}: z_max ({z_max}) <= z_min ({z_min})"
        
        volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
        assert volume > 0, f"{name}: bounding box volume is not positive: {volume}"
    
    @pytest.mark.parametrize("name,factory", ALL_DOMAIN_FACTORIES)
    def test_inside_points_within_bounds(self, name, factory):
        """Test that inside points are within bounding box."""
        domain = factory()
        bounds = domain.get_bounds()
        inside_points = sample_inside_points(domain, n_points=20)
        
        if len(inside_points) == 0:
            pytest.skip(f"Could not sample inside points for {name}")
        
        x_min, x_max = bounds[0], bounds[1]
        y_min, y_max = bounds[2], bounds[3]
        z_min, z_max = bounds[4], bounds[5]
        
        for p in inside_points:
            assert x_min <= p.x <= x_max, f"{name}: point.x={p.x} outside bounds [{x_min}, {x_max}]"
            assert y_min <= p.y <= y_max, f"{name}: point.y={p.y} outside bounds [{y_min}, {y_max}]"
            assert z_min <= p.z <= z_max, f"{name}: point.z={p.z} outside bounds [{z_min}, {z_max}]"
