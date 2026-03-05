"""
Unit tests for space colonization multi-inlet modes.

Tests the deprecation of forest_with_merge and the partitioned_xy parameter changes.
"""

import pytest
import warnings
import numpy as np
from unittest.mock import MagicMock, patch

from generation.backends.space_colonization_backend import (
    SpaceColonizationBackend,
    SpaceColonizationConfig,
)
from generation.core.domain import BoxDomain


def _can_import_generate_api():
    """Check if the generate API can be imported (requires optional dependencies)."""
    try:
        from generation.api.generate import _generate_space_colonization
        return True
    except ImportError:
        return False


class TestForestWithMergeDeprecation:
    """Test that forest_with_merge is deprecated and routes to blended."""

    def test_forest_with_merge_emits_deprecation_warning(self):
        """Test that using forest_with_merge emits a DeprecationWarning."""
        backend = SpaceColonizationBackend()
        domain = BoxDomain(
            x_min=-0.005, x_max=0.005,
            y_min=-0.005, y_max=0.005,
            z_min=-0.002, z_max=0.002,
        )
        
        config = SpaceColonizationConfig(
            multi_inlet_mode="forest_with_merge",
            max_iterations=1,
            num_attractors=10,
        )
        
        inlets = [
            {"position": [0.0, 0.0, 0.002], "radius": 0.0005},
            {"position": [0.003, 0.0, 0.002], "radius": 0.0005},
        ]
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            backend.generate_multi_inlet(
                domain=domain,
                num_outlets=10,
                inlets=inlets,
                config=config,
                rng_seed=42,
            )
            
            deprecation_warnings = [
                warning for warning in w
                if issubclass(warning.category, DeprecationWarning)
                and "forest_with_merge" in str(warning.message)
            ]
            assert len(deprecation_warnings) >= 1, (
                "Expected at least one DeprecationWarning for forest_with_merge"
            )

    def test_forest_with_merge_calls_blended_not_partitioned(self):
        """Test that forest_with_merge routes to blended, not partitioned."""
        backend = SpaceColonizationBackend()
        domain = BoxDomain(
            x_min=-0.005, x_max=0.005,
            y_min=-0.005, y_max=0.005,
            z_min=-0.002, z_max=0.002,
        )
        
        config = SpaceColonizationConfig(
            multi_inlet_mode="forest_with_merge",
            max_iterations=1,
            num_attractors=10,
        )
        
        inlets = [
            {"position": [0.0, 0.0, 0.002], "radius": 0.0005},
            {"position": [0.003, 0.0, 0.002], "radius": 0.0005},
        ]
        
        blended_called = False
        partitioned_called = False
        
        original_blended = backend._generate_multi_inlet_blended
        original_partitioned = backend._generate_multi_inlet_partitioned
        
        def mock_blended(*args, **kwargs):
            nonlocal blended_called
            blended_called = True
            return original_blended(*args, **kwargs)
        
        def mock_partitioned(*args, **kwargs):
            nonlocal partitioned_called
            partitioned_called = True
            return original_partitioned(*args, **kwargs)
        
        with patch.object(backend, '_generate_multi_inlet_blended', mock_blended):
            with patch.object(backend, '_generate_multi_inlet_partitioned', mock_partitioned):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    backend.generate_multi_inlet(
                        domain=domain,
                        num_outlets=10,
                        inlets=inlets,
                        config=config,
                        rng_seed=42,
                    )
        
        assert blended_called, "Expected _generate_multi_inlet_blended to be called"
        assert not partitioned_called, "Expected _generate_multi_inlet_partitioned NOT to be called"


class TestPartitionedXYParameters:
    """Test that partitioned_xy mode respects config parameters."""

    def test_partitioned_config_defaults(self):
        """Test that partitioned config parameters have correct defaults."""
        config = SpaceColonizationConfig()
        
        assert config.partitioned_directional_bias == 1.0
        assert config.partitioned_max_deviation_deg == 30.0
        assert config.partitioned_cone_angle_deg == 30.0
        assert config.partitioned_cylinder_radius == 0.001  # 1mm in meters

    def test_partitioned_config_custom_values(self):
        """Test that partitioned config accepts custom values."""
        config = SpaceColonizationConfig(
            multi_inlet_mode="partitioned_xy",
            partitioned_directional_bias=0.5,
            partitioned_max_deviation_deg=60.0,
            partitioned_cone_angle_deg=80.0,
            partitioned_cylinder_radius=0.004,
        )
        
        assert config.partitioned_directional_bias == 0.5
        assert config.partitioned_max_deviation_deg == 60.0
        assert config.partitioned_cone_angle_deg == 80.0
        assert config.partitioned_cylinder_radius == 0.004

    def test_partitioned_mode_uses_config_parameters(self):
        """Test that partitioned mode uses config parameters instead of hard-coded values."""
        backend = SpaceColonizationBackend()
        domain = BoxDomain(
            x_min=-0.005, x_max=0.005,
            y_min=-0.005, y_max=0.005,
            z_min=-0.002, z_max=0.002,
        )
        
        custom_bias = 0.2
        custom_deviation = 80.0
        custom_cone = 75.0
        custom_cylinder = 0.004
        
        config = SpaceColonizationConfig(
            multi_inlet_mode="partitioned_xy",
            max_iterations=1,
            num_attractors=100,
            partitioned_directional_bias=custom_bias,
            partitioned_max_deviation_deg=custom_deviation,
            partitioned_cone_angle_deg=custom_cone,
            partitioned_cylinder_radius=custom_cylinder,
        )
        
        inlets = [
            {"position": [0.0, 0.0, 0.002], "radius": 0.0005},
            {"position": [0.003, 0.0, 0.002], "radius": 0.0005},
        ]
        
        captured_sc_params = []
        captured_cylinder_radius = []
        captured_cone_angle = []
        
        original_filter_cylinder = backend._filter_tissue_points_by_cylinder
        original_filter_direction = backend._filter_tissue_points_by_direction
        
        def mock_filter_cylinder(tissue_points, inlet_position, direction, cylinder_radius=0.001):
            captured_cylinder_radius.append(cylinder_radius)
            return original_filter_cylinder(tissue_points, inlet_position, direction, cylinder_radius)
        
        def mock_filter_direction(tissue_points, origin, direction, cone_angle_deg=90.0):
            captured_cone_angle.append(cone_angle_deg)
            return original_filter_direction(tissue_points, origin, direction, cone_angle_deg)
        
        with patch.object(backend, '_filter_tissue_points_by_cylinder', mock_filter_cylinder):
            with patch.object(backend, '_filter_tissue_points_by_direction', mock_filter_direction):
                backend.generate_multi_inlet(
                    domain=domain,
                    num_outlets=10,
                    inlets=inlets,
                    config=config,
                    rng_seed=42,
                )
        
        assert len(captured_cylinder_radius) == 2, "Expected cylinder filter to be called for each inlet"
        assert all(r == custom_cylinder for r in captured_cylinder_radius), (
            f"Expected cylinder_radius={custom_cylinder}, got {captured_cylinder_radius}"
        )
        
        assert len(captured_cone_angle) == 2, "Expected direction filter to be called for each inlet"
        assert all(a == custom_cone for a in captured_cone_angle), (
            f"Expected cone_angle_deg={custom_cone}, got {captured_cone_angle}"
        )


class TestCylinderRadiusUnits:
    """Test that cylinder radius is correctly interpreted in meters."""

    def test_filter_tissue_points_by_cylinder_default_is_1mm(self):
        """Test that the default cylinder radius is 0.001 meters (1mm)."""
        backend = SpaceColonizationBackend()
        
        tissue_points = np.array([
            [0.0, 0.0, 0.0],
            [0.0005, 0.0, 0.0],  # 0.5mm from inlet - should be inside
            [0.001, 0.0, 0.0],   # 1mm from inlet - should be on boundary
            [0.002, 0.0, 0.0],   # 2mm from inlet - should be outside
        ])
        inlet_position = np.array([0.0, 0.0, 0.001])
        direction = np.array([0.0, 0.0, -1.0])
        
        filtered = backend._filter_tissue_points_by_cylinder(
            tissue_points, inlet_position, direction
        )
        
        assert len(filtered) == 3, (
            f"Expected 3 points within 1mm cylinder, got {len(filtered)}"
        )

    def test_filter_tissue_points_by_cylinder_custom_radius(self):
        """Test that custom cylinder radius works correctly."""
        backend = SpaceColonizationBackend()
        
        tissue_points = np.array([
            [0.0, 0.0, 0.0],
            [0.001, 0.0, 0.0],   # 1mm from inlet
            [0.002, 0.0, 0.0],   # 2mm from inlet
            [0.003, 0.0, 0.0],   # 3mm from inlet
            [0.004, 0.0, 0.0],   # 4mm from inlet
        ])
        inlet_position = np.array([0.0, 0.0, 0.001])
        direction = np.array([0.0, 0.0, -1.0])
        
        filtered = backend._filter_tissue_points_by_cylinder(
            tissue_points, inlet_position, direction, cylinder_radius=0.003
        )
        
        assert len(filtered) == 4, (
            f"Expected 4 points within 3mm cylinder, got {len(filtered)}"
        )


class TestGenerateAPIDeprecation:
    """Test that the generate API also handles forest_with_merge deprecation."""

    @pytest.mark.skipif(
        not _can_import_generate_api(),
        reason="Requires optional dependencies (scikit-image) for full API import"
    )
    def test_generate_space_colonization_deprecates_forest_with_merge(self):
        """Test that _generate_space_colonization handles forest_with_merge deprecation."""
        from generation.api.generate import _generate_space_colonization
        from aog_policies import GrowthPolicy
        from generation.core.domain import BoxDomain
        
        domain = BoxDomain(
            x_min=-0.005, x_max=0.005,
            y_min=-0.005, y_max=0.005,
            z_min=-0.002, z_max=0.002,
        )
        
        growth_policy = GrowthPolicy(
            backend="space_colonization",
            max_iterations=1,
            backend_params={
                "multi_inlet_mode": "forest_with_merge",
                "num_attraction_points": 10,
            },
        )
        
        ports = {
            "inlets": [
                {"position": [0.0, 0.0, 0.002], "radius": 0.0005, "direction": [0, 0, -1]},
                {"position": [0.003, 0.0, 0.002], "radius": 0.0005, "direction": [0, 0, -1]},
            ],
            "outlets": [],
        }
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _generate_space_colonization(
                domain=domain,
                ports=ports,
                growth_policy=growth_policy,
                collision_policy=None,
                seed=42,
            )
            
            deprecation_warnings = [
                warning for warning in w
                if issubclass(warning.category, DeprecationWarning)
                and "forest_with_merge" in str(warning.message)
            ]
            assert len(deprecation_warnings) >= 1, (
                "Expected at least one DeprecationWarning for forest_with_merge in generate API"
            )
