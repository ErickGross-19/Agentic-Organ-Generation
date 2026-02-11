"""Unit tests for anti-starburst branching enforcement."""

import numpy as np
import pytest
from generation.ops.anti_starburst import (
    compute_tissue_depths,
    get_visible_tissue_mask,
    enforce_max_initial_branches,
    select_growth_direction_with_branching_quota,
    check_generation_requirements,
    compute_growth_direction_exploration,
)
from generation.ops.odc_params import ODCParams
from generation.core.domain import BoxDomain


class TestComputeTissueDepths:
    def test_empty_points(self):
        domain = BoxDomain(x_min=-0.01, x_max=0.01, y_min=-0.01, y_max=0.01, z_min=-0.01, z_max=0.01)
        depths = compute_tissue_depths(np.empty((0, 3)), domain)
        assert len(depths) == 0

    def test_depth_range(self):
        domain = BoxDomain(x_min=-0.01, x_max=0.01, y_min=-0.01, y_max=0.01, z_min=-0.01, z_max=0.01)
        points = np.array([
            [0.0, 0.0, 0.01],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -0.01],
        ])
        depths = compute_tissue_depths(points, domain)
        assert len(depths) == 3
        assert all(0.0 <= d <= 1.0 for d in depths)

    def test_depth_ordering(self):
        domain = BoxDomain(x_min=-0.01, x_max=0.01, y_min=-0.01, y_max=0.01, z_min=-0.01, z_max=0.01)
        points = np.array([
            [0.0, 0.0, 0.01],
            [0.0, 0.0, -0.01],
        ])
        depths = compute_tissue_depths(points, domain, depth_axis=2)
        assert depths[0] < depths[1]

    def test_custom_depth_axis(self):
        domain = BoxDomain(x_min=-0.01, x_max=0.01, y_min=-0.01, y_max=0.01, z_min=-0.01, z_max=0.01)
        points = np.array([[0.01, 0.0, 0.0], [-0.01, 0.0, 0.0]])
        depths_x = compute_tissue_depths(points, domain, depth_axis=0)
        assert len(depths_x) == 2


class TestComputeGrowthDirectionExploration:
    def test_exploration_direction_unit_length(self):
        tip_pos = np.array([0.0, 0.0, 0.0])
        nearby = np.array([[0.0, 0.0, -0.01]])
        siblings = [np.array([0.005, 0.0, 0.0])]
        params = ODCParams()
        direction = compute_growth_direction_exploration(tip_pos, nearby, siblings, params)
        norm = np.linalg.norm(direction)
        assert abs(norm - 1.0) < 1e-6 or norm == 0.0

    def test_no_siblings(self):
        tip_pos = np.array([0.0, 0.0, 0.0])
        nearby = np.array([[0.0, 0.0, -0.01]])
        params = ODCParams()
        direction = compute_growth_direction_exploration(tip_pos, nearby, [], params)
        assert len(direction) == 3

    def test_no_attractors(self):
        tip_pos = np.array([0.0, 0.0, 0.0])
        siblings = [np.array([0.005, 0.0, 0.0])]
        params = ODCParams()
        direction = compute_growth_direction_exploration(tip_pos, np.empty((0, 3)), siblings, params)
        assert len(direction) == 3

    def test_preferred_direction_bias(self):
        tip_pos = np.array([0.0, 0.0, 0.0])
        nearby = np.array([[0.0, 0.0, -0.01]])
        params = ODCParams(
            preferred_direction=(0.0, 0.0, -1.0),
            directional_bias=0.9,
        )
        direction = compute_growth_direction_exploration(tip_pos, nearby, [], params)
        assert direction[2] < 0
