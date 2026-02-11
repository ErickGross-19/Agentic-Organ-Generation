"""Unit tests for flexible tissue distributions."""

import numpy as np
import pytest
from generation.tissue.distributions import TissueDistributionSpec
from generation.core.domain import BoxDomain


@pytest.fixture
def box_domain():
    return BoxDomain(
        x_min=-0.005, x_max=0.005,
        y_min=-0.005, y_max=0.005,
        z_min=-0.005, z_max=0.005,
    )


class TestUniformDistribution:
    def test_generates_correct_count(self, box_domain):
        spec = TissueDistributionSpec(distribution_type="uniform", n_points=50, seed=42)
        pts = spec.generate(box_domain)
        assert pts.shape[0] == 50
        assert pts.shape[1] == 3

    def test_points_in_domain(self, box_domain):
        spec = TissueDistributionSpec(distribution_type="uniform", n_points=50, seed=42)
        pts = spec.generate(box_domain)
        for pt in pts:
            assert box_domain.contains(
                __import__("generation.core.types", fromlist=["Point3D"]).Point3D(*pt)
            )

    def test_reproducibility(self, box_domain):
        spec1 = TissueDistributionSpec(distribution_type="uniform", n_points=20, seed=42)
        spec2 = TissueDistributionSpec(distribution_type="uniform", n_points=20, seed=42)
        pts1 = spec1.generate(box_domain)
        pts2 = spec2.generate(box_domain)
        np.testing.assert_array_equal(pts1, pts2)


class TestGridDistribution:
    def test_generates_grid(self, box_domain):
        spec = TissueDistributionSpec(
            distribution_type="grid",
            n_points=100,
            grid_spacing=0.002,
            seed=42,
        )
        pts = spec.generate(box_domain)
        assert pts.shape[0] > 0
        assert pts.shape[1] == 3


class TestGaussianDistribution:
    def test_gaussian_single_center(self, box_domain):
        spec = TissueDistributionSpec(
            distribution_type="gaussian",
            n_points=50,
            gaussian_centers=[(0.0, 0.0, 0.0)],
            gaussian_sigmas=[(0.002, 0.002, 0.002)],
            seed=42,
        )
        pts = spec.generate(box_domain)
        assert pts.shape[0] > 0

    def test_gaussian_multi_center(self, box_domain):
        spec = TissueDistributionSpec(
            distribution_type="gaussian",
            n_points=50,
            gaussian_centers=[(0.002, 0.0, 0.0), (-0.002, 0.0, 0.0)],
            gaussian_sigmas=[(0.001, 0.001, 0.001), (0.001, 0.001, 0.001)],
            gaussian_weights=[0.5, 0.5],
            seed=42,
        )
        pts = spec.generate(box_domain)
        assert pts.shape[0] > 0


class TestRadialDistribution:
    def test_radial_uniform(self, box_domain):
        spec = TissueDistributionSpec(
            distribution_type="radial",
            n_points=50,
            radial_profile="uniform",
            seed=42,
        )
        pts = spec.generate(box_domain)
        assert pts.shape[0] > 0


class TestDepthBiasedDistribution:
    def test_power_depth(self, box_domain):
        spec = TissueDistributionSpec(
            distribution_type="depth_biased",
            n_points=50,
            depth_power=2.0,
            seed=42,
        )
        pts = spec.generate(box_domain)
        assert pts.shape[0] > 0

    def test_beta_depth(self, box_domain):
        spec = TissueDistributionSpec(
            distribution_type="depth_biased",
            n_points=50,
            depth_distribution="beta",
            depth_beta_params=(2.0, 5.0),
            seed=42,
        )
        pts = spec.generate(box_domain)
        assert pts.shape[0] > 0


class TestShellDistribution:
    def test_shell(self, box_domain):
        spec = TissueDistributionSpec(
            distribution_type="shell",
            n_points=50,
            inner_radius=0.002,
            outer_radius=0.004,
            seed=42,
        )
        pts = spec.generate(box_domain)
        assert pts.shape[0] > 0


class TestCylindricalDistribution:
    def test_cylindrical(self, box_domain):
        spec = TissueDistributionSpec(
            distribution_type="cylindrical",
            n_points=50,
            outer_radius=0.003,
            height=0.006,
            seed=42,
        )
        pts = spec.generate(box_domain)
        assert pts.shape[0] > 0


class TestMixtureDistribution:
    def test_mixture(self, box_domain):
        sub1 = TissueDistributionSpec(distribution_type="uniform", n_points=25, seed=42)
        sub2 = TissueDistributionSpec(distribution_type="gaussian", n_points=25, seed=43,
                                      gaussian_centers=[(0.0, 0.0, 0.0)],
                                      gaussian_sigmas=[(0.002, 0.002, 0.002)])
        spec = TissueDistributionSpec(
            distribution_type="mixture",
            n_points=50,
            mixture_specs=[sub1, sub2],
            mixture_weights=[0.5, 0.5],
            seed=44,
        )
        pts = spec.generate(box_domain)
        assert pts.shape[0] > 0


class TestDistributionSerialization:
    def test_to_dict_basic(self):
        spec = TissueDistributionSpec(distribution_type="uniform", n_points=100, seed=42)
        d = spec.to_dict()
        assert d["distribution_type"] == "uniform"
        assert d["n_points"] == 100
        assert d["seed"] == 42

    def test_from_dict(self):
        d = {"distribution_type": "uniform", "n_points": 100, "seed": 42}
        spec = TissueDistributionSpec.from_dict(d)
        assert spec.distribution_type == "uniform"
        assert spec.n_points == 100
        assert spec.seed == 42

    def test_unknown_distribution_raises(self, box_domain):
        spec = TissueDistributionSpec(distribution_type="nonexistent", n_points=10)
        with pytest.raises(ValueError, match="Unknown distribution type"):
            spec.generate(box_domain)
